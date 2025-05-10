import torch
import argparse
from pathlib import Path
from gaussianimage_cholesky import GaussianImage_Cholesky
import yaml

def load_model_and_args(model_path):
    """Load model and its arguments from a checkpoint directory."""
    model_dir = Path(model_path).parent
    args_path = model_dir / "args.yaml"

    # Load arguments
    with open(args_path, 'r') as f:
        args_dict = yaml.safe_load(f)

    # Create a simple namespace object to hold arguments
    class Args:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    args = Args(**args_dict)

    # Initialize model with arguments
    model = GaussianImage_Cholesky(
        loss_type=args.loss_type if hasattr(args, 'loss_type') else "L2",
        T=args.num_frames if hasattr(args, 'num_frames') else 10,
        num_points=args.num_points,
        H=args.H if hasattr(args, 'H') else 256,
        W=args.W if hasattr(args, 'W') else 256,
        BLOCK_H=16,
        BLOCK_W=16,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        lr=args.lr,
        opt_type=args.opt_type,
        lambda_neighbor_rigidity=args.lambda_neighbor_rigidity if hasattr(args, 'lambda_neighbor_rigidity') else 0.0,
        k_neighbors=args.k_neighbors if hasattr(args, 'k_neighbors') else 5,
        polynomial_degree=args.polynomial_degree if hasattr(args, 'polynomial_degree') else 7,
        lambda_xyz_coeffs_reg=args.lambda_xyz_coeffs_reg if hasattr(args, 'lambda_xyz_coeffs_reg') else 0.0,
        lambda_cholesky_coeffs_reg=args.lambda_cholesky_coeffs_reg if hasattr(args, 'lambda_cholesky_coeffs_reg') else 0.0,
        lambda_opacity_coeffs_reg=args.lambda_opacity_coeffs_reg if hasattr(args, 'lambda_opacity_coeffs_reg') else 0.0,
        ema_decay=args.ema_decay if hasattr(args, 'ema_decay') else 0.999,
        opacity_polynomial_degree=args.opacity_polynomial_degree if hasattr(args, 'opacity_polynomial_degree') else None,
        trajectory_model_type=args.trajectory_model_type if hasattr(args, 'trajectory_model_type') else "polynomial",
        num_control_points=args.num_control_points if hasattr(args, 'num_control_points') else 5
    )

    # Load model weights
    checkpoint = torch.load(model_path, map_location=model.device)

    # Handle trajectory model type mismatch
    if model.trajectory_model_type == "bspline" and "_xyz_coeffs" in checkpoint:
        print("Converting polynomial trajectory parameters to B-spline control points...")
        # Get the polynomial coefficients
        xyz_coeffs = checkpoint["_xyz_coeffs"]
        cholesky_coeffs = checkpoint["_cholesky_coeffs"]

        # Create time points for evaluation
        t_points = torch.linspace(0, 1, model.K_control_points, device=model.device)

        # Evaluate polynomial at control points
        t_power_matrix = torch.stack([t_points.pow(i) for i in range(model.polynomial_degree + 1)], dim=1)
        xyz_control_points = torch.einsum('dnp,td->tnp', xyz_coeffs, t_power_matrix)
        cholesky_control_points = torch.einsum('dnp,td->tnp', cholesky_coeffs, t_power_matrix)

        # Update checkpoint with new parameters
        checkpoint["_xyz_control_points"] = xyz_control_points.permute(1, 0, 2)  # (N, K, 2)
        checkpoint["_cholesky_control_points"] = cholesky_control_points.permute(1, 0, 2)  # (N, K, 3)

        # Remove old parameters
        del checkpoint["_xyz_coeffs"]
        del checkpoint["_cholesky_coeffs"]

    elif model.trajectory_model_type == "polynomial" and "_xyz_control_points" in checkpoint:
        print("Converting B-spline control points to polynomial coefficients...")
        # Get the control points
        xyz_control_points = checkpoint["_xyz_control_points"]  # (N, K, 2)
        cholesky_control_points = checkpoint["_cholesky_control_points"]  # (N, K, 3)

        # Create time points for evaluation
        t_points = torch.linspace(0, 1, model.K_control_points, device=model.device)

        # Fit polynomial to control points
        t_power_matrix = torch.stack([t_points.pow(i) for i in range(model.polynomial_degree + 1)], dim=1)
        t_power_matrix_pinv = torch.linalg.pinv(t_power_matrix)

        # Convert to polynomial coefficients
        xyz_coeffs = torch.einsum('ntk,kn->tnk', xyz_control_points, t_power_matrix_pinv)
        cholesky_coeffs = torch.einsum('ntk,kn->tnk', cholesky_control_points, t_power_matrix_pinv)

        # Update checkpoint with new parameters
        checkpoint["_xyz_coeffs"] = xyz_coeffs.permute(1, 0, 2)  # (D+1, N, 2)
        checkpoint["_cholesky_coeffs"] = cholesky_coeffs.permute(1, 0, 2)  # (D+1, N, 3)

        # Remove old parameters
        del checkpoint["_xyz_control_points"]
        del checkpoint["_cholesky_control_points"]

    # Load the modified state dict
    model.load_state_dict(checkpoint)
    model.eval()

    return model, args

def main():
    parser = argparse.ArgumentParser(description="Cluster Gaussians from a trained model.")
    parser.add_argument("--model_path", type=str, required=True,
                      help="Path to the model checkpoint (.pth.tar file)")
    parser.add_argument("--n_clusters", type=int, default=5,
                      help="Number of clusters to create")
    parser.add_argument("--motion_weight", type=float, default=0.6,
                      help="Weight for motion features (0-1)")
    parser.add_argument("--spatial_weight", type=float, default=0.3,
                      help="Weight for spatial features (0-1)")
    parser.add_argument("--color_weight", type=float, default=0.1,
                      help="Weight for color features (0-1)")
    parser.add_argument("--show_velocity", action="store_true",
                      help="Show velocity vectors in visualization")
    parser.add_argument("--frame_idx", type=int, default=0,
                      help="Frame index to visualize")

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model_path}")
    model, model_args = load_model_and_args(args.model_path)

    # Create output directory
    output_dir = Path(args.model_path).parent / "clustering_results"
    output_dir.mkdir(exist_ok=True)

    # Perform clustering
    print("Performing clustering...")
    cluster_results = model.cluster_gaussians(
        n_clusters=args.n_clusters,
        motion_weight=args.motion_weight,
        spatial_weight=args.spatial_weight,
        color_weight=args.color_weight
    )

    # Visualize clusters
    print("Visualizing clusters...")
    save_path = output_dir / f"clusters_frame{args.frame_idx}_n{args.n_clusters}_m{args.motion_weight:.1f}_s{args.spatial_weight:.1f}_c{args.color_weight:.1f}.png"
    model.visualize_clusters(
        cluster_results,
        frame_idx=args.frame_idx,
        save_path=save_path,
        show_velocity=args.show_velocity
    )

    print(f"Results saved to {save_path}")

if __name__ == "__main__":
    main()
