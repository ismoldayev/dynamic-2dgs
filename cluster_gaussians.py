import torch
import argparse
from pathlib import Path
from gaussianimage_cholesky import GaussianImage_Cholesky
import yaml
import numpy as np

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

    # Load state dict with strict=False to handle any mismatches
    try:
        model.load_state_dict(checkpoint, strict=False)
    except Exception as e:
        print(f"Warning: Error loading state dict: {e}")
        print("Attempting to load parameters individually...")
        for key, value in checkpoint.items():
            if hasattr(model, key):
                try:
                    getattr(model, key).data.copy_(value)
                except Exception as e:
                    print(f"Failed to load {key}: {e}")

    # --- FORCE COPY B-SPLINE PARAMETERS IF PRESENT ---
    with torch.no_grad():
        if "_xyz_control_points" in checkpoint:
            if model._xyz_control_points.shape == checkpoint["_xyz_control_points"].shape:
                model._xyz_control_points.copy_(checkpoint["_xyz_control_points"])
            else:
                print(f"Shape mismatch for _xyz_control_points: model {model._xyz_control_points.shape}, checkpoint {checkpoint['_xyz_control_points'].shape}")

        if "_cholesky_control_points" in checkpoint:
            if model._cholesky_control_points.shape == checkpoint["_cholesky_control_points"].shape:
                model._cholesky_control_points.copy_(checkpoint["_cholesky_control_points"])
            else:
                print(f"Shape mismatch for _cholesky_control_points: model {model._cholesky_control_points.shape}, checkpoint {checkpoint['_cholesky_control_points'].shape}")

        if "_opacity_control_points" in checkpoint:
            if model._opacity_control_points.shape == checkpoint["_opacity_control_points"].shape:
                model._opacity_control_points.copy_(checkpoint["_opacity_control_points"])
            else:
                print(f"Shape mismatch for _opacity_control_points: model {model._opacity_control_points.shape}, checkpoint {checkpoint['_opacity_control_points'].shape}")

    # Verify color values after loading
    print("\nColor values after loading checkpoint:")
    with torch.no_grad():
        raw_color_logits = model._features_dc.detach().cpu().numpy()
        final_colors = model.get_features.detach().cpu().numpy()
        print("Raw Color Logits (_features_dc):")
        print(f"  Shape: {raw_color_logits.shape}")
        print(f"  Min: {np.min(raw_color_logits):.4f}, Max: {np.max(raw_color_logits):.4f}, Mean: {np.mean(raw_color_logits):.4f}, Median: {np.median(raw_color_logits):.4f}")
        print("Final Colors (after sigmoid):")
        print(f"  Shape: {final_colors.shape}")
        print(f"  Min: {np.min(final_colors):.4f}, Max: {np.max(final_colors):.4f}, Mean: {np.mean(final_colors):.4f}, Median: {np.median(final_colors):.4f}")
        for i in range(final_colors.shape[1]):
            print(f"  Channel {i} (RGB) - Min: {np.min(final_colors[:, i]):.4f}, Max: {np.max(final_colors[:, i]):.4f}, Mean: {np.mean(final_colors[:, i]):.4f}")

    # Verify opacity values after loading
    print("\nOpacity values after loading checkpoint:")
    with torch.no_grad():
        final_opacities = model.get_opacity
        print(f"Shape: {final_opacities.shape}")
        print(f"Min: {final_opacities.min().item():.4f}")
        print(f"Max: {final_opacities.max().item():.4f}")
        print(f"Mean: {final_opacities.mean().item():.4f}")
        print(f"Median: {final_opacities.median().item():.4f}")

    model.eval()
    return model, args

def parse_args():
    parser = argparse.ArgumentParser(description='Cluster Gaussians from a trained model')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the model checkpoint')
    parser.add_argument('--n_clusters', type=int, default=5,
                      help='Number of clusters to create')
    parser.add_argument('--motion_weight', type=float, default=0.6,
                      help='Weight for motion features in clustering')
    parser.add_argument('--spatial_weight', type=float, default=0.3,
                      help='Weight for spatial features in clustering')
    parser.add_argument('--color_weight', type=float, default=0.1,
                      help='Weight for color features in clustering')
    parser.add_argument('--show_velocity', action='store_true',
                      help='Show velocity vectors in visualization')
    parser.add_argument('--frame_idx', type=int, default=0,
                      help='Frame index to visualize')
    return parser.parse_args()

def main():
    args = parse_args()

    # Load model and arguments
    model, model_args = load_model_and_args(args.model_path)

    # Create output directory
    output_dir = Path(args.model_path).parent / 'clustering_results'
    output_dir.mkdir(exist_ok=True)

    # Perform clustering
    cluster_results = model.cluster_gaussians(
        n_clusters=args.n_clusters,
        motion_weight=args.motion_weight,
        spatial_weight=args.spatial_weight,
        color_weight=args.color_weight
    )

    # Generate filename based on clustering parameters
    param_str = f"n{args.n_clusters}_m{args.motion_weight:.1f}_s{args.spatial_weight:.1f}_c{args.color_weight:.1f}"
    if args.show_velocity:
        param_str += "_vel"

    # Calculate dot size based on image dimensions
    dot_size = max(1, int(min(model.H, model.W) * 0.002))  # 0.002 of minimum image dimension

    # Visualize clusters
    save_path = output_dir / f"clusters_{param_str}_frame{args.frame_idx}.png"
    model.visualize_clusters(
        cluster_results,
        frame_idx=args.frame_idx,
        save_path=save_path,
        show_velocity=args.show_velocity,
        dot_size=dot_size
    )

    print(f"Clustering results saved to {save_path}")

if __name__ == '__main__':
    main()
