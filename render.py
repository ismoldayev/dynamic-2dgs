"""
Example command:
python render.py --model_path ./checkpoints/your_video/iter30000_pts5000_totalFrames20_batchFrames20_trajbspline_ctrl5_rigidN5L1e-3_polyDeg7_opacPolyDeg7_xyzCoeffReg1e-3_cholCoeffReg1e-3_opacCoeffReg1e-3_optadan_ema9.99e-1_targetPixels65536_lr1.0e-3/gaussian_model_iter_30000.pth.tar --output_path ./rendered_video.mp4 --num_frames 100 --effect none
"""

import torch
import argparse
from pathlib import Path
from gaussianimage_cholesky import GaussianImage_Cholesky
from gsplat.gsplat.project_gaussians_2d import project_gaussians_2d
from gsplat.gsplat.rasterize_sum import rasterize_gaussians_sum
import yaml
import numpy as np
import cv2
import torchvision.transforms as transforms
from PIL import Image
import ffmpeg
import os
from tqdm import tqdm
from enum import Enum
from typing import Optional, Dict, Any
import torch.nn.functional as F

class EffectType(Enum):
    NONE = "none"
    SLOW_MOTION = "slow_motion"
    FAST_MOTION = "fast_motion"
    REVERSE = "reverse"
    LOOP = "loop"
    # Add more effects here as we implement them

class VisualizationType(Enum):
    REGULAR = "regular"  # Normal video reconstruction
    CENTERS = "centers"  # Just Gaussian centers
    INIT_RIGHT = "init_right"  # Full Gaussians but only those starting on right half

class Effect:
    """Base class for video effects."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {}

    def apply(self, frame: np.ndarray, frame_idx: int, total_frames: int) -> np.ndarray:
        """Apply the effect to a single frame.
        Args:
            frame: The input frame as a numpy array (H, W, C)
            frame_idx: Current frame index
            total_frames: Total number of frames
        Returns:
            The modified frame
        """
        return frame

class NoEffect(Effect):
    """No effect applied."""
    pass

class SlowMotionEffect(Effect):
    """Slow motion effect by interpolating frames."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.slow_factor = self.params.get('slow_factor', 2.0)  # How many times slower

    def apply(self, frame: np.ndarray, frame_idx: int, total_frames: int) -> np.ndarray:
        # For now, just return the frame as is
        # We'll implement actual slow motion interpolation later
        return frame

class FastMotionEffect(Effect):
    """Fast motion effect by skipping frames."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.speed_factor = self.params.get('speed_factor', 2.0)  # How many times faster

    def apply(self, frame: np.ndarray, frame_idx: int, total_frames: int) -> np.ndarray:
        # For now, just return the frame as is
        # We'll implement actual frame skipping later
        return frame

class ReverseEffect(Effect):
    """Reverse the video."""
    def apply(self, frame: np.ndarray, frame_idx: int, total_frames: int) -> np.ndarray:
        # For now, just return the frame as is
        # We'll implement actual reversal later
        return frame

class LoopEffect(Effect):
    """Loop the video with a smooth transition."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.loop_count = self.params.get('loop_count', 1)  # Number of times to loop

    def apply(self, frame: np.ndarray, frame_idx: int, total_frames: int) -> np.ndarray:
        # For now, just return the frame as is
        # We'll implement actual looping with transitions later
        return frame

def create_effect(effect_type: EffectType, params: Optional[Dict[str, Any]] = None) -> Effect:
    """Factory function to create effect instances."""
    effect_map = {
        EffectType.NONE: NoEffect,
        EffectType.SLOW_MOTION: SlowMotionEffect,
        EffectType.FAST_MOTION: FastMotionEffect,
        EffectType.REVERSE: ReverseEffect,
        EffectType.LOOP: LoopEffect,
    }
    effect_class = effect_map.get(effect_type)
    if effect_class is None:
        raise ValueError(f"Unknown effect type: {effect_type}")
    return effect_class(params)

def load_model_and_args(model_path):
    """Load model checkpoint and its training arguments."""
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')

    # Load args from the checkpoint directory
    args_path = Path(model_path).parent / 'args.yaml'
    with open(args_path, 'r') as f:
        args = yaml.safe_load(f)

    # Create model with loaded args
    model = GaussianImage_Cholesky(
        loss_type=args.get('loss_type', 'L2'),
        T=args.get('T', 1),
        lambda_neighbor_rigidity=args.get('lambda_neighbor_rigidity', 0.0),
        k_neighbors=args.get('k_neighbors', 5),
        ema_decay=args.get('ema_decay', 0.999),
        polynomial_degree=args.get('polynomial_degree', 7),
        lambda_xyz_coeffs_reg=args.get('lambda_xyz_coeffs_reg', 0.0),
        lambda_cholesky_coeffs_reg=args.get('lambda_cholesky_coeffs_reg', 0.0),
        lambda_opacity_coeffs_reg=args.get('lambda_opacity_coeffs_reg', 0.0),
        opacity_polynomial_degree=args.get('opacity_polynomial_degree', None),
        trajectory_model_type=args.get('trajectory_model_type', 'polynomial'),
        num_control_points=args.get('num_control_points', 5),
        num_points=args.get('num_points', 2000),
        H=args.get('H', 256),
        W=args.get('W', 256),
        BLOCK_H=args.get('BLOCK_H', 16),
        BLOCK_W=args.get('BLOCK_W', 16),
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        lr=args.get('lr', 1e-3),
        opt_type=args.get('opt_type', 'adam')
    )

    # Load state dict
    model.load_state_dict(checkpoint)
    model.eval()

    return model, args

def get_right_half_indices(model, t_values):
    """Get indices of Gaussians that start on the right half of the screen."""
    # Get positions in first frame
    if model.trajectory_model_type == "bspline":
        first_frame_positions = model._get_evaluated_bsplines(t_values[:1])[0][0].detach().cpu()
    else:
        first_frame_positions = model.get_xyz_at_time(t_values[:1])[0].detach().cpu()

    # Convert normalized coordinates to pixel coordinates
    x_coords_first_frame = (first_frame_positions[:, 0] * 0.5 + 0.5) * (model.W - 1)
    # Select Gaussians that start on the right half
    right_half_mask = x_coords_first_frame > (model.W / 2)
    right_half_indices = torch.where(right_half_mask)[0]

    return right_half_indices

def visualize_gaussian_centers(model, output_path, num_frames=100, fps=30, dot_radius=1):
    """Visualize Gaussian centers over time."""
    print("\nCreating Gaussian centers visualization...")

    # Get time values for rendering
    t_values = torch.linspace(0, 1, num_frames, device=model.device)

    # Get all xyz positions for the frames
    if model.trajectory_model_type == "bspline":
        all_xyz_normalized = model._get_evaluated_bsplines(t_values)[0].detach().cpu()
    else:
        all_xyz_normalized = model.get_xyz_at_time(t_values).detach().cpu()

    # Get static colors for all Gaussians
    static_colors_rgb_norm = model.get_features.detach().cpu().numpy()
    static_colors_bgr_uint8 = (static_colors_rgb_norm[:, [2, 1, 0]] * 255).astype(np.uint8)

    # Calculate scaled dimensions to maintain aspect ratio
    target_pixels = 6 * model.num_points  # Adjust target pixels based on number of Gaussians
    aspect_ratio = model.W / model.H
    scaled_H = int(np.sqrt(target_pixels / aspect_ratio))
    scaled_W = int(np.sqrt(target_pixels * aspect_ratio))
    print(f"Scaling video to {scaled_W}x{scaled_H}")

    # Create temporary directory for frames
    temp_frame_dir = Path(output_path).parent / "temp_frames"
    temp_frame_dir.mkdir(parents=True, exist_ok=True)
    temp_frame_paths = []

    # Render each frame
    total_dots_visible = 0
    for k in range(num_frames):
        # Create blank frame with scaled resolution
        frame = np.zeros((scaled_H, scaled_W, 3), dtype=np.uint8)

        # Get xyz positions for this frame
        xyz_k = all_xyz_normalized[k]  # (N, 2)

        # Transform normalized coordinates to pixel coordinates (scaled)
        x_coords = (xyz_k[:, 0] * 0.5 + 0.5) * (scaled_W - 1)
        y_coords = (xyz_k[:, 1] * 0.5 + 0.5) * (scaled_H - 1)

        # Draw each Gaussian center
        dots_visible_in_frame = 0
        for i in range(model.num_points):
            x, y = int(x_coords[i]), int(y_coords[i])
            if 0 <= x < scaled_W and 0 <= y < scaled_H:
                color = tuple(static_colors_bgr_uint8[i].tolist())
                cv2.circle(frame, (x, y), dot_radius, color, -1)
                dots_visible_in_frame += 1

        total_dots_visible += dots_visible_in_frame

        # Save frame as PNG
        frame_path = temp_frame_dir / f"frame_{k:04d}.png"
        cv2.imwrite(str(frame_path), frame)
        temp_frame_paths.append(str(frame_path))

    # Use ffmpeg to create video from frames
    try:
        (
            ffmpeg
            .input(f'{temp_frame_dir}/frame_%04d.png', framerate=fps)
            .output(str(output_path), pix_fmt='yuv420p', vcodec='libx264', r=fps)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        print(f"Successfully created centers visualization video at {output_path}")
    except Exception as e:
        print(f"Failed to create video using ffmpeg: {e}")
        raise

    # Clean up temporary frames
    for path in temp_frame_paths:
        try:
            os.remove(path)
        except:
            pass

    # Print summary statistics
    avg_dots_per_frame = total_dots_visible / num_frames
    print(f"Gaussian centers visualization complete:")
    print(f"Average dots per frame: {avg_dots_per_frame:.2f}")

def render_video(model, output_path, num_frames=100, fps=30, viz_type=VisualizationType.REGULAR):
    """Render a video from the trained model."""
    print(f"\nRendering {num_frames} frames at {fps} FPS...")
    print(f"Visualization type: {viz_type.value}")

    # Get time values for rendering
    t_values = torch.linspace(0, 1, num_frames, device=model.device)

    if viz_type == VisualizationType.INIT_RIGHT:
        # Get indices of Gaussians that start on the right half
        right_half_indices = get_right_half_indices(model, t_values)
        print(f"Found {len(right_half_indices)} Gaussians starting on the right half of the screen")

        # Create a mask for the right-half Gaussians
        gaussian_mask = torch.zeros(model.num_points, dtype=torch.bool, device=model.device)
        gaussian_mask[right_half_indices] = True

        # Render frames using model's forward method with the mask
        with torch.no_grad():
            out_frames_pkg = model.forward(t_values_to_render=t_values, gaussian_mask=gaussian_mask)
            out_frames = out_frames_pkg["render"].float()
    else:
        # Regular rendering
        with torch.no_grad():
            out_frames_pkg = model.forward(t_values_to_render=t_values)
            out_frames = out_frames_pkg["render"].float()

    # Create temporary directory for frames
    temp_frame_dir = Path(output_path).parent / "temp_frames"
    temp_frame_dir.mkdir(parents=True, exist_ok=True)
    temp_frame_paths = []

    # Save frames as temporary PNG files
    for i in range(num_frames):
        frame = out_frames[i].cpu().permute(1, 2, 0)
        frame_np_rgb = (frame * 255).byte().numpy()
        frame_np_bgr = cv2.cvtColor(frame_np_rgb, cv2.COLOR_RGB2BGR)

        frame_path = temp_frame_dir / f"frame_{i:04d}.png"
        cv2.imwrite(str(frame_path), frame_np_bgr)
        temp_frame_paths.append(str(frame_path))

    # Use ffmpeg to create video from frames
    try:
        (
            ffmpeg
            .input(f'{temp_frame_dir}/frame_%04d.png', framerate=fps)
            .output(str(output_path), pix_fmt='yuv420p', vcodec='libx264', r=fps)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        print(f"Successfully created video at {output_path}")
    except Exception as e:
        print(f"Failed to create video using ffmpeg: {e}")
        raise

    # Clean up temporary frames
    for path in temp_frame_paths:
        try:
            os.remove(path)
        except:
            pass

def main():
    parser = argparse.ArgumentParser(description="Render video from a trained model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--num_frames", type=int, default=100, help="Number of frames to render")
    parser.add_argument("--fps", type=float, default=None, help="FPS for the output video (defaults to original video FPS)")
    parser.add_argument("--viz_type", type=str, choices=[v.value for v in VisualizationType],
                       default=VisualizationType.REGULAR.value,
                       help="Type of visualization to generate")
    parser.add_argument("--dot_radius", type=int, default=1, help="Radius of dots in centers visualization")

    args = parser.parse_args()

    # Load model and args
    model, model_args = load_model_and_args(args.model_path)

    # Determine FPS
    if args.fps is None:
        # Try to get original FPS from model args
        fps = model_args.get('output_fps') or model_args.get('input_video_fps')
        if fps is None:
            fps = 30.0  # Default fallback
            print(f"Using default FPS: {fps}")
        else:
            print(f"Using original video FPS: {fps}")
    else:
        fps = args.fps
        print(f"Using user-specified FPS: {fps}")

    # Determine visualization type
    viz_type = VisualizationType(args.viz_type)

    # Create output path in the same directory as the model
    model_dir = Path(args.model_path).parent
    model_name = Path(args.model_path).stem  # Get just the filename without extension
    output_filename = f"{model_name}_rendered_{viz_type.value}.mp4"
    output_path = model_dir / output_filename

    if viz_type == VisualizationType.CENTERS:
        # For centers visualization, use the dedicated function
        visualize_gaussian_centers(model, output_path, args.num_frames, fps, args.dot_radius)
    else:
        # For regular and init_right visualizations, use render_video
        render_video(model, output_path, args.num_frames, fps, viz_type)

if __name__ == "__main__":
    main()
