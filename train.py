"""
Example command:
CUDA_VISIBLE_DEVICES=0 python train.py --video_path ./dynamic-2dgs/dataset/your_video.mp4 \
--num_frames 20 --num_points 5000 --iterations 30000 --lr 1e-3
"""

import math
import time
from pathlib import Path
import argparse
import yaml
import numpy as np
import torch
import sys
# from PIL import Image # Moved to utils
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from utils import *
from tqdm import tqdm
import random
import torchvision.transforms as transforms
# Import the updated model
from gaussianimage_cholesky import GaussianImage_Cholesky
from optimizer import Adan # Make sure Adan is imported if used in re-init
# Import cv2 for video writing
import cv2

# Renamed Trainer class
class VideoTrainer:
    """Trains dynamic 2d gaussians to fit video frames."""
    def __init__(
        self,
        gt_frames: torch.Tensor, # Expecting (T, C, H, W)
        video_path: Path,
        num_points: int = 2000,
        iterations:int = 30000,
        model_path = None,
        args = None,
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.gt_frames = gt_frames.to(self.device)
        self.T, self.C, self.H, self.W = self.gt_frames.shape
        self.args = args # Store args

        self.num_points = num_points
        self.video_name = video_path.stem
        BLOCK_H, BLOCK_W = 16, 16 # Keep block size fixed for now
        self.iterations = iterations
        self.save_frames = args.save_frames
        # Adjust log directory naming
        opacity_reg_str = f"_opReg{args.lambda_opacity_reg:.0e}".replace("e-0", "e-")
        temporal_xyz_str = f"_tempXYZ{args.lambda_temporal_xyz:.0e}".replace("e-0", "e-")
        temporal_chol_str = f"_tempChol{args.lambda_temporal_cholesky:.0e}".replace("e-0", "e-")
        accel_xyz_str = f"_accelXYZ{args.lambda_accel_xyz:.0e}".replace("e-0", "e-") if args.lambda_accel_xyz > 0 else ""
        accel_chol_str = f"_accelChol{args.lambda_accel_cholesky:.0e}".replace("e-0", "e-") if args.lambda_accel_cholesky > 0 else ""
        rigidity_str = f"_rigidN{args.k_neighbors}L{args.lambda_neighbor_rigidity:.0e}".replace("e-0", "e-") if args.lambda_neighbor_rigidity > 0 else ""
        # color_reg_str = f"_colorReg{args.lambda_color_reg:.0e}".replace("e-0", "e-") # Removed

        # Add more params to log_dir name for better tracking
        self.log_dir = Path(f"./checkpoints/{self.video_name}/iter{args.iterations}_pts{num_points}_frames{self.T}{opacity_reg_str}{temporal_xyz_str}{temporal_chol_str}{accel_xyz_str}{accel_chol_str}{rigidity_str}_lr{args.lr:.0e}")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        print(f"Logging to: {self.log_dir}")

        # Instantiate the updated Gaussian model with T frames
        self.gaussian_model = GaussianImage_Cholesky(
            loss_type="L2",
            T=self.T,
            num_points=self.num_points,
            H=self.H, W=self.W,
            BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
            device=self.device,
            lr=args.lr,
            opt_type=args.opt_type,
            lambda_opacity_reg=args.lambda_opacity_reg,
            lambda_temporal_xyz=args.lambda_temporal_xyz,
            lambda_temporal_cholesky=args.lambda_temporal_cholesky,
            lambda_accel_xyz=args.lambda_accel_xyz,
            lambda_accel_cholesky=args.lambda_accel_cholesky,
            lambda_neighbor_rigidity=args.lambda_neighbor_rigidity,
            k_neighbors=args.k_neighbors,
            # lambda_color_reg=args.lambda_color_reg # Removed
        ).to(self.device)

        self.logwriter = LogWriter(self.log_dir)

        # Store output fps for video writing
        self.output_fps = args.output_fps

        if model_path is not None:
            print(f"Loading model path: {model_path}")
            # Note: Loading checkpoints might require adjustments if the parameter shapes changed (e.g., _xyz, _cholesky)
            # This basic loading might only work if resuming a video training session.
            # For loading a single-image checkpoint, more logic would be needed to initialize time-varying params.
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.gaussian_model.load_state_dict(checkpoint, strict=False)
                print("Model loaded successfully.")
            except Exception as e:
                print(f"Could not load model checkpoint: {e}")
                print("Starting training from scratch.")

    def reinitialize_optimizer_scheduler(self):
        """Re-initializes the optimizer and scheduler for the Gaussian model.
        This is typically called after parameters have been added or removed (pruning/densification).
        """
        print("Re-initializing optimizer and scheduler due to parameter change.")
        current_lr = self.gaussian_model.optimizer.param_groups[0]['lr'] # Preserve current LR

        # Re-initialize optimizer
        if self.args.opt_type == "adam":
            self.gaussian_model.optimizer = torch.optim.Adam(self.gaussian_model.parameters(), lr=current_lr)
        elif self.args.opt_type == "adan": # Ensure Adan is imported in train.py
            self.gaussian_model.optimizer = Adan(self.gaussian_model.parameters(), lr=current_lr)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.args.opt_type}")

        # Re-initialize scheduler with the new optimizer
        # Preserve step_size and gamma from the old scheduler if possible
        old_step_size = self.gaussian_model.scheduler.step_size if hasattr(self.gaussian_model.scheduler, 'step_size') else 20000 # Default from GaussianImage_Cholesky
        old_gamma = self.gaussian_model.scheduler.gamma if hasattr(self.gaussian_model.scheduler, 'gamma') else 0.5 # Default

        self.gaussian_model.scheduler = torch.optim.lr_scheduler.StepLR(
            self.gaussian_model.optimizer,
            step_size=old_step_size,
            gamma=old_gamma
        )
        print(f"Optimizer and scheduler re-initialized with LR: {current_lr:.2e}, Step: {old_step_size}, Gamma: {old_gamma}")

    def train(self):
        # psnr_list will store average PSNR per iteration
        psnr_list, iter_list = [], []
        progress_bar = tqdm(range(1, self.iterations + 1), desc="Training progress")

        self.gaussian_model.train()
        start_time = time.time()
        for iter_idx in range(1, self.iterations + 1):
            # train_iter now handles looping through frames and averaging loss/psnr
            loss, psnr = self.gaussian_model.train_iter(self.gt_frames)

            psnr_list.append(psnr) # Append average PSNR for this iteration
            iter_list.append(iter_idx)

            if iter_idx % 10 == 0:
                progress_bar.set_postfix({f"Avg Loss":f"{loss:.{7}f}", "Avg PSNR":f"{psnr:.{4}f}"})
                progress_bar.update(10)

            # --- Periodic Evaluation and Saving ---
            # Also save model checkpoint here if it's a periodic eval iteration
            if iter_idx % 10000 == 0 and iter_idx != self.iterations: # Avoid double saving on last iteration
                self.evaluate_and_save_output(iter_idx, save_checkpoint=True)
            # --- End Periodic Saving ---

        end_time = time.time() - start_time
        progress_bar.close()

        # Run final evaluation and save final output (including final checkpoint)
        final_psnr, final_ms_ssim = self.evaluate_and_save_output(self.iterations, save_checkpoint=True)

        # Measure rendering speed (average over frames)
        self.gaussian_model.eval()
        with torch.no_grad():
            test_start_time = time.time()
            num_render_tests = min(10, self.T) # Render a few frames or all if T is small
            for i in range(num_render_tests):
                 # Render a single frame
                _ = self.gaussian_model(frame_index=i % self.T)
            test_end_time = (time.time() - test_start_time) / num_render_tests

        self.logwriter.write(f"Training Complete in {end_time:.4f}s")
        self.logwriter.write(f"Final Avg PSNR: {final_psnr:.4f}, Avg MS-SSIM: {final_ms_ssim:.6f}")
        self.logwriter.write(f"Avg Frame Render time: {test_end_time:.8f}s, FPS: {1/test_end_time:.4f}")

        # Save model and training log (This will now be the *final* model save, periodic ones are in evaluate_and_save_output)
        # model_save_path = self.log_dir / "gaussian_model_final.pth.tar"
        # torch.save(self.gaussian_model.state_dict(), model_save_path)
        # print(f"Model saved to {model_save_path}")
        # The above lines are now handled by the last call to evaluate_and_save_output(self.iterations, save_checkpoint=True)

        training_log_path = self.log_dir / "training_log.npy"
        np.save(training_log_path, {
            "iterations": iter_list,
            "avg_training_psnr": psnr_list,
            "training_time": end_time,
            "final_avg_psnr": final_psnr,
            "final_avg_ms_ssim": final_ms_ssim,
            "avg_rendering_time_per_frame": test_end_time,
            "avg_rendering_fps": 1/test_end_time
        })
        print(f"Training log saved to {training_log_path}")

        # Return final average metrics from the last evaluation
        return final_psnr, final_ms_ssim, end_time, test_end_time, 1/test_end_time

    # --- Refactored Evaluation and Saving Logic ---
    def evaluate_and_save_output(self, iteration, save_checkpoint=False):
        """Renders frames, calculates metrics, and saves PNGs/video for a specific iteration.
        Optionally saves a model checkpoint.
        """
        print(f"\n--- Evaluating and Saving Output for Iteration {iteration} ---")
        self.gaussian_model.eval() # Set model to evaluation mode

        total_psnr = 0
        total_ms_ssim = 0
        saved_frame_paths = []

        # --- TEMPORARY COLOR FEATURE ANALYSIS ---
        with torch.no_grad():
            raw_color_logits = self.gaussian_model._features_dc.detach().cpu().numpy()
            final_colors = self.gaussian_model.get_features.detach().cpu().numpy()
            print(f"\n--- Color Feature Analysis at Iteration {iteration} ---")
            if raw_color_logits.size > 0:
                print("Raw Color Logits (_features_dc):")
                print(f"  Shape: {raw_color_logits.shape}")
                print(f"  Min: {np.min(raw_color_logits):.4f}, Max: {np.max(raw_color_logits):.4f}, Mean: {np.mean(raw_color_logits):.4f}, Median: {np.median(raw_color_logits):.4f}")
                print("Final Colors (after sigmoid):")
                print(f"  Shape: {final_colors.shape}")
                print(f"  Min: {np.min(final_colors):.4f}, Max: {np.max(final_colors):.4f}, Mean: {np.mean(final_colors):.4f}, Median: {np.median(final_colors):.4f}")

                # Per-channel analysis for final colors might be useful too
                for i in range(final_colors.shape[1]): # Iterate over R, G, B channels
                    print(f"  Channel {i} (RGB) - Min: {np.min(final_colors[:, i]):.4f}, Max: {np.max(final_colors[:, i]):.4f}, Mean: {np.mean(final_colors[:, i]):.4f}")
            else:
                print("No color features to analyze (num_points might be 0).")
        print("--- End Color Feature Analysis ---\n")
        # --- END TEMPORARY COLOR FEATURE ANALYSIS ---

        with torch.no_grad():
            # Render all frames at once
            out_frames_pkg = self.gaussian_model() # Get dict with "render": (T, C, H, W)
            out_frames = out_frames_pkg["render"].float()

            # Prepare saving locations (if saving)
            if self.save_frames:
                # Create iteration-specific directory for frames
                frame_save_dir = self.log_dir / f"rendered_frames_iter_{iteration}"
                frame_save_dir.mkdir(parents=True, exist_ok=True)
                # Video path for this iteration
                video_save_path = self.log_dir / f"{self.video_name}_rendered_iter_{iteration}.mp4"
                transform_to_pil = transforms.ToPILImage() # Define here

                # Initialize Video Writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
                video_writer = cv2.VideoWriter(str(video_save_path), fourcc, self.output_fps, (self.W, self.H))

                print(f"Saving rendered frames to {frame_save_dir}")
                print(f"Saving rendered video to {video_save_path} (FPS: {self.output_fps})")
            else:
                video_writer = None # No video writer needed if not saving
                transform_to_pil = None # Ensure it's defined if save_frames is false

            # Process each frame
            for t in range(self.T):
                gt_frame_t = self.gt_frames[t:t+1].float()
                out_frame_t = out_frames[t:t+1] # Shape (1, C, H, W)

                # Calculate metrics
                mse_loss_t = F.mse_loss(out_frame_t, gt_frame_t)
                psnr_t = 10 * math.log10(1.0 / max(mse_loss_t.item(), 1e-10)) # Avoid log(0)
                ms_ssim_t = ms_ssim(out_frame_t, gt_frame_t, data_range=1, size_average=True).item()

                total_psnr += psnr_t
                total_ms_ssim += ms_ssim_t

                # Save individual frames and write to video if requested
                if self.save_frames and video_writer is not None:
                    # Prepare frame for saving (common logic)
                    frame_to_save = out_frame_t.squeeze(0).cpu().permute(1, 2, 0)
                    frame_np_rgb = (frame_to_save * 255).byte().numpy()

                    # --- Save PNG Frame ---
                    # Ensure transform_to_pil is available
                    if transform_to_pil is not None:
                        img_pil = transform_to_pil(frame_np_rgb)
                        frame_filename = f"frame_{t:04d}_psnr{psnr_t:.2f}.png"
                        full_frame_path = frame_save_dir / frame_filename
                        img_pil.save(full_frame_path)
                        saved_frame_paths.append(str(full_frame_path))
                    else:
                        print("Warning: transform_to_pil is None, cannot save PNG frame.")

                    # --- Save Video Frame ---
                    frame_np_bgr = cv2.cvtColor(frame_np_rgb, cv2.COLOR_RGB2BGR)
                    video_writer.write(frame_np_bgr)

            # Finalize video if saving
            if self.save_frames and video_writer is not None:
                video_writer.release()
                print(f"Finished saving output for iteration {iteration}.")

        # --- Save Model Checkpoint if requested ---
        if save_checkpoint:
            checkpoint_save_path = self.log_dir / f"gaussian_model_iter_{iteration}.pth.tar"
            torch.save(self.gaussian_model.state_dict(), checkpoint_save_path)
            print(f"Model checkpoint saved to {checkpoint_save_path}")
        # --- End Save Model Checkpoint ---

        avg_psnr = total_psnr / self.T
        avg_ms_ssim = total_ms_ssim / self.T

        # Log metrics for this evaluation
        self.logwriter.write(f"Iteration {iteration} Eval: Avg PSNR: {avg_psnr:.4f}, Avg MS_SSIM: {avg_ms_ssim:.6f}")

        self.gaussian_model.train() # Set model back to training mode
        return avg_psnr, avg_ms_ssim

    # --- Original Test Method (Now Simplified) ---
    def test(self):
        """Calculates and logs final metrics without saving frames/video."""
        self.gaussian_model.eval()
        total_psnr = 0
        total_ms_ssim = 0

        with torch.no_grad():
            # Render all frames at once
            out_frames_pkg = self.gaussian_model() # Get dict with "render": (T, C, H, W)
            out_frames = out_frames_pkg["render"].float()

            for t in range(self.T):
                gt_frame_t = self.gt_frames[t:t+1].float()
                out_frame_t = out_frames[t:t+1]

                mse_loss_t = F.mse_loss(out_frame_t, gt_frame_t)
                psnr_t = 10 * math.log10(1.0 / mse_loss_t.item())
                ms_ssim_t = ms_ssim(out_frame_t, gt_frame_t, data_range=1, size_average=True).item()

                total_psnr += psnr_t
                total_ms_ssim += ms_ssim_t

        avg_psnr = total_psnr / self.T
        avg_ms_ssim = total_ms_ssim / self.T

        self.logwriter.write(f"Test Avg PSNR: {avg_psnr:.4f}, Avg MS_SSIM: {avg_ms_ssim:.6f}")

        return avg_psnr, avg_ms_ssim

# Remove old image processing function (moved to utils)
# def image_path_to_tensor(image_path: Path):
#     ...

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Train dynamic 2D Gaussians on a video.")
    # Video input arguments
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file")
    parser.add_argument("--num_frames", type=int, default=None, help="Number of frames to process from the video (default: all)")
    # Training parameters
    parser.add_argument("--iterations", type=int, default=30000, help="Number of training iterations (default: %(default)s)")
    parser.add_argument("--num_points", type=int, default=5000, help="Number of 2D Gaussian points (default: %(default)s)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: %(default)s)")
    parser.add_argument("--opt_type", type=str, default="adan", choices=["adan", "adam"], help="Optimizer type (default: %(default)s)")
    # Other options
    parser.add_argument("--model_path", type=str, default=None, help="Path to a checkpoint to load (optional)")
    parser.add_argument("--seed", type=int, default=42, help="Set random seed for reproducibility (default: %(default)s)")
    parser.add_argument("--save_frames", action=argparse.BooleanOptionalAction, default=True, help="Save rendered frames during evaluation (default: %(default)s)")
    parser.add_argument("--output_fps", type=int, default=25, help="FPS for the output video if saving is enabled (default: %(default)s)")
    parser.add_argument("--lambda_opacity_reg", type=float, default=1e-4, help="Strength of L1 opacity regularization (default: %(default)s)")
    parser.add_argument("--lambda_temporal_xyz", type=float, default=0.1, help="Strength of temporal consistency loss for XYZ (velocity penalty) (default: %(default)s)")
    parser.add_argument("--lambda_temporal_cholesky", type=float, default=0.1, help="Strength of temporal consistency loss for Cholesky (velocity penalty) (default: %(default)s)")
    parser.add_argument("--lambda_accel_xyz", type=float, default=0.0, help="Strength of temporal acceleration loss for XYZ (default: %(default)s)")
    parser.add_argument("--lambda_accel_cholesky", type=float, default=0.0, help="Strength of temporal acceleration loss for Cholesky (default: %(default)s)")
    parser.add_argument("--lambda_neighbor_rigidity", type=float, default=0.0, help="Strength of neighbor rigidity loss (maintaining inter-Gaussian distances) (default: %(default)s)")
    parser.add_argument("--k_neighbors", type=int, default=5, help="Number of neighbors for rigidity loss (default: %(default)s)")
    # Densification arguments
    parser.add_argument("--densify_from_iter", type=int, default=500, help="Iteration to start densification (default: %(default)s)")
    parser.add_argument("--densify_until_iter", type=int, default=15000, help="Iteration to stop densification (default: %(default)s)")
    parser.add_argument("--densification_interval", type=int, default=100, help="Interval for densification (every N iterations) (default: %(default)s)")
    parser.add_argument("--size_threshold_split", type=float, default=0.01, help="Average scale threshold to split a Gaussian (default: %(default)s, placeholder value)")
    parser.add_argument("--opacity_threshold_clone", type=float, default=0.9, help="Opacity threshold to clone a small Gaussian (default: %(default)s, placeholder value)")
    parser.add_argument("--scale_factor_split_children", type=float, default=0.6, help="Scale factor for Cholesky of children when splitting (default: %(default)s)")
    parser.add_argument("--max_gaussians", type=int, default=60000, help="Maximum number of Gaussians after densification (default: %(default)s)")
    parser.add_argument("--lr_final", type=float, default=1e-05, help="Final learning rate for cosine decay (default: %(default)s)")
    parser.add_argument("--lr_delay_mult", type=float, default=0.1, help="Learning rate delay multiplier (default: %(default)s)") # From 3DGS
    parser.add_argument("--lr_delay_steps", type=int, default=0, help="Learning rate delay steps (default: %(default)s)") # From 3DGS
    parser.add_argument("--output_video_fps", type=float, default=25.0, help="FPS for the output rendered video (default: %(default)s)")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay rate for temporal regularization")

    # Remove old/unused arguments
    # parser.add_argument("--images", type=str, default="images", help="Path to training images folder (default: %(default)s)")

    args = parser.parse_args(argv)
    return args

def main(argv):
    args = parse_args(argv)
    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    # Set seed
    if args.seed is not None:
        print(f"Setting random seed to {args.seed}")
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        # Potentially add deterministic flags, but they can slow down training
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    # Load video frames using the utility function
    video_path = Path(args.video_path)
    if not video_path.is_file():
        print(f"Error: Video file not found at {video_path}")
        sys.exit(1)

    try:
        gt_frames_tensor = video_path_to_tensor(video_path, num_frames=args.num_frames)
    except Exception as e:
        print(f"Error loading video: {e}")
        sys.exit(1)

    # --- Run Training for the single video --- #
    # Instantiate the trainer
    trainer = VideoTrainer(
        gt_frames=gt_frames_tensor,
        video_path=video_path,
        num_points=args.num_points,
        iterations=args.iterations,
        args=args,
        model_path=args.model_path
    )

    # Save args to log dir
    args_save_path = trainer.log_dir / "args.yaml"
    with open(args_save_path, 'w') as f:
        f.write(args_text)
    print(f"Arguments saved to {args_save_path}")

    # Start training
    trainer.train()

    print("\nTraining finished.")

if __name__ == "__main__":
    main(sys.argv[1:])
