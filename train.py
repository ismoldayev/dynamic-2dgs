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

        self.num_points = num_points
        self.video_name = video_path.stem
        BLOCK_H, BLOCK_W = 16, 16 # Keep block size fixed for now
        self.iterations = iterations
        self.save_frames = args.save_frames
        # Adjust log directory naming for video, include lambda_opacity_reg
        opacity_reg_str = f"_opReg{args.lambda_opacity_reg:.0e}".replace("e-0", "e-") # Format like 1e-4
        temporal_xyz_str = f"_tempXYZ{args.lambda_temporal_xyz:.0e}".replace("e-0", "e-")
        temporal_chol_str = f"_tempChol{args.lambda_temporal_cholesky:.0e}".replace("e-0", "e-")

        # Add more params to log_dir name for better tracking
        self.log_dir = Path(f"./checkpoints/{self.video_name}/iter{args.iterations}_pts{num_points}_frames{self.T}{opacity_reg_str}{temporal_xyz_str}{temporal_chol_str}_lr{args.lr:.0e}")
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
            lambda_temporal_cholesky=args.lambda_temporal_cholesky
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
                progress_bar.set_postfix({f"Avg Loss":f"{loss.item():.{7}f}", "Avg PSNR":f"{psnr:.{4}f}"})
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

        # --- TEMPORARY OPACITY ANALYSIS ---
        with torch.no_grad():
            current_opacities = self.gaussian_model.get_opacity.detach().cpu().numpy().flatten()
            print(f"\n--- Opacity Analysis at Iteration {iteration} ---")
            print(f"Number of Gaussians: {len(current_opacities)}")
            if len(current_opacities) > 0:
                print(f"Min opacity: {np.min(current_opacities):.6f}")
                print(f"Max opacity: {np.max(current_opacities):.6f}")
                print(f"Mean opacity: {np.mean(current_opacities):.6f}")
                print(f"Median opacity: {np.median(current_opacities):.6f}")
                percentiles_to_check = [1, 5, 10, 25, 50, 75, 90, 95, 99]
                opacity_percentiles = np.percentile(current_opacities, percentiles_to_check)
                for p, v in zip(percentiles_to_check, opacity_percentiles):
                    print(f"{p}th percentile: {v:.6f}")
            else:
                print("No Gaussians to analyze (num_points might be 0 if pruned aggressively).")
        print("--- End Opacity Analysis ---\n")
        # --- END TEMPORARY OPACITY ANALYSIS ---

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

                # Initialize Video Writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
                video_writer = cv2.VideoWriter(str(video_save_path), fourcc, self.output_fps, (self.W, self.H))

                print(f"Saving rendered frames to {frame_save_dir}")
                print(f"Saving rendered video to {video_save_path} (FPS: {self.output_fps})")
                transform_to_pil = transforms.ToPILImage()
            else:
                video_writer = None # No video writer needed if not saving

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
                    img_pil = transform_to_pil(frame_np_rgb)
                    frame_filename = f"frame_{t:04d}_psnr{psnr_t:.2f}.png"
                    full_frame_path = frame_save_dir / frame_filename
                    img_pil.save(full_frame_path)
                    saved_frame_paths.append(str(full_frame_path))

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
    parser.add_argument("--no_save_frames", action="store_false", dest="save_frames", help="Do not save rendered frames/video (default is to save)")
    parser.add_argument("--output_fps", type=int, default=25, help="FPS for the output video if saving is enabled (default: %(default)s)")
    parser.add_argument("--lambda_opacity_reg", type=float, default=1e-4, help="Strength of L1 opacity regularization (default: %(default)s)")
    parser.add_argument("--lambda_temporal_xyz", type=float, default=0.1, help="Strength of temporal consistency loss for XYZ (default: %(default)s)")
    parser.add_argument("--lambda_temporal_cholesky", type=float, default=0.1, help="Strength of temporal consistency loss for Cholesky (default: %(default)s)")
    # parser.add_argument("--sh_degree", type=int, default=0, help="SH degree (Not used in this 2D version, default: %(default)s)") # SH degree is irrelevant for 2D

    # Remove old/unused arguments
    # parser.add_argument("--dataset", type=str, default='./datasets/kodak/', help="Training dataset")
    # parser.add_argument("--data_name", type=str, default='kodak', help="Training dataset")

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
