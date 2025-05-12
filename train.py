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
from typing import Optional
import ffmpeg
import os

# Renamed Trainer class
class VideoTrainer:
    """Trains dynamic 2d gaussians to fit video frames."""
    def __init__(
        self,
        gt_frames: torch.Tensor, # This will be ALL gt_frames now
        video_path: Path,
        num_points: int = 2000,
        iterations:int = 30000,
        model_path = None,
        input_video_fps: Optional[float] = None,
        args = None,
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.all_gt_frames = gt_frames.to(self.device) # Store all frames
        self.total_T, self.C, self.H, self.W = self.all_gt_frames.shape # T is total frames in video

        # Create normalized time values for all frames
        if self.total_T > 0:
            self.all_t_values = torch.linspace(0, 1, self.total_T, device=self.device)
        else:
            self.all_t_values = torch.empty(0, device=self.device)

        self.args = args # Store args
        self.num_frames_per_batch = args.num_frames # For training and final rendering count

        self.num_points = num_points
        self.video_name = video_path.stem
        BLOCK_H, BLOCK_W = 16, 16
        self.iterations = iterations
        self.save_frames = args.save_frames
        self.temp_frame_dir = None  # Will be used to store temporary frames

        # Adjust log directory naming - using self.total_T for total frames in video
        rigidity_str = f"_rigidN{args.k_neighbors}L{args.lambda_neighbor_rigidity:.0e}".replace("e-0", "e-") if args.lambda_neighbor_rigidity > 0 else ""
        poly_deg_str = f"_polyDeg{args.polynomial_degree}" if args.trajectory_model_type == "polynomial" else ""
        xyz_coeffs_reg_str = f"_xyzCoeffReg{args.lambda_xyz_coeffs_reg:.0e}".replace("e-0", "e-") if args.lambda_xyz_coeffs_reg > 0 else ""
        chol_coeffs_reg_str = f"_cholCoeffReg{args.lambda_cholesky_coeffs_reg:.0e}".replace("e-0", "e-") if args.lambda_cholesky_coeffs_reg > 0 else ""
        opac_coeffs_reg_str = f"_opacCoeffReg{args.lambda_opacity_coeffs_reg:.0e}".replace("e-0", "e-") if args.lambda_opacity_coeffs_reg > 0 else ""
        opac_poly_deg_str = f"_opacPolyDeg{args.opacity_polynomial_degree}" if args.opacity_polynomial_degree is not None else ""
        opt_str = f"_opt{args.opt_type}"
        ema_str = f"_ema{args.ema_decay:.0e}".replace("e-0", "e-") if args.lambda_neighbor_rigidity > 0 else ""
        target_pixels_str = f"_targetPixels{args.target_pixel_count}"
        trajectory_str = f"_traj{args.trajectory_model_type}"
        control_points_str = f"_ctrl{args.num_control_points}" if args.trajectory_model_type == "bspline" else ""

        self.log_dir = Path(f"./checkpoints/{self.video_name}/iter{args.iterations}_pts{num_points}_totalFrames{self.total_T}_batchFrames{self.num_frames_per_batch}{trajectory_str}{control_points_str}{rigidity_str}{poly_deg_str}{opac_poly_deg_str}{xyz_coeffs_reg_str}{chol_coeffs_reg_str}{opac_coeffs_reg_str}{opt_str}{ema_str}{target_pixels_str}_lr{args.lr:.0e}")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        print(f"Logging to: {self.log_dir}")

        # Instantiate the Gaussian model with total_T frames
        self.gaussian_model = GaussianImage_Cholesky(
            loss_type="L2",
            T=self.total_T, # Model's T is total frames in video
            num_points=self.num_points,
            H=self.H, W=self.W,
            BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
            device=self.device,
            lr=args.lr,
            opt_type=args.opt_type,
            lambda_neighbor_rigidity=args.lambda_neighbor_rigidity,
            k_neighbors=args.k_neighbors,
            polynomial_degree=args.polynomial_degree,
            lambda_xyz_coeffs_reg=args.lambda_xyz_coeffs_reg,
            lambda_cholesky_coeffs_reg=args.lambda_cholesky_coeffs_reg,
            lambda_opacity_coeffs_reg=args.lambda_opacity_coeffs_reg,
            ema_decay=args.ema_decay,
            opacity_polynomial_degree=args.opacity_polynomial_degree,
            # Pass GT frames and new init args
            gt_frames_for_init=self.all_gt_frames,
            initialization_logit_eps=args.initialization_logit_eps,
            # Add trajectory model type and control points
            trajectory_model_type=args.trajectory_model_type,
            num_control_points=args.num_control_points
        ).to(self.device)

        self.logwriter = LogWriter(self.log_dir)

        # Determine output FPS for video writing
        if args.output_fps is None: # User did not specify --output_fps
            if input_video_fps is not None and input_video_fps > 0:
                self.output_fps = input_video_fps
                print(f"Using duration-preserving effective FPS ({self.output_fps:.2f}) for output video as --output_fps was not specified.")
            else:
                self.output_fps = 25.0 # Fallback if input_video_fps is also invalid
                print(f"Warning: --output_fps not specified and could not determine valid input video FPS. Defaulting to {self.output_fps:.2f} FPS.")
        else: # User explicitly provided --output_fps
            self.output_fps = float(args.output_fps) # Ensure it's float
            print(f"Using user-specified --output_fps ({self.output_fps:.2f}) for output video.")

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

    def _save_frames_to_video(self, frames, output_path, fps):
        """Save a list of frames to a video file using ffmpeg."""
        if not frames:
            print("No frames to save")
            return

        # Create temporary directory for frames if it doesn't exist
        if self.temp_frame_dir is None:
            self.temp_frame_dir = self.log_dir / "temp_frames"
            self.temp_frame_dir.mkdir(parents=True, exist_ok=True)

        # Save frames as temporary PNG files
        temp_frame_paths = []
        for i, frame in enumerate(frames):
            temp_path = self.temp_frame_dir / f"frame_{i:04d}.png"
            cv2.imwrite(str(temp_path), frame)
            temp_frame_paths.append(str(temp_path))

        # Use ffmpeg to create video from frames
        try:
            print(f"Creating video with {len(temp_frame_paths)} frames at {self.output_fps} FPS")
            print(f"First frame path: {temp_frame_paths[0]}")
            print(f"Last frame path: {temp_frame_paths[-1]}")

            # First verify that the frames exist and have content
            for path in temp_frame_paths:
                if not os.path.exists(path):
                    print(f"Warning: Frame file does not exist: {path}")
                    continue
                frame_size = os.path.getsize(path)
                if frame_size == 0:
                    print(f"Warning: Empty frame file: {path}")

            # Create video with more explicit settings
            (
                ffmpeg
                .input(f'{self.temp_frame_dir}/frame_%04d.png', framerate=fps)
                .output(str(output_path), pix_fmt='yuv420p', vcodec='libx264', r=fps)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )

            # Verify the output video
            if os.path.exists(output_path):
                video_size = os.path.getsize(output_path)
                print(f"Successfully saved video to {output_path}")
                print(f"Video file size: {video_size} bytes")
            else:
                print("Error: Video file was not created")

        except ffmpeg.Error as e:
            print(f"Error saving video: {e.stderr.decode()}")
        finally:
            # Clean up temporary frames
            for path in temp_frame_paths:
                try:
                    os.remove(path)
                except:
                    pass

    def train(self):
        # psnr_list will store average PSNR per iteration
        psnr_list, iter_list = [], []
        progress_bar = tqdm(range(1, self.iterations + 1), desc="Training progress")

        self.gaussian_model.train()
        start_time = time.time()
        for iter_idx in range(1, self.iterations + 1):
            # Randomly select a batch of frames and their t_values for this iteration
            if self.total_T > 0:
                if self.total_T > self.num_frames_per_batch and self.num_frames_per_batch > 0:
                    selected_indices = torch.randperm(self.total_T, device=self.device)[:self.num_frames_per_batch]
                    # Sort indices to maintain some temporal coherence if desired, though not strictly necessary for random batch
                    # selected_indices, _ = torch.sort(selected_indices)
                    gt_frames_batch = self.all_gt_frames[selected_indices]
                    t_values_batch = self.all_t_values[selected_indices]
                else: # Use all frames if total_T is less than or equal to batch size, or batch size is 0 (use all)
                    gt_frames_batch = self.all_gt_frames
                    t_values_batch = self.all_t_values

                if gt_frames_batch.shape[0] > 0: # Ensure batch is not empty
                     loss, psnr = self.gaussian_model.train_iter(gt_frames_batch, t_values_batch)
                else: # Should not happen if logic above is correct
                    loss, psnr = 0.0, 0.0
            else: # No frames loaded
                loss, psnr = 0.0, 0.0
                if iter_idx == 1: print("Warning: No frames loaded (self.total_T is 0), skipping training iterations.")
                # break # or continue, depending on desired behavior

            psnr_list.append(psnr)
            iter_list.append(iter_idx)

            if iter_idx % 10 == 0:
                progress_bar.set_postfix({f"Avg Loss":f"{loss:.{7}f}", "Avg PSNR":f"{psnr:.{4}f}"})
                progress_bar.update(10)

            # --- Periodic Evaluation and Saving ---
            # Also save model checkpoint here if it's a periodic eval iteration
            if self.args.checkpoint_interval > 0 and iter_idx % self.args.checkpoint_interval == 0 and iter_idx != self.iterations:
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
            num_render_tests = min(10, self.total_T) if self.total_T > 0 else 0
            if num_render_tests > 0:
                for i in range(num_render_tests):
                    # Render a single frame by its t_value
                    t_idx_for_test = i % self.total_T
                    t_val_for_test = self.all_t_values[t_idx_for_test].unsqueeze(0) # Make it a batch of 1
                    _ = self.gaussian_model.forward(t_values_to_render=t_val_for_test)
                test_end_time = (time.time() - test_start_time) / num_render_tests
            else:
                test_end_time = 0 # Avoid division by zero if no frames to test

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
    def evaluate_and_save_output(
        self,
        iteration: int,
        num_eval_frames: int = 100,
        max_gaussians_to_visualize: int = 40000,
        dot_radius: int = 1,
        save_checkpoint: bool = False,
    ):
        """Renders a subset of frames, calculates metrics, and saves PNGs/video.
        Optionally saves a model checkpoint.
        """
        print(f"\n--- Evaluating and Saving Output for Iteration {iteration} ---")
        self.gaussian_model.eval() # Set model to evaluation mode

        total_psnr_eval = 0
        total_ms_ssim_eval = 0

        # Determine the frames to evaluate and render for video
        if self.total_T == 0:
            print("No frames to evaluate.")
            self.gaussian_model.train() # Set model back to training mode
            return 0.0, 0.0

        # Use all frames for evaluation
        num_eval_frames = self.total_T
        if num_eval_frames == 0: # Should be caught by total_T check, but as safeguard
            print("Number of frames to evaluate is 0. Skipping evaluation.")
            self.gaussian_model.train()
            return 0.0, 0.0

        # Select evenly spaced indices from the *total* available frames
        eval_indices = torch.linspace(0, self.total_T - 1, num_eval_frames, device='cpu').long()
        eval_t_values = self.all_t_values[eval_indices].to(self.device) # Get corresponding t_values for these frames
        gt_frames_for_eval = self.all_gt_frames[eval_indices] # GT frames for these specific indices

        # Create interpolated t_values for double the frames
        num_interpolated_frames = num_eval_frames * 2 - 1  # One less than double because we don't need to interpolate after the last frame
        # Create pairs of original frame times and interpolate between them
        original_times = eval_t_values
        interpolated_t_values = torch.zeros(num_interpolated_frames, device=self.device)
        for i in range(num_eval_frames):
            # Original frame time
            interpolated_t_values[i*2] = original_times[i]
            # Interpolated frame time (halfway between this frame and next frame)
            if i < num_eval_frames - 1:
                interpolated_t_values[i*2 + 1] = (original_times[i] + original_times[i+1]) / 2

        # --- Gaussian Centers Dynamics Visualization ---
        try:
            if self.gaussian_model.num_points > 0:
                # Get all xyz positions for the eval frames
                if self.gaussian_model.trajectory_model_type == "bspline":
                    # For B-splines, we need to evaluate the positions using the basis functions
                    all_xyz_normalized = self.gaussian_model._get_evaluated_bsplines(eval_t_values)[0].detach().cpu() # (num_eval_frames, N, 2)
                else:
                    # For polynomial model, we can use get_xyz directly
                    all_xyz_normalized = self.gaussian_model.get_xyz[eval_indices].detach().cpu() # (num_eval_frames, N, 2)

                # Get static colors for all Gaussians
                static_colors_rgb_norm = self.gaussian_model.get_features.detach().cpu().numpy()
                static_colors_bgr_uint8 = (static_colors_rgb_norm[:, [2, 1, 0]] * 255).astype(np.uint8)

                # Select subset of Gaussians if needed
                num_total_gaussians = self.gaussian_model.num_points
                if num_total_gaussians > max_gaussians_to_visualize:
                    perm = torch.randperm(num_total_gaussians, device='cpu')
                    visualized_gaussian_indices = perm[:max_gaussians_to_visualize]
                    num_gaussians_in_viz = max_gaussians_to_visualize
                    print(f"Visualizing dynamics for a random subset of {num_gaussians_in_viz} Gaussians (out of {num_total_gaussians}).")
                else:
                    visualized_gaussian_indices = torch.arange(num_total_gaussians, device='cpu')
                    num_gaussians_in_viz = num_total_gaussians
                    print(f"Visualizing dynamics for all {num_gaussians_in_viz} Gaussians.")

                # Create video writer for centers visualization
                centers_video_path = self.log_dir / f"{self.video_name}_centers_iter_{iteration}.mp4"
                print(f"\nCreating Gaussian centers visualization video...")

                # Calculate scaled dimensions to maintain aspect ratio while having width*height ≈ 3*num_gaussians
                target_pixels = 6 * self.gaussian_model.num_points
                aspect_ratio = self.W / self.H
                scaled_H = int(np.sqrt(target_pixels / aspect_ratio))
                scaled_W = int(np.sqrt(target_pixels * aspect_ratio))
                print(f"Scaling video to {scaled_W}x{scaled_H} (target: {target_pixels} pixels for {self.gaussian_model.num_points} Gaussians)")

                # Use the same approach as the reconstruction video but with scaled resolution
                video_writer = cv2.VideoWriter(str(centers_video_path),
                                             cv2.VideoWriter_fourcc(*'mp4v'),
                                             self.output_fps,
                                             (scaled_W, scaled_H),
                                             isColor=True)

                if not video_writer.isOpened():
                    print("Failed to initialize video writer with mp4v codec. Falling back to ffmpeg...")
                    try:
                        # Save frames as temporary PNG files
                        temp_frame_dir = self.log_dir / "temp_frames"
                        temp_frame_dir.mkdir(parents=True, exist_ok=True)
                        temp_frame_paths = []

                        for k in range(num_eval_frames):
                            # Create blank frame with scaled resolution
                            frame = np.zeros((scaled_H, scaled_W, 3), dtype=np.uint8)

                            # Get xyz positions for this frame
                            xyz_k = all_xyz_normalized[k] # (N, 2)
                            xyz_k_viz = xyz_k[visualized_gaussian_indices] # (num_gaussians_in_viz, 2)

                            # Transform normalized coordinates to pixel coordinates (scaled)
                            x_coords = (xyz_k_viz[:, 0] * 0.5 + 0.5) * (scaled_W - 1)
                            y_coords = (xyz_k_viz[:, 1] * 0.5 + 0.5) * (scaled_H - 1)

                            # Draw each Gaussian center
                            dots_visible_in_frame = 0
                            for i in range(num_gaussians_in_viz):
                                x, y = int(x_coords[i]), int(y_coords[i])
                                if 0 <= x < scaled_W and 0 <= y < scaled_H:
                                    color = tuple(static_colors_bgr_uint8[visualized_gaussian_indices[i]].tolist())
                                    cv2.circle(frame, (x, y), 1, color, -1)  # Keep dot radius at 1
                                    dots_visible_in_frame += 1

                            # Save frame as PNG
                            frame_path = temp_frame_dir / f"frame_{k:04d}.png"
                            cv2.imwrite(str(frame_path), frame)
                            temp_frame_paths.append(str(frame_path))

                        # Use ffmpeg to create video from frames
                        (
                            ffmpeg
                            .input(f'{temp_frame_dir}/frame_%04d.png', framerate=self.output_fps)
                            .output(str(centers_video_path), pix_fmt='yuv420p', vcodec='libx264', r=self.output_fps)
                            .overwrite_output()
                            .run(capture_stdout=True, capture_stderr=True)
                        )

                        # Clean up temporary frames
                        for path in temp_frame_paths:
                            try:
                                os.remove(path)
                            except:
                                pass
                        print("Successfully created video using ffmpeg")
                    except Exception as e:
                        print(f"Failed to create video using ffmpeg: {e}")
                        raise Exception("Failed to initialize video writer with any codec or ffmpeg")

                # Create frame save directory if needed
                if self.save_frames:
                    centers_frame_dir = self.log_dir / f"centers_frames_iter_{iteration}"
                    centers_frame_dir.mkdir(parents=True, exist_ok=True)

                # Render each frame
                total_dots_visible = 0
                for k in range(num_eval_frames):
                    # Create blank frame with scaled resolution
                    frame = np.zeros((scaled_H, scaled_W, 3), dtype=np.uint8)

                    # Get xyz positions for this frame
                    xyz_k = all_xyz_normalized[k] # (N, 2)
                    xyz_k_viz = xyz_k[visualized_gaussian_indices] # (num_gaussians_in_viz, 2)

                    # Transform normalized coordinates to pixel coordinates (scaled)
                    x_coords = (xyz_k_viz[:, 0] * 0.5 + 0.5) * (scaled_W - 1)
                    y_coords = (xyz_k_viz[:, 1] * 0.5 + 0.5) * (scaled_H - 1)

                    # Draw each Gaussian center
                    dots_visible_in_frame = 0
                    for i in range(num_gaussians_in_viz):
                        x, y = int(x_coords[i]), int(y_coords[i])
                        if 0 <= x < scaled_W and 0 <= y < scaled_H:
                            color = tuple(static_colors_bgr_uint8[visualized_gaussian_indices[i]].tolist())
                            cv2.circle(frame, (x, y), 1, color, -1)  # Keep dot radius at 1
                            dots_visible_in_frame += 1

                    total_dots_visible += dots_visible_in_frame

                    # Save frame to video
                    video_writer.write(frame)

                    # Save individual frame if requested
                    if self.save_frames:
                        frame_path = centers_frame_dir / f"centers_frame_{k:04d}.png"
                        cv2.imwrite(str(frame_path), frame)
                        if not os.path.exists(frame_path):
                            print(f"Warning: Failed to save frame {k} as PNG")

                # Release video writer
                video_writer.release()

                # Print summary statistics
                avg_dots_per_frame = total_dots_visible / num_eval_frames
                print(f"Gaussian centers visualization complete:")
                print(f"Average dots per frame: {avg_dots_per_frame:.2f}")

                # Verify the video file was created and has content
                if os.path.exists(centers_video_path):
                    video_size = os.path.getsize(centers_video_path)
                    if video_size < 1000:  # Less than 1KB
                        print("Warning: Video file is suspiciously small!")
                else:
                    print(f"Error: Video file was not created at {centers_video_path}")

                print(f"Saved Gaussian centers visualization to {centers_video_path}")
            else:
                print("Skipping centers dynamics video: No Gaussians to visualize.")
        except Exception as e:
            print(f"Could not generate Gaussian centers dynamics video: {e}")
        # --- End Gaussian Centers Dynamics Visualization ---

        # --- Color and Opacity Analysis ---
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

        # --- Opacity Distribution Analysis ---
        try:
            if self.gaussian_model.num_points > 0:
                with torch.no_grad():
                    all_opacities_tensor = self.gaussian_model.get_opacity.detach().cpu()
                    all_opacities_np = all_opacities_tensor.numpy().flatten()
                    print(f"\n--- Opacity Distribution Analysis at Iteration {iteration} (Full Model) ---")
                    if all_opacities_np.size > 0:
                        print(f"  Opacities Tensor Shape (Original T, N, 1): {all_opacities_tensor.shape}")
                        print(f"  Total Opacity Values Analyzed: {all_opacities_np.size}")
                        print(f"  Min Opacity: {np.min(all_opacities_np):.4f}")
                        print(f"  Max Opacity: {np.max(all_opacities_np):.4f}")
                        print(f"  Mean Opacity: {np.mean(all_opacities_np):.4f}")
                        print(f"  Median Opacity: {np.median(all_opacities_np):.4f}")
                        percentiles = [10, 25, 50, 75, 90]
                        perc_values = np.percentile(all_opacities_np, percentiles)
                        for p, v in zip(percentiles, perc_values):
                            print(f"  {p}th Percentile: {v:.4f}")
                    else:
                        print("  No opacity values to analyze (array is empty after processing).")
                print("--- End Opacity Distribution Analysis ---\n")
            else:
                print("Skipping opacity distribution analysis: No Gaussians.")
        except Exception as e:
            print(f"Could not perform opacity distribution analysis: {e}")
        # --- End Color and Opacity Analysis ---

        # --- Reconstruction Video Rendering ---
        with torch.no_grad():
            # Render the interpolated frames for evaluation and video output
            out_frames_pkg = self.gaussian_model.forward(t_values_to_render=interpolated_t_values)
            out_frames_eval = out_frames_pkg["render"].float() # (num_interpolated_frames, C, H, W)

            if self.save_frames:
                frame_save_dir = self.log_dir / f"rendered_frames_iter_{iteration}"
                frame_save_dir.mkdir(parents=True, exist_ok=True)
                video_save_path = self.log_dir / f"{self.video_name}_rendered_interpolated_iter_{iteration}.mp4"
                transform_to_pil = transforms.ToPILImage()
                video_writer = cv2.VideoWriter(str(video_save_path),
                                             cv2.VideoWriter_fourcc(*'mp4v'),
                                             self.output_fps * 2,  # Double the FPS for interpolated video
                                             (self.W, self.H),
                                             isColor=True)

                print(f"Saving rendered video ({num_interpolated_frames} frames) to {video_save_path}")

            # Calculate metrics only on original frames (not interpolated ones)
            for k_eval in range(num_eval_frames):
                current_gt_frame = gt_frames_for_eval[k_eval:k_eval+1] # (1, C, H, W)
                # Get the corresponding interpolated frame
                current_out_frame = out_frames_eval[k_eval*2:k_eval*2+1]   # (1, C, H, W)

                mse_loss_t = F.mse_loss(current_out_frame, current_gt_frame)
                psnr_t = 10 * math.log10(1.0 / max(mse_loss_t.item(), 1e-10))
                ms_ssim_t = ms_ssim(current_out_frame, current_gt_frame, data_range=1, size_average=True).item()

                total_psnr_eval += psnr_t
                total_ms_ssim_eval += ms_ssim_t

                if self.save_frames and video_writer is not None:
                    # Save the original frame
                    frame_to_save = out_frames_eval[k_eval*2].cpu().permute(1, 2, 0)
                    frame_np_rgb = (frame_to_save * 255).byte().numpy()
                    if transform_to_pil is not None:
                        img_pil = Image.fromarray(frame_np_rgb, 'RGB')
                        frame_filename = f"eval_frame_{k_eval:04d}_orig_origIdx{eval_indices[k_eval]:04d}_psnr{psnr_t:.2f}.png"
                        full_frame_path = frame_save_dir / frame_filename
                        img_pil.save(full_frame_path)
                    frame_np_bgr = cv2.cvtColor(frame_np_rgb, cv2.COLOR_RGB2BGR)
                    video_writer.write(frame_np_bgr)

                    # Save interpolated frame if not at the last original frame
                    if k_eval < num_eval_frames - 1:
                        frame_to_save = out_frames_eval[k_eval*2 + 1].cpu().permute(1, 2, 0)
                        frame_np_rgb = (frame_to_save * 255).byte().numpy()
                        if transform_to_pil is not None:
                            img_pil = Image.fromarray(frame_np_rgb, 'RGB')
                            frame_filename = f"eval_frame_{k_eval:04d}_interp_origIdx{eval_indices[k_eval]:04d}_psnr{psnr_t:.2f}.png"
                            full_frame_path = frame_save_dir / frame_filename
                            img_pil.save(full_frame_path)
                        frame_np_bgr = cv2.cvtColor(frame_np_rgb, cv2.COLOR_RGB2BGR)
                        video_writer.write(frame_np_bgr)

            if self.save_frames and video_writer is not None:
                video_writer.release()
                print(f"Finished saving output for iteration {iteration}.")

            # Create additional video with only original frames
            if self.save_frames:
                original_only_video_path = self.log_dir / f"{self.video_name}_rendered_original_only_iter_{iteration}.mp4"
                original_only_video_writer = cv2.VideoWriter(str(original_only_video_path),
                                                          cv2.VideoWriter_fourcc(*'mp4v'),
                                                          self.output_fps,
                                                          (self.W, self.H),
                                                          isColor=True)

                print(f"Saving original-frames-only video ({num_eval_frames} frames) to {original_only_video_path}")

                # Render only original frames
                with torch.no_grad():
                    out_frames_pkg_original = self.gaussian_model.forward(t_values_to_render=eval_t_values)
                    out_frames_original = out_frames_pkg_original["render"].float()

                    for k_eval in range(num_eval_frames):
                        frame_to_save = out_frames_original[k_eval].cpu().permute(1, 2, 0)
                        frame_np_rgb = (frame_to_save * 255).byte().numpy()
                        frame_np_bgr = cv2.cvtColor(frame_np_rgb, cv2.COLOR_RGB2BGR)
                        original_only_video_writer.write(frame_np_bgr)

                original_only_video_writer.release()
                print(f"Finished saving original-frames-only video for iteration {iteration}.")

        if save_checkpoint:
            checkpoint_save_path = self.log_dir / f"gaussian_model_iter_{iteration}.pth.tar"
            torch.save(self.gaussian_model.state_dict(), checkpoint_save_path)
            print(f"Model checkpoint saved to {checkpoint_save_path}")

        avg_psnr_eval = total_psnr_eval / num_eval_frames if num_eval_frames > 0 else 0.0
        avg_ms_ssim_eval = total_ms_ssim_eval / num_eval_frames if num_eval_frames > 0 else 0.0

        self.logwriter.write(f"Iteration {iteration} Eval ({num_eval_frames} frames): Avg PSNR: {avg_psnr_eval:.4f}, Avg MS_SSIM: {avg_ms_ssim_eval:.6f}")

        self.gaussian_model.train()
        return avg_psnr_eval, avg_ms_ssim_eval

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

            for t in range(self.total_T):
                gt_frame_t = self.all_gt_frames[t:t+1].float()
                out_frame_t = out_frames[t:t+1]

                mse_loss_t = F.mse_loss(out_frame_t, gt_frame_t)
                psnr_t = 10 * math.log10(1.0 / mse_loss_t.item())
                ms_ssim_t = ms_ssim(out_frame_t, gt_frame_t, data_range=1, size_average=True).item()

                total_psnr += psnr_t
                total_ms_ssim += ms_ssim_t

        avg_psnr = total_psnr / self.total_T
        avg_ms_ssim = total_ms_ssim / self.total_T

        self.logwriter.write(f"Test Avg PSNR: {avg_psnr:.4f}, Avg MS_SSIM: {avg_ms_ssim:.6f}")

        return avg_psnr, avg_ms_ssim

# Remove old image processing function (moved to utils)
# def image_path_to_tensor(image_path: Path):
#     ...

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Train dynamic 2D Gaussians on a video.")
    # Video input arguments
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file")
    parser.add_argument("--num_frames", type=int, default=0, help="Number of frames to randomly select for each training iteration. All frames from the video are loaded, but this controls how many are used per training step. If 0, all frames are used in each iteration. (default: %(default)s)")
    parser.add_argument("--target_pixel_count", type=int, default=65536, help="Target number of pixels for resizing frames (default: %(default)s = 256*256)")
    # Training parameters
    parser.add_argument("--iterations", type=int, default=10000, help="Number of training iterations (default: %(default)s)")
    parser.add_argument("--num_points", type=int, default=20000, help="Number of 2D Gaussian points (default: %(default)s)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: %(default)s)")
    parser.add_argument("--opt_type", type=str, default="adan", choices=["adan", "adam"], help="Optimizer type (default: %(default)s)")
    # Other options
    parser.add_argument("--model_path", type=str, default=None, help="Path to a checkpoint to load (optional)")
    parser.add_argument("--seed", type=int, default=42, help="Set random seed for reproducibility (default: %(default)s)")
    parser.add_argument("--save_frames", action=argparse.BooleanOptionalAction, default=True, help="Save rendered frames during evaluation (default: %(default)s)")
    parser.add_argument("--output_fps", type=float, default=None, help="FPS for the output video. If not specified, an effective FPS is calculated to preserve original video duration based on sampled frames. (default: %(default)s)")
    parser.add_argument("--lambda_neighbor_rigidity", type=float, default=0.0, help="Strength of neighbor rigidity loss (maintaining inter-Gaussian distances) (default: %(default)s)")
    parser.add_argument("--k_neighbors", type=int, default=5, help="Number of neighbors for rigidity loss (default: %(default)s)")
    # Learning rate arguments
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay rate for temporal regularization")
    parser.add_argument("--polynomial_degree", type=int, default=7, help="Degree of the polynomial for time representation (XYZ, Cholesky) (default: %(default)s)")
    parser.add_argument("--opacity_polynomial_degree", type=int, default=0, help="Degree of the polynomial for opacity time representation. If None, uses polynomial_degree. (default: %(default)s)")
    parser.add_argument("--checkpoint_interval", type=int, default=20000, help="Interval for saving model checkpoints (default: %(default)s, 0 for no periodic checkpoints)")
    # New coefficient regularization arguments
    parser.add_argument("--lambda_xyz_coeffs_reg", type=float, default=0.0, help="Strength of L2 regularization on XYZ polynomial coeffs (default: %(default)s)")
    parser.add_argument("--lambda_cholesky_coeffs_reg", type=float, default=0.0, help="Strength of L2 regularization on Cholesky polynomial coeffs (default: %(default)s)")
    parser.add_argument("--lambda_opacity_coeffs_reg", type=float, default=0.0, help="Strength of L2 regularization on opacity polynomial coeffs (default: %(default)s)")

    # New arguments for initialization
    parser.add_argument("--initialization_logit_eps", type=float, default=1e-6, help="Epsilon for logit calculation during initialization (default: %(default)s)")

    # Arguments for trajectory modeling
    parser.add_argument("--trajectory_model_type", type=str, default="bspline", choices=["polynomial", "bspline"], help="Type of model for trajectories (xyz, cholesky) (default: %(default)s)")
    parser.add_argument("--num_control_points", type=int, default=5, help="Number of control points K if trajectory_model_type is bspline (default: %(default)s)")

    # Remove old/unused arguments
    # parser.add_argument("--images", type=str, default="images", help="Path to training images folder (default: %(default)s)")

    args = parser.parse_args(argv)

    # Rename output_video_fps to output_fps if it exists from old args and output_fps is not set
    if hasattr(args, 'output_video_fps') and args.output_video_fps is not None and args.output_fps is None:
        print("Warning: --output_video_fps is deprecated. Using its value for --output_fps.")
        args.output_fps = args.output_video_fps
    # Remove the old attribute to avoid confusion if it was present
    if hasattr(args, 'output_video_fps'):
        delattr(args, 'output_video_fps')

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
        # Load video frames using the utility function
        # Load a specific number of evenly spaced frames based on args.num_frames
        print(f"Loading {args.num_frames if args.num_frames > 0 else 'all'} evenly spaced frames from video for training and evaluation...")
        gt_frames_tensor, input_video_fps = video_path_to_tensor(
            video_path,
            num_frames=args.num_frames if args.num_frames > 0 else None,
            target_pixel_count=args.target_pixel_count
        )

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
        model_path=args.model_path,
        input_video_fps=input_video_fps
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
