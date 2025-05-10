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
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
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
        local_rank: int = 0,
        world_size: int = 1,
    ):
        # Setup distributed training
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize distributed process group if using multiple GPUs
        if self.world_size > 1:
            dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=local_rank)
            torch.cuda.set_device(local_rank)

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

        # Adjust log directory naming - using self.total_T for total frames in video
        rigidity_str = f"_rigidN{args.k_neighbors}L{args.lambda_neighbor_rigidity:.0e}".replace("e-0", "e-") if args.lambda_neighbor_rigidity > 0 else ""
        poly_deg_str = f"_polyDeg{args.polynomial_degree}"
        xyz_coeffs_reg_str = f"_xyzCoeffReg{args.lambda_xyz_coeffs_reg:.0e}".replace("e-0", "e-") if args.lambda_xyz_coeffs_reg > 0 else ""
        chol_coeffs_reg_str = f"_cholCoeffReg{args.lambda_cholesky_coeffs_reg:.0e}".replace("e-0", "e-") if args.lambda_cholesky_coeffs_reg > 0 else ""
        opac_coeffs_reg_str = f"_opacCoeffReg{args.lambda_opacity_coeffs_reg:.0e}".replace("e-0", "e-") if args.lambda_opacity_coeffs_reg > 0 else ""
        opac_poly_deg_str = f"_opacPolyDeg{args.opacity_polynomial_degree}" if args.opacity_polynomial_degree is not None else ""
        traj_model_str = f"_traj{args.trajectory_model_type}"
        if args.trajectory_model_type == "bspline":
            traj_model_str += f"_K{args.num_control_points}"
        opt_str = f"_opt{args.opt_type}"
        ema_str = f"_ema{args.ema_decay:.3f}".replace(".", "p") # Replace . with p for filename safety
        init_eps_str = f"_initEps{args.initialization_logit_eps:.0e}".replace("e-0", "e-")
        target_pixels_str = f"_pixels{args.target_pixel_count}"

        self.log_dir = Path(f"./checkpoints/{self.video_name}/iter{args.iterations}_pts{num_points}_totalFrames{self.total_T}_batchFrames{self.num_frames_per_batch}{traj_model_str}{poly_deg_str}{opac_poly_deg_str}{opt_str}{ema_str}{init_eps_str}{rigidity_str}{xyz_coeffs_reg_str}{chol_coeffs_reg_str}{opac_coeffs_reg_str}{target_pixels_str}_lr{args.lr:.0e}")
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
            initialization_logit_eps=args.initialization_logit_eps
        ).to(self.device)

        # Wrap model with DDP if using multiple GPUs
        if self.world_size > 1:
            self.gaussian_model = DDP(self.gaussian_model, device_ids=[self.local_rank])

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

    def train(self):
        # psnr_list will store average PSNR per iteration
        psnr_list, iter_list = [], []
        progress_bar = tqdm(range(1, self.iterations + 1), desc="Training progress", disable=self.local_rank != 0)

        # Add initial evaluation at iteration 0
        if self.local_rank == 0:
            print("\nPerforming initial evaluation at iteration 0...")
            initial_psnr, initial_ms_ssim = self.evaluate_and_save_output(0, save_checkpoint=True)
            psnr_list.append(initial_psnr)
            iter_list.append(0)

        self.gaussian_model.train()
        start_time = time.time()
        for iter_idx in range(1, self.iterations + 1):
            # Randomly select a batch of frames and their t_values for this iteration
            if self.total_T > 0:
                if self.total_T > self.num_frames_per_batch and self.num_frames_per_batch > 0:
                    # Create a distributed sampler for frame selection
                    if self.world_size > 1:
                        sampler = DistributedSampler(
                            torch.arange(self.total_T, device=self.device),
                            num_replicas=self.world_size,
                            rank=self.local_rank
                        )
                        selected_indices = torch.tensor(list(sampler), device=self.device)
                    else:
                        selected_indices = torch.randperm(self.total_T, device=self.device)[:self.num_frames_per_batch]

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
                if iter_idx == 1 and self.local_rank == 0:
                    print("Warning: No frames loaded (self.total_T is 0), skipping training iterations.")

            # Only log on main process
            if self.local_rank == 0:
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
        if self.local_rank == 0:
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
        else:
            return None, None, end_time, 0, 0

    # --- Refactored Evaluation and Saving Logic ---
    def evaluate_and_save_output(self, iteration, save_checkpoint=False):
        """Renders frames for output (potentially interpolated) and a subset for metrics.
        Saves PNGs/video and optionally a model checkpoint.
        """
        print(f"\n--- Evaluating and Saving Output for Iteration {iteration} ---")
        self.gaussian_model.eval() # Set model to evaluation mode

        if self.total_T == 0:
            print("No frames to evaluate (self.total_T is 0).")
            self.gaussian_model.train() # Set model back to training mode
            return 0.0, 0.0

        # --- Parameters for OUTPUT rendering (Video, PNGs, Centers Viz) ---
        if self.num_frames_per_batch > 0:
            num_render_output_frames = 2 * self.num_frames_per_batch
        else: # if num_frames_per_batch is 0, it implies use all total_T frames as base
            num_render_output_frames = 2 * self.total_T
        num_render_output_frames = max(1, num_render_output_frames) # Ensure at least 1 frame

        # Generate t_values for these output frames (potentially interpolated)
        render_output_t_values = torch.linspace(0, 1, num_render_output_frames, device=self.device)
        print(f"Output rendering will generate {num_render_output_frames} frames (for video, PNGs, centers viz).")

        # Adjust FPS to maintain original duration
        adjusted_fps = self.output_fps * 2  # Double the FPS to maintain duration with double the frames

        # --- Gaussian Centers Dynamics Visualization ---
        try:
            if self.gaussian_model.num_points > 0:
                # Evaluate XYZ coordinates at the specific render_output_t_values
                # Ensure _get_evaluated_polynomials is available and gives (xyz, cholesky, opacity)
                if hasattr(self.gaussian_model, '_get_evaluated_polynomials'):
                    evaluated_xyz_for_viz, _, _ = self.gaussian_model._get_evaluated_polynomials(render_output_t_values)
                    xyz_for_viz = evaluated_xyz_for_viz.detach().cpu() # (num_render_output_frames, N, 2)
                else:
                    # Fallback or error if direct evaluation at t_values is not straightforwardly available
                    print("Warning: _get_evaluated_polynomials not found on model. Gaussian centers viz might not use interpolated t_values correctly for XYZ.")
                    all_xyz_full_model_t = self.gaussian_model.get_xyz.detach().cpu() # (self.total_T, N, 2)
                    viz_indices_from_model_t = torch.linspace(0, self.total_T - 1, num_render_output_frames, device='cpu').long().clamp(0, self.total_T -1)
                    xyz_for_viz = all_xyz_full_model_t[viz_indices_from_model_t]


                static_colors_rgb_norm = self.gaussian_model.get_features.detach().cpu().numpy()
                static_colors_bgr_uint8_all = (static_colors_rgb_norm[:, [2, 1, 0]] * 255).astype(np.uint8)

                max_gaussians_to_visualize = 5000
                num_total_gaussians = self.gaussian_model.num_points
                if num_total_gaussians > max_gaussians_to_visualize:
                    perm = torch.randperm(num_total_gaussians, device='cpu')
                    visualized_gaussian_indices = perm[:max_gaussians_to_visualize]
                    num_gaussians_in_viz = max_gaussians_to_visualize
                else:
                    visualized_gaussian_indices = torch.arange(num_total_gaussians, device='cpu')
                    num_gaussians_in_viz = num_total_gaussians

                if num_gaussians_in_viz > 0:
                    print(f"Visualizing dynamics for {num_gaussians_in_viz} Gaussians.")
                    static_colors_bgr_uint8_viz = static_colors_bgr_uint8_all[visualized_gaussian_indices.numpy()]

                    centers_video_path = self.log_dir / f"{self.video_name}_centers_iter_{iteration}_x{num_render_output_frames}.mp4"
                    frame_h_viz, frame_w_viz = self.H, self.W
                    centers_video_writer = cv2.VideoWriter(str(centers_video_path),
                                                           cv2.VideoWriter_fourcc(*'mp4v'),
                                                           adjusted_fps,
                                                           (frame_w_viz, frame_h_viz))

                    print(f"Saving Gaussian centers dynamics video ({num_render_output_frames} frames) to {centers_video_path}")
                    dot_radius = max(1, int(min(frame_h_viz, frame_w_viz) * 0.005))

                    for k_viz in range(num_render_output_frames):
                        frame_image = np.zeros((frame_h_viz, frame_w_viz, 3), dtype=np.uint8)
                        xyz_k_norm_all_at_t = xyz_for_viz[k_viz]
                        xyz_k_norm_viz_at_t = xyz_k_norm_all_at_t[visualized_gaussian_indices]

                        x_coords = ((xyz_k_norm_viz_at_t[:, 0] * 0.5 + 0.5) * frame_w_viz).numpy().astype(int)
                        y_coords = ((xyz_k_norm_viz_at_t[:, 1] * 0.5 + 0.5) * frame_h_viz).numpy().astype(int)

                        for i_viz in range(num_gaussians_in_viz):
                            pt_center = (x_coords[i_viz], y_coords[i_viz])
                            point_color_bgr = tuple(static_colors_bgr_uint8_viz[i_viz].tolist())
                            if 0 <= pt_center[0] < frame_w_viz and 0 <= pt_center[1] < frame_h_viz:
                                cv2.circle(frame_image, pt_center, dot_radius, point_color_bgr, -1)
                        centers_video_writer.write(frame_image)
                    centers_video_writer.release()
                else:
                    print("Skipping centers dynamics video: No Gaussians selected for visualization.")
            else:
                print("Skipping centers dynamics video: No Gaussians to visualize (num_points is 0).")
        except Exception as e:
            print(f"Could not generate Gaussian centers dynamics video: {e}")
        # --- End Gaussian Centers Dynamics Visualization ---

        # --- Color and Opacity Analysis (remains the same, analyzes full model state) ---
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
                for i in range(final_colors.shape[1]):
                    print(f"  Channel {i} (RGB) - Min: {np.min(final_colors[:, i]):.4f}, Max: {np.max(final_colors[:, i]):.4f}, Mean: {np.mean(final_colors[:, i]):.4f}")
            else:
                print("No color features to analyze (num_points might be 0).")
        print("--- End Color Feature Analysis ---\n")

        try:
            if self.gaussian_model.num_points > 0:
                with torch.no_grad():
                    all_opacities_tensor = self.gaussian_model.get_opacity.detach().cpu() # get_opacity uses self.T
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

        # --- Parameters and Rendering for METRIC calculation (PSNR, MS-SSIM) ---
        total_psnr_eval = 0
        total_ms_ssim_eval = 0

        num_metric_eval_frames = min(self.num_frames_per_batch, self.total_T) if self.num_frames_per_batch > 0 else self.total_T
        num_metric_eval_frames = max(1, num_metric_eval_frames) # Ensure at least 1 frame for metrics if possible

        if num_metric_eval_frames > 0:
            metric_eval_indices = torch.linspace(0, self.total_T - 1, num_metric_eval_frames, device='cpu').long()
            metric_t_values_for_eval = self.all_t_values[metric_eval_indices].to(self.device)
            gt_frames_for_metric_eval = self.all_gt_frames[metric_eval_indices]

            with torch.no_grad():
                # Use white background for metric evaluation
                white_background = torch.ones(3, device=self.device)
                out_frames_for_metric_pkg = self.gaussian_model.forward(
                    t_values_to_render=metric_t_values_for_eval,
                    background_color_override=white_background
                )
                out_frames_for_metric_eval = out_frames_for_metric_pkg["render"].float()

                for k_metric in range(num_metric_eval_frames):
                    current_gt_frame = gt_frames_for_metric_eval[k_metric:k_metric+1]
                    current_out_frame_metric = out_frames_for_metric_eval[k_metric:k_metric+1]

                    mse_loss_t = F.mse_loss(current_out_frame_metric, current_gt_frame)
                    psnr_t = 10 * math.log10(1.0 / max(mse_loss_t.item(), 1e-10))
                    ms_ssim_t = ms_ssim(current_out_frame_metric, current_gt_frame, data_range=1, size_average=True).item()
                    total_psnr_eval += psnr_t
                    total_ms_ssim_eval += ms_ssim_t

        avg_psnr_eval = total_psnr_eval / num_metric_eval_frames if num_metric_eval_frames > 0 else 0.0
        avg_ms_ssim_eval = total_ms_ssim_eval / num_metric_eval_frames if num_metric_eval_frames > 0 else 0.0
        self.logwriter.write(f"Iteration {iteration} Eval (Metrics on {num_metric_eval_frames} frames): Avg PSNR: {avg_psnr_eval:.4f}, Avg MS_SSIM: {avg_ms_ssim_eval:.6f}")

        # --- Rendering for SAVING video/PNGs (uses render_output_t_values) ---
        if self.save_frames:
            with torch.no_grad():
                # Use white background for final rendering
                white_background = torch.ones(3, device=self.device)
                out_frames_for_saving_pkg = self.gaussian_model.forward(
                    t_values_to_render=render_output_t_values,
                    background_color_override=white_background
                )
                out_frames_for_saving = out_frames_for_saving_pkg["render"].float() # (num_render_output_frames, C, H, W)

            frame_save_dir = self.log_dir / f"rendered_frames_iter_{iteration}_x{num_render_output_frames}"
            frame_save_dir.mkdir(parents=True, exist_ok=True)
            video_save_path = self.log_dir / f"{self.video_name}_rendered_iter_{iteration}_x{num_render_output_frames}.mp4"
            # transform_to_pil is not strictly needed if using Image.fromarray directly with numpy array

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Using adjusted_fps. If different duration is desired for the interpolated video, FPS might need adjustment.
            video_writer = cv2.VideoWriter(str(video_save_path), fourcc, adjusted_fps, (self.W, self.H))
            print(f"Saving rendered video ({num_render_output_frames} frames) to {video_save_path} (FPS: {adjusted_fps})")

            for k_save in range(num_render_output_frames):
                current_out_frame_save = out_frames_for_saving[k_save:k_save+1].squeeze(0) # (C,H,W)
                frame_np_rgb = (current_out_frame_save.cpu().permute(1, 2, 0) * 255).byte().numpy() # H,W,C

                # Save PNG Frame
                img_pil = Image.fromarray(frame_np_rgb, 'RGB')
                frame_filename = f"interpolated_frame_{k_save:04d}.png" # Filename indicates interpolated
                full_frame_path = frame_save_dir / frame_filename
                img_pil.save(full_frame_path)

                # Write to video
                frame_np_bgr = cv2.cvtColor(frame_np_rgb, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_np_bgr)

            video_writer.release()
            print(f"Finished saving output for iteration {iteration} ({num_render_output_frames} frames rendered).")

        if save_checkpoint:
            checkpoint_save_path = self.log_dir / f"gaussian_model_iter_{iteration}.pth.tar"
            torch.save(self.gaussian_model.state_dict(), checkpoint_save_path)
            print(f"Model checkpoint saved to {checkpoint_save_path}")

        self.gaussian_model.train() # Set model back to training mode
        return avg_psnr_eval, avg_ms_ssim_eval # Return metrics based on the original metric eval frame count

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
    parser.add_argument("--num_frames", type=int, default=10, help="Total number of evenly spaced frames to load from the video. These frames will be used for all training iterations and for final evaluation/rendering. If 0 or > total video frames, all actual video frames are loaded and used. (default: %(default)s)")
    parser.add_argument("--target_pixel_count", type=int, default=65536, help="Target number of pixels for resizing frames (default: %(default)s, which gives roughly 256x256)")
    # Training parameters
    parser.add_argument("--iterations", type=int, default=30000, help="Number of training iterations (default: %(default)s)")
    parser.add_argument("--num_points", type=int, default=5000, help="Number of 2D Gaussian points (default: %(default)s)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: %(default)s)")
    parser.add_argument("--opt_type", type=str, default="adan", choices=["adan", "adam"], help="Optimizer type (default: %(default)s)")
    # Other options
    parser.add_argument("--model_path", type=str, default=None, help="Path to a checkpoint to load (optional)")
    parser.add_argument("--seed", type=int, default=42, help="Set random seed for reproducibility (default: %(default)s)")
    parser.add_argument("--save_frames", action=argparse.BooleanOptionalAction, default=True, help="Save rendered frames during evaluation (default: %(default)s)")
    parser.add_argument("--output_fps", type=float, default=None, help="FPS for the output video. If not specified, an effective FPS is calculated to preserve original video duration based on sampled frames. (default: %(default)s)")
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
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay rate for temporal regularization")
    parser.add_argument("--polynomial_degree", type=int, default=7, help="Degree of the polynomial for time representation (XYZ, Cholesky) (default: %(default)s)")
    parser.add_argument("--opacity_polynomial_degree", type=int, default=None, help="Degree of the polynomial for opacity time representation. If None, uses polynomial_degree. (default: %(default)s)")
    parser.add_argument("--checkpoint_interval", type=int, default=10000, help="Interval for saving model checkpoints (default: %(default)s, 0 for no periodic checkpoints)")
    # New coefficient regularization arguments
    parser.add_argument("--lambda_xyz_coeffs_reg", type=float, default=0.0, help="Strength of L2 regularization on XYZ polynomial coeffs (default: %(default)s)")
    parser.add_argument("--lambda_cholesky_coeffs_reg", type=float, default=0.0, help="Strength of L2 regularization on Cholesky polynomial coeffs (default: %(default)s)")
    parser.add_argument("--lambda_opacity_coeffs_reg", type=float, default=0.0, help="Strength of L2 regularization on opacity polynomial coeffs (default: %(default)s)")

    # New arguments for initialization
    parser.add_argument("--initialization_logit_eps", type=float, default=1e-6, help="Epsilon for logit calculation during initialization (default: %(default)s)")

    # Arguments for trajectory modeling
    parser.add_argument("--trajectory_model_type", type=str, default="polynomial", choices=["polynomial", "bspline"], help="Type of model for trajectories (xyz, cholesky) (default: %(default)s)")
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

    # Setup distributed training
    world_size = torch.cuda.device_count()
    if world_size > 1:
        print(f"Using {world_size} GPUs for distributed training")
        mp.spawn(
            train_worker,
            args=(world_size, gt_frames_tensor, video_path, args, input_video_fps),
            nprocs=world_size,
            join=True
        )
    else:
        print("Using single GPU")
        train_worker(0, 1, gt_frames_tensor, video_path, args, input_video_fps)

def train_worker(local_rank, world_size, gt_frames_tensor, video_path, args, input_video_fps):
    # --- Run Training for the single video --- #
    # Instantiate the trainer
    trainer = VideoTrainer(
        gt_frames=gt_frames_tensor,
        video_path=video_path,
        num_points=args.num_points,
        iterations=args.iterations,
        args=args,
        model_path=args.model_path,
        input_video_fps=input_video_fps,
        local_rank=local_rank,
        world_size=world_size
    )

    # Save args to log dir (only on main process)
    if local_rank == 0:
        args_save_path = trainer.log_dir / "args.yaml"
        with open(args_save_path, 'w') as f:
            f.write(args_text)
        print(f"Arguments saved to {args_save_path}")

    # Start training
    trainer.train()

    if local_rank == 0:
        print("\nTraining finished.")

if __name__ == "__main__":
    main(sys.argv[1:])
