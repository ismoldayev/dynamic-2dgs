from gsplat.gsplat.project_gaussians_2d import project_gaussians_2d
from gsplat.gsplat.rasterize_sum import rasterize_gaussians_sum
from utils import *
import torch
import torch.nn as nn
import numpy as np
import math
# from quantize import *
from optimizer import Adan
from typing import Optional
import torch.nn.functional as F # For cosine similarity

# Helper for logit transformation
def _logit(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_clamped = torch.clamp(x, eps, 1.0 - eps)
    return torch.log(x_clamped / (1.0 - x_clamped))

class EMA:
    """Exponential Moving Average for PyTorch Models"""
    def __init__(self, model_params, decay=0.999):
        self.decay = decay
        self.shadow_params = [p.clone().detach() for p in model_params if p.requires_grad]
        # Ensure the model parameters are also a list, not a generator
        self.model_params = [p for p in model_params if p.requires_grad]
        if len(self.shadow_params) != len(self.model_params):
            raise ValueError("Mismatch between model parameters and parameters requiring gradients.")

    def update(self, new_params_data_list=None):
        """Update EMA shadow_params.
        If new_params_data_list is provided (e.g. after densification),
        it should be a list of tensor data to re-initialize specific shadow params.
        Otherwise, updates based on current model_params.
        """
        with torch.no_grad():
            if new_params_data_list is not None: # Re-initialize specific shadow params
                # This case needs careful handling if the number or order of params changes.
                # For now, assume it's a full replacement of all shadow params with new data.
                if len(new_params_data_list) != len(self.shadow_params):
                    # If a single tensor data is passed for a single param EMA (as in original plan)
                    if len(self.shadow_params) == 1 and isinstance(new_params_data_list, torch.Tensor):
                        self.shadow_params[0].copy_(new_params_data_list)
                        return # Done with re-initialization for single param EMA
                    else:
                        print("Warning: EMA update with new_params_data_list expects matching length or single tensor for single param EMA.")
                        # Fallback to standard update if lengths mismatch badly, or handle error
                        # For now, we let it proceed to the standard update if re-init fails here.
                else:
                    for i, new_data in enumerate(new_params_data_list):
                        if self.shadow_params[i].shape == new_data.shape:
                            self.shadow_params[i].copy_(new_data)
                        else:
                            # This should not happen if densify_gaussians correctly re-creates EMA objects
                            print(f"Warning: EMA shadow param {i} shape mismatch during re-init.")
                    return # Done with re-initialization

            # Standard EMA update
            for model_param, shadow_param in zip(self.model_params, self.shadow_params):
                if model_param.requires_grad:
                    shadow_param.sub_((1.0 - self.decay) * (shadow_param - model_param.data))

    def apply_shadow(self):
        """Copy shadow parameters to model parameters."""
        for model_param, shadow_param in zip(self.model_params, self.shadow_params):
            if model_param.requires_grad:
                model_param.data.copy_(shadow_param)

    def get_shadow_params(self):
        return self.shadow_params

class GaussianImage_Cholesky(nn.Module):
    def __init__(self, loss_type="L2", T=1, lambda_neighbor_rigidity=0.0, k_neighbors=5, ema_decay=0.999,
                 polynomial_degree=1, lambda_xyz_coeffs_reg=0.0, lambda_cholesky_coeffs_reg=0.0,
                 lambda_opacity_coeffs_reg=0.0, opacity_polynomial_degree=None,
                 gt_frames_for_init: Optional[torch.Tensor] = None, initialization_logit_eps: float = 1e-6,
                 trajectory_model_type: str = "polynomial", num_control_points: int = 5,
                 **kwargs):
        super().__init__()
        self.loss_type = loss_type
        self.num_points = kwargs["num_points"]
        self.H, self.W = kwargs["H"], kwargs["W"]
        self.T = T # Number of frames
        self.lambda_neighbor_rigidity = lambda_neighbor_rigidity
        self.k_neighbors = k_neighbors
        self.lambda_xyz_coeffs_reg = lambda_xyz_coeffs_reg
        self.lambda_cholesky_coeffs_reg = lambda_cholesky_coeffs_reg
        self.lambda_opacity_coeffs_reg = lambda_opacity_coeffs_reg
        self.BLOCK_W, self.BLOCK_H = kwargs["BLOCK_W"], kwargs["BLOCK_H"]
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        )
        self.device = kwargs["device"]

        self.trajectory_model_type = trajectory_model_type
        self.polynomial_degree = polynomial_degree
        self.K_control_points = num_control_points
        self.bspline_degree = 3 # Fixed cubic B-splines for now

        self.opacity_polynomial_degree = opacity_polynomial_degree if opacity_polynomial_degree is not None else polynomial_degree

        # Time values for polynomial evaluation, normalized from 0 to 1
        if self.T > 0:
            t_values = torch.linspace(0, 1, self.T, device=self.device)
        else: # Should not happen in practice, but handle gracefully
            t_values = torch.empty(0, device=self.device)
        self.register_buffer('t_values', t_values, persistent=False)

        # Create t_power_matrix: (T, num_coeffs) where each row is [t^0, t^1, ..., t^degree]
        if self.T > 0:
            t_power_matrix_list = [torch.ones_like(self.t_values).unsqueeze(1)] # t^0
            if self.polynomial_degree > 0:
                for i in range(1, self.polynomial_degree + 1):
                    t_power_matrix_list.append(self.t_values.pow(i).unsqueeze(1))
            t_power_matrix = torch.cat(t_power_matrix_list, dim=1)
        else:
            t_power_matrix = torch.empty(0, self.polynomial_degree + 1, device=self.device)
        self.register_buffer('t_power_matrix', t_power_matrix, persistent=False)

        # Create t_power_matrix for opacity if its degree is different
        if self.polynomial_degree == self.opacity_polynomial_degree:
            self.t_power_matrix_opacity = t_power_matrix
        else:
            if self.T > 0:
                t_power_matrix_opacity_list = [torch.ones_like(self.t_values).unsqueeze(1)] # t^0
                if self.opacity_polynomial_degree > 0:
                    for i in range(1, self.opacity_polynomial_degree + 1):
                        t_power_matrix_opacity_list.append(self.t_values.pow(i).unsqueeze(1))
                t_power_matrix_opacity = torch.cat(t_power_matrix_opacity_list, dim=1)
            else:
                t_power_matrix_opacity = torch.empty(0, self.opacity_polynomial_degree + 1, device=self.device)
            self.register_buffer('t_power_matrix_opacity', t_power_matrix_opacity, persistent=False)

        self.opacity_activation = torch.sigmoid
        self.rgb_activation = torch.sigmoid

        # Initialize trajectory parameters based on model type
        if self.trajectory_model_type == "polynomial":
            print(f"Using POLYNOMIAL model for XYZ/Cholesky trajectories with degree {self.polynomial_degree}.")
            # Initialize _xyz_coeffs: (polynomial_degree + 1, N, 2)
            initial_xyz_c0 = torch.atanh(2 * (torch.rand(self.num_points, 2, device=self.device) - 0.5))
            higher_order_xyz_coeffs = torch.zeros(self.polynomial_degree, self.num_points, 2, device=self.device)
            self._xyz_coeffs = nn.Parameter(torch.cat([initial_xyz_c0.unsqueeze(0), higher_order_xyz_coeffs], dim=0))

            # Initialize _cholesky_coeffs: (polynomial_degree + 1, N, 3)
            init_chol_base_c0_poly = torch.tensor([1.0, 0.0, 1.0], device=self.device).view(1, 3)
            noise_per_gaussian_c0_poly = torch.randn(self.num_points, 3, device=self.device) * 0.01
            initial_cholesky_c0_poly = init_chol_base_c0_poly.repeat(self.num_points, 1) + noise_per_gaussian_c0_poly
            higher_order_cholesky_coeffs = torch.zeros(self.polynomial_degree, self.num_points, 3, device=self.device)
            self._cholesky_coeffs = nn.Parameter(torch.cat([initial_cholesky_c0_poly.unsqueeze(0), higher_order_cholesky_coeffs], dim=0))

            self._initial_xyz_for_custom_init = initial_xyz_c0.clone().detach()

        elif self.trajectory_model_type == "bspline":
            if self.K_control_points <= self.bspline_degree:
                raise ValueError(f"Number of control points K ({self.K_control_points}) must be greater than B-spline degree p ({self.bspline_degree}).")
            print(f"Using B-SPLINE model for XYZ/Cholesky trajectories with K={self.K_control_points} control points and degree {self.bspline_degree}.")

            # Base initial values (static state for all K control points)
            initial_xyz_base_spline = torch.atanh(2 * (torch.rand(self.num_points, 2, device=self.device) - 0.5))

            _init_chol_c0_part1_spline = torch.tensor([1.0, 0.0, 1.0], device=self.device).view(1, 3).repeat(self.num_points, 1)
            _init_chol_c0_part2_spline = torch.randn(self.num_points, 3, device=self.device) * 0.01
            initial_cholesky_base_spline = _init_chol_c0_part1_spline + _init_chol_c0_part2_spline

            self._xyz_control_points = nn.Parameter(initial_xyz_base_spline.unsqueeze(1).repeat(1, self.K_control_points, 1))
            self._cholesky_control_points = nn.Parameter(initial_cholesky_base_spline.unsqueeze(1).repeat(1, self.K_control_points, 1))

            self._initial_xyz_for_custom_init = initial_xyz_base_spline.clone().detach()
        else:
            raise ValueError(f"Unknown trajectory_model_type: {self.trajectory_model_type}")

        # Initialize _opacity_coeffs: (num_coeffs_opacity, N, 1) - always polynomial for now
        # c0 (constant term) initialized to give opacity of 0.5 initially (logit(0.5)=0)
        initial_opacity_c0 = torch.zeros(self.num_points, 1, device=self.device)
        # c1...cD (higher order terms) are zero
        higher_order_opacity_coeffs = torch.zeros(self.opacity_polynomial_degree, self.num_points, 1, device=self.device)
        self._opacity_coeffs = nn.Parameter(torch.cat([initial_opacity_c0.unsqueeze(0), higher_order_opacity_coeffs], dim=0))

        # Static parameters: features_dc are still static for now
        self._features_dc = nn.Parameter(torch.randn(self.num_points, 3, device=self.device))

        # --- New Initialization Logic for Color and Opacity ---
        if gt_frames_for_init is not None and self.T > 0 and self.num_points > 0 and \
           gt_frames_for_init.shape[0] == self.T and \
           gt_frames_for_init.shape[2] == self.H and gt_frames_for_init.shape[3] == self.W:

            print(f"Using ground truth frames for Gaussian color and (cosine similarity based) opacity initialization with eps {initialization_logit_eps:.1e}.")

            initial_xyz_c0_detached = self._initial_xyz_for_custom_init # Shape (N, 2)

            # Compute pixel coordinates for all Gaussians simultaneously
            xy_normalized = torch.tanh(initial_xyz_c0_detached) # Values in [-1, 1)
            px_all = torch.clamp(torch.round((xy_normalized[:, 0] * 0.5 + 0.5) * (self.W -1)).long(), 0, self.W - 1)
            py_all = torch.clamp(torch.round((xy_normalized[:, 1] * 0.5 + 0.5) * (self.H -1)).long(), 0, self.H - 1)

            # Sample initial colors for all Gaussians from random frames using advanced indexing
            rand_frame_idx = torch.randint(0, self.T, (self.num_points,), device=self.device)
            initial_colors_all_n = gt_frames_for_init[rand_frame_idx, :, py_all, px_all] # Shape (N, 3), values in [0,1]

            # Compute the logits for _features_dc in a single operation
            self._features_dc.data = _logit(initial_colors_all_n, initialization_logit_eps)

            # For opacity:
            # Gather all relevant pixel colors from gt_frames_for_init across all time steps for all Gaussian locations
            gt_pixel_colors_at_n_locations_all_frames = gt_frames_for_init[:, :, py_all, px_all] # Shape (T, 3, N)

            # Permute gt_pixel_colors_at_n_locations_all_frames to align dimensions for broadcasting: (T, 3, N) -> (N, T, 3)
            gt_colors_permuted = gt_pixel_colors_at_n_locations_all_frames.permute(2, 0, 1)
            # initial_colors_all_n.unsqueeze(1) has shape (N, 1, 3)
            # gt_colors_permuted has shape (N, T, 3)
            cosine_similarities = F.cosine_similarity(initial_colors_all_n.unsqueeze(1), gt_colors_permuted, dim=2, eps=1e-8) # Shape (N, T)

            # Calculate the average cosine similarity and the target opacity values for all Gaussians
            avg_cosine_similarity = cosine_similarities.mean(dim=1) # Shape (N)
            target_opacity_vals = torch.clamp(1.0 - (avg_cosine_similarity / 2.0), 0.0, 1.0) # Shape (N)

            # Compute the logits for the constant term of _opacity_coeffs
            opacity_c0_logits = _logit(target_opacity_vals, initialization_logit_eps) # Shape (N)

            # Update the _opacity_coeffs parameter directly with the resulting tensor
            current_opacity_coeffs = self._opacity_coeffs.data.clone()
            current_opacity_coeffs[0] = opacity_c0_logits.unsqueeze(-1) # Add channel dim
            self._opacity_coeffs.data = current_opacity_coeffs

            print("Finished custom initialization of color and opacity.")

            # --- BEGIN INITIAL ANALYSIS AFTER CUSTOM INIT ---
            with torch.no_grad():
                # Color Analysis
                raw_color_logits_init = self._features_dc.detach().cpu().numpy()
                # Directly compute activated colors to avoid property lookup issue during __init__
                activated_colors_init_tensor = self.rgb_activation(self._features_dc)
                final_colors_init = activated_colors_init_tensor.detach().cpu().numpy()
                print(f"\n--- Color Feature Analysis (After Custom Initialization) ---")
                if raw_color_logits_init.size > 0:
                    print("Raw Color Logits (_features_dc):")
                    print(f"  Shape: {raw_color_logits_init.shape}")
                    print(f"  Min: {np.min(raw_color_logits_init):.4f}, Max: {np.max(raw_color_logits_init):.4f}, Mean: {np.mean(raw_color_logits_init):.4f}, Median: {np.median(raw_color_logits_init):.4f}")
                    print("Final Colors (after sigmoid):")
                    print(f"  Shape: {final_colors_init.shape}")
                    print(f"  Min: {np.min(final_colors_init):.4f}, Max: {np.max(final_colors_init):.4f}, Mean: {np.mean(final_colors_init):.4f}, Median: {np.median(final_colors_init):.4f}")
                    for i_chan in range(final_colors_init.shape[1]):
                        print(f"  Channel {i_chan} (RGB) - Min: {np.min(final_colors_init[:, i_chan]):.4f}, Max: {np.max(final_colors_init[:, i_chan]):.4f}, Mean: {np.mean(final_colors_init[:, i_chan]):.4f}")
                else:
                    print("No color features to analyze (num_points might be 0).")
                print("--- End Initial Color Feature Analysis ---")

                # Opacity Analysis
                if self.num_points > 0:
                    opacity_c0_logits_init_np = self._opacity_coeffs[0].detach().cpu().numpy().flatten()
                    opacity_c0_sigmoid_init_np = torch.sigmoid(self._opacity_coeffs[0].detach().cpu()).numpy().flatten()
                    all_opacities_tensor_init = self.get_opacity.detach().cpu()
                    all_opacities_init_np = all_opacities_tensor_init.numpy().flatten()

                    print(f"\n--- Opacity Distribution Analysis (After Custom Initialization) ---")
                    if opacity_c0_logits_init_np.size > 0:
                        print("Initial Opacity C0 Logits (_opacity_coeffs[0]):")
                        print(f"  Shape: {opacity_c0_logits_init_np.shape}")
                        print(f"  Min: {np.min(opacity_c0_logits_init_np):.4f}, Max: {np.max(opacity_c0_logits_init_np):.4f}, Mean: {np.mean(opacity_c0_logits_init_np):.4f}, Median: {np.median(opacity_c0_logits_init_np):.4f}")
                        print("Initial Opacity C0 Sigmoid (_opacity_coeffs[0] after sigmoid):")
                        print(f"  Shape: {opacity_c0_sigmoid_init_np.shape}")
                        print(f"  Min: {np.min(opacity_c0_sigmoid_init_np):.4f}, Max: {np.max(opacity_c0_sigmoid_init_np):.4f}, Mean: {np.mean(opacity_c0_sigmoid_init_np):.4f}, Median: {np.median(opacity_c0_sigmoid_init_np):.4f}")

                    if all_opacities_init_np.size > 0:
                        print("Full Opacity Tensor (self.get_opacity, potentially time-varying):")
                        print(f"  Opacities Tensor Shape (T, N, 1): {all_opacities_tensor_init.shape}")
                        print(f"  Total Opacity Values Analyzed: {all_opacities_init_np.size}")
                        print(f"  Min Opacity: {np.min(all_opacities_init_np):.4f}, Max Opacity: {np.max(all_opacities_init_np):.4f}, Mean Opacity: {np.mean(all_opacities_init_np):.4f}, Median Opacity: {np.median(all_opacities_init_np):.4f}")
                        percentiles_opac = [10, 25, 50, 75, 90]
                        perc_values_opac = np.percentile(all_opacities_init_np, percentiles_opac)
                        for p_perc, v_perc in zip(percentiles_opac, perc_values_opac):
                            print(f"  {p_perc}th Percentile: {v_perc:.4f}")
                    else:
                        print("  No full opacity tensor values to analyze.")
                else:
                    print("Skipping initial opacity distribution analysis: No Gaussians.")
                print("--- End Initial Opacity Distribution Analysis ---")
            # --- END INITIAL ANALYSIS AFTER CUSTOM INIT ---

        else:
            if gt_frames_for_init is not None:
                print("Warning: Skipping custom Gaussian initialization. Conditions not met (e.g., no points, T=0, or frame shape mismatch).")
            # If not using custom init, _features_dc keeps its randn init, and _opacity_coeffs keeps its zeros init for c0.
            print("Using default (random/zero) initialization for color and opacity.")

            # --- BEGIN INITIAL ANALYSIS AFTER DEFAULT INIT ---
            with torch.no_grad():
                # Color Analysis
                raw_color_logits_default = self._features_dc.detach().cpu().numpy()
                # Directly compute activated colors to avoid property lookup issue during __init__
                activated_colors_default_tensor = self.rgb_activation(self._features_dc)
                final_colors_default = activated_colors_default_tensor.detach().cpu().numpy()
                print(f"\n--- Color Feature Analysis (Default Initial State) ---")
                if raw_color_logits_default.size > 0:
                    print("Raw Color Logits (_features_dc):")
                    print(f"  Shape: {raw_color_logits_default.shape}")
                    print(f"  Min: {np.min(raw_color_logits_default):.4f}, Max: {np.max(raw_color_logits_default):.4f}, Mean: {np.mean(raw_color_logits_default):.4f}, Median: {np.median(raw_color_logits_default):.4f}")
                    print("Final Colors (after sigmoid):")
                    print(f"  Shape: {final_colors_default.shape}")
                    print(f"  Min: {np.min(final_colors_default):.4f}, Max: {np.max(final_colors_default):.4f}, Mean: {np.mean(final_colors_default):.4f}, Median: {np.median(final_colors_default):.4f}")
                    for i_chan in range(final_colors_default.shape[1]):
                        print(f"  Channel {i_chan} (RGB) - Min: {np.min(final_colors_default[:, i_chan]):.4f}, Max: {np.max(final_colors_default[:, i_chan]):.4f}, Mean: {np.mean(final_colors_default[:, i_chan]):.4f}")
                else:
                    print("No color features to analyze (num_points might be 0).")
                print("--- End Default Initial Color Feature Analysis ---")

                # Opacity Analysis
                if self.num_points > 0:
                    opacity_c0_logits_default_np = self._opacity_coeffs[0].detach().cpu().numpy().flatten()
                    opacity_c0_sigmoid_default_np = torch.sigmoid(self._opacity_coeffs[0].detach().cpu()).numpy().flatten()
                    all_opacities_tensor_default = self.get_opacity.detach().cpu()
                    all_opacities_default_np = all_opacities_tensor_default.numpy().flatten()

                    print(f"\n--- Opacity Distribution Analysis (Default Initial State) ---")
                    if opacity_c0_logits_default_np.size > 0:
                        print("Initial Opacity C0 Logits (_opacity_coeffs[0]):")
                        print(f"  Shape: {opacity_c0_logits_default_np.shape}")
                        print(f"  Min: {np.min(opacity_c0_logits_default_np):.4f}, Max: {np.max(opacity_c0_logits_default_np):.4f}, Mean: {np.mean(opacity_c0_logits_default_np):.4f}, Median: {np.median(opacity_c0_logits_default_np):.4f}")
                        print("Initial Opacity C0 Sigmoid (_opacity_coeffs[0] after sigmoid):")
                        print(f"  Shape: {opacity_c0_sigmoid_default_np.shape}")
                        print(f"  Min: {np.min(opacity_c0_sigmoid_default_np):.4f}, Max: {np.max(opacity_c0_sigmoid_default_np):.4f}, Mean: {np.mean(opacity_c0_sigmoid_default_np):.4f}, Median: {np.median(opacity_c0_sigmoid_default_np):.4f}")

                    if all_opacities_default_np.size > 0:
                        print("Full Opacity Tensor (self.get_opacity, potentially time-varying):")
                        print(f"  Opacities Tensor Shape (T, N, 1): {all_opacities_tensor_default.shape}")
                        print(f"  Total Opacity Values Analyzed: {all_opacities_default_np.size}")
                        print(f"  Min Opacity: {np.min(all_opacities_default_np):.4f}, Max Opacity: {np.max(all_opacities_default_np):.4f}, Mean Opacity: {np.mean(all_opacities_default_np):.4f}, Median Opacity: {np.median(all_opacities_default_np):.4f}")
                        percentiles_opac_def = [10, 25, 50, 75, 90]
                        perc_values_opac_def = np.percentile(all_opacities_default_np, percentiles_opac_def)
                        for p_perc, v_perc in zip(percentiles_opac_def, perc_values_opac_def):
                            print(f"  {p_perc}th Percentile: {v_perc:.4f}")
                    else:
                        print("  No full opacity tensor values to analyze.")
                else:
                    print("Skipping default initial opacity distribution analysis: No Gaussians.")
                print("--- End Default Initial Opacity Distribution Analysis ---")
            # --- END INITIAL ANALYSIS AFTER DEFAULT INIT ---

        # --- End New Initialization Logic ---

        self.last_size = (self.H, self.W)
        self.register_buffer('cholesky_bound', torch.tensor([0.5, 0, 0.5]).view(1, 1, 3).to(self.device)) # Adjusted shape for broadcasting

        # Initialize EMA objects - will be None if lambda is 0
        self.ema_xyz = None
        self.ema_cholesky = None
        self.ema_decay = ema_decay # Store decay rate

        if self.lambda_xyz_coeffs_reg > 0: # This lambda is for polynomial coefficient regularization
            if self.trajectory_model_type == "polynomial":
                self.ema_xyz = EMA([self._xyz_coeffs], decay=self.ema_decay)
            elif self.trajectory_model_type == "bspline":
                # For B-splines, lambda_xyz_coeffs_reg might be repurposed for control point smoothness later
                # For now, if a reg is on, EMA tracks control points.
                self.ema_xyz = EMA([self._xyz_control_points], decay=self.ema_decay)

        if self.lambda_cholesky_coeffs_reg > 0: # This lambda is for polynomial coefficient regularization
            if self.trajectory_model_type == "polynomial":
                self.ema_cholesky = EMA([self._cholesky_coeffs], decay=self.ema_decay)
            elif self.trajectory_model_type == "bspline":
                self.ema_cholesky = EMA([self._cholesky_control_points], decay=self.ema_decay)

        if kwargs["opt_type"] == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=kwargs["lr"])
        else:
            self.optimizer = Adan(self.parameters(), lr=kwargs["lr"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)

        # Initialize neighbor information for rigidity loss
        if self.lambda_neighbor_rigidity > 0 and self.num_points > self.k_neighbors and self.T > 1 and self.k_neighbors > 0:
            self._initialize_neighbors()

    def _initialize_neighbors(self):
        """Initialize neighbor information for rigidity loss."""
        with torch.no_grad():
            # Get initial positions from the first frame
            if self.trajectory_model_type == "polynomial":
                xyz_t0 = torch.tanh(self._xyz_coeffs[0])  # Shape (N, 2)
            else:  # bspline
                xyz_t0 = torch.tanh(self._xyz_control_points[:, 0])  # Shape (N, 2)

            # Pairwise squared Euclidean distances
            pairwise_distances = torch.cdist(xyz_t0, xyz_t0, p=2.0)  # Shape (N, N)
            pairwise_distances.fill_diagonal_(float('inf'))  # Ignore distance to self

            initial_neighbor_distances, neighbor_indices = torch.topk(
                pairwise_distances, self.k_neighbors, dim=1, largest=False
            )
            self.register_buffer('initial_neighbor_distances', initial_neighbor_distances, persistent=False)
            self.register_buffer('neighbor_indices', neighbor_indices, persistent=False)

    def _get_evaluated_polynomials(self, t_values_for_eval: torch.Tensor):
        """Helper to evaluate all time-varying polynomials for a given set of t_values."""
        if t_values_for_eval.numel() == 0:
            # Return empty tensors with correct trailing dimensions
            return (
                torch.empty(0, self.num_points, 2, device=self.device),
                torch.empty(0, self.num_points, 3, device=self.device),
                torch.empty(0, self.num_points, 1, device=self.device)
            )

        if self.trajectory_model_type == "polynomial":
            # Evaluate t_power_matrix for XYZ and Cholesky
            t_power_matrix_eval_list_xyz_chol = [torch.ones_like(t_values_for_eval).unsqueeze(1)] # t^0
            if self.polynomial_degree > 0:
                for i in range(1, self.polynomial_degree + 1):
                    t_power_matrix_eval_list_xyz_chol.append(t_values_for_eval.pow(i).unsqueeze(1))
            t_power_matrix_eval_xyz_chol = torch.cat(t_power_matrix_eval_list_xyz_chol, dim=1)

            # Evaluate t_power_matrix for Opacity
            if self.polynomial_degree == self.opacity_polynomial_degree:
                t_power_matrix_eval_opacity = t_power_matrix_eval_xyz_chol
            else:
                t_power_matrix_eval_list_opacity = [torch.ones_like(t_values_for_eval).unsqueeze(1)] # t^0
                if self.opacity_polynomial_degree > 0:
                    for i in range(1, self.opacity_polynomial_degree + 1):
                        t_power_matrix_eval_list_opacity.append(t_values_for_eval.pow(i).unsqueeze(1))
                t_power_matrix_eval_opacity = torch.cat(t_power_matrix_eval_list_opacity, dim=1)

            # xyz: (num_coeffs, N, 2) @ (T_eval, num_coeffs)^T -> (N, 2, T_eval) -> (T_eval, N, 2)
            evaluated_xyz_logits = torch.einsum('dnp,td->tnp', self._xyz_coeffs, t_power_matrix_eval_xyz_chol)
            evaluated_xyz = torch.tanh(evaluated_xyz_logits)

            # cholesky: (num_coeffs, N, 3) @ (T_eval, num_coeffs)^T -> (N, 3, T_eval) -> (T_eval, N, 3)
            evaluated_cholesky_raw = torch.einsum('dnp,td->tnp', self._cholesky_coeffs, t_power_matrix_eval_xyz_chol)
            evaluated_cholesky = evaluated_cholesky_raw + self.cholesky_bound

            # opacity: (num_coeffs, N, 1) @ (T_eval, num_coeffs)^T -> (N, 1, T_eval) -> (T_eval, N, 1)
            evaluated_opacity_logits = torch.einsum('dnp,td->tnp', self._opacity_coeffs, t_power_matrix_eval_opacity)
            evaluated_opacity = self.opacity_activation(evaluated_opacity_logits)
        else:  # bspline
            # For B-splines, we'll use linear interpolation between control points
            evaluated_xyz = torch.zeros((t_values_for_eval.shape[0], self.num_points, 2), device=self.device)
            evaluated_cholesky = torch.zeros((t_values_for_eval.shape[0], self.num_points, 3), device=self.device)
            evaluated_opacity = torch.zeros((t_values_for_eval.shape[0], self.num_points, 1), device=self.device)

            for i, t in enumerate(t_values_for_eval):
                # Normalize t to [0, K-1] range
                t_normalized = t.item() * (self.K_control_points - 1)
                t_idx = int(t_normalized)
                t_frac = t_normalized - t_idx

                if t_idx >= self.K_control_points - 1:
                    # Use last control point
                    evaluated_xyz[i] = torch.tanh(self._xyz_control_points[:, -1])
                    evaluated_cholesky[i] = self._cholesky_control_points[:, -1] + self.cholesky_bound
                    evaluated_opacity[i] = self.opacity_activation(self._opacity_control_points[:, -1])
                else:
                    # Linear interpolation between control points
                    xyz1 = torch.tanh(self._xyz_control_points[:, t_idx])
                    xyz2 = torch.tanh(self._xyz_control_points[:, t_idx + 1])
                    evaluated_xyz[i] = xyz1 * (1 - t_frac) + xyz2 * t_frac

                    chol1 = self._cholesky_control_points[:, t_idx]
                    chol2 = self._cholesky_control_points[:, t_idx + 1]
                    evaluated_cholesky[i] = (chol1 * (1 - t_frac) + chol2 * t_frac) + self.cholesky_bound

                    opac1 = self._opacity_control_points[:, t_idx]
                    opac2 = self._opacity_control_points[:, t_idx + 1]
                    evaluated_opacity[i] = self.opacity_activation(opac1 * (1 - t_frac) + opac2 * t_frac)

        return evaluated_xyz, evaluated_cholesky, evaluated_opacity

    @property
    def get_xyz(self):
        """Evaluates polynomial for means for ALL self.T frames, returns (T, N, 2)."""
        if self.T == 0:
            return torch.empty(0, self.num_points, 2, device=self.device)
        evaluated_xyz_logits = torch.einsum('dnp,td->tnp', self._xyz_coeffs, self.t_power_matrix)
        return torch.tanh(evaluated_xyz_logits)

    @property
    def get_features(self):
        """Returns colors, shape (N, 3)."""
        # Apply activation
        return self.rgb_activation(self._features_dc)

    @property
    def get_opacity(self):
        """Evaluates polynomial for opacities for ALL self.T frames, returns (T, N, 1)."""
        if self.T == 0:
            return torch.empty(0, self.num_points, 1, device=self.device)
        evaluated_opacity_logits = torch.einsum('dnp,td->tnp', self._opacity_coeffs, self.t_power_matrix_opacity)
        return self.opacity_activation(evaluated_opacity_logits)

    @property
    def get_cholesky_elements(self):
        """Evaluates polynomial for Cholesky elements for ALL self.T frames, returns (T, N, 3)."""
        if self.T == 0:
            return torch.empty(0, self.num_points, 3, device=self.device)
        evaluated_cholesky_raw = torch.einsum('dnp,td->tnp', self._cholesky_coeffs, self.t_power_matrix)
        return evaluated_cholesky_raw + self.cholesky_bound

    def get_xyz_dynamics_ranking_indices(self, top_k_percentage=0.10):
        """Calculates a metric for Gaussian XYZ dynamics based on non-constant polynomial coefficients
           and returns the indices of the top_k_percentage most dynamic Gaussians.
        """
        if self.polynomial_degree == 0 or self.num_points == 0:
            return torch.empty(0, dtype=torch.long, device=self.device) # No non-constant terms or no points

        # self._xyz_coeffs shape: (num_coeffs, N, 2)
        # Non-constant coefficients are _xyz_coeffs[1:]
        non_constant_xyz_coeffs = self._xyz_coeffs[1:] # Shape: (polynomial_degree, N, 2)

        # Calculate L2 norm of non-constant coefficients for each Gaussian
        # Sum of squares of these coefficient components for each Gaussian
        # Sum over polynomial_degree (dim 0) and coordinate dimension (dim 2)
        movement_metric = torch.sum(non_constant_xyz_coeffs**2, dim=(0, 2)) # Shape: (N)

        if movement_metric.numel() == 0: # Should be caught by num_points == 0 but as a safeguard
             return torch.empty(0, dtype=torch.long, device=self.device)

        num_top_k = min(self.num_points, max(0, int(self.num_points * top_k_percentage)))
        if num_top_k == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)

        _, top_indices = torch.topk(movement_metric, k=num_top_k)
        return top_indices

    def cluster_gaussians(self, method='temporal_spatial', n_clusters=5, use_gpu=True,
                         spatial_weight=0.3, motion_weight=0.6, color_weight=0.1):
        """Cluster Gaussians based on their temporal and spatial relationships to identify potential objects.

        Args:
            method (str): Clustering method to use. Options:
                - 'temporal_spatial': Focus on motion coherence and spatial relationships
            n_clusters (int): Number of clusters to create
            use_gpu (bool): Whether to use GPU for computations if available
            spatial_weight (float): Weight for spatial features (0-1)
            motion_weight (float): Weight for motion features (0-1)
            color_weight (float): Weight for color features (0-1)

        Returns:
            dict: Dictionary containing:
                - 'labels': Cluster labels for each Gaussian
                - 'centers': Cluster centers
                - 'features': The features used for clustering
                - 'feature_weights': The weights used for each feature type
        """
        import torch.nn.functional as F
        from sklearn.cluster import KMeans
        import numpy as np

        if self.num_points == 0:
            return {'labels': np.array([]), 'centers': None, 'features': None, 'feature_weights': None}

        # Move computations to CPU for sklearn
        device = torch.device('cpu')

        # 1. Extract spatial features (initial position and scale)
        if self.trajectory_model_type == "polynomial":
            initial_xyz = torch.tanh(self._xyz_coeffs[0]).detach().cpu().numpy()  # (N, 2)
            initial_cholesky = self._cholesky_coeffs[0].detach().cpu().numpy()  # (N, 3)
        else:  # bspline
            initial_xyz = torch.tanh(self._xyz_control_points[:, 0]).detach().cpu().numpy()  # (N, 2)
            initial_cholesky = self._cholesky_control_points[:, 0].detach().cpu().numpy()  # (N, 3)

        spatial_features = np.concatenate([initial_xyz, initial_cholesky], axis=1)  # (N, 5)

        # 2. Extract temporal features (motion patterns)
        if self.trajectory_model_type == "polynomial":
            # Get non-constant coefficients for xyz trajectories
            motion_coeffs = self._xyz_coeffs[1:].detach().cpu().numpy()  # (poly_degree, N, 2)
        else:  # bspline
            # For B-splines, we'll use the differences between consecutive control points
            # as a proxy for motion
            motion_coeffs = (self._xyz_control_points[:, 1:] - self._xyz_control_points[:, :-1]).detach().cpu().numpy()  # (N, K-1, 2)
            motion_coeffs = motion_coeffs.transpose(1, 0, 2)  # (K-1, N, 2)

        # Calculate velocity and acceleration at key points in time
        t_points = np.linspace(0, 1, 5)  # Sample 5 points in time
        velocities = []
        accelerations = []

        for t in t_points:
            if self.trajectory_model_type == "polynomial":
                # First derivative (velocity)
                vel = np.zeros((self.num_points, 2))
                for i in range(1, self.polynomial_degree + 1):
                    vel += i * motion_coeffs[i-1] * (t ** (i-1))
                velocities.append(vel)

                # Second derivative (acceleration)
                acc = np.zeros((self.num_points, 2))
                for i in range(2, self.polynomial_degree + 1):
                    acc += i * (i-1) * motion_coeffs[i-1] * (t ** (i-2))
                accelerations.append(acc)
            else:  # bspline
                # For B-splines, we'll use the control point differences as velocity
                # and their changes as acceleration
                t_idx = int(t * (self.K_control_points - 1))
                t_idx = min(t_idx, self.K_control_points - 2)  # Ensure we don't go out of bounds
                vel = motion_coeffs[t_idx]  # (N, 2)
                velocities.append(vel)

                if t_idx < self.K_control_points - 2:
                    acc = motion_coeffs[t_idx + 1] - motion_coeffs[t_idx]  # (N, 2)
                else:
                    acc = np.zeros_like(vel)
                accelerations.append(acc)

        # Stack velocities and accelerations
        velocities = np.stack(velocities, axis=1)  # (N, 5, 2)
        accelerations = np.stack(accelerations, axis=1)  # (N, 5, 2)

        # Flatten temporal features
        temporal_features = np.concatenate([
            velocities.reshape(self.num_points, -1),  # (N, 10)
            accelerations.reshape(self.num_points, -1),  # (N, 10)
            motion_coeffs.reshape(self.num_points, -1)  # (N, poly_degree * 2) or (N, (K-1) * 2)
        ], axis=1)

        # 3. Extract color features (as a secondary cue)
        colors = self.get_features.detach().cpu().numpy()  # (N, 3)

        # Normalize each feature type
        spatial_features = (spatial_features - spatial_features.mean(axis=0)) / (spatial_features.std(axis=0) + 1e-8)
        temporal_features = (temporal_features - temporal_features.mean(axis=0)) / (temporal_features.std(axis=0) + 1e-8)
        colors = (colors - colors.mean(axis=0)) / (colors.std(axis=0) + 1e-8)

        # Apply weights to each feature type
        weighted_spatial = spatial_features * spatial_weight
        weighted_temporal = temporal_features * motion_weight
        weighted_color = colors * color_weight

        # Combine all features
        features = np.concatenate([weighted_spatial, weighted_temporal, weighted_color], axis=1)

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(features)

        return {
            'labels': labels,
            'centers': kmeans.cluster_centers_,
            'features': features,
            'feature_weights': {
                'spatial': spatial_weight,
                'motion': motion_weight,
                'color': color_weight
            }
        }

    def visualize_clusters(self, cluster_results, frame_idx=0, save_path=None, show_velocity=False):
        """Visualize the clustering results by coloring Gaussians based on their cluster.

        Args:
            cluster_results (dict): Results from cluster_gaussians()
            frame_idx (int): Which frame to visualize
            save_path (str, optional): Path to save visualization
            show_velocity (bool): Whether to show velocity vectors
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if self.num_points == 0:
            print("No Gaussians to visualize")
            return

        # Get positions and velocities for the specified frame
        t_value = self.t_values[frame_idx]

        # Get positions based on trajectory model type
        if self.trajectory_model_type == "polynomial":
            # Create t_power_matrix for evaluation
            t_power_matrix = torch.stack([t_value.pow(i) for i in range(self.polynomial_degree + 1)], dim=0)
            # Evaluate polynomial
            evaluated_xyz_logits = torch.einsum('dnp,d->np', self._xyz_coeffs, t_power_matrix)
            positions = torch.tanh(evaluated_xyz_logits).detach().cpu().numpy()  # (N, 2)

            # Calculate velocity if needed
            if show_velocity:
                vel = torch.zeros((self.num_points, 2), device=self.device)
                for i in range(1, self.polynomial_degree + 1):
                    vel += i * self._xyz_coeffs[i] * (t_value ** (i-1))
                vel = vel.detach().cpu().numpy()
        else:  # bspline
            # For B-splines, we'll use the control points directly
            # Find the two control points that bound the current time
            t_normalized = t_value.item() * (self.K_control_points - 1)
            t_idx = int(t_normalized)
            t_frac = t_normalized - t_idx

            if t_idx >= self.K_control_points - 1:
                positions = torch.tanh(self._xyz_control_points[:, -1]).detach().cpu().numpy()
                if show_velocity:
                    vel = (self._xyz_control_points[:, -1] - self._xyz_control_points[:, -2]).detach().cpu().numpy()
            else:
                # Linear interpolation between control points
                pos1 = torch.tanh(self._xyz_control_points[:, t_idx])
                pos2 = torch.tanh(self._xyz_control_points[:, t_idx + 1])
                positions = (pos1 * (1 - t_frac) + pos2 * t_frac).detach().cpu().numpy()
                if show_velocity:
                    vel = (self._xyz_control_points[:, t_idx + 1] - self._xyz_control_points[:, t_idx]).detach().cpu().numpy()

        # Create figure with white background
        plt.figure(figsize=(12, 12), facecolor='white')
        plt.gca().set_facecolor('white')

        # Define vibrant colors for clusters
        n_clusters = len(np.unique(cluster_results['labels']))
        cluster_colors = [
            '#FF0000',  # Red
            '#00FF00',  # Green
            '#0000FF',  # Blue
            '#FF00FF',  # Magenta
            '#00FFFF',  # Cyan
            '#FFA500',  # Orange
            '#800080',  # Purple
            '#008000',  # Dark Green
            '#000080',  # Navy
            '#800000',  # Maroon
        ][:n_clusters]  # Take only as many colors as we have clusters

        # Plot each Gaussian as a point, colored by its cluster
        for cluster_id in range(n_clusters):
            mask = cluster_results['labels'] == cluster_id
            cluster_color = cluster_colors[cluster_id]

            # Plot points with cluster color
            plt.scatter(positions[mask, 0], positions[mask, 1],
                       c=[cluster_color], alpha=1.0, s=100,  # Much larger points, full opacity
                       label=f'Cluster {cluster_id}',
                       edgecolor='black',  # Add black outline
                       linewidth=0.5)  # Thin outline

            if show_velocity and 'vel' in locals():
                # Plot velocity vectors for this cluster
                # Scale velocity for better visualization
                vel_scale = 0.2  # Increased scale for more visible arrows
                plt.quiver(positions[mask, 0], positions[mask, 1],
                          vel[mask, 0] * vel_scale, vel[mask, 1] * vel_scale,
                          color=cluster_color, alpha=0.8, scale=30,
                          width=0.005,  # Thicker arrows
                          headwidth=5,  # Larger arrow heads
                          headlength=7)

        plt.title(f'Gaussian Clusters (Frame {frame_idx})\n'
                 f'Weights: Spatial={cluster_results["feature_weights"]["spatial"]:.1f}, '
                 f'Motion={cluster_results["feature_weights"]["motion"]:.1f}, '
                 f'Color={cluster_results["feature_weights"]["color"]:.1f}',
                 pad=20, fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')

        # Set axis limits with some padding
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
        padding = 0.1
        plt.xlim(x_min - padding, x_max + padding)
        plt.ylim(y_min - padding, y_max + padding)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close()
        else:
            plt.show()

    def forward(self, t_values_to_render: Optional[torch.Tensor] = None, background_color_override: Optional[torch.Tensor] = None):
        """Render frames specified by t_values_to_render.
        If t_values_to_render is None, renders all self.T frames.
        """
        if t_values_to_render is None:
            t_values_to_render = self.t_values # Use all stored t_values if none are provided
            if self.T == 0: # No frames to render if model T is 0
                 return {"render": torch.empty(0, 3, self.H, self.W, device=self.device)}
        elif t_values_to_render.numel() == 0:
            # If specific t_values are provided but empty, render nothing
            return {"render": torch.empty(0, 3, self.H, self.W, device=self.device)}

        num_frames_to_render = t_values_to_render.shape[0]

        # Use provided background override if given, otherwise use binary random background during training
        if background_color_override is not None:
            active_background = background_color_override
        else:
            # Generate binary random background (randomly choose between black and white)
            is_white = torch.rand(1, device=self.device) > 0.5
            active_background = torch.ones(3, device=self.device) if is_white else torch.zeros(3, device=self.device)

        # Get all parameters evaluated at the specified t_values
        evaluated_xyz, evaluated_cholesky, evaluated_opacity = self._get_evaluated_polynomials(t_values_to_render)
        static_colors = self.get_features # (N, 3), still static

        all_frames_rendered = []
        for k in range(num_frames_to_render):
            means_k = evaluated_xyz[k]             # (N, 2)
            cholesky_k = evaluated_cholesky[k]       # (N, 3)
            opacities_k = evaluated_opacity[k]         # (N, 1)

            xys, depths_from_projection, radii, conics, num_tiles_hit = project_gaussians_2d(
                means_k, cholesky_k, self.H, self.W, self.tile_bounds
            )
            effective_depths = 1.0 - opacities_k

            out_img_k = rasterize_gaussians_sum(
                xys, effective_depths, radii, conics, num_tiles_hit,
                static_colors, opacities_k,
                self.H, self.W, self.BLOCK_H, self.BLOCK_W,
                background=active_background, return_alpha=False
            )
            out_img_k = torch.clamp(out_img_k, 0, 1) # [H, W, 3]
            all_frames_rendered.append(out_img_k.permute(2,0,1)) # To C, H, W

        if not all_frames_rendered:
             return {"render": torch.empty(0, 3, self.H, self.W, device=self.device)}

        final_rendered_tensor = torch.stack(all_frames_rendered, dim=0) # (num_frames_to_render, C, H, W)
        return {"render": final_rendered_tensor}

    def train_iter(self, gt_frames_batch: torch.Tensor, t_values_batch: torch.Tensor):
        """Performs one training iteration using a batch of frames and their t_values."""
        total_loss = 0
        total_psnr = 0
        T_batch = gt_frames_batch.shape[0]

        if T_batch == 0: # Should be caught by caller, but as a safeguard
            return 0.0, 0.0

        rendered_frames_pkg = self.forward(t_values_to_render=t_values_batch)
        rendered_frames = rendered_frames_pkg["render"] # (T_batch, C, H, W)

        # Calculate loss and PSNR per frame in the batch and average
        for t_idx_in_batch in range(T_batch):
            loss_t = loss_fn(rendered_frames[t_idx_in_batch:t_idx_in_batch+1], gt_frames_batch[t_idx_in_batch:t_idx_in_batch+1], self.loss_type, lambda_value=0.7)
            total_loss += loss_t
            with torch.no_grad():
                mse_loss_t = F.mse_loss(rendered_frames[t_idx_in_batch:t_idx_in_batch+1], gt_frames_batch[t_idx_in_batch:t_idx_in_batch+1])
                psnr_t = 10 * math.log10(1.0 / max(mse_loss_t.item(), 1e-10)) # Avoid log(0)
                total_psnr += psnr_t

        average_loss = total_loss / T_batch
        average_psnr = total_psnr / T_batch

        # Add L2 regularization on polynomial coefficients (excluding constant term)
        if self.polynomial_degree > 0:
            if self.lambda_xyz_coeffs_reg > 0:
                # _xyz_coeffs shape: (num_coeffs, N, 2)
                # Regularize coeffs[1:] (linear term upwards)
                xyz_coeffs_to_reg = self._xyz_coeffs[1:]
                xyz_coeffs_l2_loss = self.lambda_xyz_coeffs_reg * torch.mean(xyz_coeffs_to_reg**2)
                average_loss += xyz_coeffs_l2_loss

            if self.lambda_cholesky_coeffs_reg > 0:
                # _cholesky_coeffs shape: (num_coeffs, N, 3)
                # Regularize coeffs[1:] (linear term upwards)
                cholesky_coeffs_to_reg = self._cholesky_coeffs[1:]
                cholesky_coeffs_l2_loss = self.lambda_cholesky_coeffs_reg * torch.mean(cholesky_coeffs_to_reg**2)
                average_loss += cholesky_coeffs_l2_loss

            if self.lambda_opacity_coeffs_reg > 0 and self.opacity_polynomial_degree > 0:
                # _opacity_coeffs shape: (num_coeffs_opacity, N, 1)
                # Regularize coeffs[1:] (linear term upwards)
                opacity_coeffs_to_reg = self._opacity_coeffs[1:]
                opacity_coeffs_l2_loss = self.lambda_opacity_coeffs_reg * torch.mean(opacity_coeffs_to_reg**2)
                average_loss += opacity_coeffs_l2_loss

        # Add neighbor rigidity loss
        if self.lambda_neighbor_rigidity > 0 and self.T > 1 and hasattr(self, 'neighbor_indices') and hasattr(self, 'initial_neighbor_distances'):
            rigidity_loss_accumulator = 0.0
            num_valid_frames_for_rigidity = 0
            all_xyz_current_frames = self.get_xyz # Get all current xyz states (T, N, 2)

            for t in range(1, self.T): # Compare frames t > 0 to the t=0 structure
                current_xyz_at_t = all_xyz_current_frames[t] # Shape (N, 2)

                # For each point i, its position is current_xyz_at_t[i]
                # Its neighbors' indices are self.neighbor_indices[i] (shape k)
                # Their positions at time t are current_xyz_at_t[self.neighbor_indices[i]] (shape k, 2)

                # Positions of center points for distance calculation (expanded for broadcasting)
                # Unsqueeze to make it (N, 1, 2) to subtract from (N, k, 2)
                center_points_pos_t = current_xyz_at_t.unsqueeze(1) # (N, 1, 2)

                # Gather positions of neighbors at current time t
                # self.neighbor_indices has shape (N, k_neighbors)
                # current_xyz_at_t has shape (N, 2)
                # current_xyz_at_t[self.neighbor_indices] directly gathers, resulting in (N, k_neighbors, 2)
                neighbor_points_pos_t = current_xyz_at_t[self.neighbor_indices]

                # Calculate squared Euclidean distances to neighbors at current time t
                dist_sq_to_neighbors_t = torch.sum((center_points_pos_t - neighbor_points_pos_t)**2, dim=2) # Shape (N, k_neighbors)
                current_distances_to_neighbors_t = torch.sqrt(dist_sq_to_neighbors_t + 1e-8) # Add epsilon for stability

                # Difference from initial distances
                distance_diff = self.initial_neighbor_distances - current_distances_to_neighbors_t
                rigidity_loss_accumulator += torch.mean(distance_diff**2)
                num_valid_frames_for_rigidity += 1

            if num_valid_frames_for_rigidity > 0:
                mean_rigidity_loss = rigidity_loss_accumulator / num_valid_frames_for_rigidity
                average_loss += self.lambda_neighbor_rigidity * mean_rigidity_loss

        # Backpropagate the average loss
        self.optimizer.zero_grad(set_to_none=True)
        average_loss.backward()
        self.optimizer.step()

        # Update EMA for temporal losses if they are active
        if self.lambda_xyz_coeffs_reg > 0 and self.ema_xyz is not None:
            self.ema_xyz.update() # EMA object internally refers to the parameter it tracks
        if self.lambda_cholesky_coeffs_reg > 0 and self.ema_cholesky is not None:
            self.ema_cholesky.update() # EMA object internally refers to the parameter it tracks

        self.scheduler.step()

        # print(f"Debug: type(average_loss)={type(average_loss)}, type(average_psnr)={type(average_psnr)}")
        return average_loss.item(), average_psnr
