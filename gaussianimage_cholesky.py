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
from scipy.interpolate import BSpline

# --- Optical Flow Loss Constants (User can modify these manually) ---
ENABLE_OPTICAL_FLOW_LOSS = True # Master switch for this feature
LAMBDA_OPTICAL_FLOW = 1e4      # Weight for the optical flow loss term
OPTICAL_FLOW_OPACITY_THRESHOLD = 0.2 # Min opacity for a Gaussian to be affected by flow loss
# --- End Optical Flow Loss Constants ---

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
            eps = 1e-6
            uniform_values = torch.rand(self.num_points, 2, device=self.device) * (1 - 2*eps) + eps
            initial_xyz_base_spline = torch.atanh(2 * uniform_values - 1)

            # Initialize with much larger covariance (1.0 standard deviation)
            _init_chol_c0_part1_spline = torch.tensor([1.0, 0.0, 1.0], device=self.device).view(1, 3).repeat(self.num_points, 1)
            _init_chol_c0_part2_spline = torch.randn(self.num_points, 3, device=self.device) * 0.01
            initial_cholesky_base_spline = _init_chol_c0_part1_spline + _init_chol_c0_part2_spline

            # Initialize control points with small random offsets
            xyz_control_points = []
            for k in range(self.K_control_points):
                if k == 0:
                    # First control point is the base position
                    xyz_control_points.append(initial_xyz_base_spline)
                else:
                    # Subsequent control points have small random offsets
                    offset = torch.randn(self.num_points, 2, device=self.device) * 0.1  # Small random offset
                    xyz_control_points.append(initial_xyz_base_spline + offset)
            self._xyz_control_points = nn.Parameter(torch.stack(xyz_control_points, dim=1))  # Shape: (N, K, 2)

            # Initialize Cholesky control points with small random offsets
            cholesky_control_points = []
            for k in range(self.K_control_points):
                if k == 0:
                    # First control point is the base position
                    cholesky_control_points.append(initial_cholesky_base_spline)
                else:
                    # Subsequent control points have small random offsets
                    offset = torch.randn(self.num_points, 3, device=self.device) * 0.01  # Smaller offset for Cholesky
                    cholesky_control_points.append(initial_cholesky_base_spline + offset)
            self._cholesky_control_points = nn.Parameter(torch.stack(cholesky_control_points, dim=1))  # Shape: (N, K, 3)

            self._initial_xyz_for_custom_init = initial_xyz_base_spline.clone().detach()

            # Pre-compute and cache basis functions for all possible t values
            self.bspline_resolution = 10000  # Increased from 1000 to handle more interpolated frames
            t_grid = torch.linspace(0, 1, self.bspline_resolution, device=self.device)

            # Create knot vector with repeated knots at start and end
            total_knots = self.K_control_points + self.bspline_degree + 1
            knots = torch.zeros(total_knots, device=self.device)

            # First p+1 knots are 0
            knots[:self.bspline_degree + 1] = 0.0

            # Last p+1 knots are 1
            knots[-self.bspline_degree - 1:] = 1.0

            # Calculate number of middle knots that need to be set
            num_middle_knots = total_knots - 2 * (self.bspline_degree + 1)
            if num_middle_knots > 0:
                # Create evenly spaced values between 0 and 1 for middle knots
                middle_values = torch.linspace(0, 1, num_middle_knots + 2, device=self.device)[1:-1]  # Exclude 0 and 1
                middle_start_idx = self.bspline_degree + 1
                knots[middle_start_idx:middle_start_idx + num_middle_knots] = middle_values

            # Create basis functions directly (vectorized)
            basis_functions = torch.zeros(self.bspline_resolution, self.K_control_points, device=self.device)
            eps = 1e-6  # Small epsilon for numerical stability

            # Compute basis functions for each control point
            for i in range(self.K_control_points):
                # Initialize degree 0 basis functions
                basis = torch.zeros(self.bspline_resolution, device=self.device)
                for j in range(self.bspline_degree + 1):
                    t_start = knots[i + j]
                    t_end = knots[i + j + 1]
                    valid_mask = (t_grid >= t_start - eps) & (t_grid < t_end + eps)
                    basis = torch.where(valid_mask, torch.ones_like(basis), basis)

                # Recursively compute higher degree basis functions
                for d in range(1, self.bspline_degree + 1):
                    new_basis = torch.zeros_like(basis)
                    for j in range(self.bspline_degree - d + 1):
                        # Get knot values for this basis function
                        t_start = knots[i + j]
                        t_end = knots[i + j + d + 1]
                        t_mid = knots[i + j + d]

                        # Calculate denominators with epsilon check
                        denom1 = t_mid - t_start
                        denom2 = t_end - t_mid

                        # Create masks for valid ranges with epsilon
                        valid_range1 = (t_grid >= t_start - eps) & (t_grid < t_mid + eps) & (denom1 > eps)
                        valid_range2 = (t_grid >= t_mid - eps) & (t_grid < t_end + eps) & (denom2 > eps)

                        # Calculate basis function values with safe division
                        term1 = torch.where(valid_range1,
                                          (t_grid - t_start) / (denom1 + eps) * basis,
                                          torch.zeros_like(basis))
                        term2 = torch.where(valid_range2,
                                          (t_end - t_grid) / (denom2 + eps) * basis,
                                          torch.zeros_like(basis))

                        new_basis = new_basis + term1 + term2

                    basis = new_basis

                # Store the basis functions for this control point
                basis_functions[:, i] = basis

            # Normalize basis functions to ensure they sum to 1 at each point
            basis_sum = basis_functions.sum(dim=1, keepdim=True)
            basis_functions = basis_functions / (basis_sum + eps)

            self.register_buffer('basis_functions_grid', basis_functions)
            self.register_buffer('t_grid', t_grid)

        else:
            raise ValueError(f"Unknown trajectory_model_type: {self.trajectory_model_type}")

        # Initialize _opacity_coeffs: (num_coeffs_opacity, N, 1) - always polynomial for now
        # c0 (constant term) initialized to give opacity of 0.5 initially (logit(0.5)=0)
        initial_opacity_c0 = torch.zeros(self.num_points, 1, device=self.device)
        # c1...cD (higher order terms) are zero
        higher_order_opacity_coeffs = torch.zeros(self.opacity_polynomial_degree, self.num_points, 1, device=self.device)
        self._opacity_coeffs = nn.Parameter(torch.cat([initial_opacity_c0.unsqueeze(0), higher_order_opacity_coeffs], dim=0))

        # Static parameters: features_dc are still static for now
        # self._opacity = nn.Parameter(torch.ones((self.num_points, 1), device=self.device))) # Old static opacity
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

            # For B-spline model, initialize all control points based on the initial position
            if self.trajectory_model_type == "bspline":
                # Initialize all control points to exactly the same position for static start
                self._xyz_control_points.data = initial_xyz_c0_detached.unsqueeze(1).repeat(1, self.K_control_points, 1)

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
        # self.register_buffer('background', torch.rand(3, device=self.device)) # Removed: background will be handled in forward
        # self.opacity_activation = torch.sigmoid
        # self.rgb_activation = torch.sigmoid
        # self.register_buffer('bound', torch.tensor([0.5, 0.5]).view(1, 2).to(self.device)) # Not used?
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
            with torch.no_grad(): # Operations here should not be part of the computation graph for _xyz init
                # For B-splines, use the first control point as initial position
                if self.trajectory_model_type == "bspline":
                    xyz_t0 = torch.tanh(self._xyz_control_points[:, 0])  # Shape (N, 2)
                else:  # polynomial
                    xyz_t0 = torch.tanh(self._xyz_coeffs[0])  # Shape (N, 2)

                # Pairwise squared Euclidean distances
                pairwise_distances = torch.cdist(xyz_t0, xyz_t0, p=2.0) # Shape (N, N)
                pairwise_distances.fill_diagonal_(float('inf')) # Ignore distance to self

                initial_neighbor_distances, neighbor_indices = torch.topk(
                    pairwise_distances, self.k_neighbors, dim=1, largest=False
                )
                self.register_buffer('initial_neighbor_distances', initial_neighbor_distances, persistent=False)
                self.register_buffer('neighbor_indices', neighbor_indices, persistent=False)
        else:
            if self.lambda_neighbor_rigidity > 0:
                print("Warning: Neighbor rigidity loss not initialized due to insufficient points, T<=1, or k_neighbors<=0.")

    def _get_evaluated_polynomials(self, t_values_for_eval: torch.Tensor):
        """Helper to evaluate all time-varying polynomials for a given set of t_values."""
        if t_values_for_eval.numel() == 0:
            num_coeffs_xyz_chol = self.polynomial_degree + 1
            num_coeffs_opacity = self.opacity_polynomial_degree + 1
            # Return empty tensors with correct trailing dimensions
            return (
                torch.empty(0, self.num_points, 2, device=self.device),
                torch.empty(0, self.num_points, 3, device=self.device),
                torch.empty(0, self.num_points, 1, device=self.device)
            )

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

        return evaluated_xyz, evaluated_cholesky, evaluated_opacity

    def _get_evaluated_bsplines(self, t_values_for_eval: torch.Tensor):
        """Helper to evaluate all time-varying B-splines for a given set of t_values using pre-computed basis functions."""
        if t_values_for_eval.numel() == 0:
            return (
                torch.empty(0, self.num_points, 2, device=self.device),
                torch.empty(0, self.num_points, 3, device=self.device),
                torch.empty(0, self.num_points, 1, device=self.device)
            )

        # Ensure t is in [0, 1] range
        t_values_for_eval = torch.clamp(t_values_for_eval, 0.0, 1.0)

        # Get grid indices with interpolation (vectorized)
        t_scaled = t_values_for_eval * (self.bspline_resolution - 1)
        t_floor = torch.floor(t_scaled).long()
        t_ceil = torch.ceil(t_scaled).long()
        t_frac = t_scaled - t_floor.float()

        # Ensure indices are within bounds (vectorized)
        t_floor = torch.clamp(t_floor, 0, self.bspline_resolution - 1)
        t_ceil = torch.clamp(t_ceil, 0, self.bspline_resolution - 1)

        # Get basis functions for both floor and ceil indices (vectorized)
        basis_floor = self.basis_functions_grid[t_floor]  # Shape: (T, K)
        basis_ceil = self.basis_functions_grid[t_ceil]    # Shape: (T, K)

        # Linear interpolation between floor and ceil basis functions (vectorized)
        basis_functions = (1 - t_frac.unsqueeze(1)) * basis_floor + t_frac.unsqueeze(1) * basis_ceil  # Shape: (T, K)

        # Evaluate XYZ positions (vectorized)
        evaluated_xyz_logits = torch.einsum('tk,nkp->tnp', basis_functions, self._xyz_control_points)  # Shape: (T, N, 2)
        evaluated_xyz = torch.tanh(evaluated_xyz_logits)

        # Evaluate Cholesky elements (vectorized)
        evaluated_cholesky_raw = torch.einsum('tk,nkp->tnp', basis_functions, self._cholesky_control_points)  # Shape: (T, N, 3)
        evaluated_cholesky = evaluated_cholesky_raw + self.cholesky_bound

        # For opacity, we still use polynomial evaluation
        # Create power matrix for opacity evaluation (vectorized)
        t_power_matrix_opacity = torch.ones_like(t_values_for_eval).unsqueeze(1)  # t^0
        if self.opacity_polynomial_degree > 0:
            for i in range(1, self.opacity_polynomial_degree + 1):
                t_power_matrix_opacity = torch.cat([t_power_matrix_opacity, t_values_for_eval.pow(i).unsqueeze(1)], dim=1)

        evaluated_opacity_logits = torch.einsum('dnp,td->tnp', self._opacity_coeffs, t_power_matrix_opacity)
        evaluated_opacity = self.opacity_activation(evaluated_opacity_logits)

        return evaluated_xyz, evaluated_cholesky, evaluated_opacity

    @property
    def get_xyz(self):
        """Evaluates means for ALL self.T frames, returns (T, N, 2)."""
        if self.T == 0:
            return torch.empty(0, self.num_points, 2, device=self.device)

        if self.trajectory_model_type == "polynomial":
            evaluated_xyz_logits = torch.einsum('dnp,td->tnp', self._xyz_coeffs, self.t_power_matrix)
            return torch.tanh(evaluated_xyz_logits)
        elif self.trajectory_model_type == "bspline":
            # Compute knot vector for cubic B-splines
            knots = torch.linspace(0, 1, self.K_control_points + self.bspline_degree + 1, device=self.device)

            # Compute basis functions for each t value
            basis_functions = torch.zeros(self.T, self.K_control_points, device=self.device)

            for t_idx, t in enumerate(self.t_values):
                # Find the knot span containing t
                span = torch.searchsorted(knots, t, right=True) - 1
                span = torch.clamp(span, self.bspline_degree, self.K_control_points - 1)

                # Initialize basis functions for degree 0
                basis = torch.zeros(self.bspline_degree + 1, device=self.device)
                basis[0] = 1.0

                # Compute basis functions for higher degrees
                for d in range(1, self.bspline_degree + 1):
                    for i in range(d + 1):
                        idx = span - d + i
                        if idx >= 0 and idx < self.K_control_points:
                            if knots[idx + d] - knots[idx] > 1e-6:
                                basis[i] = (t - knots[idx]) / (knots[idx + d] - knots[idx]) * basis[i-1]
                            if knots[idx + d + 1] - knots[idx + 1] > 1e-6:
                                basis[i] += (knots[idx + d + 1] - t) / (knots[idx + d + 1] - knots[idx + 1]) * basis[i]

                # Store the basis functions for this t value
                basis_functions[t_idx, span - self.bspline_degree:span + 1] = basis[:self.bspline_degree + 1]

            # Evaluate XYZ positions
            evaluated_xyz_logits = torch.einsum('tk,nkp->tnp', basis_functions, self._xyz_control_points)
            return torch.tanh(evaluated_xyz_logits)
        else:
            raise ValueError(f"Unknown trajectory_model_type: {self.trajectory_model_type}")

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
        active_background = background_color_override if background_color_override is not None else torch.ones(3, device=self.device)

        # Get all parameters evaluated at the specified t_values
        if self.trajectory_model_type == "polynomial":
            evaluated_xyz, evaluated_cholesky, evaluated_opacity = self._get_evaluated_polynomials(t_values_to_render)
        elif self.trajectory_model_type == "bspline":
            evaluated_xyz, evaluated_cholesky, evaluated_opacity = self._get_evaluated_bsplines(t_values_to_render)
        else:
            raise ValueError(f"Unknown trajectory_model_type: {self.trajectory_model_type}")

        static_colors = self.get_features # (N, 3), still static

        all_frames_rendered = []
        for k in range(num_frames_to_render):
            means_k = evaluated_xyz[k]             # (N, 2)
            cholesky_k = evaluated_cholesky[k]       # (N, 3)
            opacities_k = evaluated_opacity[k]         # (N, 1)

            xys, depths_from_projection, radii, conics, num_tiles_hit = project_gaussians_2d(
                means_k, cholesky_k, self.H, self.W, self.tile_bounds
            )
            # Use actual depths from projection instead of opacity-based depths
            effective_depths = depths_from_projection

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

    def train_iter(self, gt_frames_batch: torch.Tensor, t_values_batch: torch.Tensor, precomputed_flows_for_batch: Optional[torch.Tensor] = None):
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

       # --- BEGIN VECTORIZED OPTICAL FLOW LOSS CALCULATION ---
        if ENABLE_OPTICAL_FLOW_LOSS and \
           T_batch > 1 and \
           self.num_points > 0 and \
           precomputed_flows_for_batch is not None and \
           precomputed_flows_for_batch.shape[0] == T_batch - 1:

            if self.trajectory_model_type == "polynomial":
                batch_xyz_eval, _, batch_opacity_eval = self._get_evaluated_polynomials(t_values_batch)
            elif self.trajectory_model_type == "bspline":
                batch_xyz_eval, _, batch_opacity_eval = self._get_evaluated_bsplines(t_values_batch)
            else:
                batch_xyz_eval = torch.empty(T_batch, self.num_points, 2, device=self.device)
                batch_opacity_eval = torch.empty(T_batch, self.num_points, 1, device=self.device)

            xyz_t_prevs = batch_xyz_eval[:-1]
            opacity_t_prevs = batch_opacity_eval[:-1]
            xyz_t_curr_models = batch_xyz_eval[1:]

            active_gaussians_mask = (opacity_t_prevs.squeeze(-1) > OPTICAL_FLOW_OPACITY_THRESHOLD)

            if not torch.any(active_gaussians_mask):
                 mean_optical_flow_loss = torch.tensor(0.0, device=self.device)
            else:
                flows_to_sample = precomputed_flows_for_batch.permute(0, 3, 1, 2) # (T-1, 2, H, W)
                sampling_grid = xyz_t_prevs.unsqueeze(2) # (T-1, N, 1, 2)

                # Sample flow vectors
                # grid_sample output: (N_batch, C, H_out, W_out) -> (T-1, 2, N, 1)
                sampled_flow_vectors_raw = F.grid_sample(
                    flows_to_sample,
                    sampling_grid,
                    mode='bilinear',
                    padding_mode='border',
                    align_corners=True # Using True, common for normalized coords
                ) # Shape: (T-1, 2, N, 1)

                # Reshape to (T-1, N, 2)
                sampled_flow_vectors_pixel_space = sampled_flow_vectors_raw.squeeze(3).permute(0, 2, 1) # (T-1, N, 2)


                # The sampled_flow_vectors are displacements in pixel units.
                # Scale them to normalized coordinate units.
                # Flow_dx (pixels) / (W-1) * 2 gives normalized displacement_x.
                # Flow_dy (pixels) / (H-1) * 2 gives normalized displacement_y.
                # Ensure W and H are not zero to avoid division by zero
                if self.W <=1 or self.H <=1: # Should not happen if frames are loaded
                    scale_x = 0.0
                    scale_y = 0.0
                else:
                    scale_x = 2.0 / (self.W - 1)
                    scale_y = 2.0 / (self.H - 1)

                flow_scale = torch.tensor([scale_x, scale_y], device=self.device).view(1, 1, 2)
                sampled_flow_vectors_normalized_delta = sampled_flow_vectors_pixel_space * flow_scale # (T-1, N, 2)

                xyz_t_curr_flow_pred_norm = xyz_t_prevs + sampled_flow_vectors_normalized_delta # (T-1, N, 2)

                pos_diff = xyz_t_curr_models - xyz_t_curr_flow_pred_norm
                loss_all_pairs_all_gaussians = torch.sum(pos_diff**2, dim=2)
                weighted_loss_all = loss_all_pairs_all_gaussians * opacity_t_prevs.squeeze(-1)
                masked_weighted_loss = weighted_loss_all * active_gaussians_mask.float()
                total_masked_loss_sum = torch.sum(masked_weighted_loss)
                num_active_gaussians_total = torch.sum(active_gaussians_mask.float())

                if num_active_gaussians_total > 0:
                    mean_optical_flow_loss = total_masked_loss_sum / num_active_gaussians_total
                else:
                    mean_optical_flow_loss = torch.tensor(0.0, device=self.device)

            if not torch.isnan(mean_optical_flow_loss) and not torch.isinf(mean_optical_flow_loss):
                average_loss += LAMBDA_OPTICAL_FLOW * mean_optical_flow_loss
        # --- END VECTORIZED OPTICAL FLOW LOSS CALCULATION ---

        # Add L2 regularization on coefficients/control points
        if self.lambda_xyz_coeffs_reg > 0:
            if self.trajectory_model_type == "polynomial":
                # _xyz_coeffs shape: (num_coeffs, N, 2)
                # Regularize coeffs[1:] (linear term upwards)
                xyz_coeffs_to_reg = self._xyz_coeffs[1:]
                xyz_coeffs_l2_loss = self.lambda_xyz_coeffs_reg * torch.mean(xyz_coeffs_to_reg**2)
                average_loss += xyz_coeffs_l2_loss
            elif self.trajectory_model_type == "bspline":
                # For B-splines, regularize control points after the first one
                xyz_control_points_to_reg = self._xyz_control_points[:, 1:]
                xyz_coeffs_l2_loss = self.lambda_xyz_coeffs_reg * torch.mean(xyz_control_points_to_reg**2)
                average_loss += xyz_coeffs_l2_loss

        if self.lambda_cholesky_coeffs_reg > 0:
            if self.trajectory_model_type == "polynomial":
                # _cholesky_coeffs shape: (num_coeffs, N, 3)
                # Regularize coeffs[1:] (linear term upwards)
                cholesky_coeffs_to_reg = self._cholesky_coeffs[1:]
                cholesky_coeffs_l2_loss = self.lambda_cholesky_coeffs_reg * torch.mean(cholesky_coeffs_to_reg**2)
                average_loss += cholesky_coeffs_l2_loss
            elif self.trajectory_model_type == "bspline":
                # For B-splines, regularize control points after the first one
                cholesky_control_points_to_reg = self._cholesky_control_points[:, 1:]
                cholesky_coeffs_l2_loss = self.lambda_cholesky_coeffs_reg * torch.mean(cholesky_control_points_to_reg**2)
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

    def get_xyz_at_time(self, t: torch.Tensor):
        """Get XYZ positions for a specific time value or batch of time values.
        Args:
            t: Tensor of shape (B,) containing time values in [0,1]
        Returns:
            Tensor of shape (B, N, 2) containing XYZ positions
        """
        if self.trajectory_model_type == "polynomial":
            # Create power matrix for the input time values
            t_power_matrix = torch.ones_like(t).unsqueeze(1)  # t^0
            if self.polynomial_degree > 0:
                for i in range(1, self.polynomial_degree + 1):
                    t_power_matrix = torch.cat([t_power_matrix, t.pow(i).unsqueeze(1)], dim=1)

            # Evaluate XYZ positions
            evaluated_xyz_logits = torch.einsum('dnp,bd->bnp', self._xyz_coeffs, t_power_matrix)
            return torch.tanh(evaluated_xyz_logits)

        elif self.trajectory_model_type == "bspline":
            # Convert t to grid indices
            grid_indices = (t * (self.bspline_resolution - 1)).long()
            grid_indices = torch.clamp(grid_indices, 0, self.bspline_resolution - 1)

            # Get pre-computed basis functions for the requested t values
            basis_functions = self.basis_functions_grid[grid_indices]

            # Evaluate XYZ positions
            evaluated_xyz_logits = torch.einsum('tk,nkp->tnp', basis_functions, self._xyz_control_points)
            return torch.tanh(evaluated_xyz_logits)
        else:
            raise ValueError(f"Unknown trajectory_model_type: {self.trajectory_model_type}")
