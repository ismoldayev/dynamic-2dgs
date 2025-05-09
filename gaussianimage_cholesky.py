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
    def __init__(self, loss_type="L2", T=1, lambda_neighbor_rigidity=0.0, k_neighbors=5, ema_decay=0.999, polynomial_degree=1, lambda_xyz_coeffs_reg=0.0, lambda_cholesky_coeffs_reg=0.0, lambda_opacity_coeffs_reg=0.0, opacity_polynomial_degree=None, **kwargs):
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

        self.polynomial_degree = polynomial_degree # Use passed-in degree
        self.opacity_polynomial_degree = opacity_polynomial_degree if opacity_polynomial_degree is not None else polynomial_degree

        num_coeffs_xyz_chol = self.polynomial_degree + 1
        num_coeffs_opacity = self.opacity_polynomial_degree + 1

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
            t_power_matrix = torch.empty(0, num_coeffs_xyz_chol, device=self.device)
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
                t_power_matrix_opacity = torch.empty(0, num_coeffs_opacity, device=self.device)
            self.register_buffer('t_power_matrix_opacity', t_power_matrix_opacity, persistent=False)


        # Initialize _xyz_coeffs: (num_coeffs_xyz_chol, N, 2)
        # c0 (constant term) is random
        initial_xyz_c0 = torch.atanh(2 * (torch.rand(self.num_points, 2, device=self.device) - 0.5))
        # c1...cD (higher order terms) are zero for constant initialization
        higher_order_xyz_coeffs = torch.zeros(self.polynomial_degree, self.num_points, 2, device=self.device)
        self._xyz_coeffs = nn.Parameter(torch.cat([initial_xyz_c0.unsqueeze(0), higher_order_xyz_coeffs], dim=0))


        # Initialize _cholesky_coeffs: (num_coeffs_xyz_chol, N, 3)
        # c0 (constant term) is random (near identity)
        init_chol_base_c0 = torch.tensor([1.0, 0.0, 1.0], device=self.device).view(1, 3)
        noise_per_gaussian_c0 = torch.randn(self.num_points, 3, device=self.device) * 0.01
        initial_cholesky_c0 = init_chol_base_c0.repeat(self.num_points, 1) + noise_per_gaussian_c0
        # c1...cD (higher order terms) are zero
        higher_order_cholesky_coeffs = torch.zeros(self.polynomial_degree, self.num_points, 3, device=self.device)
        self._cholesky_coeffs = nn.Parameter(torch.cat([initial_cholesky_c0.unsqueeze(0), higher_order_cholesky_coeffs], dim=0))

        # Initialize _opacity_coeffs: (num_coeffs_opacity, N, 1)
        # c0 (constant term) initialized to give opacity of 0.5 initially (logit(0.5)=0)
        initial_opacity_c0 = torch.zeros(self.num_points, 1, device=self.device)
        # c1...cD (higher order terms) are zero
        higher_order_opacity_coeffs = torch.zeros(self.opacity_polynomial_degree, self.num_points, 1, device=self.device)
        self._opacity_coeffs = nn.Parameter(torch.cat([initial_opacity_c0.unsqueeze(0), higher_order_opacity_coeffs], dim=0))

        # Static parameters: features_dc are still static for now
        # self._opacity = nn.Parameter(torch.ones((self.num_points, 1), device=self.device))) # Old static opacity
        self._features_dc = nn.Parameter(torch.randn(self.num_points, 3, device=self.device))

        self.last_size = (self.H, self.W)
        # self.register_buffer('background', torch.rand(3, device=self.device)) # Removed: background will be handled in forward
        self.opacity_activation = torch.sigmoid
        self.rgb_activation = torch.sigmoid
        # self.register_buffer('bound', torch.tensor([0.5, 0.5]).view(1, 2).to(self.device)) # Not used?
        self.register_buffer('cholesky_bound', torch.tensor([0.5, 0, 0.5]).view(1, 1, 3).to(self.device)) # Adjusted shape for broadcasting

        # Initialize EMA objects - will be None if lambda is 0
        self.ema_xyz = None
        self.ema_cholesky = None
        self.ema_decay = ema_decay # Store decay rate

        if self.lambda_xyz_coeffs_reg > 0:
            self.ema_xyz = EMA([self._xyz_coeffs], decay=self.ema_decay) # Track coefficients

        if self.lambda_cholesky_coeffs_reg > 0:
            self.ema_cholesky = EMA([self._cholesky_coeffs], decay=self.ema_decay) # Track coefficients

        if kwargs["opt_type"] == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=kwargs["lr"])
        else:
            self.optimizer = Adan(self.parameters(), lr=kwargs["lr"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)

        # Initialize neighbor information for rigidity loss
        if self.lambda_neighbor_rigidity > 0 and self.num_points > self.k_neighbors and self.T > 1 and self.k_neighbors > 0:
            with torch.no_grad(): # Operations here should not be part of the computation graph for _xyz init
                # Detach explicitly if get_xyz involves operations on parameters being optimized
                xyz_t0 = self.get_xyz[0].clone().detach() # Shape (N, 2)

                # Pairwise squared Euclidean distances
                # cdist computes L2 norm (Euclidean distance), then we square it for consistency with some formulations
                # However, working with direct distances (sqrt) is fine and perhaps more intuitive.
                pairwise_distances = torch.cdist(xyz_t0, xyz_t0, p=2.0) # Shape (N, N)
                pairwise_distances.fill_diagonal_(float('inf')) # Ignore distance to self

                initial_neighbor_distances, neighbor_indices = torch.topk(
                    pairwise_distances, self.k_neighbors, dim=1, largest=False
                )
                self.register_buffer('initial_neighbor_distances', initial_neighbor_distances, persistent=False)
                self.register_buffer('neighbor_indices', neighbor_indices, persistent=False)
                # print(f"Initialized neighbor rigidity buffers. Shapes: D={self.initial_neighbor_distances.shape}, I={self.neighbor_indices.shape}")
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

    # def forward_quantize(self):
    #     l_vqm, m_bit = 0, 16*self.init_num_points*2
    #     means = torch.tanh(self.xyz_quantizer(self._xyz))
    #     cholesky_elements, l_vqs, s_bit = self.cholesky_quantizer(self._cholesky)
    #     cholesky_elements = cholesky_elements + self.cholesky_bound
    #     l_vqr, r_bit = 0, 0
    #     colors, l_vqc, c_bit = self.features_dc_quantizer(self.get_features)
    #     self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(means, cholesky_elements, self.H, self.W, self.tile_bounds)
    #     out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
    #             colors, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
    #     out_img = torch.clamp(out_img, 0, 1)
    #     out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
    #     vq_loss = l_vqm + l_vqs + l_vqr + l_vqc
    #     return {"render": out_img, "vq_loss": vq_loss, "unit_bit":[m_bit, s_bit, r_bit, c_bit]}

    # def train_iter_quantize(self, gt_image):
    #     render_pkg = self.forward_quantize()
    #     image = render_pkg["render"]
    #     loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0.7) + render_pkg["vq_loss"]
    #     loss.backward()
    #     with torch.no_grad():
    #         mse_loss = F.mse_loss(image, gt_image)
    #         psnr = 10 * math.log10(1.0 / mse_loss.item())
    #     self.optimizer.step()
    #     self.optimizer.zero_grad(set_to_none=True)
    #     self.scheduler.step()
    #     return loss, psnr

    # def compress_wo_ec(self):
    #     means = torch.tanh(self.xyz_quantizer(self._xyz))
    #     quant_cholesky_elements, cholesky_elements = self.cholesky_quantizer.compress(self._cholesky)
    #     cholesky_elements = cholesky_elements + self.cholesky_bound
    #     colors, feature_dc_index = self.features_dc_quantizer.compress(self.get_features)
    #     return {"xyz":self._xyz.half(), "feature_dc_index": feature_dc_index, "quant_cholesky_elements": quant_cholesky_elements,}

    # def decompress_wo_ec(self, encoding_dict):
    #     xyz, feature_dc_index, quant_cholesky_elements = encoding_dict["xyz"], encoding_dict["feature_dc_index"], encoding_dict["quant_cholesky_elements"]
    #     means = torch.tanh(xyz.float())
    #     cholesky_elements = self.cholesky_quantizer.decompress(quant_cholesky_elements)
    #     cholesky_elements = cholesky_elements + self.cholesky_bound
    #     colors = self.features_dc_quantizer.decompress(feature_dc_index)
    #     self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(means, cholesky_elements, self.H, self.W, self.tile_bounds)
    #     out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
    #             colors, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
    #     out_img = torch.clamp(out_img, 0, 1)
    #     out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
    #     return {"render":out_img}

    # def analysis_wo_ec(self, encoding_dict):
    #     quant_cholesky_elements, feature_dc_index = encoding_dict["quant_cholesky_elements"], encoding_dict["feature_dc_index"]
    #     total_bits = 0
    #     initial_bits, codebook_bits = 0, 0
    #     for quantizer_index, layer in enumerate(self.features_dc_quantizer.quantizer.layers):
    #         codebook_bits += layer._codebook.embed.numel()*torch.finfo(layer._codebook.embed.dtype).bits
    #     initial_bits += self.cholesky_quantizer.scale.numel()*torch.finfo(self.cholesky_quantizer.scale.dtype).bits
    #     initial_bits += self.cholesky_quantizer.beta.numel()*torch.finfo(self.cholesky_quantizer.beta.dtype).bits
    #     initial_bits += codebook_bits
    #
    #     total_bits += initial_bits
    #     total_bits += self._xyz.numel()*16
    #
    #     feature_dc_index = feature_dc_index.int().cpu().numpy()
    #     index_max = np.max(feature_dc_index)
    #     max_bit = np.ceil(np.log2(index_max)) #calculate max bit for feature_dc_index
    #     total_bits += feature_dc_index.size * max_bit #get_np_size(encoding_dict["feature_dc_index"]) * 8
    #
    #     quant_cholesky_elements = quant_cholesky_elements.cpu().numpy()
    #     total_bits += quant_cholesky_elements.size * 6 #cholesky bits
    #
    #     position_bits = self._xyz.numel()*16
    #     cholesky_bits, feature_dc_bits = 0, 0
    #     cholesky_bits += self.cholesky_quantizer.scale.numel()*torch.finfo(self.cholesky_quantizer.scale.dtype).bits
    #     cholesky_bits += self.cholesky_quantizer.beta.numel()*torch.finfo(self.cholesky_quantizer.beta.dtype).bits
    #     cholesky_bits += quant_cholesky_elements.size * 6
    #     feature_dc_bits += codebook_bits
    #     feature_dc_bits += feature_dc_index.size * max_bit
    #
    #     bpp = total_bits/self.H/self.W
    #     position_bpp = position_bits/self.H/self.W
    #     cholesky_bpp = cholesky_bits/self.H/self.W
    #     feature_dc_bpp = feature_dc_bits/self.H/self.W
    #     return {"bpp": bpp, "position_bpp": position_bpp,
    #         "cholesky_bpp": cholesky_bpp, "feature_dc_bpp": feature_dc_bpp}
    #
    # def compress(self):
    #     means = torch.tanh(self.xyz_quantizer(self._xyz))
    #     quant_cholesky_elements, cholesky_elements = self.cholesky_quantizer.compress(self._cholesky)
    #     cholesky_elements = cholesky_elements + self.cholesky_bound
    #     colors, feature_dc_index = self.features_dc_quantizer.compress(self.get_features)
    #     cholesky_compressed, cholesky_histogram_table, cholesky_unique = compress_matrix_flatten_categorical(quant_cholesky_elements.int().flatten().tolist())
    #     feature_dc_compressed, feature_dc_histogram_table, feature_dc_unique = compress_matrix_flatten_categorical(feature_dc_index.int().flatten().tolist())
    #     return {"xyz":self._xyz.half(), "feature_dc_index": feature_dc_index, "quant_cholesky_elements": quant_cholesky_elements,
    #         "feature_dc_bitstream":[feature_dc_compressed, feature_dc_histogram_table, feature_dc_unique],
    #         "cholesky_bitstream":[cholesky_compressed, cholesky_histogram_table, cholesky_unique]}
    #
    # def decompress(self, encoding_dict):
    #     xyz = encoding_dict["xyz"]
    #     num_points, device = xyz.size(0), xyz.device
    #     feature_dc_compressed, feature_dc_histogram_table, feature_dc_unique = encoding_dict["feature_dc_bitstream"]
    #     cholesky_compressed, cholesky_histogram_table, cholesky_unique = encoding_dict["cholesky_bitstream"]
    #     feature_dc_index = decompress_matrix_flatten_categorical(feature_dc_compressed, feature_dc_histogram_table, feature_dc_unique, num_points*2, (num_points, 2))
    #     quant_cholesky_elements = decompress_matrix_flatten_categorical(cholesky_compressed, cholesky_histogram_table, cholesky_unique, num_points*3, (num_points, 3))
    #     feature_dc_index = torch.from_numpy(feature_dc_index).to(device).int() #[800, 2]
    #     quant_cholesky_elements = torch.from_numpy(quant_cholesky_elements).to(device).float() #[800, 3]
    #
    #     means = torch.tanh(xyz.float())
    #     cholesky_elements = self.cholesky_quantizer.decompress(quant_cholesky_elements)
    #     cholesky_elements = cholesky_elements + self.cholesky_bound
    #     colors = self.features_dc_quantizer.decompress(feature_dc_index)
    #     self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(means, cholesky_elements, self.H, self.W, self.tile_bounds)
    #     out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
    #             colors, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
    #     out_img = torch.clamp(out_img, 0, 1)
    #     out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
    #     return {"render":out_img}
    #
    # def analysis(self, encoding_dict):
    #     quant_cholesky_elements, feature_dc_index = encoding_dict["quant_cholesky_elements"], encoding_dict["feature_dc_index"]
    #     cholesky_compressed, cholesky_histogram_table, cholesky_unique = compress_matrix_flatten_categorical(quant_cholesky_elements.int().flatten().tolist())
    #     feature_dc_compressed, feature_dc_histogram_table, feature_dc_unique = compress_matrix_flatten_categorical(feature_dc_index.int().flatten().tolist())
    #     cholesky_lookup = dict(zip(cholesky_unique, cholesky_histogram_table.astype(np.float64) / np.sum(cholesky_histogram_table).astype(np.float64)))
    #     feature_dc_lookup = dict(zip(feature_dc_unique, feature_dc_histogram_table.astype(np.float64) / np.sum(feature_dc_histogram_table).astype(np.float64)))
    #
    #     total_bits = 0
    #     initial_bits, codebook_bits = 0, 0
    #     for quantizer_index, layer in enumerate(self.features_dc_quantizer.quantizer.layers):
    #         codebook_bits += layer._codebook.embed.numel()*torch.finfo(layer._codebook.embed.dtype).bits
    #     initial_bits += self.cholesky_quantizer.scale.numel()*torch.finfo(self.cholesky_quantizer.scale.dtype).bits
    #     initial_bits += self.cholesky_quantizer.beta.numel()*torch.finfo(self.cholesky_quantizer.beta.dtype).bits
    #     initial_bits += get_np_size(cholesky_histogram_table) * 8
    #     initial_bits += get_np_size(cholesky_unique) * 8
    #     initial_bits += get_np_size(feature_dc_histogram_table) * 8
    #     initial_bits += get_np_size(feature_dc_unique) * 8
    #     initial_bits += codebook_bits
    #
    #     total_bits += initial_bits
    #     total_bits += self._xyz.numel()*16
    #     total_bits += get_np_size(cholesky_compressed) * 8
    #     total_bits += get_np_size(feature_dc_compressed) * 8
    #
    #     position_bits = self._xyz.numel()*16
    #     cholesky_bits, feature_dc_bits = 0, 0
    #     cholesky_bits += self.cholesky_quantizer.scale.numel()*torch.finfo(self.cholesky_quantizer.scale.dtype).bits
    #     cholesky_bits += self.cholesky_quantizer.beta.numel()*torch.finfo(self.cholesky_quantizer.beta.dtype).bits
    #     cholesky_bits += get_np_size(cholesky_histogram_table) * 8
    #     cholesky_bits += get_np_size(cholesky_unique) * 8
    #     cholesky_bits += get_np_size(cholesky_compressed) * 8
    #     feature_dc_bits += codebook_bits
    #     feature_dc_bits += get_np_size(feature_dc_histogram_table) * 8
    #     feature_dc_bits += get_np_size(feature_dc_unique) * 8
    #     feature_dc_bits += get_np_size(feature_dc_compressed) * 8
    #
    #     bpp = total_bits/self.H/self.W
    #     position_bpp = position_bits/self.H/self.W
    #     cholesky_bpp = cholesky_bits/self.H/self.W
    #     feature_dc_bpp = feature_dc_bits/self.H/self.W
    #     return {"bpp": bpp, "position_bpp": position_bpp,
    #         "cholesky_bpp": cholesky_bpp, "feature_dc_bpp": feature_dc_bpp,}
