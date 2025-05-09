from gsplat.gsplat.project_gaussians_2d import project_gaussians_2d
from gsplat.gsplat.rasterize_sum import rasterize_gaussians_sum
from utils import *
import torch
import torch.nn as nn
import numpy as np
import math
# from quantize import *
from optimizer import Adan

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
    def __init__(self, loss_type="L2", T=1, lambda_opacity_reg=0.0, lambda_temporal_xyz=0.0, lambda_temporal_cholesky=0.0, lambda_accel_xyz=0.0, lambda_accel_cholesky=0.0, lambda_neighbor_rigidity=0.0, k_neighbors=5, ema_decay=0.999, polynomial_degree=1, **kwargs):
        super().__init__()
        self.loss_type = loss_type
        self.num_points = kwargs["num_points"]
        self.H, self.W = kwargs["H"], kwargs["W"]
        self.T = T # Number of frames
        self.lambda_opacity_reg = lambda_opacity_reg
        self.lambda_temporal_xyz = lambda_temporal_xyz
        self.lambda_temporal_cholesky = lambda_temporal_cholesky
        self.lambda_accel_xyz = lambda_accel_xyz
        self.lambda_accel_cholesky = lambda_accel_cholesky
        self.lambda_neighbor_rigidity = lambda_neighbor_rigidity
        self.k_neighbors = k_neighbors
        self.BLOCK_W, self.BLOCK_H = kwargs["BLOCK_W"], kwargs["BLOCK_H"]
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        )
        self.device = kwargs["device"]

        self.polynomial_degree = polynomial_degree # Use passed-in degree
        num_coeffs = self.polynomial_degree + 1

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
            t_power_matrix = torch.empty(0, num_coeffs, device=self.device)
        self.register_buffer('t_power_matrix', t_power_matrix, persistent=False)


        # Initialize _xyz_coeffs: (num_coeffs, N, 2)
        # c0 (constant term) is random
        initial_xyz_c0 = torch.atanh(2 * (torch.rand(self.num_points, 2, device=self.device) - 0.5))
        # c1...cD (higher order terms) are zero for constant initialization
        higher_order_xyz_coeffs = torch.zeros(self.polynomial_degree, self.num_points, 2, device=self.device)
        self._xyz_coeffs = nn.Parameter(torch.cat([initial_xyz_c0.unsqueeze(0), higher_order_xyz_coeffs], dim=0))


        # Initialize _cholesky_coeffs: (num_coeffs, N, 3)
        # c0 (constant term) is random (near identity)
        init_chol_base_c0 = torch.tensor([1.0, 0.0, 1.0], device=self.device).view(1, 3)
        noise_per_gaussian_c0 = torch.randn(self.num_points, 3, device=self.device) * 0.01
        initial_cholesky_c0 = init_chol_base_c0.repeat(self.num_points, 1) + noise_per_gaussian_c0
        # c1...cD (higher order terms) are zero
        higher_order_cholesky_coeffs = torch.zeros(self.polynomial_degree, self.num_points, 3, device=self.device)
        self._cholesky_coeffs = nn.Parameter(torch.cat([initial_cholesky_c0.unsqueeze(0), higher_order_cholesky_coeffs], dim=0))

        # Static parameters: shape (N, ...)
        self._opacity = nn.Parameter(torch.ones((self.num_points, 1), device=self.device))
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

        if self.lambda_temporal_xyz > 0:
            self.ema_xyz = EMA([self._xyz_coeffs], decay=self.ema_decay) # Track coefficients

        if self.lambda_temporal_cholesky > 0:
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

    @property
    def get_xyz(self):
        """Evaluates polynomial for means, returns (T, N, 2)."""
        if self.T == 0:
            return torch.empty(0, self.num_points, 2, device=self.device)
        # self._xyz_coeffs: (num_coeffs, N, 2)
        # self.t_power_matrix: (T, num_coeffs)
        # einsum: d=num_coeffs, n=num_points, p=param_dim(2), t=time
        evaluated_xyz = torch.einsum('dnp,td->tnp', self._xyz_coeffs, self.t_power_matrix)
        return torch.tanh(evaluated_xyz)

    @property
    def get_features(self):
        """Returns colors, shape (N, 3)."""
        # Apply activation
        return self.rgb_activation(self._features_dc)

    @property
    def get_opacity(self):
        """Returns opacities, shape (N, 1)."""
        # Apply activation
        return self.opacity_activation(self._opacity)

    @property
    def get_cholesky_elements(self):
        """Evaluates polynomial for Cholesky elements, returns (T, N, 3)."""
        if self.T == 0:
            return torch.empty(0, self.num_points, 3, device=self.device)
        # self._cholesky_coeffs: (num_coeffs, N, 3)
        # self.t_power_matrix: (T, num_coeffs)
        # einsum: d=num_coeffs, n=num_points, p=param_dim(3), t=time
        evaluated_cholesky = torch.einsum('dnp,td->tnp', self._cholesky_coeffs, self.t_power_matrix)
        return evaluated_cholesky + self.cholesky_bound

    def forward(self, frame_index=None, background_color_override=None):
        """Render either a specific frame or all frames.

        Args:
            frame_index (int, optional): If specified, render only this frame.
                                       Otherwise, render all frames.
            background_color_override (torch.Tensor, optional): If provided, use this as the background color.
                                                              Otherwise, defaults to white.

        Returns:
            dict: Dictionary containing the rendered image(s).
                  If frame_index is given, "render" has shape (1, C, H, W).
                  Otherwise, "render" has shape (T, C, H, W).
        """

        if background_color_override is not None:
            active_background = background_color_override
        else:
            active_background = torch.ones(3, device=self.device) # Default to white (was torch.zeros for black)

        if frame_index is not None:
            # Render a single frame t
            means_t = self.get_xyz[frame_index] # (N, 2)
            cholesky_t = self.get_cholesky_elements[frame_index] # (N, 3)
            colors = self.get_features # (N, 3)
            opacities = self.get_opacity # (N, 1)

            # gsplat expects N x D input, project_gaussians_2d takes means and cholesky separately
            xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d(
                means_t, cholesky_t, self.H, self.W, self.tile_bounds
            )
            # rasterize_gaussians_sum expects N x D input for colors and opacities
            out_img = rasterize_gaussians_sum(
                xys, depths, radii, conics, num_tiles_hit,
                colors, opacities, self.H, self.W, self.BLOCK_H, self.BLOCK_W,
                background=active_background, return_alpha=False
            )
            out_img = torch.clamp(out_img, 0, 1) #[H, W, 3]
            # Reshape to (1, C, H, W)
            out_img = out_img.view(1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
            return {"render": out_img}
        else:
            # Render all frames (can be memory intensive)
            all_frames_rendered = []
            for t in range(self.T):
                # Pass the background_color_override to recursive calls
                frame_render_pkg = self.forward(frame_index=t, background_color_override=active_background)
                all_frames_rendered.append(frame_render_pkg["render"])
            # Stack along the time dimension
            all_frames_tensor = torch.cat(all_frames_rendered, dim=0) # (T, C, H, W)
            return {"render": all_frames_tensor}

    def train_iter(self, gt_frames):
        """Performs one training iteration using all frames.

        Args:
            gt_frames (torch.Tensor): Ground truth video frames (T, C, H, W).

        Returns:
            tuple: Average loss and average PSNR across all frames.
        """
        total_loss = 0
        total_psnr = 0

        # Generate a random background color for this iteration
        # iter_random_background = torch.rand(3, device=self.device) # Removed for always-white background

        # Render all frames using the default background (now white)
        rendered_frames_pkg = self.forward() # Removed background_color_override
        rendered_frames = rendered_frames_pkg["render"]

        # Calculate loss and PSNR per frame and average
        for t in range(self.T):
            loss_t = loss_fn(rendered_frames[t:t+1], gt_frames[t:t+1], self.loss_type, lambda_value=0.7)
            total_loss += loss_t
            with torch.no_grad():
                mse_loss_t = F.mse_loss(rendered_frames[t:t+1], gt_frames[t:t+1])
                psnr_t = 10 * math.log10(1.0 / mse_loss_t.item())
                total_psnr += psnr_t

        average_loss = total_loss / self.T
        average_psnr = total_psnr / self.T

        # Add L1 opacity regularization
        # We penalize the raw logits _opacity to encourage them to be negative (pushing sigmoid towards 0)
        # Or, penalize the output of get_opacity directly (simpler to reason about scale)
        if self.lambda_opacity_reg > 0:
            opacity_values = self.get_opacity # These are already sigmoid-ed
            opacity_reg_loss = self.lambda_opacity_reg * torch.mean(opacity_values)
            average_loss += opacity_reg_loss

        # Add temporal consistency regularization for XYZ
        if self.lambda_temporal_xyz > 0 and self.T > 1:
            xyz_params = self.get_xyz # Shape (T, N, 2)
            xyz_diff = xyz_params[1:] - xyz_params[:-1] # Differences between frame t and t-1
            temporal_xyz_loss = self.lambda_temporal_xyz * torch.mean(xyz_diff**2)
            average_loss += temporal_xyz_loss

        # Add temporal consistency regularization for Cholesky components
        if self.lambda_temporal_cholesky > 0 and self.T > 1:
            cholesky_params = self.get_cholesky_elements # Shape (T, N, 3)
            cholesky_diff = cholesky_params[1:] - cholesky_params[:-1]
            temporal_cholesky_loss = self.lambda_temporal_cholesky * torch.mean(cholesky_diff**2)
            average_loss += temporal_cholesky_loss

        # Add temporal acceleration regularization for XYZ
        if self.lambda_accel_xyz > 0 and self.T > 2:
            xyz_params = self.get_xyz # Shape (T, N, 2)
            # P[t] - 2*P[t-1] + P[t-2]
            xyz_accel = xyz_params[2:] - 2 * xyz_params[1:-1] + xyz_params[:-2]
            temporal_accel_xyz_loss = self.lambda_accel_xyz * torch.mean(xyz_accel**2)
            average_loss += temporal_accel_xyz_loss

        # Add temporal acceleration regularization for Cholesky components
        if self.lambda_accel_cholesky > 0 and self.T > 2:
            cholesky_params = self.get_cholesky_elements # Shape (T, N, 3)
            # P[t] - 2*P[t-1] + P[t-2]
            cholesky_accel = cholesky_params[2:] - 2 * cholesky_params[1:-1] + cholesky_params[:-2]
            temporal_accel_cholesky_loss = self.lambda_accel_cholesky * torch.mean(cholesky_accel**2)
            average_loss += temporal_accel_cholesky_loss

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
        if self.lambda_temporal_xyz > 0 and self.ema_xyz is not None:
            self.ema_xyz.update() # EMA object internally refers to the parameter it tracks
        if self.lambda_temporal_cholesky > 0 and self.ema_cholesky is not None:
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
