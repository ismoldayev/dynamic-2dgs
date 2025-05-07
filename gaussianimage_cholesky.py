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
    def __init__(self, loss_type="L2", T=1, lambda_opacity_reg=0.0, lambda_temporal_xyz=0.0, lambda_temporal_cholesky=0.0, lambda_color_reg=0.0, ema_decay=0.999, **kwargs):
        super().__init__()
        self.loss_type = loss_type
        self.num_points = kwargs["num_points"]
        self.H, self.W = kwargs["H"], kwargs["W"]
        self.T = T # Number of frames
        self.lambda_opacity_reg = lambda_opacity_reg
        self.lambda_temporal_xyz = lambda_temporal_xyz
        self.lambda_temporal_cholesky = lambda_temporal_cholesky
        self.lambda_color_reg = lambda_color_reg
        self.BLOCK_W, self.BLOCK_H = kwargs["BLOCK_W"], kwargs["BLOCK_H"]
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        ) #
        self.device = kwargs["device"]

        # Time-varying parameters: shape (T, N, ...)
        self._xyz = nn.Parameter(torch.atanh(2 * (torch.rand(self.T, self.num_points, 2, device=self.device) - 0.5)))
        # Initialize cholesky near identity (small off-diag, positive diag)
        init_chol = torch.tensor([1.0, 0.0, 1.0], device=self.device).view(1, 1, 3)
        self._cholesky = nn.Parameter(init_chol.repeat(self.T, self.num_points, 1) + torch.randn(self.T, self.num_points, 3, device=self.device) * 0.01)

        # Static parameters: shape (N, ...)
        self._opacity = nn.Parameter(torch.ones((self.num_points, 1), device=self.device))
        self._features_dc = nn.Parameter(torch.randn(self.num_points, 3, device=self.device))

        self.last_size = (self.H, self.W)
        self.register_buffer('background', torch.zeros(3, device=self.device))
        self.opacity_activation = torch.sigmoid
        self.rgb_activation = torch.sigmoid
        # self.register_buffer('bound', torch.tensor([0.5, 0.5]).view(1, 2).to(self.device)) # Not used?
        self.register_buffer('cholesky_bound', torch.tensor([0.5, 0, 0.5]).view(1, 1, 3).to(self.device)) # Adjusted shape for broadcasting

        # Initialize EMA objects - will be None if lambda is 0
        self.ema_xyz = None
        self.ema_cholesky = None
        self.ema_decay = ema_decay # Store decay rate

        if self.lambda_temporal_xyz > 0:
            # EMA expects a list of parameters. For self._xyz, it's a single parameter.
            self.ema_xyz = EMA([self._xyz], decay=self.ema_decay)

        if self.lambda_temporal_cholesky > 0:
            self.ema_cholesky = EMA([self._cholesky], decay=self.ema_decay)

        if kwargs["opt_type"] == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=kwargs["lr"])
        else:
            self.optimizer = Adan(self.parameters(), lr=kwargs["lr"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)

    # def _init_data(self):
    #     self.cholesky_quantizer._init_data(self._cholesky)

    @property
    def get_xyz(self):
        """Returns means for all frames, shape (T, N, 2)."""
        return torch.tanh(self._xyz)

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
        """Returns Cholesky elements for all frames, shape (T, N, 3)."""
        # Add bound - Note: This was likely for initialization constraints, may need adjustment.
        # Ensure cholesky_bound broadcasts correctly: (1, 1, 3)
        return self._cholesky + self.cholesky_bound

    def forward(self, frame_index=None):
        """Render either a specific frame or all frames.

        Args:
            frame_index (int, optional): If specified, render only this frame.
                                       Otherwise, render all frames.

        Returns:
            dict: Dictionary containing the rendered image(s).
                  If frame_index is given, "render" has shape (1, C, H, W).
                  Otherwise, "render" has shape (T, C, H, W).
        """
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
                background=self.background, return_alpha=False
            )
            out_img = torch.clamp(out_img, 0, 1) #[H, W, 3]
            # Reshape to (1, C, H, W)
            out_img = out_img.view(1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
            return {"render": out_img}
        else:
            # Render all frames (can be memory intensive)
            all_frames_rendered = []
            for t in range(self.T):
                frame_render_pkg = self.forward(frame_index=t)
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

        # Render all frames
        rendered_frames_pkg = self.forward() # Gets dict with "render": (T, C, H, W)
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

        # Add L2 regularization on color features (_features_dc)
        if self.lambda_color_reg > 0:
            color_reg_loss = self.lambda_color_reg * torch.mean(self._features_dc**2)
            average_loss += color_reg_loss

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

    def densify_gaussians(
        self,
        gt_frames: torch.Tensor, # (T, C, H, W) - for potential future use with gradients
        size_threshold_split: float,
        opacity_threshold_clone: float,
        scale_factor_split_children: float,
        max_gaussians_scene: int
    ) -> int:
        """Densifies Gaussians by splitting large ones or cloning small, opaque ones."""

        num_gaussians_before = self.num_points
        if num_gaussians_before == 0: # Can't densify if there are no Gaussians
            return 0
        if num_gaussians_before >= max_gaussians_scene:
            print(f"Max Gaussians ({max_gaussians_scene}) reached. Skipping densification.")
            return 0

        num_added_total = 0

        with torch.no_grad():
            # Calculate current scales of Gaussians
            # self._cholesky diagonal elements are log-scales.
            # Consider only the first two diagonal elements for 2D scale.
            current_raw_scales_t_n_dim = torch.exp(self._cholesky[:, :, torch.arange(2), torch.arange(2)]) # Shape (T, N, 2)

            # Use the maximum of the two scale components at each time step, then average over time
            max_scale_component_t_n = torch.max(current_raw_scales_t_n_dim[:,:,0], current_raw_scales_t_n_dim[:,:,1]) # Shape (T, N)
            avg_max_scale_n = torch.mean(max_scale_component_t_n, dim=0) # Shape (N,)

            # --- Stage 1: Identify Gaussians to split (large Gaussians) ---
            split_candidate_mask_n = avg_max_scale_n > size_threshold_split

            # Also ensure we don't try to split if it would exceed max_gaussians_scene
            # Each split adds 1 Gaussian (parent removed, 2 children added)
            # Available slots for new Gaussians from splitting = max_gaussians_scene - num_gaussians_before
            # If num_to_split > available_slots, we must cap num_to_split

            potential_splits_indices = torch.where(split_candidate_mask_n)[0]
            num_potential_splits = potential_splits_indices.shape[0]

            available_slots = max_gaussians_scene - num_gaussians_before

            if num_potential_splits > available_slots and available_slots > 0:
                # If we want to split more than we have slots for, prioritize splitting the largest ones
                print(f"Capping splits: {num_potential_splits} candidates, but only {available_slots} slots available.")
                sorted_split_candidates_by_size_indices = torch.argsort(avg_max_scale_n[potential_splits_indices], descending=True)
                gaussians_to_split_indices = potential_splits_indices[sorted_split_candidates_by_size_indices[:available_slots]]
            elif available_slots <= 0 :
                 gaussians_to_split_indices = torch.empty(0, dtype=torch.long, device=self.device)
            else:
                gaussians_to_split_indices = potential_splits_indices

            num_to_split = gaussians_to_split_indices.shape[0]

            # --- Stage 2: Identify Gaussians to clone (small but high opacity Gaussians) ---
            # These should not be candidates for splitting
            not_splitting_mask_n = ~torch.isin(torch.arange(self.num_points, device=self.device), gaussians_to_split_indices)

            current_opacities_n = self.get_opacity.squeeze(-1) # Shape (N,)

            # "Small" means their average max scale is NOT above size_threshold_split
            # (or we could use a different, smaller threshold for "smallness" if desired)
            is_small_n = avg_max_scale_n <= size_threshold_split

            clone_candidate_mask_n = not_splitting_mask_n & is_small_n & (current_opacities_n > opacity_threshold_clone)

            potential_clones_indices = torch.where(clone_candidate_mask_n)[0]
            num_potential_clones = potential_clones_indices.shape[0]

            # Each clone adds 1 Gaussian.
            # Available slots after considering splits = max_gaussians_scene - (num_gaussians_before + num_to_split)
            available_slots_for_cloning = max_gaussians_scene - (num_gaussians_before + num_to_split)

            if num_potential_clones > available_slots_for_cloning and available_slots_for_cloning > 0:
                # If we want to clone more than we have slots for, prioritize cloning the most opaque ones
                print(f"Capping clones: {num_potential_clones} candidates, but only {available_slots_for_cloning} slots available.")
                sorted_clone_candidates_by_opacity_indices = torch.argsort(current_opacities_n[potential_clones_indices], descending=True)
                gaussians_to_clone_indices = potential_clones_indices[sorted_clone_candidates_by_opacity_indices[:available_slots_for_cloning]]
            elif available_slots_for_cloning <= 0:
                gaussians_to_clone_indices = torch.empty(0, dtype=torch.long, device=self.device)
            else:
                gaussians_to_clone_indices = potential_clones_indices

            num_to_clone = gaussians_to_clone_indices.shape[0]

            if num_to_split == 0 and num_to_clone == 0:
                return 0 # No changes made

            # Lists to store parameters of all Gaussians that will exist after densification
            all_final_xyz = []
            all_final_cholesky = []
            all_final_dc = []
            all_final_opacity = []

            current_xyz_data = self._xyz.data.clone()
            current_cholesky_data = self._cholesky.data.clone()
            current_features_dc_data = self._features_dc.data.clone()
            current_opacity_data = self._opacity.data.clone()

            # --- Add parameters of Gaussians that are kept (i.e., not split) ---
            keep_mask = torch.ones(self.num_points, dtype=torch.bool, device=self.device)
            if num_to_split > 0:
                keep_mask[gaussians_to_split_indices] = False

            if keep_mask.any(): # Only add if some are actually kept
                all_final_xyz.append(current_xyz_data[:, keep_mask, :])
                all_final_cholesky.append(current_cholesky_data[:, keep_mask, :, :])
                all_final_dc.append(current_features_dc_data[keep_mask, :])
                all_final_opacity.append(current_opacity_data[keep_mask, :])

            # --- Perform Splitting and add children ---
            if num_to_split > 0:
                print(f"Splitting {num_to_split} large Gaussians.")
                split_parents_xyz = current_xyz_data[:, gaussians_to_split_indices, :]         # T, num_split, 3
                split_parents_cholesky = current_cholesky_data[:, gaussians_to_split_indices, :, :] # T, num_split, 3, 3
                split_parents_dc = current_features_dc_data[gaussians_to_split_indices, :]       # num_split, 3
                split_parents_opacity = current_opacity_data[gaussians_to_split_indices, :]    # num_split, 1

                child1_cholesky = split_parents_cholesky.clone()
                child1_cholesky[:, :, torch.arange(2), torch.arange(2)] += torch.log(torch.tensor(scale_factor_split_children, device=self.device))
                child2_cholesky = split_parents_cholesky.clone()
                child2_cholesky[:, :, torch.arange(2), torch.arange(2)] += torch.log(torch.tensor(scale_factor_split_children, device=self.device))

                parent_scales_t_s_dim = torch.exp(split_parents_cholesky[:, :, torch.arange(2), torch.arange(2)]) # T, num_split, 2
                offset_xy_t_s_dim = (torch.rand_like(parent_scales_t_s_dim) - 0.5) * 0.1 * parent_scales_t_s_dim
                offset_t_s_3 = torch.cat([offset_xy_t_s_dim, torch.zeros_like(offset_xy_t_s_dim[..., :1])], dim=-1)

                child1_xyz = split_parents_xyz - offset_t_s_3
                child2_xyz = split_parents_xyz + offset_t_s_3

                all_final_xyz.extend([child1_xyz, child2_xyz])
                all_final_cholesky.extend([child1_cholesky, child2_cholesky])
                all_final_dc.extend([split_parents_dc, split_parents_dc])
                all_final_opacity.extend([split_parents_opacity, split_parents_opacity])

            # --- Perform Cloning and add clones ---
            if num_to_clone > 0:
                print(f"Cloning {num_to_clone} small, opaque Gaussians.")
                cloned_xyz = current_xyz_data[:, gaussians_to_clone_indices, :]
                cloned_cholesky = current_cholesky_data[:, gaussians_to_clone_indices, :, :]
                cloned_dc = current_features_dc_data[gaussians_to_clone_indices, :]
                cloned_opacity = current_opacity_data[gaussians_to_clone_indices, :]

                all_final_xyz.append(cloned_xyz)
                all_final_cholesky.append(cloned_cholesky)
                all_final_dc.append(cloned_dc)
                all_final_opacity.append(cloned_opacity)

            # Concatenate all parameters to form the new set
            if not all_final_dc: # Should not happen if num_gaussians_before > 0 and (num_split > 0 or num_clone > 0 or keep_mask.any())
                 print("Warning: Densification resulted in zero Gaussians. No changes applied.")
                 # num_points remains num_gaussians_before, num_added_total will be 0 due to later calculation
            else:
                 self._xyz = nn.Parameter(torch.cat(all_final_xyz, dim=1))
                 self._cholesky = nn.Parameter(torch.cat(all_final_cholesky, dim=1))
                 self._features_dc = nn.Parameter(torch.cat(all_final_dc, dim=0))
                 self._opacity = nn.Parameter(torch.cat(all_final_opacity, dim=0))
                 self.num_points = self._features_dc.shape[0] # N is dim 0 for static params

        num_added_total = self.num_points - num_gaussians_before

        if num_added_total > 0:
            # Re-initialize EMA buffers if they exist, for the new set of Gaussians
            if hasattr(self, 'ema_xyz') and self.ema_xyz is not None:
                print("Re-initializing EMA for _xyz")
                self.ema_xyz = EMA([self._xyz], decay=self.ema_decay) # Use stored self.ema_decay
            if hasattr(self, 'ema_cholesky') and self.ema_cholesky is not None:
                print("Re-initializing EMA for _cholesky")
                self.ema_cholesky = EMA([self._cholesky], decay=self.ema_decay) # Use stored self.ema_decay

        return num_added_total

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
