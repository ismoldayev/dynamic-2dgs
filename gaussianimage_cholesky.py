from gsplat.gsplat.project_gaussians_2d import project_gaussians_2d
from gsplat.gsplat.rasterize_sum import rasterize_gaussians_sum
from utils import *
import torch
import torch.nn as nn
import numpy as np
import math
# from quantize import *
from optimizer import Adan

class GaussianImage_Cholesky(nn.Module):
    def __init__(self, loss_type="L2", T=1, lambda_opacity_reg=0.0, lambda_temporal_xyz=0.0, lambda_temporal_cholesky=0.0, **kwargs):
        super().__init__()
        self.loss_type = loss_type
        self.num_points = kwargs["num_points"]
        self.H, self.W = kwargs["H"], kwargs["W"]
        self.T = T # Number of frames
        self.lambda_opacity_reg = lambda_opacity_reg
        self.lambda_temporal_xyz = lambda_temporal_xyz
        self.lambda_temporal_cholesky = lambda_temporal_cholesky
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

        # Backpropagate the average loss
        self.optimizer.zero_grad(set_to_none=True)
        average_loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return average_loss, average_psnr

    def prune_gaussians(self, opacity_threshold):
        """Prunes Gaussians whose opacity is below the given threshold.

        Returns:
            int: Number of Gaussians pruned.
        """
        if self.num_points == 0:
            print("No Gaussians to prune.")
            return 0

        with torch.no_grad():
            current_opacities = self.get_opacity # Access property, shape (N, 1)
            # Ensure it's 1D for boolean indexing across the N dimension
            keep_mask = (current_opacities > opacity_threshold).squeeze(dim=-1) # Squeeze the last dim to make it (N)

            num_to_keep = torch.sum(keep_mask).item()
            num_pruned = self.num_points - num_to_keep

            if num_pruned == 0:
                print(f"Pruning: No Gaussians below threshold {opacity_threshold:.6f}. All {self.num_points} remain.")
                return 0

            print(f"Pruning: Keeping {num_to_keep} of {self.num_points} Gaussians (pruning {num_pruned}). Opacity threshold: {opacity_threshold:.6f}")

            # Create new tensors from the masked data
            new_xyz_data = self._xyz.data[:, keep_mask, :].clone()
            new_cholesky_data = self._cholesky.data[:, keep_mask, :].clone()
            new_opacity_data = self._opacity.data[keep_mask, :].clone()
            new_features_dc_data = self._features_dc.data[keep_mask, :].clone()

            # It's important that the optimizer is re-initialized *after* these parameters are replaced.
            # The train.py script already handles optimizer re-initialization.

            # Detach old parameters from the graph and allow them to be garbage collected
            # by removing them as attributes before reassigning.
            del self._xyz
            del self._cholesky
            del self._opacity
            del self._features_dc

            # Assign new nn.Parameter objects. This correctly registers them with the module.
            self._xyz = nn.Parameter(new_xyz_data)
            self._cholesky = nn.Parameter(new_cholesky_data)
            self._opacity = nn.Parameter(new_opacity_data)
            self._features_dc = nn.Parameter(new_features_dc_data)

            # Ensure requires_grad is set if it was lost (it should be inherited by default by nn.Parameter)
            # For safety, could do: self._xyz.requires_grad_(True), etc. if issues arise, but usually not needed.

            self.num_points = num_to_keep

        return num_pruned

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

        # total_bits += initial_bits
        # total_bits += self._xyz.numel()*16

        # feature_dc_index = feature_dc_index.int().cpu().numpy()
        # index_max = np.max(feature_dc_index)
        # max_bit = np.ceil(np.log2(index_max)) #calculate max bit for feature_dc_index
        # total_bits += feature_dc_index.size * max_bit #get_np_size(encoding_dict["feature_dc_index"]) * 8

        # quant_cholesky_elements = quant_cholesky_elements.cpu().numpy()
        # total_bits += quant_cholesky_elements.size * 6 #cholesky bits

        # position_bits = self._xyz.numel()*16
        # cholesky_bits, feature_dc_bits = 0, 0
        # cholesky_bits += self.cholesky_quantizer.scale.numel()*torch.finfo(self.cholesky_quantizer.scale.dtype).bits
        # cholesky_bits += self.cholesky_quantizer.beta.numel()*torch.finfo(self.cholesky_quantizer.beta.dtype).bits
        # cholesky_bits += quant_cholesky_elements.size * 6
        # feature_dc_bits += codebook_bits
        # feature_dc_bits += feature_dc_index.size * max_bit

        # bpp = total_bits/self.H/self.W
        # position_bpp = position_bits/self.H/self.W
        # cholesky_bpp = cholesky_bits/self.H/self.W
        # feature_dc_bpp = feature_dc_bits/self.H/self.W
        # return {"bpp": bpp, "position_bpp": position_bpp,
        #     "cholesky_bpp": cholesky_bpp, "feature_dc_bpp": feature_dc_bpp}

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

    # def decompress(self, encoding_dict):
    #     xyz = encoding_dict["xyz"]
    #     num_points, device = xyz.size(0), xyz.device
    #     feature_dc_compressed, feature_dc_histogram_table, feature_dc_unique = encoding_dict["feature_dc_bitstream"]
    #     cholesky_compressed, cholesky_histogram_table, cholesky_unique = encoding_dict["cholesky_bitstream"]
    #     feature_dc_index = decompress_matrix_flatten_categorical(feature_dc_compressed, feature_dc_histogram_table, feature_dc_unique, num_points*2, (num_points, 2))
    #     quant_cholesky_elements = decompress_matrix_flatten_categorical(cholesky_compressed, cholesky_histogram_table, cholesky_unique, num_points*3, (num_points, 3))
    #     feature_dc_index = torch.from_numpy(feature_dc_index).to(device).int() #[800, 2]
    #     quant_cholesky_elements = torch.from_numpy(quant_cholesky_elements).to(device).float() #[800, 3]

        # means = torch.tanh(xyz.float())
        # cholesky_elements = self.cholesky_quantizer.decompress(quant_cholesky_elements)
        # cholesky_elements = cholesky_elements + self.cholesky_bound
        # colors = self.features_dc_quantizer.decompress(feature_dc_index)
        # self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(means, cholesky_elements, self.H, self.W, self.tile_bounds)
        # out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
        #         colors, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        # out_img = torch.clamp(out_img, 0, 1)
        # out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        # return {"render":out_img}

    # def analysis(self, encoding_dict):
    #     quant_cholesky_elements, feature_dc_index = encoding_dict["quant_cholesky_elements"], encoding_dict["feature_dc_index"]
    #     cholesky_compressed, cholesky_histogram_table, cholesky_unique = compress_matrix_flatten_categorical(quant_cholesky_elements.int().flatten().tolist())
    #     feature_dc_compressed, feature_dc_histogram_table, feature_dc_unique = compress_matrix_flatten_categorical(feature_dc_index.int().flatten().tolist())
    #     cholesky_lookup = dict(zip(cholesky_unique, cholesky_histogram_table.astype(np.float64) / np.sum(cholesky_histogram_table).astype(np.float64)))
    #     feature_dc_lookup = dict(zip(feature_dc_unique, feature_dc_histogram_table.astype(np.float64) / np.sum(feature_dc_histogram_table).astype(np.float64)))

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

    #     total_bits += initial_bits
    #     total_bits += self._xyz.numel()*16
    #     total_bits += get_np_size(cholesky_compressed) * 8
    #     total_bits += get_np_size(feature_dc_compressed) * 8

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

    #     bpp = total_bits/self.H/self.W
    #     position_bpp = position_bits/self.H/self.W
    #     cholesky_bpp = cholesky_bits/self.H/self.W
    #     feature_dc_bpp = feature_dc_bits/self.H/self.W
    #     return {"bpp": bpp, "position_bpp": position_bpp,
    #         "cholesky_bpp": cholesky_bpp, "feature_dc_bpp": feature_dc_bpp,}
