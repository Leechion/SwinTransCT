import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FBPNet(nn.Module):
    """
    Stable, device-safe Filtered Backprojection (FBP) module for parallel-beam CT.

    Key features:
    - Accepts either an image [B,1,H,W] (then predicts sinogram via small conv)
      or a sinogram [B, num_angles, det_count] (set input_is_sinogram=True).
    - Uses a learned small conv to predict sinograms when needed.
    - Applies Ram-Lak filtering in frequency domain (fixed filter stored as buffer).
    - Vectorized, dtype/device-safe linear interpolation for backprojection (no grid_sample).
    - Compatible with PyTorch 1.x and 2.x (no endpoint arg; custom gaussian blur).

    Usage examples are included in the __main__ block.
    """

    def __init__(self, img_size=256, num_angles=180, det_count=256, proj_kernel=3):
        super().__init__()
        self.img_size = int(img_size)
        self.num_angles = int(num_angles)
        self.det_count = int(det_count)

        # small learned projection conv (image -> sinogram-like)
        self.proj_conv = nn.Conv2d(1, self.num_angles, kernel_size=proj_kernel,
                                   padding=proj_kernel // 2, bias=True)
        nn.init.xavier_normal_(self.proj_conv.weight)
        nn.init.zeros_(self.proj_conv.bias)

        # Ram-Lak freq response as buffer (float32)
        # use rfftfreq length = det_count with sample spacing = 1
        freq = np.fft.rfftfreq(self.det_count, d=1.0)
        ramlak = np.abs(freq).astype(np.float32)
        self.register_buffer('ramlak_freq', torch.from_numpy(ramlak))

    def _predict_sinogram(self, img):
        # img: [B,1,H,W]
        proj_feat = self.proj_conv(img)  # [B, num_angles, H, W]
        proj_feat = F.relu(proj_feat)
        # assume detector axis corresponds to width (W)
        proj = proj_feat.sum(dim=2)  # [B, num_angles, W]
        # if W != det_count, resample along last dim
        if proj.shape[-1] != self.det_count:
            proj = F.interpolate(proj.unsqueeze(1), size=(self.num_angles, self.det_count),
                                 mode='bilinear', align_corners=False).squeeze(1)
        # per-sample normalization (preserve angle relationships)
        mean = proj.mean(dim=(-2, -1), keepdim=True)
        std = proj.std(dim=(-2, -1), keepdim=True) + 1e-6
        proj = (proj - mean) / std
        return proj

    def _ramlak_filter(self, sinogram):
        # sinogram: [B, num_angles, det_count]
        proj_fft = torch.fft.rfft(sinogram, dim=-1)
        ramlak = self.ramlak_freq.to(proj_fft.dtype).to(sinogram.device)
        filtered_fft = proj_fft * ramlak.unsqueeze(0).unsqueeze(0)
        filtered = torch.fft.irfft(filtered_fft, n=self.det_count, dim=-1)
        return filtered

    @staticmethod
    def _gaussian_blur(input, kernel_size=3, sigma=0.5):
        # device/dtype-safe small gaussian blur implemented with conv2d
        device = input.device
        dtype = input.dtype
        k = int(kernel_size)
        assert k % 2 == 1, "kernel_size must be odd"
        half = k // 2
        coords = torch.arange(-half, half + 1, device=device, dtype=dtype)
        kernel_1d = torch.exp(-0.5 * (coords / float(sigma)) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # [1,1,k,k]
        kernel_2d = kernel_2d.expand(input.shape[1], 1, k, k).contiguous()
        return F.conv2d(input, kernel_2d, padding=half, groups=input.shape[1])

    def forward(self, x, input_is_sinogram=False):
        """
        Forward reconstruction.
        - If input_is_sinogram=False, x is [B,1,H,W] image and we predict sinogram.
        - If input_is_sinogram=True, x is [B,num_angles,det_count] sinogram and used directly.

        Returns: [B,1,img_size,img_size]
        """
        # ensure module is on same device as input
        device = x.device
        # move projection conv weights to device if needed (no-op if already)
        self.proj_conv.to(device)

        if input_is_sinogram:
            sinogram = x
            assert sinogram.ndim == 3 and sinogram.shape[1] == self.num_angles and sinogram.shape[2] == self.det_count, \
                f"Expected sinogram shape [B,{self.num_angles},{self.det_count}]"
        else:
            # image input: ensure shape and dtype
            assert x.ndim == 4 and x.shape[1] == 1, "Expected image tensor [B,1,H,W]"
            sinogram = self._predict_sinogram(x)

        # filter
        filtered = self._ramlak_filter(sinogram)

        # precompute image grid (in normalized coordinates)
        H = self.img_size
        W = self.img_size
        x_lin = torch.linspace(-1.0, 1.0, steps=W, device=device, dtype=filtered.dtype)
        y_lin = torch.linspace(-1.0, 1.0, steps=H, device=device, dtype=filtered.dtype)
        yy, xx = torch.meshgrid(y_lin, x_lin, indexing='ij')  # [H,W]

        angles = torch.linspace(0.0, np.pi - np.pi / self.num_angles, steps=self.num_angles, device=device, dtype=filtered.dtype)

        # normalized factor: t ranges in [-sqrt(2), sqrt(2)] for image coords in [-1,1]
        norm_factor = float(np.sqrt(2.0))

        B = filtered.shape[0]
        n_pixels = H * W

        # prepare flattened coords (1, n_pixels)
        # we'll compute u indices per angle and sample via gather
        fbp = torch.zeros(B, 1, H, W, device=device, dtype=filtered.dtype)

        # flatten grid once
        # xx_flat: [n_pixels], yy_flat: [n_pixels]
        xx_flat = xx.reshape(-1)
        yy_flat = yy.reshape(-1)

        # For each angle compute detector coordinate u and sample
        det_N = float(self.det_count)
        for i in range(self.num_angles):
            theta = angles[i]
            # t in [-sqrt(2), sqrt(2)]
            t = (xx_flat * torch.cos(theta) + yy_flat * torch.sin(theta)) / norm_factor  # [n_pixels]
            # map to detector index in [0, det_count-1]
            u = (t + 1.0) * 0.5 * (det_N - 1.0)
            u = torch.clamp(u, 0.0, det_N - 1.0)

            # prepare indexing for batch: [B, n_pixels]
            u_exp = u.unsqueeze(0).expand(B, -1)  # float indices
            idx0 = torch.floor(u_exp).long()
            idx1 = torch.clamp(idx0 + 1, max=self.det_count - 1)
            w = (u_exp - idx0.to(u_exp.dtype))  # weights for interpolation

            proj_i = filtered[:, i, :]  # [B, det_count]
            vals0 = torch.gather(proj_i, 1, idx0)
            vals1 = torch.gather(proj_i, 1, idx1)
            sampled = vals0 * (1.0 - w) + vals1 * w  # [B, n_pixels]

            sampled = sampled.reshape(B, 1, H, W)
            fbp = fbp + sampled

        # scale: detector pixel size * angle step
        det_pixel_size = 2.0 / float(self.det_count)  # detector covers [-1,1]
        angle_step = float(np.pi / self.num_angles)
        fbp = fbp * det_pixel_size * np.pi / float(self.num_angles)

        # mild gaussian blur
        fbp = self._gaussian_blur(fbp, kernel_size=3, sigma=0.5)
        fbp = torch.clamp(fbp, min=-1.0, max=1.0)
        return fbp


# ------------------ quick test ------------------
# if __name__ == '__main__':
#     # auto device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = FBPNet(img_size=256, num_angles=180, det_count=256).to(device)
#
#     # image input test
#     img = (torch.rand(2, 1, 256, 256, device=device) * 2.0 - 1.0).to(dtype=torch.float32)
#     with torch.no_grad():
#         out = model(img, input_is_sinogram=False)
#     print('image input ->', out.shape, out.dtype, out.device)
#
#     # sinogram input test
#     sino = torch.randn(2, 180, 256, device=device, dtype=torch.float32)
#     with torch.no_grad():
#         out2 = model(sino, input_is_sinogram=True)
#     print('sinogram input ->', out2.shape, out2.dtype, out2.device)
