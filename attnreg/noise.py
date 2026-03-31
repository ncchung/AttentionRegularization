import numpy as np
import matplotlib.pyplot as plt
from typing import Protocol, Optional
import torch

from attnreg.attention import ViTWrapper
from attnreg.bootstrap import BootstrapFunc, bootstrap_normal
from attnreg.regularization import attention_uncertainty, regularize_attention_all
from attnreg.plotting import plot_statistics_in_ROI, plot_attention_with_frame


class NoiseFunc(Protocol):
    def __call__(self, mask: torch.tensor, **kwargs) -> torch.tensor:
        ...
    """
    Callable protocol for noise injection functions
    """


def sample_rgb_gaussian_noise(mask: torch.tensor, image: torch.tensor, seed: int | None = None, **kwargs):
    """
    Much simpler: generate full noise image, then select masked values.
    """
    device, dtype = image.device, image.dtype

    # if seed is not None:
    #     gen = torch.Generator(device=device)
    #     gen.manual_seed(seed)
    if seed is not None:
        torch.manual_seed(seed)

    mean = image.mean(dim=(0, 2, 3), keepdim=True)  # [1,C,1,1]
    std  = image.std(dim=(0, 2, 3), keepdim=True)

    #noise_img = torch.randn_like(image, generator=gen) * std + mean
    noise_img = torch.randn_like(image) * std + mean

    return noise_img[mask]

# def sample_rgb_gaussian_noise(mask: torch.tensor, image: torch.tensor, seed: int | None = None, **kwargs):
#     """
#     Much simpler: generate full noise image, then select masked values.
#     """
#     device, dtype = image.device, image.dtype

#     # if seed is not None:
#     #     gen = torch.Generator(device=device)
#     #     gen.manual_seed(seed)
#     if seed is not None:
#         np.random.seed(seed)

#     img_np = image.squeeze().permute(1,2,0).detach().numpy()

#     # Separate channels
#     R, G, B = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]

#     # Calculate mean and std per channel
#     mean_r, std_r = R.mean(), R.std()
#     mean_g, std_g = G.mean(), G.std()
#     mean_b, std_b = B.mean(), B.std()

#     # Sample new image from normal distribution per channel
#     sampled_r = np.random.normal(mean_r, std_r, R.shape)
#     sampled_g = np.random.normal(mean_g, std_g, G.shape)
#     sampled_b = np.random.normal(mean_b, std_b, B.shape)

#     # Stack channels and clip to valid range [0, 255]
#     sampled_img = np.stack([sampled_r, sampled_g, sampled_b], axis=2)

#     noise_img = torch.from_numpy(sampled_img).float().permute(2,0,1).unsqueeze(0).to(device)

#     return noise_img[mask]

""" ======================================= Masks ======================================="""

class NoiseMask(Protocol):
    def __call__(self, height: int, width: int, size: int, **kwargs) -> torch.tensor:
        ...
    """
    Callable protocol for noise masks
    """

def square_mask(
        height: int,
        width: int,
        size: int,
        loc_x: int | None = None,
        loc_y: int | None = None,
        seed: int | None = None,
        device=None,
    ) -> torch.tensor:
        """
        Create a boolean square mask.
    
        Parameters
        ----------
        height, width : image dimensions
        size          : square side length
        loc_x, loc_y  : top-left corner (optional). Random if None.
        device        : torch device
    
        Returns
        -------
        mask : (H, W) bool tensor
        """

        if seed is not None:
            torch.manual_seed(seed)
    
        if size > height or size > width:
            raise ValueError("Square size must fit inside the image")
    
        device = device or "cpu"
    
        # Random location if not provided
        if loc_x is None:
            loc_x = torch.randint(0, width - size + 1, (1,), device=device).item()
        if loc_y is None:
            loc_y = torch.randint(0, height - size + 1, (1,), device=device).item()
    
        mask = torch.zeros((height, width), dtype=torch.bool, device=device)
        mask[loc_y:loc_y + size, loc_x:loc_x + size] = True
    
        return mask

def diffuse_mask(
    height: int, 
    width: int,
    size: int,
    cluster_scale: float = 20.0,
    seed: int | None = None,
    device=None
) -> torch.tensor:
    """
    Create a clustered mask using a smooth random field.
    """

    total_pixels = size**2

    field = np.random.randn(height, width)

    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]

    gaussian = np.exp(-(fx**2 + fy**2) * (cluster_scale ** 2))

    field_fft = np.fft.fft2(field)
    field_smooth = np.fft.ifft2(field_fft * gaussian).real

    field_smooth -= field_smooth.min()
    field_smooth /= field_smooth.max()

    flat = field_smooth.flatten()

    idx = np.argpartition(flat, -total_pixels)[-total_pixels:]

    #mask = np.zeros(h * w, dtype=np.uint8)
    mask = torch.zeros(height * width, dtype=torch.bool, device=device)
    mask[idx] = True

    return mask.reshape(height, width)
    

def apply_noise_with_mask(
    image: torch.tensor,
    mask: torch.tensor,
    noise_fn: NoiseFunc,
    noise_kwargs: dict | None = None,
) -> torch.tensor:
    """
    Apply noise to pixels where mask == True.

    Parameters
    ----------
    image : [B, C, H, W]
    mask  : [H, W] or [B, H, W] (bool)
    noise_fn : NoiseFunc
        Must return values shaped like image[mask_exp]

    Returns
    -------
    [B, C, H, W] tensor (copy)
    """

    if noise_kwargs is None:
        noise_kwargs = {}

    if mask.dtype != torch.bool:
        raise ValueError("mask must be boolean")

    B, C, H, W = image.shape

    out = image.clone()

    mask_exp = mask.clone()
    
    # ---- expand mask to [B,C,H,W] ----
    if mask.ndim == 2:           # [H,W]
        mask_exp = mask.unsqueeze(0).unsqueeze(0)   # [1,1,H,W]
    elif mask.ndim == 3:         # [B,H,W]
        mask_exp = mask.unsqueeze(1)                # [B,1,H,W]
    else:
        raise ValueError("mask must be [H,W] or [B,H,W]")

    mask_exp = mask_exp.expand(B, C, H, W)

    # ---- generate replacement values ----
    noise_vals = noise_fn(mask_exp, image=image, **noise_kwargs)

    # ---- replace only masked entries ----
    out[mask_exp] = noise_vals

    return out
