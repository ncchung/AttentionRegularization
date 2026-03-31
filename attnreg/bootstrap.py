import torch
import matplotlib.pyplot as plt
import gc
import numpy as np
from typing import Protocol
from collections.abc import Sequence


class BootstrapFunc(Protocol):
    def __call__(self, sample_img: torch.tensor, bootstrap_N, **kwargs) -> torch.tensor:
        ...
    """
    Callable protocol for bootstrap method functions
    """  

def bootstrap_normal(
    sample_img: torch.Tensor,
    bootstrap_N=1,
    all_channels=True,
    std_factor: float = 1.0,
    draw=False,
    savename=None,
    seed: int | None = None,
    **kwargs,
) -> torch.Tensor:
    """
    Parametric normal bootstrapping (batched).

    Parameters
    ----------
    sample_img : Tensor [B, C, H, W]
    bootstrap_N : number of bootstrap samples per image
    std_factor : multiplier in front of std (allows to shrink or widen the width)

    Returns
    -------
    Tensor [B, bootstrap_N, C, H, W]
    """

    if seed is not None:
        torch.manual_seed(seed)

    if sample_img.ndim != 4:
        raise ValueError("Input must have shape [B, C, H, W]")

    B, C, H, W = sample_img.shape
    device = sample_img.device
    dtype = sample_img.dtype

    # ---------- compute stats per image ----------
    if all_channels:
        # per-image, per-channel
        mean = sample_img.mean(dim=(2, 3), keepdim=True)  # [B,C,1,1]
        std  = sample_img.std(dim=(2, 3), keepdim=True)
    else:
        # per-image global
        mean = sample_img.mean(dim=(1, 2, 3), keepdim=True)  # [B,1,1,1]
        std  = sample_img.std(dim=(1, 2, 3), keepdim=True)

    # shape: [B, bootstrap_N, C, H, W]
    noise = torch.randn(
        (B, bootstrap_N, C, H, W),
        device=device,
        dtype=dtype,
    )

    boot = noise * std_factor * std.unsqueeze(1) + mean.unsqueeze(1)

    if draw:
        
        img_np = boot[0, 0].permute(1, 2, 0).detach().cpu().numpy()
        plt.imshow(img_np)
        plt.axis("off")

        if savename:
            plt.savefig(
                f"../results/plots/bootstrap_normal_{savename}.pdf", 
                format="pdf", 
                bbox_inches="tight"
            )
            
        plt.show()

    return boot # [B, bootstrap_N, C, H, W]


def bootstrap_pixel(
    sample_img: torch.Tensor,
    bootstrap_N: int = 1,
    normalize: bool = False,
    draw: bool = False,
    savename=None,
    seed: int | None = None,
    **kwargs,
) -> torch.Tensor:
    """
    Non-parametric pixel bootstrap (batched).

    Pixels are resampled with replacement while preserving channel structure.

    Parameters
    ----------
    sample_img : Tensor [B, C, H, W]
    bootstrap_N : number of bootstrap samples per image

    Returns
    -------
    Tensor [B, bootstrap_N, C, H, W]
    """

    if seed is not None:
        torch.manual_seed(seed)

    if sample_img.ndim != 4:
        raise ValueError("Input must have shape [B, C, H, W]")

    B, C, H, W = sample_img.shape
    device = sample_img.device
    dtype = sample_img.dtype

    # Flatten pixels while preserving channel structure
    # [B, H*W, C]
    flat = sample_img.permute(0, 2, 3, 1).reshape(B, H * W, C)

    # Sample pixel indices with replacement
    # [B, bootstrap_N, H*W]
    idx = torch.randint(
        0,
        H * W,
        (B, bootstrap_N, H * W),
        device=device,
    )

    # Expand flat pixels so gather can be used
    # [B, bootstrap_N, H*W, C]
    flat_exp = flat.unsqueeze(1).expand(-1, bootstrap_N, -1, -1)

    # Gather pixels
    boot = torch.gather(
        flat_exp,
        dim=2,
        index=idx.unsqueeze(-1).expand(-1, -1, -1, C),
    )

    # Reshape back to images
    boot = boot.reshape(B, bootstrap_N, H, W, C).permute(0, 1, 4, 2, 3)

    if draw:
        img_np = boot[0, 0].permute(1, 2, 0).detach().cpu().numpy()

        if normalize:
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

        plt.imshow(img_np)
        plt.axis("off")

        if savename:
            plt.savefig(
                f"../results/plots/bootstrap_pixel_{savename}.pdf", 
                format="pdf", 
                bbox_inches="tight"
            )
            
        plt.show()

    return boot
    

def bootstrap_patch(
    sample_img: torch.tensor, 
    patch: int = 8, 
    draw=False,
    seed: int | None = None,
    **kwargs,
) -> torch.tensor:
    """
    Non-parametric bootstrapping via reshuffling patches of the image of size 'patch'. 
    """

    if seed is not None:
        torch.manual_seed(seed)

    _, channels, height, width = sample_img.shape
    
    # check divisibility
    assert height % patch == 0 and width % patch == 0, "Image size must be divisible by patch_size"

    sample_img_pert = sample_img.clone()
    
    # number of patches in each dimension
    nH = height // patch
    nW = width // patch
    num_patches = nH * nW
    
    # ---- 1. Extract patches: shape [1, C, num_patches_h, num_patches_w, patch, patch]
    patches = sample_img_pert.unfold(2, patch, patch).unfold(3, patch, patch)

    # Move patch dims together → [num_patches, C, patch, patch]
    patches = patches.contiguous().view(1, channels, nH, nW, patch, patch)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(num_patches, channels, patch, patch)

    # Shuffle patches
    perm = torch.randperm(num_patches)
    patches_shuffled = patches[perm]

    # Reconstruct image
    # Reshape back to grid
    patches_grid = patches_shuffled.reshape(nH, nW, channels, patch, patch)
    
    # Move dims: [C, H, W]
    img_shuffled = patches_grid.permute(2, 0, 3, 1, 4).reshape(channels, height, width)

    # Add batch dimension
    sample_img_pert = img_shuffled.unsqueeze(0)

    # Optionally draw the perturbed image with bounding boxes around the perturbed pixels
    if draw:
        fig, ax = plt.subplots(1)
        img_np = sample_img_pert.squeeze().permute(1, 2, 0).cpu().detach().numpy()
        if normalize:
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())  # Normalize for display
        ax.imshow(img_np)
        plt.show()

    gc.collect()
    return sample_img_pert

def bootstrap_uniform(
    sample_img: torch.Tensor,
    bootstrap_N: int = 1,
    min_val: float | None = None,
    max_val: float | None = None,
    all_channels: bool = True,
    normalization: tuple[Sequence[float], Sequence[float]] = (
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    ),
    draw: bool = False,
    seed: int | None = None,
    **kwargs,
) -> torch.Tensor:
    """
    Parametric uniform bootstrapping (batched).

    Parameters
    ----------
    sample_img : Tensor [B, C, H, W]
    bootstrap_N : number of bootstrap samples per image
    min_val, max_val : bounds of uniform distribution (inferred if None )
    all_channels : whether to compute bounds per channel or globally

    Returns
    -------
    Tensor [B, bootstrap_N, C, H, W]
    """

    if seed is not None:
        torch.manual_seed(seed)

    if sample_img.ndim != 4:
        raise ValueError("Input must have shape [B, C, H, W]")

    B, C, H, W = sample_img.shape
    device = sample_img.device
    dtype = sample_img.dtype

    # bounds
    if min_val is None or max_val is None:
        if all_channels:
            min_val = sample_img.amin(dim=(2, 3), keepdim=True)  # [B,C,1,1]
            max_val = sample_img.amax(dim=(2, 3), keepdim=True)
        else:
            min_val = sample_img.amin(dim=(1, 2, 3), keepdim=True)  # [B,1,1,1]
            max_val = sample_img.amax(dim=(1, 2, 3), keepdim=True)
    else:
        min_val = torch.tensor(min_val, device=device, dtype=dtype).view(1, 1, 1, 1)
        max_val = torch.tensor(max_val, device=device, dtype=dtype).view(1, 1, 1, 1)

    # sample uniform noise
    noise = torch.rand(
        (B, bootstrap_N, C, H, W),
        device=device,
        dtype=dtype,
    )

    boot = noise * (max_val.unsqueeze(1) - min_val.unsqueeze(1)) + min_val.unsqueeze(1)

    if normalization is not None:
        mean, std = normalization
        mean = torch.tensor(mean, device=device, dtype=dtype).view(1, 1, C, 1, 1)
        std = torch.tensor(std, device=device, dtype=dtype).view(1, 1, C, 1, 1)
        boot = (boot - mean) / std

    if draw:
        img_np = boot[0, 0].permute(1, 2, 0).detach().cpu().numpy()
        plt.imshow(img_np)
        plt.axis("off")
        plt.show()

    return boot  # [B, bootstrap_N, C, H, W]
    