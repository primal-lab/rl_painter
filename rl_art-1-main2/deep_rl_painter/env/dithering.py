# dithering.py
import torch

# Canonical 8×8 Bayer matrix (values 0..63)
_BAYER8 = torch.tensor([
    [ 0,48,12,60, 3,51,15,63],
    [32,16,44,28,35,19,47,31],
    [ 8,56, 4,52,11,59, 7,55],
    [40,24,36,20,43,27,39,23],
    [ 2,50,14,62, 1,49,13,61],
    [34,18,46,30,33,17,45,29],
    [10,58, 6,54, 9,57, 5,53],
    [42,26,38,22,41,25,37,21]
], dtype=torch.float32)

def ordered_dither(gray01: torch.Tensor, cell: int = 8) -> torch.Tensor:
    """
    gray01: [B,1,H,W] float in [0,1]
    returns: [B,1,H,W] float in {0,1}
    """
    assert gray01.ndim == 4 and gray01.shape[1] == 1, \
        f"ordered_dither expects [B,1,H,W], got {tuple(gray01.shape)}"
    device = gray01.device
    B, _, H, W = gray01.shape

    if cell != 8:
        # For now we support 8×8 Bayer. You can extend later if needed.
        raise ValueError(f"Only cell=8 supported right now, got {cell}")

    # Normalize thresholds to (0,1)
    T = ((_BAYER8.to(device) + 0.5) / 64.0)  # +0.5 centers thresholds
    # Tile to image size
    rep_h = (H + 7) // 8
    rep_w = (W + 7) // 8
    thresh = T.repeat(rep_h, rep_w)[:H, :W].unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

    return (gray01 >= thresh).float()

def _save_png01_nchw(x: torch.Tensor, path: str):
    """Save a [B,1,H,W] or [1,H,W] tensor in [0,1] as an 8-bit PNG."""
    import os
    from PIL import Image
    import numpy as np

    os.makedirs(os.path.dirname(path), exist_ok=True)
    x = x.detach().clamp(0, 1).cpu()

    if x.ndim == 4:  # [B,1,H,W] or [B,C,H,W]
        x = x[0]
    if x.ndim == 3:  # [1,H,W] or [C,H,W]
        x = x[0]
    arr = (x.numpy() * 255).astype(np.uint8)  # [H,W]
    Image.fromarray(arr).save(path)

