import torch
import torch.nn.functional as F

def build_gpu_preprocess(n_px, device):
    """
    GPU version of CLIP _transform(n_px), matching behavior:
    - Resize so the shortest side == n_px (preserving aspect ratio)
    - CenterCrop to [n_px, n_px]
    - Convert grayscale -> RGB by channel repeat
    - Scale to [0,1] if needed
    - Normalize with CLIP stats
    Expects [C,H,W] or [B,C,H,W] tensors.
    """

    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    def preprocess(x: torch.Tensor) -> torch.Tensor:
        # --- Make sure we have [B,C,H,W] ---
        if x.dim() == 3:
            x = x.unsqueeze(0)  # [1,C,H,W]
        if x.size(1) == 1:      # _convert_image_to_rgb equivalent (grayscale -> 3ch)
            x = x.repeat(1, 3, 1, 1)
        elif x.size(1) != 3:
            raise ValueError(f"Expected 1 or 3 channels, got {x.size(1)}.")

        # Move to GPU, float
        x = x.to(device, dtype=torch.float32, non_blocking=True)

        # --- ToTensor() equivalent scaling ---
        # If values look like [0,255], scale to [0,1]
        if x.max() > 1.5:
            x = x / 255.0

        # --- Resize(n_px, keep aspect) ---
        # Compute size so min(H,W) == n_px
        _, _, H, W = x.shape
        short, long = (H, W) if H <= W else (W, H)
        scale = n_px / float(short)
        newH = int(round(H * scale))
        newW = int(round(W * scale))
        x = F.interpolate(x, size=(newH, newW), mode="bicubic", align_corners=False)

        # --- CenterCrop(n_px) ---
        _, _, H2, W2 = x.shape
        top = max((H2 - n_px) // 2, 0)
        left = max((W2 - n_px) // 2, 0)
        x = x[:, :, top:top + n_px, left:left + n_px]
        # (If numeric rounding makes side smaller than n_px, pad — rare. Optional:
        #  padH = n_px - x.shape[2]; padW = n_px - x.shape[3]; handle if >0.)

        # --- Normalize(mean, std) ---
        x = (x - mean) / std
        return x

    return preprocess
