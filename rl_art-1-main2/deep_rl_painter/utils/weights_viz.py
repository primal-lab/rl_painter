# utils/weights_viz.py
# visualisation of weights
import math
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

# ---------- Core stats ----------
@torch.no_grad()
def layer_stats(model):
    """
    Per-parameter stats (mean, std, L2, maxabs) + grad stats if present.
    Return: List[dict]
    """
    stats = []
    for name, p in model.named_parameters():
        if p.data is None:
            continue
        d = {
            "name": name,
            "shape": list(p.shape),
            "w_mean": p.data.mean().item(),
            "w_std":  p.data.std(unbiased=False).item(),
            "w_l2":   p.data.norm().item(),
            "w_maxabs": p.data.abs().max().item(),
        }
        if p.grad is not None:
            g = p.grad
            d.update({
                "g_mean": g.mean().item(),
                "g_std":  g.std(unbiased=False).item(),
                "g_l2":   g.norm().item(),
                "g_maxabs": g.abs().max().item(),
            })
        stats.append(d)
    return stats

# ---------- Weight update ratio ----------
class WeightChangeTracker:
    """
    Tracks how much each layer changes per step/epoch.
    update_ratio = ||ΔW|| / (||W|| + eps)
    """
    def __init__(self, model):
        self.prev = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def step(self, model, eps=1e-12):
        ratios = {}
        curr = model.state_dict()
        for k, v in curr.items():
            if not torch.is_floating_point(v):
                continue
            dw = v - self.prev[k]
            denom = v.norm().item() + eps
            ratios[k] = (dw.norm().item() / denom) if denom > 0 else 0.0
        self.prev = {k: v.detach().clone() for k, v in curr.items()}
        return ratios

# ---------- Gradient flow ----------
@torch.no_grad()
def gradient_flow(model):
    """
    Returns (names, mean_abs_grad_per_param) for the latest backward pass.
    """
    names, means = [], []
    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            names.append(n)
            means.append(p.grad.abs().mean().item())
    return names, means

# ---------- Conv filter grids ----------
@torch.no_grad()
def conv_filter_grid(conv_weight, max_filters=64, normalize=True, padding=1):
    """
    Visualize early conv filters as an image grid.
    conv_weight: (out_c, in_c, kH, kW)
    """
    W = conv_weight.clone().cpu()
    out_c, in_c, kH, kW = W.shape

    if in_c == 3:
        imgs = W[:max_filters]
    elif in_c == 1:
        imgs = W[:max_filters].repeat(1, 3, 1, 1)
    else:
        W = W.sum(dim=1, keepdim=True)
        imgs = W[:max_filters].repeat(1, 3, 1, 1)

    grid = make_grid(
        imgs,
        nrow=int(math.sqrt(min(max_filters, imgs.size(0)))) or 1,
        normalize=normalize,
        padding=padding,
    )
    return grid  # (3, Hgrid, Wgrid)

# ---------- Activation recorder (hooks) ----------
class ActivationRecorder:
    """
    Register forward hooks on chosen modules (e.g., first convs, merged MLP layers).
    Saves last-batch activations per module name for feature-map grids & dead-ReLU rates.
    """
    def __init__(self, modules_dict):
        self.handles = []
        self.cache = {}
        for name, m in modules_dict.items():
            if m is None:
                continue
            h = m.register_forward_hook(self._hook(name))
            self.handles.append(h)

    def _hook(self, name):
        def fn(module, inp, out):
            self.cache[name] = out.detach().to('cpu')
        return fn

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    @torch.no_grad()
    def dead_rate(self, name):
        x = self.cache.get(name)
        if x is None:
            return None
        total = x.numel()
        zeros = (x == 0).sum().item()
        return zeros / max(total, 1)

    @torch.no_grad()
    def feature_map_grid(self, name, max_maps=64, normalize=True):
        """
        For (B,C,H,W) activations: returns a multi-channel image grid (first item in batch).
        For (B,D) activations (e.g., MLP): return None (use histograms from layer_stats instead).
        """
        x = self.cache.get(name)
        if x is None:
            return None
        if x.dim() == 4:  # (B, C, H, W)
            fm = x[0]                      # (C, H, W)
            C = fm.shape[0]
            Csel = min(C, max_maps)
            fm = fm[:Csel].unsqueeze(1)    # (Csel,1,H,W)
            fm = fm.repeat(1, 3, 1, 1)     # to 3ch for logging
            grid = make_grid(fm, nrow=int(math.sqrt(Csel)) or 1,
                             normalize=normalize, padding=1)
            return grid
        return None

# ---------- Actor/Critic specific ----------
@torch.no_grad()
def actor_distribution(actor_output, discrete=True):
    """
    For discrete heads: pass in logits -> returns probs histogram data.
    For continuous heads: returns per-dimension tensors for hist logging.
    """
    if discrete:
        if actor_output.dim() == 2:
            probs = F.softmax(actor_output, dim=-1)
            return {"probs": probs.flatten().cpu()}
        return {"probs": actor_output.flatten().cpu()}
    else:
        if actor_output.dim() == 2:
            return {f"act_dim_{i}": actor_output[:, i].cpu()
                    for i in range(actor_output.size(1))}
        return {"actions": actor_output.flatten().cpu()}

@torch.no_grad()
def td_error_hist(td_errors):
    return td_errors.detach().flatten().cpu()

@torch.no_grad()
def snapshot_vector(model):
    v = []
    for p in model.parameters():
        if torch.is_floating_point(p):
            v.append(p.flatten().cpu())
    return torch.cat(v) if v else torch.tensor([])

# ===== Ultra-light, GPU-friendly helpers =====

@torch.no_grad()
def flat_params(model, device):
    """
    One flat vector of trainable float params ON GPU (no CPU copy).
    """
    vecs = []
    for p in model.parameters():
        if p.requires_grad and torch.is_floating_point(p):
            vecs.append(p.detach().view(-1))
    if not vecs:
        return torch.zeros(1, device=device)
    return torch.cat(vecs, dim=0).to(device)

@torch.no_grad()
def grad_flow_mean(model, device):
    """
    Single mean(|grad|) over all params ON GPU; returns a Python float.
    """
    total_abs = torch.zeros((), device=device)
    total_n   = torch.zeros((), device=device)
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        total_abs += g.abs().sum()
        total_n   += torch.tensor(g.numel(), device=device)
    return (total_abs / total_n.clamp_min(1)).item()

@torch.no_grad()
def update_ratio_gpu(prev_vec, curr_vec, eps=1e-12):
    """
    ||ΔW|| / (||W|| + eps) computed ON GPU; returns a Python float.
    """
    delta = (curr_vec - prev_vec)
    return (delta.norm() / (curr_vec.norm() + eps)).item()

# Backward-compatible aliases (if you referenced the underscored names)
_flat_params      = flat_params
_grad_flow_mean   = grad_flow_mean
_update_ratio_gpu = update_ratio_gpu
