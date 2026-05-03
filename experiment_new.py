"""
Reproduces Figure D.3 from:
  "Generalization Bounds for Rank-sparse Neural Networks"
  Ledent, Alves, Lei. NeurIPS 2025.

Architecture (Section D.3):
  - 3 convolutional layers, kernel size (3,3)
  - 5 fully connected layers
  - Output layer width = 10 (CIFAR-10)
  - Intermediary FC layer widths varied in {200, 600, 1000}
  - Spectral regularization: spectral norms of all weight matrices constrained to 1
  - Margin set dynamically to ensure <=1% accuracy loss vs margin=0
  - Reference matrices M_ell = 0

Bounds computed (dominant term only, ignoring constants and log factors):
  - Bartlett et al. 2017      (Eq. C1)
  - Golowich et al. 2018 x2  (Eq. C6, C7)
  - Neyshabur et al. 2018    (Eq. C5, deduced from [3]/[1])
  - Long & Sedghi 2020       (Eq. C15)  -- parameter counting
  - Graf et al. 2022         (Eq. C16)  -- parameter counting
  - Pinto et al. 2024 x3     (Eq. C9/C10 with C1=1 and C1=2)
  - Ledent et al. 2025       (without loss augmentation)
  - Ledent et al. 2025       (with loss augmentation)

Usage:
  python cifar10_cnn_bounds.py [--widths 200 600 1000] [--epochs 50]
                               [--save_dir ./saved_models] [--seed 42]
"""

import argparse
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json


# ──────────────────────────────────────────────────────────────
# 1.  Spectral normalisation helper
# ──────────────────────────────────────────────────────────────

def spectral_norm_matrix(W: torch.Tensor) -> torch.Tensor:
    """Largest singular value of W (viewed as a 2-D matrix)."""
    W2 = W.view(W.shape[0], -1)
    # power iteration is cheaper than full SVD for large matrices
    if W2.shape[0] == 1 or W2.shape[1] == 1:
        return W2.norm()
    u = torch.randn(W2.shape[0], 1, device=W.device, dtype=W.dtype)
    u = u / u.norm()
    for _ in range(10):
        v = W2.t().mv(u.squeeze())
        v = v / v.norm()
        u_new = W2.mv(v)
        sigma = u_new.norm()
        u = u_new / sigma
    return sigma


def project_spectral_norm_(module: nn.Module, max_sigma: float = 1.0):
    """In-place projection: rescale weight so spectral norm <= max_sigma."""
    with torch.no_grad():
        for name, param in module.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                sigma = spectral_norm_matrix(param.data)
                if sigma > max_sigma:
                    param.data.mul_(max_sigma / sigma)


# ──────────────────────────────────────────────────────────────
# 2.  Network definition
# ──────────────────────────────────────────────────────────────

class CNN(nn.Module):
    """
    3 conv layers (3x3 kernel) + 5 FC layers.
    Conv channels are fixed; FC width is the hyperparameter.
    BatchNorm added after every layer.
    """

    CONV_CHANNELS = [3, 64, 128, 256]

    def __init__(self, fc_width: int, num_classes: int = 10):
        super().__init__()
        self.fc_width = fc_width

        # --- convolutional part ---
        self.conv1 = nn.Conv2d(3,   64,  kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64,  128, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(256)

        # After pooling: 32 → 16 → 8 → 4
        flat_size = 256 * 4 * 4

        # --- fully connected part ---
        self.fc1 = nn.Linear(flat_size, fc_width, bias=False)
        self.bn4 = nn.BatchNorm1d(fc_width)

        self.fc2 = nn.Linear(fc_width, fc_width, bias=False)
        self.bn5 = nn.BatchNorm1d(fc_width)

        self.fc3 = nn.Linear(fc_width, fc_width, bias=False)
        self.bn6 = nn.BatchNorm1d(fc_width)

        self.fc4 = nn.Linear(fc_width, fc_width, bias=False)
        self.bn7 = nn.BatchNorm1d(fc_width)

        self.fc5 = nn.Linear(fc_width, num_classes, bias=False)
        self.bn8 = nn.BatchNorm1d(num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = F.relu(self.bn6(self.fc3(x)))
        x = F.relu(self.bn7(self.fc4(x)))
        x = self.bn8(self.fc5(x))  # no ReLU on final logits

        return x

    def intermediate_activations(self, x):
        acts = []

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        acts.append(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        acts.append(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        acts.append(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.bn4(self.fc1(x)))
        acts.append(x)

        x = F.relu(self.bn5(self.fc2(x)))
        acts.append(x)

        x = F.relu(self.bn6(self.fc3(x)))
        acts.append(x)

        x = F.relu(self.bn7(self.fc4(x)))
        acts.append(x)

        x = self.bn8(self.fc5(x))
        acts.append(x)

        return acts, x

# ──────────────────────────────────────────────────────────────
# 3.  Training
# ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device, max_sigma=1.0, lambda_reg=1e-3):
    model.train()
    total_loss, correct, total = 0., 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)

        # Compute loss
        loss = F.cross_entropy(out, labels)

        # Compute L1-reg
        l1_norm = sum(param.abs().sum() for param in model.parameters())

        # Total regularized loss
        loss = loss + lambda_reg * l1_norm
        loss.backward()
        optimizer.step()
        
        # project spectral norms after each step
        project_spectral_norm_(model, max_sigma)
        total_loss += loss.item() * imgs.size(0)
        correct += out.argmax(1).eq(labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        correct += out.argmax(1).eq(labels).sum().item()
        total += imgs.size(0)
    return correct / total


# ──────────────────────────────────────────────────────────────
# 4.  Margin calibration
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def find_margin(model, loader, device, acc_drop_tol=0.01):
    """
    Find largest gamma such that accuracy with margin gamma is
    within acc_drop_tol of accuracy with gamma=0.
    Uses bisection on the training set.
    """
    model.eval()
    # collect all scores and labels
    all_scores, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        scores = model(imgs)
        all_scores.append(scores.cpu())
        all_labels.append(labels)
    scores = torch.cat(all_scores)   # (N, C)
    labels = torch.cat(all_labels)   # (N,)
    N = scores.shape[0]

    base_acc = (scores.argmax(1) == labels).float().mean().item()

    # margin of each sample: score_y - max_{c != y} score_c
    true_scores = scores[range(N), labels]
    max_other = scores.clone()
    max_other[range(N), labels] = -1e9
    max_other_scores = max_other.max(1).values
    margins = true_scores - max_other_scores   # (N,)

    lo, hi = 0.0, margins.max().item()
    for _ in range(50):
        mid = (lo + hi) / 2.0
        # fraction correctly classified with margin mid
        acc_margin = (margins >= mid).float().mean().item()
        if base_acc - acc_margin <= acc_drop_tol:
            lo = mid
        else:
            hi = mid
    gamma = lo
    I_gamma = (margins < gamma).float().mean().item()   # fraction of samples with margin < gamma
    return gamma, I_gamma


# ──────────────────────────────────────────────────────────────
# 5.  Schatten-p quasi-norm
# ──────────────────────────────────────────────────────────────

def schatten_p(W: torch.Tensor, p: float) -> float:
    """||W||_{sc,p}^p  (sum of singular values^p)."""
    W2 = W.detach().view(W.shape[0], -1).float()
    sv = torch.linalg.svdvals(W2)
    if p == 0:
        return float((sv > 1e-10).sum().item())
    return float((sv ** p).sum().item())


def schatten_p_norm(W: torch.Tensor, p: float) -> float:
    """||W||_{sc,p}  (p-th root of schatten_p)."""
    if p == 0:
        return schatten_p(W, 0)   # = rank
    return schatten_p(W, p) ** (1.0 / p)


def entrywise_lp(W: torch.Tensor, p: float) -> float:
    """
    Entry-wise L_p quasi-norm: ||W||_{p}^p = sum_{i,j} |W_{ij}|^p
    (viewing W as a 2-D matrix).  For p=0 returns the number of non-zero entries.
    """
    W2 = W.detach().view(W.shape[0], -1).float()
    if p == 0:
        return float((W2.abs() > 1e-10).sum().item())
    return float((W2.abs() ** p).sum().item())


def conv_spectral_norm(weight, input_shape, stride=1, padding=0, n_iters=20):
    """
    weight: (C_out, C_in, kH, kW)
    input_shape: (C_in, H, W)
    """
    C_in, H, W = input_shape

    # initialize random vector (like vec(x))
    v = torch.randn(1, C_in, H, W).to(weight.device)
    v = v / v.norm()

    for _ in range(n_iters):
        # forward: T v
        u = F.conv2d(v, weight, stride=stride, padding=padding)
        u = u / u.norm()

        # adjoint: T^T u
        v = F.conv_transpose2d(u, weight, stride=stride, padding=padding)
        v = v / v.norm()

    # Rayleigh quotient
    Tv = F.conv2d(v, weight, stride=stride, padding=padding)
    sigma = Tv.norm()

    return sigma.item()

# ──────────────────────────────────────────────────────────────
# 6.  Network weight statistics
# ──────────────────────────────────────────────────────────────

def get_weight_stats(model):
    """
    Returns per-layer dicts with:
      W            : weight tensor
      spectral_norm: ||A||  (spectral norm, = op_norm for conv via op(A))
      frobenius    : ||A||_F
      schatten_fn  : function(p) -> ||A||_{sc,p}^p
      shape        : (out, in_flat)
      is_conv      : bool
      patches      : number of spatial patches (for conv layers)
      U, d, w      : channels_out, patch_dim, spatial_dim (for conv)
    """
    stats = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            is_conv = isinstance(m, nn.Conv2d)
            W = m.weight
            W2 = W.detach().view(W.shape[0], -1).float()

            sv = torch.linalg.svdvals(W2)
            if is_conv:
                spec = conv_spectral_norm(W, (W.shape[1], 32, 32))
            else:
                spec = sv.max().item()
            frob = W2.norm().item()

            entry = {
                'name': name,
                'W': W,
                'spectral_norm': spec,
                'frobenius': frob,
                'sv': sv.cpu().numpy(),
                'shape': tuple(W2.shape),
                'is_conv': is_conv,
            }
            if is_conv:
                # U = out_channels, d = in_channels * kH * kW, w = spatial output size
                U = W.shape[0]
                d = W.shape[1] * W.shape[2] * W.shape[3]
                entry['U'] = U
                entry['d_patch'] = d
                # spatial dim after pooling depends on layer index; we store kernel info
                entry['kernel'] = (W.shape[2], W.shape[3])
            else:
                entry['out'] = W.shape[0]
                entry['in'] = W.shape[1]
            stats.append(entry)
    return stats


# ──────────────────────────────────────────────────────────────
# 7.  Spatial dimensions for CIFAR-10 after pooling
# ──────────────────────────────────────────────────────────────
# Input: 32x32
# After conv1 + MaxPool2d(2): 16x16 → O1 = 16*16 = 256,  w1 = 16*16
# After conv2 + MaxPool2d(2): 8x8  → O2 = 8*8   = 64,   w2 = 8*8
# After conv3 + MaxPool2d(2): 4x4  → O3 = 4*4   = 16,   w3 = 4*4

SPATIAL_W = [16*16, 8*8, 4*4]   # w_ell for conv layers 1,2,3


# ──────────────────────────────────────────────────────────────
# 8.  Activation norms for loss augmentation (B_{ell-1, A})
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_B_ells(model, loader, device):
    """
    B_{ell-1, A} = max over training samples i, patches o of ||[F^{0->ell-1}(x_i)]_{patch o}||

    For conv layers: patch norm = L2 norm of a spatial patch (channel slice).
    For FC layers: just the vector norm of the activation.

    We return one value per layer (L values total), where layer index 0 is the first
    convolutional layer.  B_0 = max_i max_o ||[x_i]_{S_{0,o}}||  (input patch norms).
    """
    model.eval()
    L = 8  # 3 conv + 5 FC

    # We'll store running maxima
    B = [0.0] * L

    for imgs, _ in loader:
        imgs = imgs.to(device)
        N = imgs.shape[0]

        # B_0: max patch norm of input
        # Input patches: conv1 has kernel 3x3, padding=1, so each patch = 3*3*3 = 27 values
        # Padding makes it so the spatial output has the same size as input before pooling.
        # We approximate: max_i ||x_i|| (full image norm) as an upper bound.
        # More precisely, B_{0,A} = max_i max_o ||[x_i]_{S_{0,o}}||
        # For 3x3 conv with padding=1: each patch is a 3x3x3 block
        # We unfold the image to get all patches
        x = imgs  # (N, 3, 32, 32)
        # unfold: extract 3x3 patches
        patches = x.unfold(2, 3, 1).unfold(3, 3, 1)   # (N, 3, 32, 32, 3, 3)
        patch_norms = patches.contiguous().view(N, -1, 27).norm(dim=2)   # (N, n_patches)
        B[0] = max(B[0], patch_norms.max().item())

        # Now propagate through the network layers and compute B_ell
        acts, _ = model.intermediate_activations(x)

        # acts[0]: after conv1+relu+pool  shape (N, 64, 16, 16)   → B_1 = max patch norm
        # acts[1]: after conv2+relu+pool  shape (N, 128, 8, 8)
        # acts[2]: after conv3+relu+pool  shape (N, 256, 4, 4)
        # For conv layers: patch norm = norm of the channel vector at each spatial location
        for ci, act in enumerate(acts[:3]):
            # act: (N, C, H, W) — patch at each spatial loc is the C-dim vector
            # ||act[:, :, h, w]||_2
            norms = act.norm(dim=1)   # (N, H, W)
            B[ci + 1] = max(B[ci + 1], norms.max().item())

        # acts[3..6]: FC intermediate activations, shape (N, fc_width)
        # B_ell = max_i ||F^{0->ell-1}(x_i)||
        for fi, act in enumerate(acts[3:7]):
            norms = act.norm(dim=1)   # (N,)
            B[fi + 4] = max(B[fi + 4], norms.max().item())

        # acts[7]: final output (N, 10) — not needed for B

    return B   # length L = 8


# ──────────────────────────────────────────────────────────────
# 9.  Bound computation
# ──────────────────────────────────────────────────────────────

def get_layer_params(model, stats, fc_width):
    """
    Extract per-layer parameters needed for all bounds.
    Layer ordering: conv1, conv2, conv3, fc1, fc2, fc3, fc4, fc5
    L = 8 total layers.

    Returns dict with lists indexed by layer 0..L-1.
    """
    L = 8
    # spectral norms ||A_ell|| = ||op(A_ell)|| (for conv, spectral norm of the operator)
    # After spectral normalization, these should all be ≤ 1
    spec_norms = [s['spectral_norm'] for s in stats]
    frob_norms = [s['frobenius'] for s in stats]
    shapes = [s['shape'] for s in stats]   # (out, in_flat)

    # Widths w_ell: output dimension of each layer
    # For conv: w_ell = U_ell * spatial_w_ell
    # For FC: w_ell = out_features
    # These are pre-activation widths
    conv_U = [64, 128, 256]
    conv_spatial_w = SPATIAL_W   # [256, 64, 16]
    fc_out = [fc_width] * 4 + [10]

    w_ell = []
    for i in range(3):
        w_ell.append(conv_U[i] * conv_spatial_w[i])
    for i in range(5):
        w_ell.append(fc_out[i])

    # U_ell and d_{ell-1} for CNN bounds (conv layers only)
    # U_ell = out_channels; d_{ell-1} = in_channels * kH * kW
    U_ell = conv_U + [None] * 5
    d_ell_minus1 = [3*3*3, 64*3*3, 128*3*3] + [None] * 5  # [27, 576, 1152]
    W_ell_spatial = SPATIAL_W + [1] * 5  # W_ell = U_ell * w_ell; WL = 1

    # For FC layers: w_ell and w_{ell-1} (width dimensions for bound terms)
    # w_ell_bar = w_ell + w_{ell-1}
    fc_widths_in = [256*4*4, fc_width, fc_width, fc_width, fc_width]  # input dims for each FC layer

    return {
        'L': L,
        'spec_norms': spec_norms,
        'frob_norms': frob_norms,
        'shapes': shapes,
        'w_ell': w_ell,         # total width at each layer
        'U_ell': U_ell,
        'd_ell_minus1': d_ell_minus1,
        'W_ell_spatial': W_ell_spatial,
        'fc_widths_in': fc_widths_in,
        'stats': stats,
    }


def compute_schatten_ratios(stats, p_ells):
    """
    Compute ||A_ell||_{sc,p_ell}^{p_ell} / ||A_ell||^{p_ell}
    i.e. the rank proxy r_ell = (||A-M||_{sc,p}/||A||)^p  with M=0.
    """
    ratios = []
    for i, s in enumerate(stats):
        p = p_ells[i]
        W = s['W']
        sv = torch.linalg.svdvals(W.detach().view(W.shape[0], -1).float()).cpu().numpy()
        spec = sv.max()
        if p == 0:
            rank = (sv > 1e-8 * spec).sum()
            ratios.append(float(rank))
        else:
            schatten_p_val = (sv ** p).sum()
            spec_p = spec ** p
            ratios.append(float(schatten_p_val / spec_p) if spec_p > 0 else 0.0)
    return ratios


def optimize_p_ells(stats, layer_params, B_max):
    """
    Optimize p_ell in [0,2] per layer to minimize the dominant term in Theorem 3.6.
    We grid-search over a discrete set of p values and pick the combination
    that minimizes the sum inside the square root of R^C_{FA}.

    For simplicity we optimize each layer independently (the cross terms factorize
    given the product of spectral norms is fixed).
    """
    L = layer_params['L']
    spec_norms = layer_params['spec_norms']
    # product of spectral norms (all ≤ 1 after projection)
    prod_spec = math.prod(max(s, 1e-15) for s in spec_norms)
    p_grid = np.linspace(0, 1, 41)

    best_p = []
    for ell in range(L):
        s = stats[ell]
        W = s['W']
        sv = torch.linalg.svdvals(W.detach().view(W.shape[0], -1).float()).cpu().numpy()
        spec = sv.max()
        U = layer_params['U_ell'][ell] if ell < 3 else layer_params['shapes'][ell][0]
        d = layer_params['d_ell_minus1'][ell] if ell < 3 else layer_params['shapes'][ell][1]
        W_sp = layer_params['W_ell_spatial'][ell]

        best_val = float('inf')
        best_p_ell = 0.0
        for p in p_grid:
            if p == 0:
                sch_ratio = float((sv > 1e-8 * spec).sum())
            else:
                sch = (sv ** p).sum()
                sch_ratio = float(sch / (spec ** p + 1e-30))

            # Term contribution to R^C_{FA} (ignoring the product of spec norms factor)
            # [B * prod_spec]^{2p/(p+2)} * r_ell^{2/(p+2)} * (U+d)^{2/(p+2)} * W_sp^{p/(p+2)}
            norm_factor = (B_max * prod_spec) ** (2 * p / (p + 2))
            rank_factor = sch_ratio ** (2 / (p + 2))
            dim_factor = (U + d) ** (2 / (p + 2)) * (W_sp ** (p / (p + 2)))
            val = norm_factor * rank_factor * dim_factor
            if val < best_val:
                best_val = val
                best_p_ell = p
        best_p.append(best_p_ell)
    return best_p


# ──────────────────────────────────────────────────────────────
# 10. Individual bound formulas (dominant term, no constants/logs)
# ──────────────────────────────────────────────────────────────
def bound_long_sedghi2020(stats, layer_params, N):
    """
    Eq. C15 / C14: Long & Sedghi 2020 parameter counting
    ~ sqrt(W * L / N)   where W = total params, L = depth
    """
    W_total = sum(s['shape'][0] * s['shape'][1] for s in stats)
    L = layer_params['L']
    return math.sqrt(W_total * L / N)


def bound_graf2022(stats, layer_params, N):
    """
    Eq. C16 / C15: Graf et al. 2022 parameter counting
    ~ sqrt(W * L / N)
    """
    return bound_long_sedghi2020(stats, layer_params, N)


def bound_pinto2024_C9(stats, layer_params, N, B_max, C1=1.0):
    """
    Eq. C9 / C10: Pinto et al. 2024
    ~ C1^L * prod(||A_ell||) * L * r * sqrt(W_max / N)
    where r = min rank of weight matrices.
    """
    spec = layer_params['spec_norms']
    L = layer_params['L']
    prod_spec = math.prod(max(s, 1e-15) for s in spec)
    # rank: use actual matrix rank
    ranks = []
    for s in stats:
        W = s['W'].detach().view(s['W'].shape[0], -1).float()
        sv = torch.linalg.svdvals(W).cpu().numpy()
        spec_val = sv.max()
        rank = int((sv > 1e-6 * spec_val).sum())
        ranks.append(rank)
    r = min(ranks)
    W_max = max(s['shape'][0] * s['shape'][1] for s in stats)
    return (C1 ** L) * prod_spec * L * r * math.sqrt(W_max / N)


def ledent_thm36(stats, layer_params, N, B_max, p_ells=None):
    """
    Theorem 3.6 (without loss augmentation):
    R^C_{FA} = [sum_ell { [B * prod_i rho_i ||A_i||]^{2p_ell/(p_ell+2)}
                          * [||A_ell - M_ell||_{sc,p_ell} / ||op(A_ell)||^{p_ell}]^{2/(p_ell+2)}
                          * (U_ell + d_{ell-1})^{2/(p_ell+2)} * W_ell^{p_ell/(p_ell+2)} }]^{1/2}

    With M_ell = 0 and rho_ell = 1 (ReLU), B = B_max (max input norm).
    Dominant term: sqrt(L/N) * R^C_{FA}
    """
    L = layer_params['L']
    spec_norms = layer_params['spec_norms']
    prod_spec = math.prod(max(s, 1e-15) for s in spec_norms)

    if p_ells is None:
        p_ells = optimize_p_ells(stats, layer_params, B_max)

    R_sum = 0.0
    for ell in range(L):
        p = p_ells[ell]
        s = stats[ell]
        W = s['W']
        sv = torch.linalg.svdvals(W.detach().view(W.shape[0], -1).float()).cpu().numpy()
        spec = sv.max()

        # ||A_ell||_{sc,p}^p / ||op(A)||^p  (M=0)
        if p == 0:
            rank = float((sv > 1e-8 * spec).sum())
            sch_ratio_p = rank
        else:
            sch_p = (sv ** p).sum()
            sch_ratio_p = float(sch_p / (spec ** p + 1e-30))

        U = layer_params['U_ell'][ell] if ell < 3 else layer_params['shapes'][ell][0]
        d = layer_params['d_ell_minus1'][ell] if ell < 3 else layer_params['shapes'][ell][1]
        W_sp = layer_params['W_ell_spatial'][ell]

        # [B * prod_spec]^{2p/(p+2)}
        norm_factor = (B_max * prod_spec) ** (2 * p / (p + 2))
        # [sch_ratio_p]^{2/(p+2)}
        rank_factor = sch_ratio_p ** (2 / (p + 2))
        # (U + d)^{2/(p+2)} * W_sp^{p/(p+2)}
        dim_factor = (U + d) ** (2 / (p + 2)) * (W_sp ** (p / (p + 2)))

        R_sum += norm_factor * rank_factor * dim_factor

    R = math.sqrt(R_sum)
    return math.sqrt(L / N) * R


def ledent_thm37(stats, layer_params, N, B_max, B_ells, p_ells=None):
    """
    Theorem 3.7 (with loss augmentation):
    Same as 3.6 but B * prod_{i>=ell} ||A_i|| replaced by
    B_{ell-1, A} * prod_{i>=ell} rho_i ||A_i||.
    """
    L = layer_params['L']
    spec_norms = layer_params['spec_norms']

    if p_ells is None:
        p_ells = optimize_p_ells(stats, layer_params, B_max)

    R_sum = 0.0
    for ell in range(L):
        p = p_ells[ell]
        s = stats[ell]
        W = s['W']
        sv = torch.linalg.svdvals(W.detach().view(W.shape[0], -1).float()).cpu().numpy()
        spec = sv.max()

        if p == 0:
            rank = float((sv > 1e-8 * spec).sum())
            sch_ratio_p = rank
        else:
            sch_p = (sv ** p).sum()
            sch_ratio_p = float(sch_p / (spec ** p + 1e-30))

        U = layer_params['U_ell'][ell] if ell < 3 else layer_params['shapes'][ell][0]
        d = layer_params['d_ell_minus1'][ell] if ell < 3 else layer_params['shapes'][ell][1]
        W_sp = layer_params['W_ell_spatial'][ell]

        # B_{ell-1,A} * prod_{i>=ell} ||A_i||
        B_ell_prev = B_ells[ell]   # B_{ell-1, A}
        prod_spec_from_ell = math.prod(max(spec_norms[i], 1e-15) for i in range(ell, L))
        aug_factor = B_ell_prev * prod_spec_from_ell

        norm_factor = aug_factor ** (2 * p / (p + 2))
        rank_factor = sch_ratio_p ** (2 / (p + 2))
        dim_factor = (U + d) ** (2 / (p + 2)) * (W_sp ** (p / (p + 2)))

        R_sum += norm_factor * rank_factor * dim_factor

    R = math.sqrt(R_sum)
    return math.sqrt(L / N) * R


def optimize_p_ells_entrywise(stats, layer_params, B_max):
    """
    Same as optimize_p_ells but uses the entry-wise L_p quasi-norm instead of
    the Schatten-p quasi-norm as the rank proxy.

    The bound term per layer is structurally identical to Theorem 3.6 / 3.7 of
    Ledent 2025, with the sole substitution:

        ||A_ell||_{sc,p}^p / ||op(A)||^p   →   ||A_ell||_{entry,p}^p / ||op(A)||^p

    where  ||A||_{entry,p}^p = sum_{i,j} |A_{ij}|^p.

    Per-layer independent optimisation over p in [0, 1].
    """
    L = layer_params['L']
    spec_norms = layer_params['spec_norms']
    prod_spec = math.prod(max(s, 1e-15) for s in spec_norms)
    p_grid = np.linspace(0, 1, 41)

    best_p = []
    for ell in range(L):
        s = stats[ell]
        W = s['W']
        W2 = W.detach().view(W.shape[0], -1).float()
        spec = W2.norm(dim=1).max().item()          # row-wise max as proxy; real spec from stats
        spec = max(stats[ell]['spectral_norm'], 1e-15)

        U = layer_params['U_ell'][ell] if ell < 3 else layer_params['shapes'][ell][0]
        d = layer_params['d_ell_minus1'][ell] if ell < 3 else layer_params['shapes'][ell][1]
        W_sp = layer_params['W_ell_spatial'][ell]

        best_val = float('inf')
        best_p_ell = 0.0
        for p in p_grid:
            if p == 0:
                entry_ratio = float((W2.abs() > 1e-10).sum().item())
            else:
                entry_p = float((W2.abs() ** p).sum().item())
                entry_ratio = entry_p / (spec ** p + 1e-30)

            norm_factor = (B_max * prod_spec) ** (2 * p / (p + 2))
            rank_factor = entry_ratio ** (2 / (p + 2))
            dim_factor  = (U + d) ** (2 / (p + 2)) * (W_sp ** (p / (p + 2)))
            val = norm_factor * rank_factor * dim_factor
            if val < best_val:
                best_val = val
                best_p_ell = p
        best_p.append(best_p_ell)
    return best_p


def ours_thm36_entrywise(stats, layer_params, N, B_max, p_ells=None):
    """
    Ours (without loss augmentation) – entry-wise L_p variant of Theorem 3.6.

    Identical to ledent_thm36 except the Schatten-p quasi-norm ratio

        ||A_ell||_{sc,p}^p / ||op(A)||^p

    is replaced by the entry-wise L_p quasi-norm ratio

        ||A_ell||_{entry,p}^p / ||op(A)||^p  =  (sum_{i,j} |A_{ij}|^p) / ||op(A)||^p

    with M_ell = 0, rho_ell = 1, B = B_max.
    """
    L = layer_params['L']
    spec_norms = layer_params['spec_norms']
    prod_spec = math.prod(max(s, 1e-15) for s in spec_norms)

    if p_ells is None:
        p_ells = optimize_p_ells_entrywise(stats, layer_params, B_max)

    R_sum = 0.0
    for ell in range(L):
        p = p_ells[ell]
        s = stats[ell]
        W = s['W']
        W2 = W.detach().view(W.shape[0], -1).float()
        spec = max(s['spectral_norm'], 1e-15)

        # entry-wise L_p ratio: ||A||_{entry,p}^p / ||op(A)||^p
        if p == 0:
            entry_ratio_p = float((W2.abs() > 1e-10).sum().item())
        else:
            entry_p = float((W2.abs() ** p).sum().item())
            entry_ratio_p = entry_p / (spec ** p + 1e-30)

        U = layer_params['U_ell'][ell] if ell < 3 else layer_params['shapes'][ell][0]
        d = layer_params['d_ell_minus1'][ell] if ell < 3 else layer_params['shapes'][ell][1]
        W_sp = layer_params['W_ell_spatial'][ell]

        norm_factor = (B_max * prod_spec) ** (2 * p / (p + 2))
        rank_factor = entry_ratio_p ** (2 / (p + 2))
        dim_factor  = (U + d) ** (2 / (p + 2)) * (W_sp ** (p / (p + 2)))

        R_sum += norm_factor * rank_factor * dim_factor

    R = math.sqrt(R_sum)
    return math.sqrt(L / N) * R


def ours_thm37_entrywise(stats, layer_params, N, B_max, B_ells, p_ells=None):
    """
    Ours (with loss augmentation) – entry-wise L_p variant of Theorem 3.7.

    Identical to ledent_thm37 except the Schatten-p quasi-norm ratio is replaced
    by the entry-wise L_p quasi-norm ratio (see ours_thm36_entrywise).
    The loss-augmentation substitution

        B * prod_{i} ||A_i||   →   B_{ell-1,A} * prod_{i>=ell} ||A_i||

    is applied in the same way as Theorem 3.7.
    """
    L = layer_params['L']
    spec_norms = layer_params['spec_norms']

    if p_ells is None:
        p_ells = optimize_p_ells_entrywise(stats, layer_params, B_max)

    R_sum = 0.0
    for ell in range(L):
        p = p_ells[ell]
        s = stats[ell]
        W = s['W']
        W2 = W.detach().view(W.shape[0], -1).float()
        spec = max(s['spectral_norm'], 1e-15)

        if p == 0:
            entry_ratio_p = float((W2.abs() > 1e-10).sum().item())
        else:
            entry_p = float((W2.abs() ** p).sum().item())
            entry_ratio_p = entry_p / (spec ** p + 1e-30)

        U = layer_params['U_ell'][ell] if ell < 3 else layer_params['shapes'][ell][0]
        d = layer_params['d_ell_minus1'][ell] if ell < 3 else layer_params['shapes'][ell][1]
        W_sp = layer_params['W_ell_spatial'][ell]

        # loss-augmented prefactor: B_{ell-1,A} * prod_{i>=ell} ||A_i||
        B_ell_prev = B_ells[ell]
        prod_spec_from_ell = math.prod(max(spec_norms[i], 1e-15) for i in range(ell, L))
        aug_factor = B_ell_prev * prod_spec_from_ell

        norm_factor = aug_factor ** (2 * p / (p + 2))
        rank_factor = entry_ratio_p ** (2 / (p + 2))
        dim_factor  = (U + d) ** (2 / (p + 2)) * (W_sp ** (p / (p + 2)))

        R_sum += norm_factor * rank_factor * dim_factor

    R = math.sqrt(R_sum)
    return math.sqrt(L / N) * R


# ──────────────────────────────────────────────────────────────
# 11. Main experiment loop
# ──────────────────────────────────────────────────────────────

def run_experiment(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # --- data ---
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    train_data = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=True, transform=transform_train)
    test_data = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_data,  batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)
    N = len(train_data)

    # For bound computation: a separate loader with no augmentation
    train_data_noaug = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=False, transform=transform_test)
    bound_loader = DataLoader(train_data_noaug, batch_size=256,
                              shuffle=False, num_workers=2, pin_memory=True)

    results = {}

    for fc_width in args.widths:
        print(f"\n{'='*60}")
        print(f"  FC width = {fc_width}")
        print(f"{'='*60}")

        ckpt_path = os.path.join(args.save_dir, f'model_w{fc_width}_withl1reg.pt')
        model = CNN(fc_width=fc_width, num_classes=10).to(device)

        if os.path.exists(ckpt_path) and not args.retrain:
            print(f"  Loading saved weights from {ckpt_path}")
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state['model_state'])
            best_acc = state.get('train_acc', None)
            print(f"  Loaded. Train acc (at save time): {best_acc}")
        else:
            # --- training ---
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

            for epoch in range(1, args.epochs + 1):
                tr_loss, tr_acc = train_one_epoch(
                    model, train_loader, optimizer, device, max_sigma=1.0)
                scheduler.step()
                if epoch % 10 == 0 or epoch == args.epochs:
                    te_acc = evaluate(model, test_loader, device)
                    print(f"  Epoch {epoch:3d}: loss={tr_loss:.4f}  "
                          f"train_acc={tr_acc:.4f}  test_acc={te_acc:.4f}")

            torch.save({'model_state': model.state_dict(),
                        'train_acc': tr_acc,
                        'fc_width': fc_width},
                       ckpt_path)
            print(f"  Saved to {ckpt_path}")

        # --- evaluation ---
        model.eval()
        train_acc = evaluate(model, bound_loader, device)
        test_acc  = evaluate(model, test_loader,  device)
        print(f"  Final train acc: {train_acc:.4f},  test acc: {test_acc:.4f}")

        gamma, I_gamma = find_margin(model, bound_loader, device, acc_drop_tol=0.01)
        print(f"  Margin gamma={gamma:.5f},  I_gamma={I_gamma:.4f}")

        # --- weight statistics ---
        stats = get_weight_stats(model)
        layer_params = get_layer_params(model, stats, fc_width)

        print("  Spectral norms: " +
              ", ".join(f"{s['spectral_norm']:.4f}" for s in stats))
        print("  Frobenius norms: " +
              ", ".join(f"{s['frobenius']:.4f}" for s in stats))

        # --- max input norm on training set ---
        B_max = 0.0
        for imgs, _ in bound_loader:
            B_max = max(
                B_max,
                imgs.view(imgs.size(0), -1).norm(dim=1).max().item()
            )
            # B_max = max(B_max, imgs.norm(dim=(1, 2, 3)).max().item())

        print(f"  B_max (max input L2 norm): {B_max:.4f}")

        # --- B_{ell,A} for loss augmentation ---
        B_ells = compute_B_ells(model, bound_loader, device)
        print(f"  B_ells: {[f'{b:.4f}' for b in B_ells]}")

        # --- total parameters ---
        W_total = sum(p.numel() for p in model.parameters())
        W_max_layer = max(s['shape'][0] * s['shape'][1] for s in stats)
        L = layer_params['L']
        print(f"  Total params W={W_total},  W_max_layer={W_max_layer},  L={L}")

        # --- optimized p_ells (Schatten-p, for Ledent 2025) ---
        p_ells_opt = optimize_p_ells(stats, layer_params, B_max)
        print(f"  Optimal p_ells (Schatten): {[f'{p:.3f}' for p in p_ells_opt]}")

        # --- optimized p_ells (entry-wise Lp, for Ours) ---
        p_ells_opt_ew = optimize_p_ells_entrywise(stats, layer_params, B_max)
        print(f"  Optimal p_ells (entry-wise): {[f'{p:.3f}' for p in p_ells_opt_ew]}")

        # ---- compute all bounds (dominant term only) ----
        bounds = {}
        b_val = bound_long_sedghi2020(stats, layer_params, N)
        bounds['LongSedghi2020_C15'] = b_val
        print(f"  Long & Sedghi 2020 (C15):        {b_val:.4e}")

        b_val = bound_graf2022(stats, layer_params, N)
        bounds['Graf2022_C16'] = b_val
        print(f"  Graf et al. 2022 (C16):          {b_val:.4e}")

        b_val = bound_pinto2024_C9(stats, layer_params, N, B_max, C1=1.0)
        bounds['Pinto2024_C9_C1eq1'] = b_val
        print(f"  Pinto et al. 2024 (C9, C1=1):    {b_val:.4e}")

        b_val = bound_pinto2024_C9(stats, layer_params, N, B_max, C1=2.0)
        bounds['Pinto2024_C9_C1eq2'] = b_val
        print(f"  Pinto et al. 2024 (C9, C1=2):    {b_val:.4e}")

        b_val = ledent_thm36(stats, layer_params, N, B_max, p_ells_opt)
        bounds['Ledent2025_Thm36'] = b_val
        print(f"  Ledent et al. 2025 (no loss aug):           {b_val:.4e}")

        b_val = ledent_thm37(stats, layer_params, N, B_max, B_ells, p_ells_opt)
        bounds['Ledent2025_Thm37'] = b_val
        print(f"  Ledent et al. 2025 (with loss aug):         {b_val:.4e}")

        b_val = ours_thm36_entrywise(stats, layer_params, N, B_max, p_ells_opt_ew)
        bounds['Ours_Thm36_EntryLp'] = b_val
        print(f"  Ours (no loss aug,  entry-wise Lp):         {b_val:.4e}")

        b_val = ours_thm37_entrywise(stats, layer_params, N, B_max, B_ells, p_ells_opt_ew)
        bounds['Ours_Thm37_EntryLp'] = b_val
        print(f"  Ours (with loss aug, entry-wise Lp):        {b_val:.4e}")

        results[fc_width] = {
            'train_acc': train_acc,
            'test_acc':  test_acc,
            'gamma':     gamma,
            'I_gamma':   I_gamma,
            'B_max':     B_max,
            'B_ells':    B_ells,
            'W_total':   W_total,
            'bounds':    bounds,
            'p_ells_opt': p_ells_opt,
            'p_ells_opt_entrywise': p_ells_opt_ew,
            'spec_norms': layer_params['spec_norms'],
            'frob_norms': layer_params['frob_norms'],
        }

    # Save results
    results_path = os.path.join(args.save_dir, 'bound_results.json')
    # Convert to JSON-serializable
    results_json = {}
    for w, r in results.items():
        results_json[str(w)] = {k: (v if not isinstance(v, (np.floating, np.integer))
                                    else float(v))
                                for k, v in r.items()}
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2, default=float)
    print(f"\nResults saved to {results_path}")

    return results


# ──────────────────────────────────────────────────────────────
# 12. Plotting (Figure D.3 style)
# ──────────────────────────────────────────────────────────────

def plot_results(results, save_path='figure_D3.png'):
    widths = sorted(results.keys())
    bound_keys = [
        ('LongSedghi2020_C15',    'Long & Sedghi, 2020\nEq. C15'),
        ('Graf2022_C16',          'Graf et al., 2022\nEq. C16'),
        ('Pinto2024_C9_C1eq1',    'Pinto et al., 2024\nEq. C9 · C1=1'),
        ('Pinto2024_C9_C1eq2',    'Pinto et al., 2024\nEq. C9 · C1=2'),
        ('Ledent2025_Thm36',      'Ledent et al., 2025\n(without loss-aug)'),
        ('Ledent2025_Thm37',      'Ledent et al., 2025\n(with loss-aug)'),
        ('Ours_Thm36_EntryLp',    'Ours\n(without loss-aug)'),
        ('Ours_Thm37_EntryLp',    'Ours\n(with loss-aug)'),
    ]

    n_methods = len(bound_keys)
    n_widths   = len(widths)
    width_colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F'][:n_widths]

    x = np.arange(n_methods)
    bar_w = 0.18
    offsets = np.linspace(-(n_widths - 1) / 2, (n_widths - 1) / 2, n_widths) * bar_w

    fig, ax = plt.subplots(figsize=(18, 5))

    for wi, (w, color) in enumerate(zip(widths, width_colors)):
        vals = [results[w]['bounds'].get(k, np.nan) for k, _ in bound_keys]
        # log scale: take log10
        log_vals = [math.log10(v) if v > 0 else float('nan') for v in vals]
        ax.bar(x + offsets[wi], log_vals, bar_w,
               label=f'Width (w)={w}', color=color, edgecolor='k', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in bound_keys], fontsize=7.5)
    ax.set_ylabel('Bound (Log₁₀ Scale)', fontsize=10)
    ax.set_title('Figure D.3: Numerical Comparison of Bounds with Literature (CNNs, CIFAR-10)',
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {save_path}")


# ──────────────────────────────────────────────────────────────
# 13. Entry point
# ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Reproduce Figure D.3: CNN Generalization Bounds on CIFAR-10')
    parser.add_argument('--widths', type=int, nargs='+', default=[200, 600, 1000],
                        help='FC layer widths to evaluate (default: 200 600 1000)')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='./saved_models',
                        help='Directory to save/load model checkpoints')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--retrain', action='store_true',
                        help='Force retraining even if checkpoint exists')
    parser.add_argument('--plot_path', type=str, default='./figure_D3.png')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print("Configuration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    results = run_experiment(args)
    plot_results(results, save_path=args.plot_path)
    print('ALL DONE!')
