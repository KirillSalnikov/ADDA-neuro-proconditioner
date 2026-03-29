"""Training script v7 — Universal ConvSAI with 3D geometry encoder.

Key difference from v6: instead of discrete shape_id, the model takes a 3D
binary occupancy grid of the particle shape. This enables ONE model for ALL
shapes, including shapes never seen during training.

Training generates diverse random shapes:
  - Spheres
  - Ellipsoids with random aspect ratios (continuous, not discrete)
  - Cubes
  - Cylinders
  - Capsules

Usage:
    python train_v7/train.py --name universal_r7 --device 0 --save \
        --loss adversarial --r_cut 7 --hidden_size 512 --num_layers 4 \
        --num_steps 40000 --lr 5e-4 --grid_min 8 --grid_max 32 \
        --curriculum_frac 0.3 --m_re_min 1.5 --m_re_max 4.0
"""
import os
import sys
import copy
import datetime
import argparse
import pprint
import time
import math

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neuralif.fft_matvec import FFTMatVec
from neuralif.utils import count_parameters, save_dict_to_file
from neuralif.logger import TrainResults

from apps.adda_matrix import (
    make_sphere_dipoles, make_cube_dipoles, make_ellipsoid_dipoles,
    make_cylinder_dipoles, make_capsule_dipoles,
)
from neural_precond.model import (ConvSAI_Universal, ConvSAI_Multigrid,
                                  ConvSAI_Separable, ConvSAI_Hybrid,
                                  ConvSAI_Spectral, positions_to_occupancy)
from neural_precond.loss import (
    conv_sai_probe_loss,
    conv_sai_adversarial_probe_loss,
    conv_sai_right_probe_loss,
    conv_sai_gmres_loss,
)


# ---------------------------------------------------------------------------
# Squared kernel wrapper: M = K² in frequency domain
# ---------------------------------------------------------------------------

class SquaredConvSAI(nn.Module):
    """Wrapper that composes the kernel with itself: M = K·K.

    In frequency domain: M_hat = K_hat @ K_hat (3x3 matrix multiply per freq).
    Effective r_cut doubles (e.g. 7→14) with 0 extra params and 0 extra eval cost.
    """

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.r_cut = base.r_cut
        self.n_stencil = base.n_stencil
        self.stencil = base.stencil
        # Forward attributes needed by training code
        self.shape_encoder = base.shape_encoder
        self.mlp = base.mlp

    def forward(self, *args, **kwargs):
        return self.base(*args, **kwargs)

    def build_M_hat(self, kernel, fft_matvec):
        K_hat = self.base.build_M_hat(kernel, fft_matvec)
        return torch.einsum('ijxyz,jkxyz->ikxyz', K_hat, K_hat)

    def make_precond_fn(self, kernel, fft_matvec):
        kernel_c128 = kernel.detach().to(dtype=torch.complex128, device='cpu')
        box = fft_matvec.box
        gx = 2 * box[0].item()
        gy = 2 * box[1].item()
        gz = 2 * box[2].item()

        stencil = self.stencil.cpu()
        wi = stencil[:, 0] % gx
        wj = stencil[:, 1] % gy
        wk = stencil[:, 2] % gz

        M_grid = torch.zeros(3, 3, gx, gy, gz, dtype=torch.complex128)
        M_grid[:, :, wi, wj, wk] = kernel_c128.permute(1, 2, 0)
        K_hat = torch.fft.fftn(M_grid, dim=(2, 3, 4))
        M_hat = torch.einsum('ijxyz,jkxyz->ikxyz', K_hat, K_hat)

        pos = fft_matvec.pos_shifted.cpu()
        pi, pj, pk = pos[:, 0], pos[:, 1], pos[:, 2]
        N = fft_matvec.N
        n = fft_matvec.n

        def precond_fn(v):
            with torch.no_grad():
                v_grid = torch.zeros(3, gx, gy, gz, dtype=torch.complex128)
                v_grid[:, pi, pj, pk] = v.reshape(N, 3).T
                v_hat = torch.fft.fftn(v_grid, dim=(1, 2, 3))
                result_hat = torch.einsum('ijxyz,jxyz->ixyz', M_hat, v_hat)
                result_grid = torch.fft.ifftn(result_hat, dim=(1, 2, 3))
                return result_grid[:, pi, pj, pk].T.reshape(n)

        return precond_fn


# ---------------------------------------------------------------------------
# Random shape generation
# ---------------------------------------------------------------------------
SHAPE_TYPES = ['sphere', 'ellipsoid', 'cube', 'cylinder', 'capsule']

# Weights: ellipsoids get more weight since they're the most diverse
SHAPE_WEIGHTS = [0.15, 0.40, 0.15, 0.15, 0.15]


def generate_random_shape(rng, grid, only_shape=None):
    """Generate random shape and return (positions, shape_name).

    Ellipsoids get random continuous aspect ratios — this is the key
    advantage of the universal model over discrete shape_id.

    If only_shape is set, generate only that shape type.
    """
    if only_shape is not None:
        shape_type = only_shape
    else:
        shape_type = rng.choice(SHAPE_TYPES, p=SHAPE_WEIGHTS)

    if shape_type == 'sphere':
        positions = make_sphere_dipoles(grid)
        name = 'sphere'

    elif shape_type == 'ellipsoid':
        # Random aspect ratios: ay in [0.3, 1.0], az in [0.3, 3.0]
        ay = rng.uniform(0.3, 1.0)
        az = rng.uniform(0.3, 3.0)
        positions = make_ellipsoid_dipoles(grid, (1.0, ay, az))
        name = f'ell_{ay:.2f}_{az:.2f}'

    elif shape_type == 'cube':
        positions = make_cube_dipoles(grid)
        name = 'cube'

    elif shape_type == 'cylinder':
        positions = make_cylinder_dipoles(grid)
        name = 'cylinder'

    elif shape_type == 'capsule':
        positions = make_capsule_dipoles(grid)
        name = 'capsule'

    else:
        raise ValueError(f"Unknown shape: {shape_type}")

    return positions, name


# ---------------------------------------------------------------------------
# Fixed validation configs
# ---------------------------------------------------------------------------
FIXED_VAL_CONFIGS_STANDARD = [
    # (m_re, m_im, kd, shape_generator, grid, name)
    (2.0, 0.0, 0.42, lambda g: make_sphere_dipoles(g), 10, 'sphere'),
    (2.5, 0.0, 0.42, lambda g: make_sphere_dipoles(g), 10, 'sphere'),
    (3.0, 0.0, 0.42, lambda g: make_sphere_dipoles(g), 12, 'sphere'),
    (2.0, 0.0, 0.42, lambda g: make_cube_dipoles(g), 10, 'cube'),
    (2.5, 0.0, 0.42, lambda g: make_cube_dipoles(g), 10, 'cube'),
    (2.0, 0.0, 0.42, lambda g: make_ellipsoid_dipoles(g, (1.0, 1.0, 2.0)), 12, 'ell_1:1:2'),
    (2.5, 0.0, 0.42, lambda g: make_ellipsoid_dipoles(g, (1.0, 1.0, 2.0)), 12, 'ell_1:1:2'),
    (3.0, 0.0, 0.42, lambda g: make_ellipsoid_dipoles(g, (1.0, 1.0, 2.0)), 12, 'ell_1:1:2'),
    (2.0, 0.0, 0.42, lambda g: make_ellipsoid_dipoles(g, (1.0, 1.0, 0.5)), 12, 'ell_1:1:0.5'),
    (2.5, 0.0, 0.42, lambda g: make_ellipsoid_dipoles(g, (1.0, 1.0, 0.5)), 12, 'ell_1:1:0.5'),
    (3.0, 0.0, 0.42, lambda g: make_ellipsoid_dipoles(g, (1.0, 1.0, 0.5)), 12, 'ell_1:1:0.5'),
    (2.5, 0.0, 0.42, lambda g: make_ellipsoid_dipoles(g, (1.0, 0.8, 0.6)), 12, 'ell_1:0.8:0.6'),
    (2.5, 0.0, 0.42, lambda g: make_ellipsoid_dipoles(g, (1.0, 1.0, 3.0)), 12, 'ell_1:1:3'),
    (3.0, 0.0, 0.42, lambda g: make_ellipsoid_dipoles(g, (1.0, 1.0, 3.0)), 16, 'ell_1:1:3'),
    (2.5, 0.0, 0.42, lambda g: make_ellipsoid_dipoles(g, (1.0, 0.7, 1.5)), 12, 'ell_1:0.7:1.5'),
    (2.5, 0.0, 0.42, lambda g: make_cylinder_dipoles(g), 10, 'cylinder'),
    (3.0, 0.0, 0.42, lambda g: make_cylinder_dipoles(g), 10, 'cylinder'),
    (2.5, 0.0, 0.42, lambda g: make_capsule_dipoles(g), 10, 'capsule'),
    # Unseen ellipsoid aspect ratios (test generalization)
    (2.5, 0.0, 0.42, lambda g: make_ellipsoid_dipoles(g, (1.0, 0.5, 0.5)), 12, 'ell_unseen_0.5_0.5'),
    (2.5, 0.0, 0.42, lambda g: make_ellipsoid_dipoles(g, (1.0, 0.9, 2.5)), 12, 'ell_unseen_0.9_2.5'),
]

# Validation configs for high-m regime (m=4-6), smaller kd for accuracy
FIXED_VAL_CONFIGS_HIGH_M = [
    # kd=0.20 (dpl≈31) — safe for m up to 5: |m|*kd = 1.0
    (4.0, 0.0, 0.20, lambda g: make_sphere_dipoles(g), 10, 'sphere'),
    (5.0, 0.0, 0.20, lambda g: make_sphere_dipoles(g), 10, 'sphere'),
    (6.0, 0.0, 0.15, lambda g: make_sphere_dipoles(g), 10, 'sphere'),
    (4.0, 0.0, 0.20, lambda g: make_cube_dipoles(g), 10, 'cube'),
    (5.0, 0.0, 0.20, lambda g: make_cube_dipoles(g), 10, 'cube'),
    (4.0, 0.0, 0.20, lambda g: make_ellipsoid_dipoles(g, (1.0, 1.0, 2.0)), 12, 'ell_1:1:2'),
    (5.0, 0.0, 0.20, lambda g: make_ellipsoid_dipoles(g, (1.0, 1.0, 2.0)), 12, 'ell_1:1:2'),
    (4.0, 0.0, 0.20, lambda g: make_ellipsoid_dipoles(g, (1.0, 1.0, 0.5)), 12, 'ell_1:1:0.5'),
    (5.0, 0.0, 0.20, lambda g: make_ellipsoid_dipoles(g, (1.0, 1.0, 0.5)), 12, 'ell_1:1:0.5'),
    (4.0, 0.0, 0.20, lambda g: make_ellipsoid_dipoles(g, (1.0, 1.0, 3.0)), 12, 'ell_1:1:3'),
    (5.0, 0.0, 0.15, lambda g: make_ellipsoid_dipoles(g, (1.0, 1.0, 3.0)), 12, 'ell_1:1:3'),
    (4.0, 0.0, 0.20, lambda g: make_cylinder_dipoles(g), 10, 'cylinder'),
    (5.0, 0.0, 0.20, lambda g: make_capsule_dipoles(g), 10, 'capsule'),
    # Unseen
    (4.5, 0.0, 0.20, lambda g: make_ellipsoid_dipoles(g, (1.0, 0.5, 0.5)), 12, 'ell_unseen_0.5_0.5'),
    (4.5, 0.0, 0.20, lambda g: make_ellipsoid_dipoles(g, (1.0, 0.9, 2.5)), 12, 'ell_unseen_0.9_2.5'),
]

FIXED_VAL_CONFIGS_BOX = [
    # Cube at m=3.0 across various grids and kd
    (3.0, 0.0, 0.42, lambda g: make_cube_dipoles(g), 8, 'cube'),
    (3.0, 0.0, 0.42, lambda g: make_cube_dipoles(g), 10, 'cube'),
    (3.0, 0.0, 0.42, lambda g: make_cube_dipoles(g), 12, 'cube'),
    (3.0, 0.0, 0.42, lambda g: make_cube_dipoles(g), 14, 'cube'),
    (3.0, 0.0, 0.42, lambda g: make_cube_dipoles(g), 16, 'cube'),
    (3.0, 0.0, 0.42, lambda g: make_cube_dipoles(g), 20, 'cube'),
    (3.0, 0.0, 0.30, lambda g: make_cube_dipoles(g), 10, 'cube'),
    (3.0, 0.0, 0.30, lambda g: make_cube_dipoles(g), 14, 'cube'),
    (3.0, 0.0, 0.60, lambda g: make_cube_dipoles(g), 10, 'cube'),
    (3.0, 0.0, 0.60, lambda g: make_cube_dipoles(g), 14, 'cube'),
]

# Large-grid configs for multigrid (grid >= 30, all coarse levels active)
# NOTE: g48 on CPU is too slow for inline validation (~minutes per BiCGStab call)
FIXED_VAL_CONFIGS_MULTIGRID = [
    # Small grid — tests fine level only (fast, baseline)
    (3.0, 0.0, 0.42, lambda g: make_sphere_dipoles(g), 12, 'sphere'),
    (3.0, 0.0, 0.42, lambda g: make_ellipsoid_dipoles(g, (1.0, 1.0, 0.5)), 12, 'ell_1:1:0.5'),
    # Medium grid — tests 2 levels (grid >= 15 for stride-2)
    (2.0, 0.0, 0.42, lambda g: make_sphere_dipoles(g), 16, 'sphere_g16'),
    (2.0, 0.0, 0.42, lambda g: make_ellipsoid_dipoles(g, (1.0, 1.0, 2.0)), 16, 'ell_1:1:2_g16'),
    # Large grid — tests ALL 3 levels, use m=2.0 to keep BiCGStab fast
    (2.0, 0.0, 0.42, lambda g: make_sphere_dipoles(g), 30, 'sphere_g30'),
    (2.5, 0.0, 0.42, lambda g: make_sphere_dipoles(g), 30, 'sphere_g30'),
    (2.0, 0.0, 0.42, lambda g: make_cube_dipoles(g), 30, 'cube_g30'),
    (2.0, 0.0, 0.42, lambda g: make_ellipsoid_dipoles(g, (1.0, 1.0, 2.0)), 30, 'ell_1:1:2_g30'),
    (2.5, 0.0, 0.42, lambda g: make_ellipsoid_dipoles(g, (1.0, 1.0, 0.5)), 30, 'ell_1:1:0.5_g30'),
]

FIXED_VAL_CONFIGS = FIXED_VAL_CONFIGS_STANDARD  # overridden by --high_m or --only_shape or --multigrid


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def sample_parameters(rng, config, step=None, num_steps=None):
    """Sample random physical parameters with weighted m_re distribution."""
    hard_frac = config.get('hard_sample_frac', 0.5)
    m_re_min = config['m_re_min']
    m_re_max = config['m_re_max']

    if m_re_min == m_re_max:
        m_re = m_re_min
    else:
        m_re_mid = (m_re_min + m_re_max) / 2.0
        if rng.random() < hard_frac:
            m_re = rng.uniform(m_re_mid, m_re_max)
        else:
            m_re = rng.uniform(m_re_min, m_re_max)

    m_im = rng.uniform(config['m_im_min'], config['m_im_max'])

    log_kd_min = np.log(config['kd_min'])
    log_kd_max = np.log(config['kd_max'])
    kd = float(np.exp(rng.uniform(log_kd_min, log_kd_max)))

    grid_min = config['grid_min']
    grid_max = config['grid_max']
    curriculum_frac = config.get('curriculum_frac', 0.5)
    if step is not None and num_steps is not None and curriculum_frac > 0:
        progress = min(step / (num_steps * curriculum_frac), 1.0)
        current_grid_max = int(grid_min + (grid_max - grid_min) * progress)
        current_grid_max = max(current_grid_max, grid_min)
    else:
        current_grid_max = grid_max

    grid = rng.randint(grid_min, current_grid_max + 1)
    return m_re, m_im, kd, grid


def build_fft_matvec_from_positions(positions, m_re, m_im, kd, device):
    """Build FFTMatVec from dipole positions.

    Uses ADDA convention: k=1, d=kd (so k*d=kd).
    This makes Python's system matrix match ADDA's exactly.
    """
    d = kd
    k = 1.0
    m = complex(m_re, m_im)
    return FFTMatVec(positions, k, m, d=d, device=device)


# ---------------------------------------------------------------------------
# Loss dispatch
# ---------------------------------------------------------------------------
def compute_loss(model, kernel, fft_mv, config):
    """Compute training loss based on config['loss'] setting."""
    loss_type = config['loss']
    num_probes = config.get('num_probes', 5)

    if loss_type == 'probe':
        return conv_sai_probe_loss(model, kernel, fft_mv, num_probes=num_probes)
    elif loss_type == 'adversarial':
        return conv_sai_adversarial_probe_loss(
            model, kernel, fft_mv,
            num_probes=num_probes,
            adversarial_iters=config.get('adversarial_iters', 10),
        )
    elif loss_type == 'right_probe':
        return conv_sai_right_probe_loss(
            model, kernel, fft_mv, num_probes=num_probes,
        )
    elif loss_type == 'gmres':
        return conv_sai_gmres_loss(
            model, kernel, fft_mv,
            gmres_iters=config.get('gmres_iters', 10),
            num_rhs=config.get('gmres_rhs', 2),
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@torch.no_grad()
def validate_fixed(model, device, encoder_resolution, rtol=1e-5, max_iter=1000):
    """Validate on FIXED configs with BiCGStab."""
    from krylov.bicgstab import bicgstab

    model.eval()

    total_precond = 0
    total_unprecond = 0
    counted = 0
    details = []

    for m_re, m_im, kd, gen_fn, grid, name in FIXED_VAL_CONFIGS:
        try:
            positions = gen_fn(grid)
            fft_mv = build_fft_matvec_from_positions(positions, m_re, m_im, kd, 'cpu')
        except Exception:
            continue

        n = fft_mv.n

        def A_op(v):
            v2d = v.unsqueeze(1) if v.dim() == 1 else v
            return fft_mv(v2d).squeeze(1)

        torch.manual_seed(hash((m_re, m_im, kd, name, grid)) % 2**31)
        b = torch.randn(n, dtype=torch.complex128) + \
            1j * torch.randn(n, dtype=torch.complex128)
        b = b / torch.linalg.vector_norm(b)

        # Build occupancy grid and predict kernel
        occ_grid = positions_to_occupancy(positions, grid_size=grid, device='cpu')
        kernel = model(m_re, m_im, kd, occ_grid, grid)
        precond_fn = model.make_precond_fn(kernel, fft_mv)

        res_p, _ = bicgstab(A_op, b, M=precond_fn, rtol=rtol, max_iter=max_iter)
        res_u, _ = bicgstab(A_op, b, rtol=rtol, max_iter=max_iter)

        ip = len(res_p) - 1
        iu = len(res_u) - 1
        total_precond += ip
        total_unprecond += iu
        counted += 1
        spd = iu / max(ip, 1)
        details.append(f"{name} g{grid} m={m_re}+{m_im}i kd={kd:.2f}: "
                       f"{iu}->{ip} ({spd:.1f}x)")

    if counted == 0:
        print("Validation: no valid configs")
        return 1.0

    avg_p = total_precond / counted
    avg_u = total_unprecond / counted
    speedup = avg_u / max(avg_p, 1)
    inv_speedup = avg_p / max(avg_u, 1)

    print(f"Validation fixed ({counted} configs)\t"
          f"precond: {avg_p:.1f}\t unprecond: {avg_u:.1f}\t speedup: {speedup:.2f}x")

    worst = sorted(details, key=lambda s: float(s.split('(')[1].split('x')[0]))
    for w in worst[:3]:
        print(f"  worst: {w}")

    return inv_speedup


@torch.no_grad()
def validate_probe(model, rng, config, device, num_val_steps=50):
    """Validate by measuring probe loss (fast, for LR scheduling)."""
    model.eval()
    total_loss = 0.0

    val_config = dict(config)
    # For multigrid, keep large grids so coarse levels are tested
    probe_grid_cap = 32 if config.get('multigrid_levels', 0) > 1 else 20
    val_config['grid_max'] = min(config.get('grid_max', 24), probe_grid_cap)
    val_config['grid_min'] = min(config.get('grid_min', 8), val_config['grid_max'])

    counted = 0
    for _ in range(num_val_steps):
        m_re, m_im, kd, grid = sample_parameters(rng, val_config)
        try:
            positions, _ = generate_random_shape(rng, grid)
            fft_mv = build_fft_matvec_from_positions(positions, m_re, m_im, kd, device)
            occ_grid = positions_to_occupancy(positions, grid_size=grid, device=device)
            kernel = model(m_re, m_im, kd, occ_grid, grid)
            loss = conv_sai_probe_loss(model, kernel, fft_mv, num_probes=10)
            total_loss += loss.item()
            counted += 1
        except (torch.OutOfMemoryError, Exception):
            torch.cuda.empty_cache()
            continue

    avg = total_loss / max(counted, 1)
    print(f"Validation probe:\t{avg:.6f}")
    return avg


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def main(config):
    device = config['device']
    encoder_resolution = config.get('encoder_resolution', 32)

    if config["save"]:
        os.makedirs(config['folder'], exist_ok=True)
        config_save = {k: (str(v) if isinstance(v, torch.device) else v)
                       for k, v in config.items()}
        save_dict_to_file(config_save, os.path.join(config['folder'], "config.json"))

    torch.manual_seed(config["seed"])
    rng = np.random.RandomState(config["seed"])

    if config.get("spectral", False):
        base_model = ConvSAI_Spectral(
            freq_hidden=config.get("freq_hidden", 64),
            freq_layers=config.get("freq_layers", 3),
            global_hidden=config.get("global_hidden", 256),
            global_layers=config.get("global_layers", 3),
            shape_embed_dim=config.get("shape_embed_dim", 16),
            activation=config.get("activation", "relu"),
            squared=config.get("squared_kernel", False),
            freq_coords=not config.get("no_freq_coords", False),
            encoder_resolution=encoder_resolution,
            encoder_channels=tuple(config.get("encoder_channels", [16, 32, 64])),
        )
        print(f"Spectral: freq_hidden={config.get('freq_hidden', 64)}, "
              f"freq_layers={config.get('freq_layers', 3)}")
    elif config.get("hybrid", False):
        # Hybrid: pre-trained near (ConvSAI_Universal) + trainable far (Separable)
        near_model = ConvSAI_Universal(
            r_cut=config.get("r_cut", 7),
            hidden_size=config.get("hidden_size", 512),
            num_layers=config.get("num_layers", 4),
            shape_embed_dim=config.get("shape_embed_dim", 16),
            activation=config.get("activation", "relu"),
            scale_by_stencil=config.get("scale_by_stencil", True),
            encoder_resolution=encoder_resolution,
            encoder_channels=tuple(config.get("encoder_channels", [16, 32, 64])),
        )
        far_model = ConvSAI_Separable(
            axis_range=config.get("axis_range", 30),
            hidden_size=config.get("hidden_size", 512),
            num_layers=config.get("num_layers", 4),
            shape_embed_dim=config.get("shape_embed_dim", 16),
            activation=config.get("activation", "relu"),
            scale_by_stencil=config.get("scale_by_stencil", True),
            encoder_resolution=encoder_resolution,
            encoder_channels=tuple(config.get("encoder_channels", [16, 32, 64])),
        )
        base_model = ConvSAI_Hybrid(near_model, far_model,
                                     squared=config.get("squared_kernel", False))
        sq = config.get("squared_kernel", False)
        print(f"Hybrid: near r_cut={config.get('r_cut', 7)} + "
              f"far separable range={config.get('axis_range', 30)}, K²={sq}")
    elif config.get("separable", False):
        base_model = ConvSAI_Separable(
            axis_range=config.get("axis_range", 30),
            hidden_size=config.get("hidden_size", 512),
            num_layers=config.get("num_layers", 4),
            shape_embed_dim=config.get("shape_embed_dim", 16),
            activation=config.get("activation", "relu"),
            scale_by_stencil=config.get("scale_by_stencil", True),
            squared=config.get("squared_kernel", False),
            encoder_resolution=encoder_resolution,
            encoder_channels=tuple(config.get("encoder_channels", [16, 32, 64])),
        )
        eff = config.get("axis_range", 30)
        if config.get("squared_kernel", False):
            eff *= 2
        print(f"Separable 1D kernels: range={config.get('axis_range', 30)}, "
              f"eff_range={eff}, K²={config.get('squared_kernel', False)}")
    else:
        base_model = ConvSAI_Universal(
            r_cut=config.get("r_cut", 7),
            hidden_size=config.get("hidden_size", 512),
            num_layers=config.get("num_layers", 4),
            shape_embed_dim=config.get("shape_embed_dim", 16),
            activation=config.get("activation", "relu"),
            scale_by_stencil=config.get("scale_by_stencil", True),
            encoder_resolution=encoder_resolution,
            encoder_channels=tuple(config.get("encoder_channels", [16, 32, 64])),
        )

    # Wrap with K² or multigrid if requested (BEFORE loading weights)
    if config.get("multigrid_levels", 0) > 1:
        num_levels = config["multigrid_levels"]
        bottleneck = config.get("mg_bottleneck", 0)
        mg_squared = config.get("mg_squared", False)
        model = ConvSAI_Multigrid(base_model, num_levels=num_levels,
                                   bottleneck=bottleneck, squared=mg_squared)
        eff_r = model.r_cut * (2 ** (num_levels - 1))
        if mg_squared:
            eff_r *= 2
        print(f"Multigrid: {num_levels} levels (stride 1..{2**(num_levels-1)}), "
              f"effective r_cut={eff_r}, bottleneck={bottleneck}, K²={mg_squared}")
    elif config.get("squared_kernel", False) and not isinstance(base_model, (ConvSAI_Hybrid, ConvSAI_Separable, ConvSAI_Spectral)):
        model = SquaredConvSAI(base_model)
        print(f"Squared kernel: M = K² (effective r_cut={model.r_cut * 2})")
    else:
        model = base_model

    # Load near-field weights for hybrid mode
    if config.get("hybrid") and config.get("near_checkpoint"):
        near_state = torch.load(config["near_checkpoint"], map_location="cpu",
                                weights_only=True)
        base_model.near.load_state_dict(near_state, strict=False)
        print(f"Loaded near-field from {config['near_checkpoint']}")
        if config.get("freeze_near", False):
            for p in base_model.near.parameters():
                p.requires_grad = False
            print("  Near-field FROZEN")

    if config.get("resume"):
        state = torch.load(config["resume"], map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing and not unexpected:
            base_model.load_state_dict(state, strict=False)
            print(f"Loaded base weights from {config['resume']}")
        else:
            print(f"Loaded weights from {config['resume']} "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})")

    model.to(device)
    loss_type = config['loss']

    # EMA model
    ema_decay = config.get("ema_decay", 0.0)
    ema_model = None
    if ema_decay > 0:
        ema_model = copy.deepcopy(model)
        ema_model.eval()
        print(f"EMA enabled: decay={ema_decay}")

    print(f"Parameters: {count_parameters(model):,}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable:,}")
    if hasattr(model, 'r_cut'):
        print(f"Stencil: r_cut={model.r_cut}, {model.n_stencil} displacements")
    elif hasattr(model, 'axis_range'):
        print(f"Separable: axis_range={model.axis_range}, {model.n_axis} entries/axis")
    elif hasattr(model, 'near'):
        print(f"Hybrid: near r_cut={model.near.r_cut}, far range={model.far.axis_range}")
    print(f"Encoder resolution: {encoder_resolution}^3")
    print(f"Shape embed dim: {config.get('shape_embed_dim', 16)}")
    print(f"Loss: {loss_type}")
    if loss_type == 'adversarial':
        print(f"  adversarial_iters: {config.get('adversarial_iters', 10)}")
    only = config.get('only_shape')
    if only:
        print(f"Shapes: {only} ONLY")
    else:
        print(f"Shapes: random (sphere, ellipsoid[continuous], cube, cylinder, capsule)")
    print(f"Ranges: m_re=[{config['m_re_min']}, {config['m_re_max']}], "
          f"m_im=[{config['m_im_min']}, {config['m_im_max']}], "
          f"kd=[{config['kd_min']}, {config['kd_max']}]")
    print(f"Grid: [{config['grid_min']}, {config['grid_max']}], "
          f"curriculum={config.get('curriculum_frac', 0.5)}")
    print()

    # Separate param groups: higher LR for multigrid coarse heads
    if isinstance(model, ConvSAI_Multigrid):
        freeze_base = config.get("mg_freeze_base", False)
        coarse_lr = config["lr"] * config.get("coarse_lr_mult", 3.0)
        if freeze_base:
            # Freeze base model — only train coarse heads
            for p in model.base.parameters():
                p.requires_grad = False
            optimizer = torch.optim.AdamW(
                model.coarse_heads.parameters(), lr=coarse_lr,
                weight_decay=config.get("weight_decay", 1e-4))
            trainable = sum(p.numel() for p in model.coarse_heads.parameters())
            print(f"Optimizer: FROZEN base, coarse_lr={coarse_lr}, "
                  f"trainable={trainable:,}")
        else:
            coarse_ids = set(id(p) for p in model.coarse_heads.parameters())
            base_params = [p for p in model.parameters() if id(p) not in coarse_ids]
            optimizer = torch.optim.AdamW([
                {'params': base_params, 'lr': config["lr"]},
                {'params': list(model.coarse_heads.parameters()), 'lr': coarse_lr},
            ], weight_decay=config.get("weight_decay", 1e-4))
            print(f"Optimizer: base_lr={config['lr']}, coarse_lr={coarse_lr}")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                       weight_decay=config.get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=config.get("patience", 500))

    best_val = float("inf")
    logger = TrainResults(config['folder'])

    grad_clip = config.get("gradient_clipping", 1.0)
    val_interval = config.get("val_interval", 500)
    log_interval = config.get("log_interval", 50)
    num_steps = config.get("num_steps", 40000)
    solve_val_interval = config.get("solve_val_interval", 2000)
    warmup_steps = config.get("warmup_steps", 0)
    base_lr = config["lr"]

    running_loss = 0.0
    start_total = time.perf_counter()

    for step in range(1, num_steps + 1):
        model.train()
        start = time.perf_counter()

        # LR warmup
        if warmup_steps > 0 and step <= warmup_steps:
            warmup_factor = step / warmup_steps
            for pg in optimizer.param_groups:
                pg['lr'] = base_lr * warmup_factor

        m_re, m_im, kd, grid = sample_parameters(
            rng, config, step=step, num_steps=num_steps)

        try:
            positions, shape_name = generate_random_shape(
                rng, grid, only_shape=config.get('only_shape'))
            fft_mv = build_fft_matvec_from_positions(positions, m_re, m_im, kd, device)
        except Exception:
            continue

        # Build occupancy grid
        occ_grid = positions_to_occupancy(positions, grid_size=grid, device=device)

        # Forward: predict kernel from physical params + geometry
        kernel = model(m_re, m_im, kd, occ_grid, grid)

        try:
            loss = compute_loss(model, kernel, fft_mv, config)
        except (torch.OutOfMemoryError, RuntimeError) as e:
            if 'out of memory' in str(e).lower():
                torch.cuda.empty_cache()
                print(f"  OOM at step {step} (grid={grid}), skipping")
                optimizer.zero_grad()
                continue
            raise

        loss.backward()
        running_loss += loss.item()

        if grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        else:
            grad_norm = 0.0

        optimizer.step()
        optimizer.zero_grad()

        # EMA update
        if ema_model is not None:
            with torch.no_grad():
                for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

        step_time = time.perf_counter() - start
        logger.log(loss.item(),
                   grad_norm if isinstance(grad_norm, float) else grad_norm.item(),
                   step_time)

        if step % log_interval == 0:
            avg_loss = running_loss / log_interval
            elapsed = time.perf_counter() - start_total
            steps_per_sec = step / elapsed
            print(f"  [{step}/{num_steps}] loss={avg_loss:.6f} "
                  f"gnorm={grad_norm if isinstance(grad_norm, float) else grad_norm.item():.4f} "
                  f"lr={optimizer.param_groups[0]['lr']:.1e} "
                  f"{steps_per_sec:.1f} steps/s "
                  f"({shape_name} g{grid} m={m_re:.2f}+{m_im:.2f}i kd={kd:.2f})")
            running_loss = 0.0

        if step % val_interval == 0:
            val_rng = np.random.RandomState(config["seed"] + step)
            probe_val = validate_probe(model, val_rng, config, device)
            scheduler.step(probe_val)
            logger.log_val(None, probe_val)

        if step % solve_val_interval == 0:
            val_metric = validate_fixed(model, device, encoder_resolution)

            if val_metric < best_val:
                if config["save"]:
                    save_model = model.base if isinstance(model, SquaredConvSAI) else model
                    torch.save(save_model.state_dict(),
                               os.path.join(config['folder'], "best_model.pt"))
                best_val = val_metric
                print(f"  >>> New best (raw): {best_val:.6f} (speedup {1/best_val:.2f}x)")

            # Also evaluate EMA model
            if ema_model is not None:
                ema_val = validate_fixed(ema_model, device, encoder_resolution)
                if ema_val < best_val:
                    if config["save"]:
                        save_ema = ema_model.base if isinstance(ema_model, SquaredConvSAI) else ema_model
                        torch.save(save_ema.state_dict(),
                                   os.path.join(config['folder'], "best_model.pt"))
                    best_val = ema_val
                    print(f"  >>> New best (EMA): {best_val:.6f} (speedup {1/best_val:.2f}x)")

        if config["save"] and step % config.get("save_interval", 5000) == 0:
            torch.save(model.state_dict(),
                       os.path.join(config['folder'], f"model_step{step}.pt"))

    total_time = time.perf_counter() - start_total
    print(f"\nTraining complete: {num_steps} steps in {total_time:.1f}s "
          f"({num_steps / total_time:.1f} steps/s)")

    if config["save"]:
        logger.save_results()
        save_model = model.base if isinstance(model, SquaredConvSAI) else model
        torch.save(save_model.state_dict(),
                   os.path.join(config['folder'], "final_model.pt"))
        if ema_model is not None:
            save_ema = ema_model.base if isinstance(ema_model, SquaredConvSAI) else ema_model
            torch.save(save_ema.state_dict(),
                       os.path.join(config['folder'], "ema_model.pt"))

    print(f"Best validation (inv_speedup): {best_val:.6f} (speedup {1/max(best_val, 1e-6):.2f}x)")

    print("\n=== Final evaluation (raw) ===")
    validate_fixed(model, device, encoder_resolution)

    if ema_model is not None:
        print("\n=== Final evaluation (EMA) ===")
        validate_fixed(ema_model, device, encoder_resolution)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ConvSAI v7 — Universal model with 3D geometry encoder")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--save", action='store_true')

    # Loss function
    parser.add_argument("--loss", type=str, default="adversarial",
                        choices=["probe", "adversarial", "right_probe", "gmres"])
    parser.add_argument("--adversarial_iters", type=int, default=10)
    parser.add_argument("--gmres_iters", type=int, default=10)
    parser.add_argument("--gmres_rhs", type=int, default=2)

    # Training
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_steps", type=int, default=40000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gradient_clipping", type=float, default=1.0)
    parser.add_argument("--num_probes", type=int, default=5)
    parser.add_argument("--val_interval", type=int, default=500)
    parser.add_argument("--solve_val_interval", type=int, default=2000)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--patience", type=int, default=500)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--curriculum_frac", type=float, default=0.3)
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="LR warmup steps (0 = disabled)")
    parser.add_argument("--ema_decay", type=float, default=0.0,
                        help="EMA decay (0 = disabled, 0.999 recommended)")
    parser.add_argument("--squared_kernel", action='store_true',
                        help="Use M=K² frequency-domain composition")
    parser.add_argument("--spectral", action='store_true',
                        help="Pointwise spectral preconditioner (uses D_hat)")
    parser.add_argument("--freq_hidden", type=int, default=64,
                        help="Hidden dim for per-frequency MLP (spectral)")
    parser.add_argument("--freq_layers", type=int, default=3,
                        help="Layers for per-frequency MLP (spectral)")
    parser.add_argument("--global_hidden", type=int, default=256,
                        help="Hidden dim for global encoder (spectral)")
    parser.add_argument("--global_layers", type=int, default=3,
                        help="Layers for global encoder (spectral)")
    parser.add_argument("--no_freq_coords", action='store_true',
                        help="Disable frequency coordinates input (spectral)")
    parser.add_argument("--separable", action='store_true',
                        help="Use separable 1D kernel architecture")
    parser.add_argument("--hybrid", action='store_true',
                        help="Hybrid: 3D near-field + separable far-field")
    parser.add_argument("--near_checkpoint", type=str, default=None,
                        help="Pre-trained near-field checkpoint (hybrid mode)")
    parser.add_argument("--freeze_near", action='store_true',
                        help="Freeze near-field model (hybrid mode)")
    parser.add_argument("--axis_range", type=int, default=30,
                        help="1D kernel range per axis (separable/hybrid mode)")
    parser.add_argument("--multigrid_levels", type=int, default=0,
                        help="Multigrid levels (0=disabled, 3=stride 1,2,4)")
    parser.add_argument("--mg_bottleneck", type=int, default=0,
                        help="Bottleneck dim for coarse heads (0=direct linear)")
    parser.add_argument("--mg_squared", action='store_true',
                        help="Apply K² on top of multigrid")
    parser.add_argument("--mg_freeze_base", action='store_true',
                        help="Freeze base model, train only coarse heads")
    parser.add_argument("--coarse_lr_mult", type=float, default=3.0,
                        help="LR multiplier for coarse heads")

    # Model
    parser.add_argument("--r_cut", type=int, default=7)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--shape_embed_dim", type=int, default=16)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--no_scale_by_stencil", action='store_true')
    parser.add_argument("--encoder_resolution", type=int, default=32)
    parser.add_argument("--encoder_channels", type=int, nargs='+', default=[16, 32, 64])

    # Sampling ranges
    parser.add_argument("--m_re_min", type=float, default=1.5)
    parser.add_argument("--m_re_max", type=float, default=4.0)
    parser.add_argument("--m_im_min", type=float, default=0.0)
    parser.add_argument("--m_im_max", type=float, default=0.5)
    parser.add_argument("--kd_min", type=float, default=0.2)
    parser.add_argument("--kd_max", type=float, default=0.8)
    parser.add_argument("--grid_min", type=int, default=8)
    parser.add_argument("--grid_max", type=int, default=32)
    parser.add_argument("--hard_sample_frac", type=float, default=0.5)
    parser.add_argument("--high_m", action='store_true',
                        help="Use high-m validation configs (m=4-6, smaller kd)")
    parser.add_argument("--only_shape", type=str, default=None,
                        choices=SHAPE_TYPES,
                        help="Train on only this shape type")

    args = parser.parse_args()

    if args.device is None:
        device = "cpu"
        print("Warning: CPU training. Use --device <id> for GPU.")
    else:
        device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    if args.name is not None:
        folder = "results/" + args.name
    else:
        folder = "results/v7_" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    print(f"Device: {device}")
    config = vars(args)
    config['device'] = device
    config['folder'] = folder
    config['scale_by_stencil'] = not config.pop('no_scale_by_stencil', False)

    if args.high_m:
        FIXED_VAL_CONFIGS.clear()
        FIXED_VAL_CONFIGS.extend(FIXED_VAL_CONFIGS_HIGH_M)
        print("Using HIGH-M validation configs (m=4-6, small kd)")
    elif args.multigrid_levels > 1:
        FIXED_VAL_CONFIGS.clear()
        FIXED_VAL_CONFIGS.extend(FIXED_VAL_CONFIGS_MULTIGRID)
        print("Using MULTIGRID validation configs (small + large grid)")
    elif args.only_shape == 'cube':
        FIXED_VAL_CONFIGS.clear()
        FIXED_VAL_CONFIGS.extend(FIXED_VAL_CONFIGS_BOX)
        print("Using BOX-only validation configs")

    pprint.pprint(config)
    print()

    main(config)
