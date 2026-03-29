"""Export ConvSAI_Universal preconditioner to binary .precond format for ADDA.

The universal model uses a 3D CNN geometry encoder instead of discrete shape_id.
It takes an occupancy grid of the actual shape → shape embedding → kernel.

Usage:
    python apps/export_universal_precond.py \
        --checkpoint results/universal_r7/best_model.pt \
        --shape sphere --grid 24 --m_re 2.0 --m_im 0.0 --kd 0.42 \
        --output /tmp/sphere24_universal.precond

    # Squared kernel (K²) — effective r_cut doubled:
    python apps/export_universal_precond.py \
        --checkpoint results/universal_r7_k2/best_model.pt \
        --squared_kernel \
        --shape sphere --grid 24 --m_re 2.0 --kd 0.42 \
        --output /tmp/sphere24_k2.precond
"""
import argparse
import struct
import sys
import os

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from neural_precond.model import ConvSAI_Universal, ConvSAI_Multigrid, positions_to_occupancy
from apps.export_sai_precond import (
    make_sphere_dipoles_adda, make_cube_dipoles_adda, make_ellipsoid_dipoles_adda,
)

PRECOND_MAGIC = 0x4E49464C
CONVSAI_MODE = 3


def make_generic_shape_dipoles(shape, grid, ay=1.0, az=1.0):
    """Create dipole positions for cylinder, capsule, prism, plate via occupancy test."""
    positions = []
    cx, cy, cz = grid / 2.0, grid / 2.0, grid / 2.0
    r = grid / 2.0  # base radius in grid units

    for iz in range(grid):
        for iy in range(grid):
            for ix in range(grid):
                # Centered coords normalized to [-1, 1]
                x = (ix + 0.5 - cx) / r
                y = (iy + 0.5 - cy) / r
                z = (iz + 0.5 - cz) / r
                inside = False

                if shape == 'cylinder':
                    # cylinder with h/d = az, axis along z
                    h_half = az  # h/d ratio, half-height = az * r
                    if x*x + y*y <= 1.0 and abs(z) <= az:
                        inside = True
                elif shape == 'capsule':
                    # capsule: cylinder with hemispherical caps, h/d = az
                    h_half = az
                    if abs(z) <= az:
                        if x*x + y*y <= 1.0:
                            inside = True
                    else:
                        dz = abs(z) - az
                        if x*x + y*y + dz*dz <= 1.0:
                            inside = True
                elif shape.startswith('prism'):
                    # Regular n-sided prism, h/Dx = az
                    import math
                    n_sides = int(ay)  # ay stores number of sides
                    h_half = az
                    if abs(z) <= az:
                        # Check if (x,y) inside regular n-gon inscribed in unit circle
                        angle = math.atan2(y, x)
                        dist = math.sqrt(x*x + y*y)
                        sector = 2 * math.pi / n_sides
                        # Distance to nearest edge
                        angle_in_sector = abs((angle % sector) - sector/2)
                        r_edge = math.cos(math.pi / n_sides) / math.cos(angle_in_sector)
                        if dist <= r_edge:
                            inside = True
                elif shape == 'plate':
                    # Circular plate with h/d = az
                    if x*x + y*y <= 1.0 and abs(z) <= az:
                        inside = True

                if inside:
                    positions.append([ix, iy, iz])

    if not positions:
        raise ValueError(f"No dipoles generated for shape={shape} grid={grid}")
    return np.array(positions, dtype=np.int32)


def make_shape_positions(shape, grid, ay=1.0, az=1.0):
    """Create dipole positions for a given shape."""
    if shape == 'sphere':
        return make_sphere_dipoles_adda(grid)
    elif shape == 'cube':
        return make_cube_dipoles_adda(grid)
    elif shape == 'ellipsoid':
        return make_ellipsoid_dipoles_adda(grid, aspect_ratios=(1.0, ay, az))
    elif shape in ('cylinder', 'capsule', 'plate') or shape.startswith('prism'):
        return make_generic_shape_dipoles(shape, grid, ay, az)
    else:
        raise ValueError(f"Unknown shape: {shape}")


def export_convsai_fft(stencil_np, kernel_np, n_dipoles, output_path):
    """Write mode=3 (CONVSAI) preconditioner."""
    n = 3 * n_dipoles
    n_stencil = len(stencil_np)

    with open(output_path, 'wb') as f:
        f.write(struct.pack('<5Q', PRECOND_MAGIC, n, n_stencil, CONVSAI_MODE, 0))
        stencil_i32 = stencil_np.astype(np.int32)
        f.write(stencil_i32.tobytes())
        kernel_flat = kernel_np.reshape(n_stencil * 9)
        interleaved = np.empty(n_stencil * 18, dtype=np.float64)
        interleaved[0::2] = kernel_flat.real
        interleaved[1::2] = kernel_flat.imag
        f.write(interleaved.tobytes())

    file_size = os.path.getsize(output_path)
    print(f"Exported ConvSAI Universal FFT: n={n}, n_stencil={n_stencil}, "
          f"file={file_size/1024:.1f} KB -> {output_path}")


def compute_squared_kernel(stencil_np, kernel_np, r_cut):
    """Compute K² kernel in frequency domain, return wider stencil + kernel.

    M_hat = K_hat @ K_hat (3x3 matrix multiply per frequency point).
    IFFT back to spatial domain gives a kernel with effective r_cut = 2*r_cut.

    Args:
        stencil_np: (n_stencil, 3) int — displacement vectors
        kernel_np: (n_stencil, 3, 3) complex — kernel values
        r_cut: int — original r_cut

    Returns:
        new_stencil: (n_stencil2, 3) int
        new_kernel: (n_stencil2, 3, 3) complex
    """
    # Grid size: must be >= 2*r_cut_eff + 1 = 4*r_cut + 1 for convolution
    g = 4 * r_cut + 2
    # Ensure even
    if g % 2 != 0:
        g += 1

    # Build spatial kernel on grid
    # kernel_np[s, a, b] is the (a,b) entry of the 3x3 block at stencil[s]
    # ADDA stores kernel_raw[9*s + 3*a + b] → same layout
    # For FFT: K_grid[a, b, x, y, z] should be kernel_np[s, a, b]
    K_grid = np.zeros((3, 3, g, g, g), dtype=np.complex128)
    for s in range(len(stencil_np)):
        di, dj, dk = stencil_np[s]
        gi = di % g
        gj = dj % g
        gk = dk % g
        K_grid[:, :, gi, gj, gk] = kernel_np[s]  # (3, 3) block, no transpose

    # FFT
    K_hat = np.fft.fftn(K_grid, axes=(2, 3, 4))

    # M_hat = K_hat @ K_hat (3x3 matmul per frequency)
    M_hat = np.einsum('ijxyz,jkxyz->ikxyz', K_hat, K_hat)

    # IFFT back
    M_grid = np.fft.ifftn(M_hat, axes=(2, 3, 4))

    # Extract non-zero entries as new stencil
    r_eff = 2 * r_cut
    new_stencil = []
    new_kernel = []
    threshold = 1e-12

    for di in range(-r_eff, r_eff + 1):
        for dj in range(-r_eff, r_eff + 1):
            for dk in range(-r_eff, r_eff + 1):
                gi = di % g
                gj = dj % g
                gk = dk % g
                block = M_grid[:, :, gi, gj, gk]  # (3, 3)
                if np.max(np.abs(block)) > threshold:
                    new_stencil.append([di, dj, dk])
                    new_kernel.append(block)  # (3,3) — same layout as kernel_np

    new_stencil = np.array(new_stencil, dtype=np.int32)
    new_kernel = np.array(new_kernel, dtype=np.complex128)

    print(f"Squared kernel: {len(stencil_np)} -> {len(new_stencil)} stencil entries "
          f"(r_cut {r_cut} -> {r_eff})")
    return new_stencil, new_kernel


def compute_multigrid_stencil(base_stencil_np, kernels_np, r_cut, num_levels):
    """Combine multi-level stencils into a single stencil for ADDA export.

    Each level's entries are placed at stride 2^level. Overlapping entries
    (e.g. (0,0,0) present in all levels) are summed.

    Args:
        base_stencil_np: (n_stencil, 3) int — base stencil displacements
        kernels_np: list of (n_stencil, 3, 3) complex arrays, one per level
        r_cut: int — base r_cut
        num_levels: int — number of levels

    Returns:
        combined_stencil: (M, 3) int
        combined_kernel: (M, 3, 3) complex
    """
    combined = {}  # (di, dj, dk) -> (3, 3) complex

    for level in range(num_levels):
        scale = 2 ** level
        kernel = kernels_np[level]
        for s in range(len(base_stencil_np)):
            di = int(base_stencil_np[s, 0]) * scale
            dj = int(base_stencil_np[s, 1]) * scale
            dk = int(base_stencil_np[s, 2]) * scale
            key = (di, dj, dk)
            if key in combined:
                combined[key] = combined[key] + kernel[s]
            else:
                combined[key] = kernel[s].copy()

    # Sort for deterministic output
    keys = sorted(combined.keys())
    stencil = np.array(keys, dtype=np.int32)
    kernel_arr = np.array([combined[k] for k in keys], dtype=np.complex128)

    eff_r = r_cut * (2 ** (num_levels - 1))
    print(f"Multigrid stencil: {num_levels} levels, "
          f"{len(base_stencil_np)} base -> {len(stencil)} combined entries "
          f"(r_cut {r_cut} -> eff {eff_r})")
    return stencil, kernel_arr


def main():
    parser = argparse.ArgumentParser(
        description='Export ConvSAI_Universal preconditioner to .precond for ADDA')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--shape', default='sphere')
    parser.add_argument('--ay', type=float, default=1.0,
                        help='Ellipsoid y/x aspect ratio')
    parser.add_argument('--az', type=float, default=1.0,
                        help='Ellipsoid z/x aspect ratio')
    parser.add_argument('--grid', type=int, required=True)
    parser.add_argument('--m_re', type=float, default=1.5)
    parser.add_argument('--m_im', type=float, default=0.0)
    parser.add_argument('--kd', type=float, default=1.0)
    parser.add_argument('--r_cut', type=int, default=7)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--shape_embed_dim', type=int, default=16)
    parser.add_argument('--encoder_resolution', type=int, default=32)
    parser.add_argument('--n_dipoles', type=int, default=None)
    parser.add_argument('--squared_kernel', action='store_true',
                        help='Apply K² composition (M = K·K in freq domain)')
    parser.add_argument('--multigrid_levels', type=int, default=0,
                        help='Multigrid levels (0=disabled, 3=stride 1,2,4)')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    # Build positions for occupancy grid
    positions = make_shape_positions(args.shape, args.grid, args.ay, args.az)
    N = args.n_dipoles if args.n_dipoles is not None else len(positions)

    # Build occupancy grid
    occ_grid = positions_to_occupancy(positions, grid_size=args.grid, device='cpu')

    # Load base model
    base_model = ConvSAI_Universal(
        r_cut=args.r_cut, hidden_size=args.hidden_size,
        num_layers=args.num_layers, shape_embed_dim=args.shape_embed_dim,
        encoder_resolution=args.encoder_resolution,
        scale_by_stencil=True)

    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)

    if args.multigrid_levels > 1:
        # Load multigrid model (checkpoint contains base + coarse_heads)
        model = ConvSAI_Multigrid(base_model, num_levels=args.multigrid_levels)
        model.load_state_dict(state)
        model.eval()

        print(f"Shape: {args.shape}, grid: {args.grid}, N_dip: {N}")
        eff_r = args.r_cut * (2 ** (args.multigrid_levels - 1))
        print(f"Multigrid: {args.multigrid_levels} levels, "
              f"r_cut={args.r_cut}, effective r_cut={eff_r}")

        with torch.no_grad():
            kernels = model(args.m_re, args.m_im, args.kd, occ_grid, args.grid)

        kernels_np = [k.numpy() for k in kernels]
        stencil_np = model.stencil.numpy()

        stencil_np, kernel_np = compute_multigrid_stencil(
            stencil_np, kernels_np, args.r_cut, args.multigrid_levels)
    else:
        base_model.load_state_dict(state)
        base_model.eval()
        model = base_model

        print(f"Shape: {args.shape}, grid: {args.grid}, N_dip: {N}")
        if args.shape == 'ellipsoid':
            print(f"  Aspect ratios: ay={args.ay}, az={args.az}")
        print(f"Stencil: {model.n_stencil} displacements, r_cut={args.r_cut}")

        with torch.no_grad():
            kernel = model(args.m_re, args.m_im, args.kd, occ_grid, args.grid)
        kernel_np = kernel.squeeze(0).numpy()
        stencil_np = model.stencil.numpy()

        if args.squared_kernel:
            stencil_np, kernel_np = compute_squared_kernel(
                stencil_np, kernel_np, args.r_cut)

    export_convsai_fft(stencil_np, kernel_np, N, args.output)


if __name__ == '__main__':
    main()
