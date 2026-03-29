"""Export ConvSAI_Spectral preconditioner to binary .precond format for ADDA.

The spectral model works in frequency domain: for each frequency point,
a small MLP predicts M_hat(k) from D_hat(k) + global conditioning.
To export, we compute M_hat for the specific problem, IFFT to spatial domain,
and extract the stencil.

Usage:
    python apps/export_spectral_precond.py \
        --checkpoint results/spectral_v2/best_model.pt \
        --shape sphere --grid 24 --m_re 2.0 --m_im 0.0 --kd 0.42 \
        --output /tmp/sphere24_spectral.precond
"""
import argparse
import struct
import sys
import os

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from neural_precond.model import ConvSAI_Spectral, positions_to_occupancy
from core.fft_matvec import FFTMatVec
from apps.export_universal_precond import (
    make_shape_positions, export_convsai_fft,
)

PRECOND_MAGIC = 0x4E49464C
CONVSAI_MODE = 3


def main():
    parser = argparse.ArgumentParser(
        description='Export ConvSAI_Spectral preconditioner to .precond for ADDA')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--shape', default='sphere')
    parser.add_argument('--ay', type=float, default=1.0)
    parser.add_argument('--az', type=float, default=1.0)
    parser.add_argument('--grid', type=int, required=True)
    parser.add_argument('--m_re', type=float, default=1.5)
    parser.add_argument('--m_im', type=float, default=0.0)
    parser.add_argument('--kd', type=float, default=1.0)
    parser.add_argument('--freq_hidden', type=int, default=256)
    parser.add_argument('--freq_layers', type=int, default=5)
    parser.add_argument('--global_hidden', type=int, default=256)
    parser.add_argument('--global_layers', type=int, default=3)
    parser.add_argument('--threshold', type=float, default=1e-10,
                        help='Threshold for stencil entry magnitude')
    parser.add_argument('--n_dipoles', type=int, default=None)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    # Build positions
    positions = make_shape_positions(args.shape, args.grid, args.ay, args.az)
    N = args.n_dipoles if args.n_dipoles is not None else len(positions)

    # Build occupancy grid
    occ_grid = positions_to_occupancy(positions, grid_size=args.grid, device='cpu')

    # Build FFTMatVec to get D_hat
    pos_torch = torch.tensor(positions, dtype=torch.long)
    d = 1.0
    k = args.kd / d
    m = complex(args.m_re, args.m_im)
    fft_mv = FFTMatVec(pos_torch, k, m, d=d, device='cpu')

    # Load model
    model = ConvSAI_Spectral(
        freq_hidden=args.freq_hidden,
        freq_layers=args.freq_layers,
        global_hidden=args.global_hidden,
        global_layers=args.global_layers,
        squared=True,  # spectral_v2 uses squared
        freq_coords=True,
    )

    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()

    print(f"Shape: {args.shape}, grid: {args.grid}, N_dip: {N}")
    if args.shape == 'ellipsoid':
        print(f"  Aspect ratios: ay={args.ay}, az={args.az}")

    # Compute M_hat
    with torch.no_grad():
        cond = model(args.m_re, args.m_im, args.kd, occ_grid, args.grid)
        M_hat = model.build_M_hat(cond, fft_mv)  # (3, 3, gx, gy, gz) complex

    M_hat_np = M_hat.numpy().astype(np.complex128)
    gx, gy, gz = M_hat_np.shape[2], M_hat_np.shape[3], M_hat_np.shape[4]

    # IFFT to spatial domain
    M_spatial = np.fft.ifftn(M_hat_np, axes=(2, 3, 4))  # (3, 3, gx, gy, gz)

    # Extract stencil: for each displacement, check if block is significant
    stencil = []
    kernel = []

    for di in range(-(gx // 2), gx // 2 + 1):
        for dj in range(-(gy // 2), gy // 2 + 1):
            for dk in range(-(gz // 2), gz // 2 + 1):
                gi = di % gx
                gj = dj % gy
                gk = dk % gz
                block = M_spatial[:, :, gi, gj, gk]  # (3, 3) complex
                if np.max(np.abs(block)) > args.threshold:
                    stencil.append([di, dj, dk])
                    kernel.append(block)

    stencil_np = np.array(stencil, dtype=np.int32)
    kernel_np = np.array(kernel, dtype=np.complex128)

    print(f"M_hat grid: {gx}x{gy}x{gz}")
    print(f"Stencil entries: {len(stencil_np)} (threshold={args.threshold})")

    export_convsai_fft(stencil_np, kernel_np, N, args.output)


if __name__ == '__main__':
    main()
