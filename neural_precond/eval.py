#!/usr/bin/env python3
"""Evaluate trained NeuralPrecond model.

For each test sample:
  1. encode_geometry() → cache
  2. Build precond_fn from cache
  3. BiCGStab(A, b, M=None) → iters_baseline
  4. BiCGStab(A, b, M=precond_fn) → iters_precond
  5. Compare iteration counts and wall-clock time

The key metric is BiCGStab iteration count reduction.
"""
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from train_sai import SAIDataset, is_fft_mode
from core.fft_matvec import FFTMatVec
from core.utils import count_parameters
from torch_geometric.loader import DataLoader
from krylov.bicgstab import bicgstab

from neural_precond.model import NeuralPrecond, PolyPrecond, PolyPrecondMLP, ConvSAI_MLP


def cast_data_f32(data):
    """Cast graph features to float32 for GNN forward pass."""
    data_f32 = data.clone()
    data_f32.x = data_f32.x.float()
    data_f32.edge_attr = data_f32.edge_attr.float()
    if hasattr(data_f32, 'global_features') and data_f32.global_features is not None:
        data_f32.global_features = data_f32.global_features.float()
    return data_f32


def evaluate_precond(model_path, data_dir, device='cpu', max_samples=50,
                     max_nodes=None, rtol=1e-6, max_iter=5000,
                     latent_size=64, latent_size_apply=32,
                     message_passing_steps=6, apply_mp_steps=1,
                     model_type='neural', poly_degree=3,
                     hidden_size=128, num_layers=4, num_shapes=3,
                     shape_embed_dim=8, r_cut=3):
    """Full evaluation of NeuralPrecond, PolyPrecond, PolyPrecondMLP, or ConvSAI_MLP.

    Everything runs on CPU for consistency — BiCGStab needs CPU anyway.
    The encode_geometry can optionally run on GPU, but for eval simplicity
    we keep everything on one device.
    """

    # Load model on eval device
    if model_type == 'conv_sai':
        model = ConvSAI_MLP(
            r_cut=r_cut,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_shapes=num_shapes,
            shape_embed_dim=shape_embed_dim,
            activation='relu',
        )
    elif model_type == 'mlp':
        model = PolyPrecondMLP(
            poly_degree=poly_degree,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_shapes=num_shapes,
            shape_embed_dim=shape_embed_dim,
            activation='relu',
        )
    elif model_type == 'poly':
        model = PolyPrecond(
            poly_degree=poly_degree,
            latent_size=latent_size,
            message_passing_steps=message_passing_steps,
            activation='relu',
        )
    else:
        model = NeuralPrecond(
            latent_size=latent_size,
            latent_size_apply=latent_size_apply,
            message_passing_steps=message_passing_steps,
            apply_mp_steps=apply_mp_steps,
            activation='relu',
        )
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    total_params = count_parameters(model)
    print(f"Model type: {model_type}")
    print(f"Model: {total_params} total parameters")
    if model_type == 'conv_sai':
        print(f"Stencil: r_cut={r_cut}, {model.n_stencil} displacements")
        print(f"Hidden size: {hidden_size}, Layers: {num_layers}")
        print(f"Apply-phase: FFT convolution (1 matvec cost)")
    elif model_type == 'mlp':
        print(f"Polynomial degree K={poly_degree}")
        print(f"Hidden size: {hidden_size}, Layers: {num_layers}")
        print(f"Apply-phase params: 0 (Horner via FFTMatVec)")
    elif model_type == 'poly':
        print(f"Polynomial degree K={poly_degree}")
        print(f"Apply-phase params: 0 (Horner via FFTMatVec)")
    else:
        apply_params = (
            sum(p.numel() for p in model.apply_input.parameters()) +
            sum(p.numel() for p in model.apply_mp.parameters()) +
            sum(p.numel() for p in model.apply_output.parameters())
        )
        print(f"Apply-phase params: {apply_params}")
    print(f"Checkpoint: {model_path}")
    if model_type != 'mlp':
        print(f"L={latent_size}, encoder_MP={message_passing_steps}")
    print(f"Eval device: {device}")
    print()

    # Load test data
    test_dataset = SAIDataset(os.path.join(data_dir, 'test'), max_nodes=max_nodes)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    n_test = min(max_samples, len(test_dataset))
    print(f"Test samples: {len(test_dataset)} total, evaluating {n_test}")
    print()

    # ── BiCGStab iteration count comparison ────────────────
    print("=" * 70)
    print("BiCGStab ITERATION COUNT: Preconditioned vs Unpreconditioned")
    print("=" * 70)

    results = []

    for i, data in enumerate(test_loader):
        if i >= n_test:
            break

        data = data.to(device)
        if not is_fft_mode(data):
            continue

        data_f32 = cast_data_f32(data)
        n = data_f32.x.shape[0]  # DOF nodes

        # Build FFT matvec on CPU for solver
        positions_cpu = data.positions.cpu() if data.positions.device.type != 'cpu' else data.positions
        fft_mv = FFTMatVec(positions_cpu, data.k_val.item(),
                           complex(data.m_re.item(), data.m_im.item()),
                           d=data.d_val.item(), device='cpu')
        n_dof = fft_mv.n

        def A_op(v):
            v2d = v.unsqueeze(1) if v.dim() == 1 else v
            return fft_mv(v2d).squeeze(1)

        # Encode geometry and create preconditioner
        with torch.no_grad():
            if model_type == 'conv_sai':
                # ConvSAI: extract physical params, predict kernel
                m_re = data.m_re.item()
                m_im = data.m_im.item()
                kd_val = data.kd.item() if hasattr(data, 'kd') else data.k_val.item() * data.d_val.item()
                shape_id = data.shape_id.item() if hasattr(data, 'shape_id') else 0
                grid_val = data.grid.item() if hasattr(data, 'grid') else int(round(data.n_dipoles.item() ** (1/3)))
                kernel = model(m_re, m_im, kd_val, shape_id, grid_val)
                precond_fn = model.make_precond_fn(kernel, fft_mv)
            elif model_type == 'mlp':
                # MLP: extract physical params from data, no graph needed
                m_re = data.m_re.item()
                m_im = data.m_im.item()
                kd_val = data.kd.item() if hasattr(data, 'kd') else data.k_val.item() * data.d_val.item()
                # Infer shape_id and grid from data
                shape_id = data.shape_id.item() if hasattr(data, 'shape_id') else 0
                grid_val = data.grid.item() if hasattr(data, 'grid') else int(round(data.n_dipoles.item() ** (1/3)))
                coefficients = model(m_re, m_im, kd_val, shape_id, grid_val)
                precond_fn = model.make_precond_fn(coefficients, fft_mv)
            elif model_type == 'poly':
                coefficients = model.encode_geometry(data_f32)
                precond_fn = model.make_precond_fn(coefficients, fft_mv)
            else:
                cache = model.encode_geometry(data_f32)
                cache_cpu = {k: v.cpu() if v.device.type != 'cpu' else v
                             for k, v in cache.items()}
                if not hasattr(evaluate_precond, '_model_solver'):
                    import copy
                    evaluate_precond._model_solver = copy.deepcopy(model).cpu().double()
                    evaluate_precond._model_solver.eval()
                else:
                    evaluate_precond._model_solver.load_state_dict(
                        {k: v.cpu().double() for k, v in model.state_dict().items()})
                precond_fn = evaluate_precond._model_solver.make_precond_fn(cache_cpu)

        # Random RHS
        b = torch.randn(n_dof, dtype=torch.complex128) + \
            1j * torch.randn(n_dof, dtype=torch.complex128)
        b = b / torch.linalg.vector_norm(b)

        # ── Unpreconditioned BiCGStab ──
        t0 = time.perf_counter()
        res_u, _ = bicgstab(A_op, b, rtol=rtol, max_iter=max_iter)
        t_unprecond = time.perf_counter() - t0
        iters_unprecond = len(res_u) - 1

        # ── Preconditioned BiCGStab ──
        t0 = time.perf_counter()
        res_p, _ = bicgstab(A_op, b, M=precond_fn, rtol=rtol, max_iter=max_iter)
        t_precond = time.perf_counter() - t0
        iters_precond = len(res_p) - 1

        results.append({
            'n': n_dof,
            'iters_unprecond': iters_unprecond,
            'iters_precond': iters_precond,
            'time_unprecond': t_unprecond,
            'time_precond': t_precond,
            'final_res_unprecond': res_u[-1],
            'final_res_precond': res_p[-1],
        })

        speedup = iters_unprecond / max(iters_precond, 1)
        print(f"  [{i+1}/{n_test}] n={n_dof}: "
              f"unprecond={iters_unprecond}, precond={iters_precond} iters  "
              f"speedup={speedup:.2f}x  "
              f"time: {t_unprecond:.2f}s vs {t_precond:.2f}s")

    if not results:
        print("No valid test samples found.")
        return

    # ── Summary ────────────────────────────────────────────
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    iters_u = [r['iters_unprecond'] for r in results]
    iters_p = [r['iters_precond'] for r in results]
    times_u = [r['time_unprecond'] for r in results]
    times_p = [r['time_precond'] for r in results]

    print(f"\n  {'Method':<25} {'Avg iters':>12} {'Median iters':>14} {'Std':>10}")
    print(f"  {'-'*61}")
    print(f"  {'Unpreconditioned':<25} {np.mean(iters_u):>12.1f} "
          f"{np.median(iters_u):>14.1f} {np.std(iters_u):>10.1f}")
    label = {'poly': 'Poly precond', 'mlp': 'MLP precond', 'neural': 'Neural precond',
             'conv_sai': 'ConvSAI precond'}[model_type]
    print(f"  {label:<25} {np.mean(iters_p):>12.1f} "
          f"{np.median(iters_p):>14.1f} {np.std(iters_p):>10.1f}")

    avg_speedup = np.mean(iters_u) / max(np.mean(iters_p), 1)
    median_speedup = np.median(iters_u) / max(np.median(iters_p), 1)

    print(f"\n  >>> Iteration speedup (avg): {avg_speedup:.2f}x <<<")
    print(f"  >>> Iteration speedup (median): {median_speedup:.2f}x <<<")

    print(f"\n  {'Method':<25} {'Avg time (s)':>14} {'Median time (s)':>16}")
    print(f"  {'-'*55}")
    print(f"  {'Unpreconditioned':<25} {np.mean(times_u):>14.3f} "
          f"{np.median(times_u):>16.3f}")
    print(f"  {label:<25} {np.mean(times_p):>14.3f} "
          f"{np.median(times_p):>16.3f}")

    wall_speedup = np.mean(times_u) / max(np.mean(times_p), 1e-10)
    print(f"\n  >>> Wall-clock speedup: {wall_speedup:.2f}x <<<")

    # Per-sample speedups
    per_sample_speedups = [r['iters_unprecond'] / max(r['iters_precond'], 1)
                           for r in results]
    print(f"\n  Per-sample iteration speedups: "
          f"min={min(per_sample_speedups):.2f}x, "
          f"max={max(per_sample_speedups):.2f}x, "
          f"mean={np.mean(per_sample_speedups):.2f}x")

    # Convergence check
    converged_u = sum(1 for r in results if r['final_res_unprecond'] < rtol)
    converged_p = sum(1 for r in results if r['final_res_precond'] < rtol)
    print(f"\n  Converged: unprecond={converged_u}/{len(results)}, "
          f"precond={converged_p}/{len(results)} (rtol={rtol})")

    print()
    print("Evaluation complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate NeuralPrecond model")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./data/SAI_v2/")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for model inference (cpu recommended for eval)")
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--max_nodes", type=int, default=None)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--max_iter", type=int, default=5000)

    # Model architecture (must match training config)
    parser.add_argument("--model_type", type=str, default="neural",
                        choices=["neural", "poly", "mlp", "conv_sai"],
                        help="Model type: 'neural', 'poly' (GNN), 'mlp' (PolyPrecondMLP), or 'conv_sai' (ConvSAI_MLP)")
    parser.add_argument("--poly_degree", type=int, default=3,
                        help="Polynomial degree K")
    parser.add_argument("--latent_size", type=int, default=64)
    parser.add_argument("--latent_size_apply", type=int, default=32)
    parser.add_argument("--message_passing_steps", type=int, default=6)
    parser.add_argument("--apply_mp_steps", type=int, default=1)
    # MLP-specific
    parser.add_argument("--hidden_size", type=int, default=128,
                        help="MLP hidden size (PolyPrecondMLP only)")
    parser.add_argument("--num_layers", type=int, default=4,
                        help="MLP number of layers (PolyPrecondMLP only)")
    parser.add_argument("--num_shapes", type=int, default=3,
                        help="Number of shape types (PolyPrecondMLP only)")
    parser.add_argument("--shape_embed_dim", type=int, default=8,
                        help="Shape embedding dim (PolyPrecondMLP/ConvSAI_MLP only)")
    # ConvSAI_MLP specific
    parser.add_argument("--r_cut", type=int, default=3,
                        help="Stencil radius in grid units (ConvSAI_MLP only)")

    args = parser.parse_args()

    evaluate_precond(
        model_path=args.model_path,
        data_dir=args.data_dir,
        device=args.device,
        max_samples=args.max_samples,
        max_nodes=args.max_nodes,
        rtol=args.rtol,
        max_iter=args.max_iter,
        latent_size=args.latent_size,
        latent_size_apply=args.latent_size_apply,
        message_passing_steps=args.message_passing_steps,
        apply_mp_steps=args.apply_mp_steps,
        model_type=args.model_type,
        poly_degree=args.poly_degree,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_shapes=args.num_shapes,
        shape_embed_dim=args.shape_embed_dim,
        r_cut=args.r_cut,
    )
