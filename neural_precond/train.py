"""Training script for NeuralPrecond (GNN-based callable preconditioner).

Reuses SAIDataset and FFTMatVecCache from train_sai.py.

For each training step:
  1. Load geometry graph from SAI dataset
  2. encode_geometry() once (expensive: encoder + processor MP)
  3. precond_probe_loss() with P random probes (cheap per-probe apply_precond)
  4. Backprop through apply-phase parameters

Validation: probe loss on held-out geometries + optional BiCGStab iteration count.

Usage:
    python -m neural_precond.train --save --name precond_v1 --device 0 \
        --data_dir ./data/SAI_v2/ --max_nodes 3000 --num_probes 10 \
        --latent_size 64 --latent_size_apply 32 --apply_mp_steps 1
"""
import os
import sys
import datetime
import argparse
import pprint
import time

import numpy as np
import torch
import torch_geometric
from torch_geometric.loader import DataLoader

# Reuse from existing codebase
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from train_sai import SAIDataset, is_fft_mode, FFTMatVecCache
from core.utils import count_parameters, save_dict_to_file
from core.logger import TrainResults

from neural_precond.model import NeuralPrecond, PolyPrecond
from neural_precond.loss import precond_probe_loss, poly_precond_probe_loss


def cast_data_f32(data):
    """Cast graph features to float32 for GNN forward pass."""
    data_f32 = data.clone()
    data_f32.x = data_f32.x.float()
    data_f32.edge_attr = data_f32.edge_attr.float()
    if hasattr(data_f32, 'global_features') and data_f32.global_features is not None:
        data_f32.global_features = data_f32.global_features.float()
    return data_f32


@torch.no_grad()
def validate(model, val_loader, device, num_probes=10, fft_cache=None,
             model_type='neural'):
    """Validate by measuring probe loss on validation set."""
    model.eval()
    total_loss = 0.0
    count = 0

    for data in val_loader:
        data = data.to(device)
        if not is_fft_mode(data):
            continue

        data_f32 = cast_data_f32(data)

        # Get FFT matvec
        fft_mv = fft_cache.get_or_build(data, device) if fft_cache else None
        if fft_mv is None:
            continue

        if model_type == 'poly':
            coefficients = model.encode_geometry(data_f32)
            loss = _poly_probe_loss_no_grad(coefficients, fft_mv, num_probes)
        else:
            cache = model.encode_geometry(data_f32)
            loss = _probe_loss_no_grad(model, cache, fft_mv, num_probes)

        total_loss += loss
        count += 1

    avg_loss = total_loss / max(count, 1)
    print(f"Validation loss:\t{avg_loss:.6f}")
    return avg_loss


def _probe_loss_no_grad(model, cache, fft_matvec, num_probes):
    """Compute probe loss without gradients (for validation) — NeuralPrecond."""
    n = cache['node_cache'].shape[0]
    device = cache['node_cache'].device

    z_re = torch.randn(n, num_probes, device=device, dtype=torch.float32)
    z_im = torch.randn(n, num_probes, device=device, dtype=torch.float32)

    z_complex = torch.complex(z_re.double(), z_im.double())
    w = fft_matvec(z_complex.to(torch.complex128))
    w_re = w.real.float()
    w_im = w.imag.float()

    total_loss = 0.0
    for p in range(num_probes):
        out_re, out_im = model.apply_precond(cache, w_re[:, p], w_im[:, p])

        res_re = out_re - z_re[:, p]
        res_im = out_im - z_im[:, p]

        res_norm_sq = (res_re.pow(2).sum() + res_im.pow(2).sum()).item()
        z_norm_sq = (z_re[:, p].pow(2).sum() + z_im[:, p].pow(2).sum()).item()

        total_loss += res_norm_sq / (z_norm_sq + 1e-8)

    return total_loss / num_probes


def _poly_probe_loss_no_grad(coefficients, fft_matvec, num_probes):
    """Compute probe loss without gradients (for validation) — PolyPrecond."""
    n = fft_matvec.n
    device = coefficients.device

    z_re = torch.randn(n, num_probes, device=device, dtype=torch.float32)
    z_im = torch.randn(n, num_probes, device=device, dtype=torch.float32)
    z = torch.complex(z_re.double(), z_im.double()).to(torch.complex128)

    w = fft_matvec(z)  # A·z
    h = PolyPrecond.apply_poly(coefficients.to(torch.complex128), fft_matvec, w)

    residual = h - z.to(h.dtype)
    res_norm_sq = (residual.real.pow(2) + residual.imag.pow(2)).sum(dim=0)
    z_norm_sq = (z.real.pow(2) + z.imag.pow(2)).sum(dim=0)

    return ((res_norm_sq / (z_norm_sq + 1e-8)).mean()).item()


@torch.no_grad()
def validate_bicgstab(model, val_loader, device, max_val_samples=10,
                      fft_cache=None, rtol=1e-6, max_iter=5000,
                      model_type='neural'):
    """Validate by running BiCGStab with and without preconditioner.

    This is the ground-truth metric: iteration count reduction.
    """
    from krylov.bicgstab import bicgstab
    from core.fft_matvec import FFTMatVec

    model.eval()
    total_iters_precond = 0
    total_iters_unprecond = 0
    count = 0

    for i, data in enumerate(val_loader):
        if i >= max_val_samples:
            break

        data = data.to(device)
        if not is_fft_mode(data):
            continue

        data_f32 = cast_data_f32(data)

        # Build FFT matvec on CPU for solver
        fft_mv_cpu = FFTMatVec(data.positions, data.k_val.item(),
                               complex(data.m_re.item(), data.m_im.item()),
                               d=data.d_val.item(), device='cpu')
        n = fft_mv_cpu.n

        def A_op(v):
            v2d = v.unsqueeze(1) if v.dim() == 1 else v
            return fft_mv_cpu(v2d).squeeze(1)

        # Random RHS
        b = torch.randn(n, dtype=torch.complex128) + \
            1j * torch.randn(n, dtype=torch.complex128)
        b = b / torch.linalg.vector_norm(b)

        if model_type == 'poly':
            # PolyPrecond: encode → coefficients, make_precond_fn with CPU fft_mv
            coefficients = model.encode_geometry(data_f32)
            precond_fn = model.make_precond_fn(coefficients, fft_mv_cpu)
        else:
            # NeuralPrecond: encode → cache, make_precond_fn from cache
            cache = model.encode_geometry(data_f32)
            cache_cpu = {k: v.cpu() for k, v in cache.items()}

            if not hasattr(validate_bicgstab, '_model_cpu'):
                import copy
                validate_bicgstab._model_cpu = copy.deepcopy(model).cpu().double()
            validate_bicgstab._model_cpu.load_state_dict(
                {k: v.cpu().double() for k, v in model.state_dict().items()})
            validate_bicgstab._model_cpu.eval()
            precond_fn = validate_bicgstab._model_cpu.make_precond_fn(cache_cpu)

        # Preconditioned BiCGStab
        res_p, _ = bicgstab(A_op, b, M=precond_fn, rtol=rtol, max_iter=max_iter)
        n_iters_p = len(res_p) - 1

        # Unpreconditioned BiCGStab
        res_u, _ = bicgstab(A_op, b, rtol=rtol, max_iter=max_iter)
        n_iters_u = len(res_u) - 1

        total_iters_precond += n_iters_p
        total_iters_unprecond += n_iters_u
        count += 1

    avg_precond = total_iters_precond / max(count, 1)
    avg_unprecond = total_iters_unprecond / max(count, 1)
    speedup = avg_unprecond / max(avg_precond, 1)
    print(f"Validation BiCGStab\t precond: {avg_precond:.1f} iters\t "
          f"unprecond: {avg_unprecond:.1f} iters\t speedup: {speedup:.2f}x")
    return avg_precond


def main(config):
    device = config['device']

    if config["save"]:
        os.makedirs(config['folder'], exist_ok=True)
        config_save = {k: (str(v) if isinstance(v, torch.device) else v)
                       for k, v in config.items()}
        save_dict_to_file(config_save, os.path.join(config['folder'], "config.json"))

    torch_geometric.seed_everything(config["seed"])

    # Create model
    model_type = config.get("model_type", "neural")

    if model_type == "poly":
        model = PolyPrecond(
            poly_degree=config.get("poly_degree", 3),
            latent_size=config["latent_size"],
            message_passing_steps=config["message_passing_steps"],
            activation=config["activation"],
            use_checkpoint=config.get("checkpoint", False),
        )
    else:
        model = NeuralPrecond(
            latent_size=config["latent_size"],
            latent_size_apply=config["latent_size_apply"],
            message_passing_steps=config["message_passing_steps"],
            apply_mp_steps=config["apply_mp_steps"],
            activation=config["activation"],
            use_checkpoint=config.get("checkpoint", False),
        )

    if config.get("resume"):
        print(f"Loading weights from {config['resume']}")
        model.load_state_dict(torch.load(config["resume"], map_location="cpu",
                                         weights_only=True))

    model.to(device)
    print(f"Model type: {model_type}")
    print(f"Number params in model: {count_parameters(model)}")

    if model_type == "poly":
        print(f"Polynomial degree K={model.poly_degree}")
        print(f"Apply-phase params: 0 (Horner via FFTMatVec)")
    else:
        apply_params = (
            sum(p.numel() for p in model.apply_input.parameters()) +
            sum(p.numel() for p in model.apply_mp.parameters()) +
            sum(p.numel() for p in model.apply_output.parameters())
        )
        print(f"Apply-phase params: {apply_params}")

    if config.get("compile"):
        model = torch.compile(model, dynamic=True)
        print("torch.compile enabled (dynamic=True)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20
    )

    use_amp = config.get("amp", False) and (device != "cpu")
    scaler = torch.amp.GradScaler(enabled=use_amp)
    if use_amp:
        print("AMP enabled (bfloat16)")

    # Datasets
    data_dir = config["data_dir"]
    max_nodes = config.get("max_nodes")
    train_dataset = SAIDataset(os.path.join(data_dir, "train"), max_nodes=max_nodes)
    val_dataset = SAIDataset(os.path.join(data_dir, "val"), max_nodes=max_nodes)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}"
          + (f" (max_nodes={max_nodes})" if max_nodes else ""))

    num_workers = config.get("num_workers", 4)
    use_gpu = (device != "cpu")
    pin_memory = use_gpu and num_workers > 0
    persistent = num_workers > 0

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory,
                              persistent_workers=persistent)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory,
                            persistent_workers=persistent)

    # Verify FFT mode
    sample = train_dataset[0]
    if not is_fft_mode(sample):
        raise RuntimeError("NeuralPrecond requires v2 (FFT) data format "
                           "with data.positions. Found v1 (legacy) format.")

    # FFT MatVec cache
    fft_cache = FFTMatVecCache(max_size=config.get("fft_cache_size", 50))

    best_val = float("inf")
    logger = TrainResults(config['folder'])

    num_probes = config.get("num_probes", 10)
    grad_clip = config.get("gradient_clipping", 1.0)
    val_interval = config.get("val_interval", 500)
    solve_val = config.get("solve_val", False)

    total_it = 0

    for epoch in range(config["num_epochs"]):
        running_loss = 0.0
        start_epoch = time.perf_counter()

        for it, data in enumerate(train_loader):
            total_it += 1
            model.train()

            start = time.perf_counter()
            data = data.to(device)

            if not is_fft_mode(data):
                continue

            data_f32 = cast_data_f32(data)

            # Get FFT matvec for this geometry
            fft_mv = fft_cache.get_or_build(data, device)

            # Probe loss: ||M(A*z) - z||² / ||z||²
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                if model_type == 'poly':
                    coefficients = model.encode_geometry(data_f32)
                    loss = poly_precond_probe_loss(model, coefficients, fft_mv,
                                                   num_probes=num_probes)
                else:
                    cache = model.encode_geometry(data_f32)
                    loss = precond_probe_loss(model, cache, fft_mv,
                                              num_probes=num_probes)

            scaler.scale(loss).backward()
            running_loss += loss.item()

            if grad_clip > 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip
                )
            else:
                grad_norm = 0.0

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            logger.log(loss.item(),
                       grad_norm if isinstance(grad_norm, float) else grad_norm.item(),
                       time.perf_counter() - start)

            # Periodic logging
            if total_it % 50 == 0:
                print(f"  [it {total_it}] loss={loss.item():.6f} "
                      f"grad_norm={grad_norm if isinstance(grad_norm, float) else grad_norm.item():.4f}")

            # Periodic validation
            if (total_it + 1) % val_interval == 0:
                if solve_val:
                    val_metric = validate_bicgstab(
                        model, val_loader, device, fft_cache=fft_cache,
                        model_type=model_type)
                else:
                    val_metric = validate(
                        model, val_loader, device,
                        num_probes=num_probes, fft_cache=fft_cache,
                        model_type=model_type)

                scheduler.step(val_metric)
                logger.log_val(None, val_metric)

                if val_metric < best_val:
                    if config["save"]:
                        torch.save(model.state_dict(),
                                   os.path.join(config['folder'], "best_model.pt"))
                    best_val = val_metric

        epoch_time = time.perf_counter() - start_epoch

        if config["save"]:
            torch.save(model.state_dict(),
                       os.path.join(config['folder'], f"model_epoch{epoch+1}.pt"))

        print(f"Epoch {epoch+1}\t loss: {running_loss / max(len(train_loader), 1):.6f}\t "
              f"time: {epoch_time:.1f}s")

    # Cleanup
    fft_cache.clear()

    if config["save"]:
        logger.save_results()
        torch.save(model.state_dict(),
                   os.path.join(config['folder'], "final_model.pt"))

    print(f"\nBest validation metric: {best_val}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NeuralPrecond model")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--save", action='store_true')

    # Training
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gradient_clipping", type=float, default=1.0)
    parser.add_argument("--num_probes", type=int, default=10,
                        help="Number of random probe vectors per step")
    parser.add_argument("--val_interval", type=int, default=500,
                        help="Validate every N iterations")
    parser.add_argument("--solve_val", action='store_true',
                        help="Use BiCGStab iterations for validation (slower)")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--fft_cache_size", type=int, default=50,
                        help="Max FFTMatVec objects to cache on GPU")

    # Model
    parser.add_argument("--model_type", type=str, default="neural",
                        choices=["neural", "poly"],
                        help="Model type: 'neural' (NeuralPrecond) or 'poly' (PolyPrecond)")
    parser.add_argument("--poly_degree", type=int, default=3,
                        help="Polynomial degree K for PolyPrecond")
    parser.add_argument("--latent_size", type=int, default=64,
                        help="Encoder latent dimension L")
    parser.add_argument("--latent_size_apply", type=int, default=32,
                        help="Apply-phase latent dimension La (NeuralPrecond only)")
    parser.add_argument("--message_passing_steps", type=int, default=6,
                        help="Encoder processor MP steps")
    parser.add_argument("--apply_mp_steps", type=int, default=1,
                        help="Apply-phase MP steps (NeuralPrecond only)")
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--checkpoint", action='store_true',
                        help="Enable gradient checkpointing for large grids")
    parser.add_argument("--amp", action='store_true',
                        help="Enable automatic mixed precision (float16)")
    parser.add_argument("--compile", action='store_true',
                        help="torch.compile the model (dynamic shapes)")

    # Data
    parser.add_argument("--data_dir", type=str, default="./data/SAI_v2/",
                        help="Root data directory with train/val/test subdirs")
    parser.add_argument("--max_nodes", type=int, default=None,
                        help="Max n (3*N_dipoles) to include in training")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker processes for async data loading")

    args = parser.parse_args()

    if args.device is None:
        device = "cpu"
        print("Warning: Using CPU training. Use --device <id> for GPU.")
    else:
        device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available()
                              else "cpu")

    if args.name is not None:
        folder = "results/" + args.name
    else:
        folder = "results/" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    print(f"Using device: {device}")
    config = vars(args)
    config['device'] = device
    config['folder'] = folder
    pprint.pprint(config)
    print()

    main(config)
