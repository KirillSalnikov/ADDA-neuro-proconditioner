"""Training script for MLP-based preconditioners (no graphs needed).

Supports two model types:
  - PolyPrecondMLP: MLP → polynomial coefficients → Horner apply (K matvecs)
  - ConvSAI_MLP: MLP → convolution kernel → FFT apply (1 matvec)

No dataset files needed — parameters are sampled on the fly.
No graph construction — only positions + FFTMatVec.

For each training step:
  1. Sample random physical parameters (m_re, m_im, kd, shape, grid)
  2. Generate dipole positions (no graph building)
  3. Build FFTMatVec (lightweight on small grids)
  4. MLP forward: params → preconditioner representation
  5. Probe loss: ||M·A·z - z||² / ||z||²
  6. Backward + optimizer step

Usage:
    python -m neural_precond.train_mlp --model_type conv_sai --save --name sai_v1 --device 0
    python -m neural_precond.train_mlp --model_type mlp --save --name poly_v1 --device 0
"""
import os
import sys
import datetime
import argparse
import pprint
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.fft_matvec import FFTMatVec
from core.utils import count_parameters, save_dict_to_file
from core.logger import TrainResults

from apps.adda_matrix import (
    make_sphere_dipoles, make_cube_dipoles, make_ellipsoid_dipoles,
    make_cylinder_dipoles, make_capsule_dipoles,
)
from neural_precond.model import PolyPrecondMLP, PolyPrecond, ConvSAI_MLP
from neural_precond.loss import (
    poly_precond_probe_loss, conv_sai_probe_loss,
    conv_sai_bicgstab_loss, conv_sai_spectral_loss,
)


# Shape name → function mapping
SHAPE_GENERATORS = {
    0: make_sphere_dipoles,    # sphere
    1: make_cube_dipoles,      # cube
    2: make_ellipsoid_dipoles, # ellipsoid
    3: make_cylinder_dipoles,  # cylinder
    4: make_capsule_dipoles,   # capsule
}
SHAPE_NAMES = {0: 'sphere', 1: 'cube', 2: 'ellipsoid', 3: 'cylinder', 4: 'capsule'}


def sample_parameters(rng, config, step=None, num_steps=None):
    """Sample random physical parameters for one training step.

    With curriculum learning: grid range grows linearly from grid_min
    to grid_max over the first curriculum_frac of training.

    Returns:
        m_re, m_im, kd: float — physical parameters
        shape_id: int — shape index
        grid: int — grid size
    """
    m_re = rng.uniform(config['m_re_min'], config['m_re_max'])
    m_im = rng.uniform(config['m_im_min'], config['m_im_max'])
    # Log-uniform sampling for kd: equal weight to each decade
    log_kd_min = np.log(config['kd_min'])
    log_kd_max = np.log(config['kd_max'])
    kd = float(np.exp(rng.uniform(log_kd_min, log_kd_max)))
    shape_id = rng.randint(0, config['num_shapes'])

    grid_min = config['grid_min']
    grid_max = config['grid_max']

    # Curriculum: ramp grid_max over first curriculum_frac of training
    curriculum_frac = config.get('curriculum_frac', 0.5)
    if step is not None and num_steps is not None and curriculum_frac > 0:
        progress = min(step / (num_steps * curriculum_frac), 1.0)
        current_grid_max = int(grid_min + (grid_max - grid_min) * progress)
        current_grid_max = max(current_grid_max, grid_min)
    else:
        current_grid_max = grid_max

    grid = rng.randint(grid_min, current_grid_max + 1)
    return m_re, m_im, kd, shape_id, grid


def build_fft_matvec(shape_id, grid, m_re, m_im, kd, device):
    """Generate dipole positions and build FFTMatVec (no graph needed).

    Args:
        shape_id: int — 0=sphere, 1=cube, 2=ellipsoid
        grid: int — grid size
        m_re, m_im: float — refractive index components
        kd: float — size parameter
        device: torch device

    Returns:
        fft_mv: FFTMatVec instance
    """
    d = 1.0
    k = kd / d
    m = complex(m_re, m_im)

    gen_fn = SHAPE_GENERATORS[shape_id]
    positions = gen_fn(grid)

    fft_mv = FFTMatVec(positions, k, m, d=d, device=device)
    return fft_mv


@torch.no_grad()
def validate(model, rng, config, device, num_val_steps=50):
    """Validate by measuring probe loss on random configurations."""
    model.eval()
    total_loss = 0.0
    model_type = config.get('model_type', 'mlp')

    # Cap validation grid to avoid OOM on large grids
    val_config = dict(config)
    val_config['grid_max'] = min(config.get('grid_max', 24), 60)

    counted = 0
    for _ in range(num_val_steps):
        m_re, m_im, kd, shape_id, grid = sample_parameters(rng, val_config)
        try:
            fft_mv = build_fft_matvec(shape_id, grid, m_re, m_im, kd, device)

            n = fft_mv.n
            num_probes = config.get('num_probes', 10)

            z_re = torch.randn(n, num_probes, device=device, dtype=torch.float32)
            z_im = torch.randn(n, num_probes, device=device, dtype=torch.float32)
            z = torch.complex(z_re.double(), z_im.double()).to(torch.complex128)

            w = fft_mv(z)  # A·z

            if model_type == 'conv_sai':
                kernel = model(m_re, m_im, kd, shape_id, grid)
                M_hat = model.build_M_hat(kernel, fft_mv)
                # Apply M·w via FFT convolution
                N_dip = fft_mv.N
                box = fft_mv.box
                gx, gy, gz = 2*box[0].item(), 2*box[1].item(), 2*box[2].item()
                pos = fft_mv.pos_shifted
                pi, pj, pk = pos[:, 0], pos[:, 1], pos[:, 2]

                w_reshaped = w.reshape(N_dip, 3, num_probes)
                w_grid = torch.zeros(3, num_probes, gx, gy, gz,
                                     dtype=torch.complex128, device=device)
                w_grid[:, :, pi, pj, pk] = w_reshaped.permute(1, 2, 0)
                w_hat = torch.fft.fftn(w_grid, dim=(2, 3, 4))

                M_hat_c128 = M_hat.to(torch.complex128)
                result_hat = torch.einsum('ijxyz,jpxyz->ipxyz', M_hat_c128, w_hat)
                result_grid = torch.fft.ifftn(result_hat, dim=(2, 3, 4))
                h = result_grid[:, :, pi, pj, pk].permute(2, 0, 1).reshape(n, num_probes)
            else:
                coefficients = model(m_re, m_im, kd, shape_id, grid)
                h = PolyPrecond.apply_poly(coefficients.to(torch.complex128), fft_mv, w)

            residual = h - z.to(h.dtype)
            res_norm_sq = (residual.real.pow(2) + residual.imag.pow(2)).sum(dim=0)
            z_norm_sq = (z.real.pow(2) + z.imag.pow(2)).sum(dim=0)

            loss = ((res_norm_sq / (z_norm_sq + 1e-8)).mean()).item()
            total_loss += loss
            counted += 1
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue

    avg_loss = total_loss / max(counted, 1)
    print(f"Validation loss:\t{avg_loss:.6f}")
    return avg_loss


@torch.no_grad()
def validate_bicgstab(model, rng, config, device, num_val_steps=5,
                      rtol=1e-5, max_iter=2000):
    """Validate by running BiCGStab with and without MLP preconditioner.

    Returns inverse speedup (iter_precond / iter_unprecond) as the metric
    to minimize. Values < 0.5 mean >2x iteration speedup (wall-clock gain
    with FFT apply). Lower is better.
    """
    from krylov.bicgstab import bicgstab

    model.eval()
    model_type = config.get('model_type', 'mlp')
    total_iters_precond = 0
    total_iters_unprecond = 0

    # Use moderate grids for validation (balance cost vs relevance)
    val_config = dict(config)
    val_config['grid_min'] = max(config.get('grid_min', 6), 10)
    val_config['grid_max'] = min(config.get('grid_max', 24), 20)

    counted = 0
    for _ in range(num_val_steps):
        m_re, m_im, kd, shape_id, grid = sample_parameters(rng, val_config)
        try:
            fft_mv_cpu = build_fft_matvec(shape_id, grid, m_re, m_im, kd, 'cpu')
        except Exception:
            continue
        n = fft_mv_cpu.n

        def A_op(v):
            v2d = v.unsqueeze(1) if v.dim() == 1 else v
            return fft_mv_cpu(v2d).squeeze(1)

        b = torch.randn(n, dtype=torch.complex128) + \
            1j * torch.randn(n, dtype=torch.complex128)
        b = b / torch.linalg.vector_norm(b)

        if model_type == 'conv_sai':
            kernel = model(m_re, m_im, kd, shape_id, grid)
            precond_fn = model.make_precond_fn(kernel, fft_mv_cpu)
        else:
            coefficients = model(m_re, m_im, kd, shape_id, grid)
            precond_fn = model.make_precond_fn(coefficients, fft_mv_cpu)

        res_p, _ = bicgstab(A_op, b, M=precond_fn, rtol=rtol, max_iter=max_iter)
        res_u, _ = bicgstab(A_op, b, rtol=rtol, max_iter=max_iter)

        total_iters_precond += len(res_p) - 1
        total_iters_unprecond += len(res_u) - 1
        counted += 1

    if counted == 0:
        print("Validation BiCGStab\t no valid configurations")
        return 1.0

    avg_precond = total_iters_precond / counted
    avg_unprecond = total_iters_unprecond / counted
    speedup = avg_unprecond / max(avg_precond, 1)
    inv_speedup = avg_precond / max(avg_unprecond, 1)
    print(f"Validation BiCGStab\t precond: {avg_precond:.1f} iters\t "
          f"unprecond: {avg_unprecond:.1f} iters\t speedup: {speedup:.2f}x")
    return inv_speedup


def main(config):
    device = config['device']

    if config["save"]:
        os.makedirs(config['folder'], exist_ok=True)
        config_save = {k: (str(v) if isinstance(v, torch.device) else v)
                       for k, v in config.items()}
        save_dict_to_file(config_save, os.path.join(config['folder'], "config.json"))

    torch.manual_seed(config["seed"])
    rng = np.random.RandomState(config["seed"])

    # Create model
    model_type = config.get('model_type', 'mlp')

    if model_type == 'conv_sai':
        model = ConvSAI_MLP(
            r_cut=config.get("r_cut", 3),
            hidden_size=config.get("hidden_size", 256),
            num_layers=config.get("num_layers", 4),
            num_shapes=config.get("num_shapes", 3),
            shape_embed_dim=config.get("shape_embed_dim", 8),
            activation=config.get("activation", "relu"),
            scale_by_stencil=config.get("scale_by_stencil", True),
        )
    else:
        model = PolyPrecondMLP(
            poly_degree=config.get("poly_degree", 3),
            hidden_size=config.get("hidden_size", 128),
            num_layers=config.get("num_layers", 4),
            num_shapes=config.get("num_shapes", 3),
            shape_embed_dim=config.get("shape_embed_dim", 8),
            activation=config.get("activation", "relu"),
        )

    if config.get("resume"):
        print(f"Loading weights from {config['resume']}")
        model.load_state_dict(torch.load(config["resume"], map_location="cpu",
                                         weights_only=True))

    model.to(device)
    print(f"Model: {type(model).__name__}")
    print(f"Number params: {count_parameters(model)}")
    if model_type == 'conv_sai':
        print(f"Stencil: r_cut={model.r_cut}, {model.n_stencil} displacements")
        print(f"Kernel output: {model.n_stencil} × 3×3 complex = {model.n_stencil * 18} real values")
    else:
        print(f"Polynomial degree K={model.poly_degree}")
    print(f"Hidden size: {model.hidden_size}, Num layers: {model.num_layers}")
    print(f"Shape embed dim: {model.shape_embed_dim}")
    print(f"Parameter ranges: m_re=[{config['m_re_min']}, {config['m_re_max']}], "
          f"m_im=[{config['m_im_min']}, {config['m_im_max']}], "
          f"kd=[{config['kd_min']}, {config['kd_max']}]")
    print(f"Grid range: [{config['grid_min']}, {config['grid_max']}]")
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                   weight_decay=config.get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=config.get("patience", 500)
    )

    best_val = float("inf")
    logger = TrainResults(config['folder'])

    num_probes = config.get("num_probes", 10)
    grad_clip = config.get("gradient_clipping", 1.0)
    val_interval = config.get("val_interval", 500)
    log_interval = config.get("log_interval", 50)
    solve_val = config.get("solve_val", False)
    num_steps = config.get("num_steps", 50000)
    loss_type = config.get("loss_type", "probe")
    bicgstab_iters = config.get("bicgstab_iters", 30)
    spectral_power_iters = config.get("spectral_power_iters", 20)
    spectral_weight = config.get("spectral_weight", 0.1)

    if loss_type != "probe":
        print(f"Loss type: {loss_type}")
        if loss_type == "bicgstab":
            print(f"  BiCGStab unroll iters: {bicgstab_iters}")
        elif loss_type == "spectral":
            print(f"  Power iters: {spectral_power_iters}")
        elif loss_type == "combined":
            print(f"  Probe + spectral (weight={spectral_weight})")
            print(f"  Power iters: {spectral_power_iters}")

    running_loss = 0.0
    start_total = time.perf_counter()

    for step in range(1, num_steps + 1):
        model.train()
        start = time.perf_counter()

        # 1. Sample random parameters (with curriculum for grid)
        m_re, m_im, kd, shape_id, grid = sample_parameters(
            rng, config, step=step, num_steps=num_steps)

        # 2. Build FFTMatVec (no graph!)
        fft_mv = build_fft_matvec(shape_id, grid, m_re, m_im, kd, device)

        # 3. MLP forward + loss
        if model_type == 'conv_sai':
            kernel = model(m_re, m_im, kd, shape_id, grid)

            if loss_type == "probe":
                loss = conv_sai_probe_loss(model, kernel, fft_mv,
                                           num_probes=num_probes)
            elif loss_type == "bicgstab":
                loss = conv_sai_bicgstab_loss(model, kernel, fft_mv,
                                              num_iters=bicgstab_iters,
                                              num_rhs=max(1, num_probes // 5))
            elif loss_type == "spectral":
                loss = conv_sai_spectral_loss(model, kernel, fft_mv,
                                              num_power_iters=spectral_power_iters,
                                              num_vectors=max(1, num_probes // 3))
            elif loss_type == "combined":
                loss_probe = conv_sai_probe_loss(model, kernel, fft_mv,
                                                 num_probes=num_probes)
                loss_spectral = conv_sai_spectral_loss(
                    model, kernel, fft_mv,
                    num_power_iters=spectral_power_iters,
                    num_vectors=max(1, num_probes // 3))
                loss = loss_probe + spectral_weight * loss_spectral
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")
        else:
            coefficients = model(m_re, m_im, kd, shape_id, grid)
            loss = poly_precond_probe_loss(model, coefficients, fft_mv,
                                            num_probes=num_probes)

        # 5. Backward + step
        loss.backward()
        running_loss += loss.item()

        if grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip
            )
        else:
            grad_norm = 0.0

        optimizer.step()
        optimizer.zero_grad()

        step_time = time.perf_counter() - start
        logger.log(loss.item(),
                   grad_norm if isinstance(grad_norm, float) else grad_norm.item(),
                   step_time)

        # Periodic logging
        if step % log_interval == 0:
            avg_loss = running_loss / log_interval
            elapsed = time.perf_counter() - start_total
            steps_per_sec = step / elapsed
            print(f"  [step {step}/{num_steps}] loss={avg_loss:.6f} "
                  f"grad_norm={grad_norm if isinstance(grad_norm, float) else grad_norm.item():.4f} "
                  f"lr={optimizer.param_groups[0]['lr']:.1e} "
                  f"steps/s={steps_per_sec:.1f} "
                  f"({SHAPE_NAMES[shape_id]} g{grid} m={m_re:.2f}+{m_im:.2f}i kd={kd:.2f})")
            running_loss = 0.0

        # Periodic validation
        if step % val_interval == 0:
            val_rng = np.random.RandomState(config["seed"] + step)
            if solve_val:
                val_metric = validate_bicgstab(
                    model, val_rng, config, device)
            else:
                val_metric = validate(
                    model, val_rng, config, device)

            scheduler.step(val_metric)
            logger.log_val(None, val_metric)

            if val_metric < best_val:
                if config["save"]:
                    torch.save(model.state_dict(),
                               os.path.join(config['folder'], "best_model.pt"))
                best_val = val_metric
                print(f"  >>> New best: {best_val:.6f}")

        # Periodic checkpoints
        if config["save"] and step % config.get("save_interval", 5000) == 0:
            torch.save(model.state_dict(),
                       os.path.join(config['folder'], f"model_step{step}.pt"))

    total_time = time.perf_counter() - start_total
    print(f"\nTraining complete: {num_steps} steps in {total_time:.1f}s "
          f"({num_steps / total_time:.1f} steps/s)")

    if config["save"]:
        logger.save_results()
        torch.save(model.state_dict(),
                   os.path.join(config['folder'], "final_model.pt"))

    print(f"Best validation metric: {best_val:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train MLP-based preconditioner (PolyPrecondMLP or ConvSAI_MLP)")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--save", action='store_true')
    parser.add_argument("--model_type", type=str, default="mlp",
                        choices=["mlp", "conv_sai"],
                        help="Model type: 'mlp' (polynomial) or 'conv_sai' (FFT convolution)")

    # Training
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_steps", type=int, default=50000,
                        help="Total training steps")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gradient_clipping", type=float, default=1.0)
    parser.add_argument("--num_probes", type=int, default=10,
                        help="Number of random probe vectors per step")
    parser.add_argument("--val_interval", type=int, default=500,
                        help="Validate every N steps")
    parser.add_argument("--log_interval", type=int, default=50,
                        help="Log every N steps")
    parser.add_argument("--save_interval", type=int, default=5000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--patience", type=int, default=500,
                        help="LR scheduler patience (in val intervals)")
    parser.add_argument("--solve_val", action='store_true',
                        help="Use BiCGStab iterations for validation (slower)")
    parser.add_argument("--loss_type", type=str, default="probe",
                        choices=["probe", "bicgstab", "spectral", "combined"],
                        help="Loss function: probe (default), bicgstab (unrolled), "
                             "spectral (power iteration), combined (probe+spectral)")
    parser.add_argument("--bicgstab_iters", type=int, default=30,
                        help="Number of BiCGStab iterations to unroll (bicgstab loss)")
    parser.add_argument("--spectral_power_iters", type=int, default=20,
                        help="Power iteration steps (spectral/combined loss)")
    parser.add_argument("--spectral_weight", type=float, default=0.1,
                        help="Weight of spectral loss in combined mode")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--curriculum_frac", type=float, default=0.5,
                        help="Fraction of training to ramp grid from grid_min to grid_max (0=no curriculum)")

    # Model architecture (shared)
    parser.add_argument("--hidden_size", type=int, default=None,
                        help="MLP hidden layer size (default: 128 for poly, 256 for conv_sai)")
    parser.add_argument("--num_layers", type=int, default=4,
                        help="Number of MLP layers")
    parser.add_argument("--num_shapes", type=int, default=3,
                        help="Number of shape types (sphere, cube, ellipsoid)")
    parser.add_argument("--shape_embed_dim", type=int, default=8,
                        help="Shape embedding dimension")
    parser.add_argument("--activation", type=str, default="relu",
                        choices=["relu", "gelu"])
    # PolyPrecondMLP specific
    parser.add_argument("--poly_degree", type=int, default=3,
                        help="Polynomial degree K (PolyPrecondMLP only)")
    # ConvSAI_MLP specific
    parser.add_argument("--r_cut", type=int, default=3,
                        help="Stencil radius in grid units (ConvSAI_MLP only)")
    parser.add_argument("--no_scale_by_stencil", action='store_true',
                        help="Use 1/sqrt(18) scaling instead of 1/sqrt(n_stencil*18) (for large r_cut)")

    # Sampling ranges (hard regime)
    parser.add_argument("--m_re_min", type=float, default=1.1)
    parser.add_argument("--m_re_max", type=float, default=2.5)
    parser.add_argument("--m_im_min", type=float, default=0.0)
    parser.add_argument("--m_im_max", type=float, default=0.1)
    parser.add_argument("--kd_min", type=float, default=0.3)
    parser.add_argument("--kd_max", type=float, default=1.5)
    parser.add_argument("--grid_min", type=int, default=6,
                        help="Minimum grid size for training")
    parser.add_argument("--grid_max", type=int, default=12,
                        help="Maximum grid size for training")

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
        folder = "results/mlp_" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    print(f"Using device: {device}")
    config = vars(args)
    config['device'] = device
    config['folder'] = folder

    # Set default hidden_size based on model type
    if config['hidden_size'] is None:
        config['hidden_size'] = 256 if config['model_type'] == 'conv_sai' else 128

    config['scale_by_stencil'] = not config.pop('no_scale_by_stencil', False)

    pprint.pprint(config)
    print()

    main(config)
