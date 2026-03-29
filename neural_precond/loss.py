"""Loss functions for neural preconditioner training.

Three families of loss:

1. **Probe loss** (original): ||M·A·z - z||² / ||z||²
   Measures average quality of M ≈ A⁻¹ on random vectors.
   Fast but doesn't correlate well with BiCGStab convergence.

2. **Differentiable BiCGStab loss** (new): unroll K iterations of BiCGStab,
   minimize log(||r_hat_K||² / ||r_hat_0||²).
   Directly optimizes what we care about — convergence speed.

3. **Spectral loss** (new): minimize spectral radius of (I - M·A) via
   power iteration. Lower spectral radius = faster Krylov convergence.
"""
import torch


# ---------------------------------------------------------------------------
# Probe losses (original)
# ---------------------------------------------------------------------------

def precond_probe_loss(model, cache, fft_matvec, num_probes=10):
    """Compute probe-based preconditioner loss: ||M(A*z) - z||² / ||z||².

    For P random complex probe vectors z:
      1. w = A*z (FFT, no grad, complex128)
      2. M(w) via model.apply_precond (with grad, float32)
      3. loss = ||M(w) - z||² / ||z||²

    Args:
        model: NeuralPrecond instance (must be in train mode for gradients)
        cache: dict from model.encode_geometry() — geometry cache
        fft_matvec: FFTMatVec instance for computing A*z
        num_probes: number of random probe vectors P

    Returns:
        loss: scalar float32 — average normalized probe loss
    """
    n = cache['node_cache'].shape[0]  # number of DOF nodes (= 3*N_dipoles)
    device = cache['node_cache'].device

    # Random complex probe vectors (float32 for efficiency)
    z_re = torch.randn(n, num_probes, device=device, dtype=torch.float32)
    z_im = torch.randn(n, num_probes, device=device, dtype=torch.float32)

    # w = A*z via FFT (no gradients, high precision)
    with torch.no_grad():
        z_complex = torch.complex(z_re.double(), z_im.double())
        # FFTMatVec handles (3N, P) directly
        w = fft_matvec(z_complex.to(torch.complex128))  # (3N, P)
        w_re = w.real.float()  # (n, P)
        w_im = w.imag.float()  # (n, P)

    # Apply preconditioner M to each probe: M(w_p) for p=1..P
    total_loss = torch.tensor(0.0, device=device)

    for p in range(num_probes):
        # M(w_p)
        out_re, out_im = model.apply_precond(cache, w_re[:, p], w_im[:, p])

        # Residual: M(A*z) - z
        res_re = out_re - z_re[:, p]
        res_im = out_im - z_im[:, p]

        # ||residual||²
        res_norm_sq = res_re.pow(2).sum() + res_im.pow(2).sum()

        # ||z||²
        z_norm_sq = z_re[:, p].pow(2).sum() + z_im[:, p].pow(2).sum()

        total_loss = total_loss + res_norm_sq / (z_norm_sq + 1e-8)

    return total_loss / num_probes


def poly_precond_probe_loss(model, coefficients, fft_matvec, num_probes=10):
    """Compute probe-based loss for polynomial preconditioner.

    For P random complex probe vectors z:
      1. w = A·z (FFT, no grad, complex128)
      2. h = p(A)·w via Horner (with grad through coefficients)
      3. loss = ||h - z||² / ||z||²

    Gradients flow through the polynomial coefficients c_0...c_K only.
    FFTMatVec calls are exact arithmetic — no approximation error.

    The Horner evaluation is batched: FFTMatVec handles (n, P) directly,
    so all probes are processed in K matvec calls total.

    Args:
        model: PolyPrecond instance (unused in apply, but kept for API consistency)
        coefficients: (K+1,) complex — from model.encode_geometry(), WITH grad
        fft_matvec: FFTMatVec instance for computing A·z
        num_probes: number of random probe vectors P

    Returns:
        loss: scalar — average normalized probe loss
    """
    from neural_precond.model import PolyPrecond

    # Infer n from fft_matvec
    n = fft_matvec.n  # 3 * N_dipoles
    device = coefficients.device

    # Random complex probe vectors z: (n, P)
    z_re = torch.randn(n, num_probes, device=device, dtype=torch.float32)
    z_im = torch.randn(n, num_probes, device=device, dtype=torch.float32)
    z = torch.complex(z_re.double(), z_im.double()).to(torch.complex128)

    # w = A·z (no grad — A is fixed)
    with torch.no_grad():
        w = fft_matvec(z)  # (n, P) complex128

    # h = p(A)·w via Horner — grad flows through coefficients
    h = PolyPrecond.apply_poly(coefficients, fft_matvec, w)  # (n, P)

    # Target: z (cast to same dtype as h for comparison)
    z_target = z.to(h.dtype)

    # Residual: h - z
    residual = h - z_target  # (n, P)

    # ||residual||² / ||z||² per probe, then average
    res_norm_sq = (residual.real.pow(2) + residual.imag.pow(2)).sum(dim=0)  # (P,)
    z_norm_sq = (z_target.real.pow(2) + z_target.imag.pow(2)).sum(dim=0)    # (P,)

    loss = (res_norm_sq / (z_norm_sq + 1e-8)).mean()

    return loss


def conv_sai_probe_loss(model, kernel, fft_matvec, num_probes=5):
    """Compute probe-based SAI loss for ConvSAI_MLP.

    loss = E_z[ ||M·A·z - z||² / ||z||² ]

    M is applied via FFT convolution with the learned kernel.
    Gradients flow through: kernel → M_hat → result → loss.

    A·z is computed without gradients (A is fixed physics).
    The FFT of the kernel (build_M_hat) is differentiable via out-of-place scatter.

    Args:
        model: ConvSAI_MLP instance (for build_M_hat method)
        kernel: (n_stencil, 3, 3) complex — from model.forward(), WITH grad
        fft_matvec: FFTMatVec instance
        num_probes: number of random probe vectors

    Returns:
        loss: scalar — average normalized probe loss
    """
    N = fft_matvec.N
    n = fft_matvec.n
    device = kernel[0].device if isinstance(kernel, (list, tuple)) else kernel.device

    # Build M_hat from kernel (differentiable)
    M_hat = model.build_M_hat(kernel, fft_matvec)  # (3, 3, gx, gy, gz)

    box = fft_matvec.box
    gx = 2 * box[0].item()
    gy = 2 * box[1].item()
    gz = 2 * box[2].item()
    pos = fft_matvec.pos_shifted
    pi, pj, pk = pos[:, 0], pos[:, 1], pos[:, 2]

    # Random probes z: (n, P) complex128
    z_re = torch.randn(n, num_probes, device=device, dtype=torch.float32)
    z_im = torch.randn(n, num_probes, device=device, dtype=torch.float32)
    z = torch.complex(z_re.double(), z_im.double()).to(torch.complex128)

    # w = A·z (no grad — A is fixed)
    with torch.no_grad():
        w = fft_matvec(z)  # (n, P) complex128

    # Scatter w to grid (no grad through w)
    w_reshaped = w.reshape(N, 3, num_probes)  # (N, 3, P)
    w_grid = torch.zeros(3, num_probes, gx, gy, gz,
                         dtype=torch.complex128, device=device)
    w_grid[:, :, pi, pj, pk] = w_reshaped.permute(1, 2, 0)  # (3, P, N)

    w_hat = torch.fft.fftn(w_grid, dim=(2, 3, 4))  # (3, P, gx, gy, gz)

    # M·w via FFT convolution (grad flows through M_hat → kernel)
    # result_hat[i, p, x, y, z] = sum_j M_hat[i, j, x, y, z] * w_hat[j, p, x, y, z]
    M_hat_c128 = M_hat.to(torch.complex128)
    result_hat = torch.einsum('ijxyz,jpxyz->ipxyz', M_hat_c128, w_hat)

    result_grid = torch.fft.ifftn(result_hat, dim=(2, 3, 4))  # (3, P, gx, gy, gz)

    # Gather from dipole positions
    Mw = result_grid[:, :, pi, pj, pk]  # (3, P, N)
    Mw = Mw.permute(2, 0, 1).reshape(n, num_probes)  # (n, P)

    # Loss: ||M·A·z - z||² / ||z||²
    z_target = z.to(Mw.dtype)
    residual = Mw - z_target
    res_norm_sq = (residual.real.pow(2) + residual.imag.pow(2)).sum(dim=0)  # (P,)
    z_norm_sq = (z_target.real.pow(2) + z_target.imag.pow(2)).sum(dim=0)    # (P,)

    loss = (res_norm_sq / (z_norm_sq + 1e-8)).mean()

    return loss


# ---------------------------------------------------------------------------
# Helper: apply M·v via FFT convolution (differentiable through M_hat)
# ---------------------------------------------------------------------------

def _apply_M_conv(M_hat, v, fft_matvec):
    """Apply preconditioner M·v via FFT convolution.

    Differentiable through M_hat (for training the kernel).
    v is treated as input data — no gradient through v needed.

    Args:
        M_hat: (3, 3, gx, gy, gz) complex — FFT of kernel, WITH grad
        v: (n,) complex128 — input vector
        fft_matvec: FFTMatVec instance (for grid geometry)

    Returns:
        (n,) complex128 — M·v
    """
    N = fft_matvec.N
    n = fft_matvec.n
    box = fft_matvec.box
    gx = 2 * box[0].item()
    gy = 2 * box[1].item()
    gz = 2 * box[2].item()
    pos = fft_matvec.pos_shifted
    pi, pj, pk = pos[:, 0], pos[:, 1], pos[:, 2]
    device = M_hat.device

    # Scatter v to grid: (n,) → (N, 3) → (3, N) → grid
    v_3 = v.reshape(N, 3).T  # (3, N)
    v_grid = torch.zeros(3, gx, gy, gz, dtype=torch.complex128, device=device)
    v_grid[:, pi, pj, pk] = v_3

    # FFT → multiply by M_hat → IFFT
    v_hat = torch.fft.fftn(v_grid, dim=(1, 2, 3))
    result_hat = torch.einsum('ijxyz,jxyz->ixyz', M_hat.to(torch.complex128), v_hat)
    result_grid = torch.fft.ifftn(result_hat, dim=(1, 2, 3))

    # Gather from grid → flatten
    return result_grid[:, pi, pj, pk].T.reshape(n)


# ---------------------------------------------------------------------------
# Differentiable BiCGStab loss
# ---------------------------------------------------------------------------

def conv_sai_bicgstab_loss(model, kernel, fft_matvec, num_iters=30, num_rhs=2):
    """Differentiable BiCGStab loss for ConvSAI preconditioner.

    Unrolls num_iters of left-preconditioned BiCGStab and minimizes the
    log-ratio of final to initial preconditioned residual norms:

        loss = log(||r_hat_K||² / ||r_hat_0||²)

    Lower loss = faster convergence = better preconditioner.
    At perfect convergence in K steps, loss → -inf.

    Gradients flow through M·v applications at each iteration
    (via M_hat → kernel → MLP). A·v is detached (fixed physics).

    Memory: ~2 FFT convolutions stored per iteration × num_iters.
    For grid=12: ~180 MB for 30 iterations. For grid=24: ~1.2 GB.

    Args:
        model: ConvSAI_MLP instance
        kernel: (n_stencil, 3, 3) complex — from model.forward(), WITH grad
        fft_matvec: FFTMatVec instance
        num_iters: number of BiCGStab iterations to unroll (default 30)
        num_rhs: number of random right-hand sides to average over

    Returns:
        loss: scalar — log residual reduction (lower is better)
    """
    n = fft_matvec.n
    device = kernel[0].device if isinstance(kernel, (list, tuple)) else kernel.device

    # Build M_hat once (differentiable through kernel)
    M_hat = model.build_M_hat(kernel, fft_matvec)

    total_loss = torch.tensor(0.0, device=device)
    eps = 1e-30

    for _ in range(num_rhs):
        # Random normalized RHS
        b = torch.randn(n, dtype=torch.complex128, device=device)
        b = b / torch.linalg.vector_norm(b)

        # Initial: x=0, r=b
        r = b.clone()

        # Preconditioned initial residual (grad through M)
        r_hat = _apply_M_conv(M_hat, r, fft_matvec)
        r_hat_0_norm_sq = (r_hat.real.pow(2).sum()
                           + r_hat.imag.pow(2).sum()).detach()

        # Shadow residual — fixed, detached (standard BiCGStab choice)
        r_tilde = r_hat.detach().clone()

        # BiCGStab scalars
        rho = torch.tensor(1.0, dtype=torch.complex128, device=device)
        alpha = torch.tensor(1.0, dtype=torch.complex128, device=device)
        omega = torch.tensor(1.0, dtype=torch.complex128, device=device)

        v_vec = torch.zeros(n, dtype=torch.complex128, device=device)
        p = torch.zeros(n, dtype=torch.complex128, device=device)

        for _ in range(num_iters):
            # rho_new = <r_tilde, r_hat>
            rho_new = torch.dot(r_tilde.conj(), r_hat)
            if rho_new.abs().item() < eps:
                break  # breakdown

            beta = (rho_new / (rho + eps)) * (alpha / (omega + eps))
            p = r_hat + beta * (p - omega * v_vec)

            # v = M · (A · p) — A detached, M with grad
            with torch.no_grad():
                Ap = fft_matvec(p.unsqueeze(1)).squeeze(1)
            v_vec = _apply_M_conv(M_hat, Ap, fft_matvec)

            sigma = torch.dot(r_tilde.conj(), v_vec)
            if sigma.abs().item() < eps:
                break
            alpha = rho_new / sigma

            s = r_hat - alpha * v_vec

            # t = M · (A · s) — A detached, M with grad
            with torch.no_grad():
                As = fft_matvec(s.unsqueeze(1)).squeeze(1)
            t = _apply_M_conv(M_hat, As, fft_matvec)

            tt = torch.dot(t.conj(), t)
            if tt.abs().item() < eps:
                break
            omega = torch.dot(t.conj(), s) / tt

            # Update preconditioned residual
            r_hat = s - omega * t
            rho = rho_new

        # Loss: log ratio of final to initial preconditioned residual
        r_hat_K_norm_sq = r_hat.real.pow(2).sum() + r_hat.imag.pow(2).sum()
        loss = torch.log(r_hat_K_norm_sq / (r_hat_0_norm_sq + eps) + eps)
        total_loss = total_loss + loss

    return total_loss / num_rhs


# ---------------------------------------------------------------------------
# Spectral loss via power iteration
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Batched helpers (differentiable)
# ---------------------------------------------------------------------------

def _apply_M_batched(M_hat, v, fft_matvec):
    """Apply preconditioner M to batched vectors v: (n, P) -> (n, P).

    Differentiable through M_hat. v can also carry gradients.
    """
    N = fft_matvec.N
    n = fft_matvec.n
    P = v.shape[1] if v.dim() > 1 else 1
    if v.dim() == 1:
        v = v.unsqueeze(1)

    box = fft_matvec.box
    gx = 2 * box[0].item()
    gy = 2 * box[1].item()
    gz = 2 * box[2].item()
    pos = fft_matvec.pos_shifted
    pi, pj, pk = pos[:, 0], pos[:, 1], pos[:, 2]
    device = M_hat.device

    v_reshaped = v.reshape(N, 3, P)
    v_grid = torch.zeros(3, P, gx, gy, gz, dtype=torch.complex128, device=device)
    v_grid[:, :, pi, pj, pk] = v_reshaped.permute(1, 2, 0)

    v_hat = torch.fft.fftn(v_grid, dim=(2, 3, 4))

    M_hat_c128 = M_hat.to(torch.complex128)
    result_hat = torch.einsum('ijxyz,jpxyz->ipxyz', M_hat_c128, v_hat)

    result_grid = torch.fft.ifftn(result_hat, dim=(2, 3, 4))

    result = result_grid[:, :, pi, pj, pk]  # (3, P, N)
    return result.permute(2, 0, 1).reshape(n, P)


def _apply_A_batched(v, fft_matvec):
    """Apply A·v for batched vectors: (n, P) -> (n, P).

    Differentiable through v. Uses out-of-place ops for autograd safety.
    """
    N = fft_matvec.N
    n = fft_matvec.n
    P = v.shape[1] if v.dim() > 1 else 1
    if v.dim() == 1:
        v = v.unsqueeze(1)

    box = fft_matvec.box
    gx = 2 * box[0].item()
    gy = 2 * box[1].item()
    gz = 2 * box[2].item()
    pos = fft_matvec.pos_shifted
    pi, pj, pk = pos[:, 0], pos[:, 1], pos[:, 2]
    device = v.device

    v_reshaped = v.reshape(N, 3, P)
    v_grid = torch.zeros(3, P, gx, gy, gz, dtype=torch.complex128, device=device)
    v_grid[:, :, pi, pj, pk] = v_reshaped.permute(1, 2, 0)

    v_hat = torch.fft.fftn(v_grid, dim=(2, 3, 4))

    D = fft_matvec.D_hat  # (6, gx, gy, gz)
    # Out-of-place: avoid in-place assignment issues
    y0 = D[0].unsqueeze(0)*v_hat[0] + D[1].unsqueeze(0)*v_hat[1] + D[2].unsqueeze(0)*v_hat[2]
    y1 = D[1].unsqueeze(0)*v_hat[0] + D[3].unsqueeze(0)*v_hat[1] + D[4].unsqueeze(0)*v_hat[2]
    y2 = D[2].unsqueeze(0)*v_hat[0] + D[4].unsqueeze(0)*v_hat[1] + D[5].unsqueeze(0)*v_hat[2]
    y_hat = torch.stack([y0, y1, y2], dim=0)  # (3, P, gx, gy, gz)

    y_grid = torch.fft.ifftn(y_hat, dim=(2, 3, 4))
    conv = y_grid[:, :, pi, pj, pk].permute(2, 0, 1).reshape(n, P)

    return v - fft_matvec.alpha * conv


# ---------------------------------------------------------------------------
# Adversarial probe loss
# ---------------------------------------------------------------------------

def conv_sai_adversarial_probe_loss(model, kernel, fft_matvec,
                                     num_probes=5, adversarial_iters=10):
    """Adversarial probe loss: find worst-case z via power iteration on (I-MA).

    Phase 1 (NO grad): power iteration finds z that maximizes ||MAz - z||/||z||.
    Phase 2 (WITH grad): compute probe loss on these worst-case vectors.

    Focuses training on the worst eigenmodes of (I - MA), unlike random probes
    which optimize the average (Frobenius norm).

    Args:
        model: ConvSAI_MLP instance
        kernel: (n_stencil, 3, 3) complex — from model.forward(), WITH grad
        fft_matvec: FFTMatVec instance
        num_probes: number of adversarial vectors
        adversarial_iters: power iteration steps to find worst-case z

    Returns:
        loss: scalar — probe loss on adversarial vectors
    """
    n = fft_matvec.n
    device = kernel[0].device if isinstance(kernel, (list, tuple)) else kernel.device

    M_hat = model.build_M_hat(kernel, fft_matvec)

    # Phase 1: Find adversarial vectors via power iteration (NO grad)
    with torch.no_grad():
        M_hat_det = M_hat.detach()
        z = torch.randn(n, num_probes, dtype=torch.complex128, device=device)
        z = z / torch.linalg.vector_norm(z, dim=0, keepdim=True)

        for _ in range(adversarial_iters):
            # w = (I - MA)·z
            Az = fft_matvec(z)
            MAz = _apply_M_batched(M_hat_det, Az, fft_matvec)
            w = z - MAz
            norms = torch.linalg.vector_norm(w, dim=0, keepdim=True)
            z = w / (norms + 1e-30)

    z = z.detach()

    # Phase 2: Compute probe loss on adversarial vectors (WITH grad)
    with torch.no_grad():
        w = fft_matvec(z)  # A·z, no grad

    # M·(A·z) with grad through M_hat
    Mw = _apply_M_batched(M_hat, w, fft_matvec)

    # Loss: ||MA·z - z||² / ||z||²
    z_target = z.to(Mw.dtype)
    residual = Mw - z_target
    res_norm_sq = (residual.real.pow(2) + residual.imag.pow(2)).sum(dim=0)
    z_norm_sq = (z_target.real.pow(2) + z_target.imag.pow(2)).sum(dim=0)

    loss = (res_norm_sq / (z_norm_sq + 1e-8)).mean()
    return loss


# ---------------------------------------------------------------------------
# Right preconditioning probe loss
# ---------------------------------------------------------------------------

def conv_sai_right_probe_loss(model, kernel, fft_matvec, num_probes=5):
    """Right preconditioning probe loss: ||A·M·z - z||² / ||z||².

    For right preconditioning, we want A·M ≈ I (vs left: M·A ≈ I).
    Right-preconditioned BiCGStab solves A·M·y = b, then x = M·y.

    May work better than left preconditioning for nonsymmetric systems
    where left and right spectra differ significantly.

    Gradient flow: z → M(z) [through M_hat] → A(Mz) [through Mz → M_hat].
    """
    n = fft_matvec.n
    device = kernel[0].device if isinstance(kernel, (list, tuple)) else kernel.device

    M_hat = model.build_M_hat(kernel, fft_matvec)

    # Random probes z: (n, P) complex128 (no grad)
    z_re = torch.randn(n, num_probes, device=device, dtype=torch.float32)
    z_im = torch.randn(n, num_probes, device=device, dtype=torch.float32)
    z = torch.complex(z_re.double(), z_im.double()).to(torch.complex128)

    # Step 1: M·z (grad through M_hat)
    Mz = _apply_M_batched(M_hat, z, fft_matvec)  # (n, P)

    # Step 2: A·(M·z) (grad flows through Mz → M_hat)
    AMz = _apply_A_batched(Mz, fft_matvec)  # (n, P)

    # Loss: ||A·M·z - z||² / ||z||²
    z_target = z.to(AMz.dtype)
    residual = AMz - z_target
    res_norm_sq = (residual.real.pow(2) + residual.imag.pow(2)).sum(dim=0)
    z_norm_sq = (z_target.real.pow(2) + z_target.imag.pow(2)).sum(dim=0)

    loss = (res_norm_sq / (z_norm_sq + 1e-8)).mean()
    return loss


# ---------------------------------------------------------------------------
# GMRES loss (unrolled Arnoldi process)
# ---------------------------------------------------------------------------

def conv_sai_gmres_loss(model, kernel, fft_matvec,
                         gmres_iters=10, num_rhs=2):
    """Differentiable GMRES loss for ConvSAI preconditioner.

    Unrolls K steps of left-preconditioned GMRES (Arnoldi process) and
    minimizes the final residual norm via least squares on the Hessenberg matrix.

    GMRES is more stable than BiCGStab for backpropagation:
    - Residual decreases monotonically (no erratic oscillations)
    - No divisions by potentially small scalars (rho, omega)
    - Well-conditioned least squares problem

    loss = log(||r_K|| / ||r_0||)

    Gradients flow through M applications at each Arnoldi step.
    A·v is detached (fixed physics). Memory: ~1 M application per Arnoldi step.

    Args:
        model: ConvSAI_MLP instance
        kernel: (n_stencil, 3, 3) complex — from model.forward(), WITH grad
        fft_matvec: FFTMatVec instance
        gmres_iters: number of Arnoldi steps K (default 10)
        num_rhs: number of random RHS to average over

    Returns:
        loss: scalar — log residual reduction (lower is better)
    """
    n = fft_matvec.n
    device = kernel[0].device if isinstance(kernel, (list, tuple)) else kernel.device

    M_hat = model.build_M_hat(kernel, fft_matvec)

    total_loss = torch.tensor(0.0, device=device)
    counted = 0
    k = gmres_iters

    for _ in range(num_rhs):
        # Random normalized RHS
        b = torch.randn(n, dtype=torch.complex128, device=device)
        b = b / torch.linalg.vector_norm(b)

        # r0 = M·b (grad through M_hat)
        r0 = _apply_M_conv(M_hat, b.detach(), fft_matvec)
        beta = torch.linalg.vector_norm(r0)

        if beta.real.item() < 1e-30:
            continue

        # Arnoldi process — all H entries stored as Python lists (out-of-place)
        V = [r0 / beta]
        H_columns = []  # list of column tensors, built out-of-place

        k_actual = k
        for j in range(k):
            # w = M · (A · v_j)  — A detached, M with grad
            with torch.no_grad():
                Avj = fft_matvec(V[j].detach().unsqueeze(1)).squeeze(1)
            w = _apply_M_conv(M_hat, Avj, fft_matvec)

            # Modified Gram-Schmidt — collect entries in Python list
            h_col_entries = []
            for i in range(j + 1):
                h_ij = torch.dot(V[i].conj(), w)
                h_col_entries.append(h_ij)
                w = w - h_ij * V[i]

            h_norm = torch.linalg.vector_norm(w)
            h_col_entries.append(h_norm)

            # Pad column to k+1 entries with zeros
            zero = torch.tensor(0.0, dtype=torch.complex128, device=device)
            while len(h_col_entries) < k + 1:
                h_col_entries.append(zero)

            H_columns.append(torch.stack(h_col_entries))  # out-of-place

            if h_norm.real.item() < 1e-14:
                k_actual = j + 1
                break

            V.append(w / h_norm)

        # Build H_k out-of-place: (k_actual+1, k_actual)
        H_k = torch.stack(H_columns[:k_actual], dim=1)[:k_actual + 1, :]

        # Solve least squares: min ||beta * e1 - H_k @ y||
        e1 = torch.cat([beta.unsqueeze(0),
                         torch.zeros(k_actual, dtype=torch.complex128, device=device)])

        y = torch.linalg.lstsq(H_k, e1.unsqueeze(1)).solution.squeeze(1)

        # Residual in Hessenberg space
        residual_vec = e1 - H_k @ y
        res_norm = torch.linalg.vector_norm(residual_vec)

        # Loss: log ratio (detach beta in denominator)
        loss = torch.log(res_norm / (beta.detach() + 1e-30) + 1e-30)
        total_loss = total_loss + loss
        counted += 1

    if counted == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return total_loss / counted


def conv_sai_spectral_loss(model, kernel, fft_matvec,
                           num_power_iters=20, num_vectors=3):
    """Spectral loss: minimize spectral radius of (I - M·A).

    The convergence rate of any Krylov method is bounded by the spectral
    radius rho(I - M·A). Lower rho → faster convergence.

    Algorithm:
      1. Power iteration (WITHOUT grad) to find the dominant eigenvector
         of (I - M·A) — the worst-case direction for the preconditioner.
      2. One final application of (I - M·A) WITH grad through M.
      3. Loss = ||result||² ≈ |lambda_max|² (spectral radius squared).

    This focuses the gradient on reducing the WORST eigenvalue, unlike
    probe loss which optimizes the average (Frobenius norm).

    Args:
        model: ConvSAI_MLP instance
        kernel: (n_stencil, 3, 3) complex — from model.forward(), WITH grad
        fft_matvec: FFTMatVec instance
        num_power_iters: iterations of power method (more = better eigenvalue estimate)
        num_vectors: number of independent starting vectors (finds multiple eigenvalues)

    Returns:
        loss: scalar — estimated spectral radius squared of (I - M·A)
    """
    n = fft_matvec.n
    device = kernel[0].device if isinstance(kernel, (list, tuple)) else kernel.device

    M_hat = model.build_M_hat(kernel, fft_matvec)

    total_loss = torch.tensor(0.0, device=device)

    for _ in range(num_vectors):
        # Random unit starting vector
        v = torch.randn(n, dtype=torch.complex128, device=device)
        v = v / torch.linalg.vector_norm(v)

        # Power iteration: find dominant eigenvector of (I - M·A)
        # All WITHOUT gradients — just finding the worst direction
        with torch.no_grad():
            M_hat_detached = M_hat.detach()
            for _ in range(num_power_iters):
                # w = (I - M·A)·v
                Av = fft_matvec(v.unsqueeze(1)).squeeze(1)
                MAv = _apply_M_conv(M_hat_detached, Av, fft_matvec)
                w = v - MAv

                w_norm = torch.linalg.vector_norm(w)
                if w_norm.item() < 1e-30:
                    break
                v = w / w_norm

        # Final application WITH grad through M_hat
        # v is the approximate dominant eigenvector (detached)
        v = v.detach()
        with torch.no_grad():
            Av = fft_matvec(v.unsqueeze(1)).squeeze(1)
        MAv = _apply_M_conv(M_hat, Av, fft_matvec)  # grad through M_hat
        w = v - MAv  # (I - M·A)·v

        # ||w||² ≈ |lambda_max|² — spectral radius squared
        spectral_sq = w.real.pow(2).sum() + w.imag.pow(2).sum()
        total_loss = total_loss + spectral_sq

    return total_loss / num_vectors
