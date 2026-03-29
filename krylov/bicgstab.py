"""Left-preconditioned BiCGStab for nonsymmetric complex systems.

BiCGStab (Bi-Conjugate Gradient Stabilized) works for general (non-symmetric)
linear systems. When left-preconditioned with M ≈ A⁻¹, it solves:
    M·A·x = M·b

This is needed for NeuralSAI because M·A is not complex symmetric,
so COCR/CSYM (which require complex symmetry) cannot be used.

Reference:
  van der Vorst, "Bi-CGSTAB: A Fast and Smoothly Converging Variant
  of Bi-CG for the Solution of Nonsymmetric Linear Systems", SIAM J.
  Sci. Stat. Comput., 1992.
"""
import torch


def _norm(v):
    """||v||_2 for real or complex vectors."""
    return torch.linalg.vector_norm(v)


def bicgstab(A, b, x0=None, M=None, rtol=1e-8, max_iter=100000):
    """Left-preconditioned BiCGStab.

    Solves A·x = b, optionally left-preconditioned as M·A·x = M·b.

    Args:
        A: matrix (sparse or dense) supporting A @ x, or callable
        b: right-hand side vector (complex or real)
        x0: initial guess (default: zero)
        M: left preconditioner — callable M(v) or sparse matrix.
           If None, no preconditioning (identity).
        rtol: relative tolerance ||r||/||b|| for stopping
        max_iter: maximum number of iterations

    Returns:
        residuals: list of relative residual norms at each iteration
        x: approximate solution
    """
    # Setup matrix-vector product
    if callable(A) and not hasattr(A, '__matmul__'):
        matvec_A = A
    else:
        matvec_A = lambda v: A @ v

    # Setup preconditioner
    if M is None:
        precond = lambda v: v
    elif callable(M) and not hasattr(M, '__matmul__'):
        precond = M
    else:
        precond = lambda v: M @ v

    # Initial guess
    x = x0.clone() if x0 is not None else torch.zeros_like(b)

    # Initial residual
    r = b - matvec_A(x)
    b_norm = _norm(b)
    if b_norm < 1e-30:
        return [0.0], x

    # Apply left preconditioner to initial residual
    r_hat = precond(r)

    # Choose r_tilde (shadow residual) — use preconditioned initial residual
    r_tilde = r_hat.clone()

    rho_old = torch.tensor(1.0, dtype=b.dtype, device=b.device)
    alpha = torch.tensor(1.0, dtype=b.dtype, device=b.device)
    omega = torch.tensor(1.0, dtype=b.dtype, device=b.device)

    v = torch.zeros_like(b)
    p = torch.zeros_like(b)

    residuals = [(_norm(r) / b_norm).item()]

    for iteration in range(max_iter):
        if residuals[-1] < rtol:
            break

        # rho = (r_tilde, r_hat)
        rho_new = torch.dot(r_tilde.conj().resolve_conj(), r_hat) if b.is_complex() \
            else torch.dot(r_tilde, r_hat)

        if torch.abs(rho_new) < 1e-30:
            break  # Breakdown

        # beta = (rho_new / rho_old) * (alpha / omega)
        beta = (rho_new / rho_old) * (alpha / omega)

        # p = r_hat + beta * (p - omega * v)
        p = r_hat + beta * (p - omega * v)

        # v = M · A · p
        v = precond(matvec_A(p))

        # alpha = rho_new / (r_tilde, v)
        sigma = torch.dot(r_tilde.conj().resolve_conj(), v) if b.is_complex() \
            else torch.dot(r_tilde, v)

        if torch.abs(sigma) < 1e-30:
            break  # Breakdown

        alpha = rho_new / sigma

        # s = r_hat - alpha * v (preconditioned half-step residual)
        s = r_hat - alpha * v

        # Check convergence at half-step
        # s corresponds to preconditioned residual; compute actual residual
        x_half = x + alpha * p
        r_half = b - matvec_A(x_half)
        res_half = (_norm(r_half) / b_norm).item()
        if res_half < rtol:
            x = x_half
            residuals.append(res_half)
            break

        # t = M · A · s
        # First compute A·s_unpreconditioned — but s is in preconditioned space
        # Actually in left-preconditioned BiCGStab, we work with preconditioned vectors
        t = precond(matvec_A(s))

        # omega = (t, s) / (t, t)
        if b.is_complex():
            ts = torch.dot(t.conj().resolve_conj(), s)
            tt = torch.dot(t.conj().resolve_conj(), t)
        else:
            ts = torch.dot(t, s)
            tt = torch.dot(t, t)

        if torch.abs(tt) < 1e-30:
            break  # Breakdown

        omega = ts / tt

        # Update solution
        x = x + alpha * p + omega * s

        # Update preconditioned residual
        r_hat = s - omega * t

        # Compute actual residual for stopping criterion
        r = b - matvec_A(x)
        res = (_norm(r) / b_norm).item()
        residuals.append(res)

        rho_old = rho_new

    return residuals, x
