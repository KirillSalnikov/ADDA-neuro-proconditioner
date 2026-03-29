"""FFT-based matrix-vector product for DDA interaction matrices.

Computes A·z = z - alpha * G_conv(z) using FFT convolution on a doubled grid,
avoiding explicit storage of the dense N²-element interaction matrix.

For a grid of size (gx, gy, gz), the doubled grid is (2gx, 2gy, 2gz).
Memory usage: O(grid³) instead of O(N²) for dense storage.

Usage:
    fft_mv = FFTMatVec(positions, k, m, d, device='cuda')
    Az = fft_mv(z)  # z: (3N, P) complex, P = num probes
"""
import torch
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'apps'))
try:
    from apps.adda_matrix import ldr_polarizability
except ImportError:
    from adda_matrix import ldr_polarizability


class FFTMatVec:
    """FFT-based A·z computation for DDA interaction matrices.

    Precomputes D_hat = FFT(Green's tensor on doubled grid), then computes
    A·z = z - alpha * IFFT(D_hat * FFT(z_grid)) via convolution.

    Args:
        positions: (N, 3) integer grid positions of dipoles
        k: wavenumber
        m: complex refractive index
        d: interdipole spacing (default 1.0)
        device: torch device
    """

    def __init__(self, positions, k, m, d=1.0, device='cpu'):
        self.device = torch.device(device)
        self.k = k
        self.m = m
        self.d = d

        # Compute LDR polarizability
        alpha_np = ldr_polarizability(m, d, k)
        self.alpha = torch.tensor(alpha_np, dtype=torch.complex128, device=self.device)

        # Integer positions (N, 3)
        if isinstance(positions, torch.Tensor):
            self.positions = positions.long().to(self.device)
        else:
            self.positions = torch.tensor(positions, dtype=torch.long, device=self.device)

        self.N = self.positions.shape[0]
        self.n = 3 * self.N  # total DOFs

        # Determine bounding box
        pos_min = self.positions.min(dim=0).values  # (3,)
        pos_max = self.positions.max(dim=0).values  # (3,)
        self.box = (pos_max - pos_min + 1).cpu()  # (3,) — box size in each dim

        # Shift positions to start at 0
        self.pos_shifted = self.positions - pos_min.unsqueeze(0)  # (N, 3)

        # Doubled grid dimensions for circular convolution
        self.gx = 2 * self.box[0].item()
        self.gy = 2 * self.box[1].item()
        self.gz = 2 * self.box[2].item()

        # Build and precompute D_hat
        self._build_D_hat()

    def _build_D_hat(self):
        """Build FFT of Green's tensor on doubled grid.

        For displacements (di, dj, dk) in range [-(box-1), box-1],
        compute G(displacement * d, k) and store in doubled grid with
        negative index wrapping.

        D_hat shape: (6, gx, gy, gz) complex128
        Components: xx, xy, xz, yy, yz, zz (symmetric tensor → 6 unique)
        """
        bx, by, bz = self.box[0].item(), self.box[1].item(), self.box[2].item()
        gx, gy, gz = self.gx, self.gy, self.gz
        k = self.k
        d = self.d

        # All displacement indices: -(box-1) to (box-1) in each dim
        di_range = torch.arange(-(bx - 1), bx, device=self.device, dtype=torch.long)
        dj_range = torch.arange(-(by - 1), by, device=self.device, dtype=torch.long)
        dk_range = torch.arange(-(bz - 1), bz, device=self.device, dtype=torch.long)

        # Meshgrid of all displacements
        di, dj, dk = torch.meshgrid(di_range, dj_range, dk_range, indexing='ij')
        # Flatten for vectorized computation
        di_flat = di.reshape(-1)
        dj_flat = dj.reshape(-1)
        dk_flat = dk.reshape(-1)

        # Physical displacement vectors
        rx = di_flat.double() * d
        ry = dj_flat.double() * d
        rz = dk_flat.double() * d

        r = torch.sqrt(rx**2 + ry**2 + rz**2)

        # Mask for nonzero displacements
        nonzero = r > 1e-15

        # Initialize Green's tensor components (6 unique for symmetric 3x3)
        n_disp = di_flat.shape[0]
        G_xx = torch.zeros(n_disp, dtype=torch.complex128, device=self.device)
        G_xy = torch.zeros(n_disp, dtype=torch.complex128, device=self.device)
        G_xz = torch.zeros(n_disp, dtype=torch.complex128, device=self.device)
        G_yy = torch.zeros(n_disp, dtype=torch.complex128, device=self.device)
        G_yz = torch.zeros(n_disp, dtype=torch.complex128, device=self.device)
        G_zz = torch.zeros(n_disp, dtype=torch.complex128, device=self.device)

        # Compute Green's tensor for nonzero displacements
        r_nz = r[nonzero]
        rx_nz = rx[nonzero]
        ry_nz = ry[nonzero]
        rz_nz = rz[nonzero]

        # Unit vectors
        rhat_x = rx_nz / r_nz
        rhat_y = ry_nz / r_nz
        rhat_z = rz_nz / r_nz

        kr = k * r_nz

        # G(r) = exp(ikr)/r * [k²*(I - r̂r̂) + (1-ikr)/r² * (3r̂r̂ - I)]
        # NO 1/(4*pi) factor — matches ADDA's InterTerm_poi (DDA convention)
        phase = torch.exp(1j * kr) / r_nz

        # Coefficient for identity term: k² - (1-ikr)/r²
        factor_I = k**2 - (1 - 1j * kr) / r_nz**2

        # Coefficient for r̂r̂ term: -k² + 3(1-ikr)/r²
        factor_rr = -k**2 + 3 * (1 - 1j * kr) / r_nz**2

        # Outer product components r̂_μ r̂_ν
        rr_xx = rhat_x * rhat_x
        rr_xy = rhat_x * rhat_y
        rr_xz = rhat_x * rhat_z
        rr_yy = rhat_y * rhat_y
        rr_yz = rhat_y * rhat_z
        rr_zz = rhat_z * rhat_z

        # G_μν = phase * (factor_I * δ_μν + factor_rr * r̂_μ r̂_ν)
        G_xx[nonzero] = phase * (factor_I * 1.0 + factor_rr * rr_xx)
        G_xy[nonzero] = phase * (factor_I * 0.0 + factor_rr * rr_xy)
        G_xz[nonzero] = phase * (factor_I * 0.0 + factor_rr * rr_xz)
        G_yy[nonzero] = phase * (factor_I * 1.0 + factor_rr * rr_yy)
        G_yz[nonzero] = phase * (factor_I * 0.0 + factor_rr * rr_yz)
        G_zz[nonzero] = phase * (factor_I * 1.0 + factor_rr * rr_zz)

        # Place into doubled grid with negative index wrapping
        # Wrapped indices: negative indices wrap around
        wi = di_flat % gx
        wj = dj_flat % gy
        wk = dk_flat % gz

        # Allocate G_real on doubled grid: (6, gx, gy, gz)
        G_real = torch.zeros(6, gx, gy, gz, dtype=torch.complex128, device=self.device)

        # Scatter into grid
        G_real[0, wi, wj, wk] = G_xx
        G_real[1, wi, wj, wk] = G_xy
        G_real[2, wi, wj, wk] = G_xz
        G_real[3, wi, wj, wk] = G_yy
        G_real[4, wi, wj, wk] = G_yz
        G_real[5, wi, wj, wk] = G_zz

        # FFT to get D_hat
        self.D_hat = torch.fft.fftn(G_real, dim=(1, 2, 3))  # (6, gx, gy, gz)

    def matvec(self, z):
        """Compute A·z = z - alpha * G_conv(z) via FFT convolution.

        Args:
            z: (n, P) complex tensor, n = 3*N DOFs, P = number of probe vectors

        Returns:
            Az: (n, P) complex tensor
        """
        N = self.N
        P = z.shape[1] if z.dim() > 1 else 1
        if z.dim() == 1:
            z = z.unsqueeze(1)

        z = z.to(dtype=torch.complex128, device=self.device)

        gx, gy, gz = self.gx, self.gy, self.gz
        pos = self.pos_shifted  # (N, 3)

        # Scatter z into grid: z(3N, P) → z_grid(3, P, gx, gy, gz)
        z_reshaped = z.reshape(N, 3, P)  # (N, 3, P)
        z_grid = torch.zeros(3, P, gx, gy, gz, dtype=torch.complex128, device=self.device)

        # Advanced indexing to scatter
        pi = pos[:, 0]  # (N,)
        pj = pos[:, 1]
        pk = pos[:, 2]

        # z_grid[:, :, pi, pj, pk] = z_reshaped.permute(1, 2, 0)  → (3, P, N)
        z_grid[:, :, pi, pj, pk] = z_reshaped.permute(1, 2, 0)

        # FFT of z_grid
        z_hat = torch.fft.fftn(z_grid, dim=(2, 3, 4))  # (3, P, gx, gy, gz)

        # Multiply D_hat (6, gx, gy, gz) with z_hat (3, P, gx, gy, gz)
        # Symmetric matrix-vector: D is symmetric 3x3 stored as 6 components
        # D = [[xx, xy, xz],
        #      [xy, yy, yz],
        #      [xz, yz, zz]]
        D = self.D_hat  # (6, gx, gy, gz)

        y_hat = torch.zeros_like(z_hat)  # (3, P, gx, gy, gz)

        # y[0] = D_xx * z[0] + D_xy * z[1] + D_xz * z[2]
        y_hat[0] = D[0].unsqueeze(0) * z_hat[0] + D[1].unsqueeze(0) * z_hat[1] + D[2].unsqueeze(0) * z_hat[2]
        # y[1] = D_xy * z[0] + D_yy * z[1] + D_yz * z[2]
        y_hat[1] = D[1].unsqueeze(0) * z_hat[0] + D[3].unsqueeze(0) * z_hat[1] + D[4].unsqueeze(0) * z_hat[2]
        # y[2] = D_xz * z[0] + D_yz * z[1] + D_zz * z[2]
        y_hat[2] = D[2].unsqueeze(0) * z_hat[0] + D[4].unsqueeze(0) * z_hat[1] + D[5].unsqueeze(0) * z_hat[2]

        # IFFT
        y_grid = torch.fft.ifftn(y_hat, dim=(2, 3, 4))  # (3, P, gx, gy, gz)

        # Gather from grid positions
        conv = y_grid[:, :, pi, pj, pk]  # (3, P, N)
        conv = conv.permute(2, 0, 1).reshape(self.n, P)  # (3N, P)

        # A·z = z - alpha * conv
        result = z - self.alpha * conv

        return result

    def __call__(self, z):
        """Shorthand for matvec."""
        return self.matvec(z)
