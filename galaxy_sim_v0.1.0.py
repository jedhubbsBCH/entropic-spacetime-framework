#!/usr/bin/env python3
"""
galactic_simulation.py

2‑D Galactic Disc Simulation in the Entropic Spacetime Framework
================================================================

Version 2.4.2 — fixed unmatched brackets in plotting section; snapshots save as
PNG (dpi = 120, tight bbox); AMP via `torch.amp.autocast`.
"""

import os, math, numpy as np, torch
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# AMP setup (mixed precision)
# ---------------------------------------------------------------------------
torch.set_default_dtype(torch.float32)
scaler = torch.cuda.amp.GradScaler(enabled=True)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
nx = ny = 512
Lx = Ly = 50.0          # kpc

dt      = 0.01
n_steps = 80_000
output_snaps = 15

# baryonic exponential disc
disk_on   = True
Sigma0    = 50.0
R_d       = 3.0
G_const   = 1.0

# oscillatory coupling
osc_coupling_on = False
alpha_osc = 0.5
R_wave    = 6.0

# entropic‑field coeffs
aT, bT, lambdaT = 0.0,  1.0, 0.1
aS, bS, lambdaS = 0.0, -1.0, 0.1
kappa_base = 0.05

# ---------------------------------------------------------------------------
# GRIDS & DEVICE
# ---------------------------------------------------------------------------
dx, dy = Lx/nx, Ly/ny
xv = torch.linspace(-Lx/2, Lx/2, nx)
yv = torch.linspace(-Ly/2, Ly/2, ny)
X, Y = torch.meshgrid(xv, yv, indexing='ij')
R = torch.sqrt(X**2 + Y**2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on {device}")

base_out = 'output'
run_id   = sum(d.startswith(base_out) for d in os.listdir())
output_dir = f"{base_out}_{run_id:02d}"
os.makedirs(output_dir, exist_ok=True)
print(f"Snapshots → {output_dir}")

# ---------------------------------------------------------------------------
# FIELD INITIALISATION
# ---------------------------------------------------------------------------
S_T = torch.zeros(1, 1, nx, ny, device=device)
S_S = torch.zeros(1, 1, nx, ny, device=device)

sigma0 = 5.0
S_S += torch.exp(-(X**2 + Y**2)/(2*sigma0**2)).to(device).unsqueeze(0).unsqueeze(0)
S_T += 1e-3 * torch.randn_like(S_T)

v_T = torch.zeros_like(S_T)
v_S = torch.zeros_like(S_S)

# ---------------------------------------------------------------------------
# FFT LAPLACIAN
# ---------------------------------------------------------------------------
kx = torch.fft.fftfreq(nx, d=dx, device=device)*2*math.pi
ky = torch.fft.fftfreq(ny, d=dy, device=device)*2*math.pi
KX, KY = torch.meshgrid(kx, ky, indexing='ij')
K2 = -(KX**2 + KY**2)

def laplacian(field: torch.Tensor) -> torch.Tensor:
    fhat = torch.fft.fftn(field.squeeze(0).squeeze(0))
    lap  = torch.fft.ifftn(fhat*K2).real
    return lap.unsqueeze(0).unsqueeze(0)

# ---------------------------------------------------------------------------
# COUPLING FUNCTION (linear in S_T)
# ---------------------------------------------------------------------------
def coupling_function(ST, SS, Rgrid):
    k_eff = kappa_base * (1 + torch.exp(-Rgrid/10))
    if osc_coupling_on:
        k_eff *= 1 + alpha_osc*torch.sin(Rgrid/R_wave)
    return k_eff * ST * SS**2

# ---------------------------------------------------------------------------
# STELLAR DISC FORCE
# ---------------------------------------------------------------------------
try:
    import scipy.special as sc
except ImportError:
    sc = None

def K0_torch(x):
    if hasattr(torch.special, 'k0'):
        return torch.special.k0(x)
    if sc is None:
        raise RuntimeError('SciPy required for Bessel k0')
    return torch.from_numpy(sc.k0(x.cpu().numpy())).to(x.device)

def K1_torch(x):
    if hasattr(torch.special, 'k1'):
        return torch.special.k1(x)
    if sc is None:
        raise RuntimeError('SciPy required for Bessel k1')
    return torch.from_numpy(sc.k1(x.cpu().numpy())).to(x.device)

def dPhi_disk_dR(Rgrid):
    if not disk_on:
        return torch.zeros_like(Rgrid)
    x = Rgrid/(2*R_d) + 1e-12
    I0, I1 = torch.special.i0(x), torch.special.i1(x)
    K0, K1 = K0_torch(x), K1_torch(x)
    v2 = 4*math.pi*G_const*Sigma0*R_d*x**2*(I0*K0 - I1*K1)
    return v2/(Rgrid+1e-6)

# ---------------------------------------------------------------------------
# POTENTIAL & GRADIENTS
# ---------------------------------------------------------------------------
def potential(ST, SS):
    return (aT*ST + bT*ST**2 + lambdaT*ST**4 +
            aS*SS + bS*SS**2 + lambdaS*SS**4 +
            coupling_function(ST, SS, R.to(device)))

def potential_derivs(ST, SS):
    dT = aT + 2*bT*ST + 4*lambdaT*ST**3
    dS = aS + 2*bS*SS + 4*lambdaS*SS**3
    if disk_on:
        dS += dPhi_disk_dR(R.to(device)).unsqueeze(0).unsqueeze(0)
    ST_r, SS_r = ST.requires_grad_(True), SS.requires_grad_(True)
    mix = coupling_function(ST_r, SS_r, R.to(device)).sum()
    gT, gS = torch.autograd.grad(mix, (ST_r, SS_r))
    ST.requires_grad_(False); SS.requires_grad_(False)
    return dT + gT, dS + gS

# ---------------------------------------------------------------------------
# ROTATION CURVE
# ---------------------------------------------------------------------------
Xcpu, Ycpu = X.cpu().numpy(), Y.cpu().numpy()

def compute_rotation_curve(SS):
    rho = np.clip(SS.squeeze().cpu().numpy(), 0, None)
    radii = np.linspace(0.5, min(nx, ny)//2, 100)
    v = []
    for r in radii:
        m = rho[(Xcpu**2 + Ycpu**2) <= (r*dx)**2].sum()
        v.append(math.sqrt(m/(r*dx + 1e-6)))
    return radii*dx, v

# ---------------------------------------------------------------------------
# SNAPSHOT SCHEDULE
# ---------------------------------------------------------------------------
exp_steps = np.unique(np.geomspace(1, n_steps, output_snaps, dtype=int))
lin_steps = np.linspace(286, 582, 30, dtype=int)
snapshot_steps = sorted(set(np.concatenate([exp_steps, lin_steps])))

# ---------------------------------------------------------------------------
# MAIN INTEGRATION LOOP
# ---------------------------------------------------------------------------
for step in range(1, n_steps+1):
    with torch.amp.autocast(device_type='cuda'):
        lap_T = laplacian(S_T)
        lap_S = laplacian(S_S)
        dVdT, dVdS = potential_derivs(S_T, S_S)
        v_T += dt * (lap_T - dVdT)
        v_S += dt * (lap_S - dVdS)
        S_T += dt * v_T
        S_S += dt * v_S

    if step in snapshot_steps:
        idx = snapshot_steps.index(step)

        ST2 = S_T.squeeze().cpu().numpy()
        SS2 = S_S.squeeze().cpu().numpy()
        V2  = potential(S_T, S_S).squeeze().cpu().numpy()
        SC  = ST2 + coupling_function(S_T, S_S, R.to(device)).squeeze().cpu().numpy()

        fig, axs = plt.subplots(3, 4, figsize=(20, 12), gridspec_kw={'height_ratios':[1,1,0.8]})

        # Row 0: fields & potential
        im0 = axs[0,0].imshow(ST2, origin='lower', extent=[-Lx/2, Lx/2, -Ly/2, Ly/2],
                             vmin=ST2.min(), vmax=ST2.max())
        axs[0,0].set_title(f"S_T step {step}")
        fig.colorbar(im0, ax=axs[0,0], fraction=0.046, pad=0.04)

        im1 = axs[0,1].imshow(SS2, origin='lower',
                              extent=[-Lx/2, Lx/2, -Ly/2, Ly/2],
                              vmin=SS2.min(), vmax=SS2.max())
        axs[0,1].set_title(f"S_S step {step}")
        fig.colorbar(im1, ax=axs[0,1], fraction=0.046, pad=0.04)

        im2 = axs[0,2].imshow(SC, origin='lower',
                              extent=[-Lx/2, Lx/2, -Ly/2, Ly/2])
        axs[0,2].set_title("S_T + coupling")
        fig.colorbar(im2, ax=axs[0,2], fraction=0.046, pad=0.04)

        im3 = axs[0,3].imshow(V2, origin='lower',
                              extent=[-Lx/2, Lx/2, -Ly/2, Ly/2])
        axs[0,3].set_title("Potential V")
        fig.colorbar(im3, ax=axs[0,3], fraction=0.046, pad=0.04)

        # ─── Row 1: 2-D FFT of S_S, radial PSD of S_S, rotation curve ───
        fft_SS = np.fft.fftshift(np.fft.fft2(SS2))
        P2_SS  = np.abs(fft_SS)**2
        im4 = axs[1,0].imshow(np.log(P2_SS + 1e-6), origin='lower')
        axs[1,0].set_title("log-FFT  S_S")
        fig.colorbar(im4, ax=axs[1,0], fraction=0.046, pad=0.04)

        ny2, nx2 = P2_SS.shape
        yk = np.arange(ny2) - ny2//2
        xk = np.arange(nx2) - nx2//2
        Xk, Yk = np.meshgrid(xk, yk)
        Rk = np.sqrt(Xk**2 + Yk**2).astype(int)
        tbin = np.bincount(Rk.ravel(), P2_SS.ravel())
        nr   = np.bincount(Rk.ravel())
        radial_SS = tbin / (nr + 1e-6)
        axs[1,1].plot(radial_SS)
        axs[1,1].set_title("Radial PSD  S_S")

        radii, vrot = compute_rotation_curve(S_S)
        axs[1,2].plot(radii, vrot)
        axs[1,2].set_title("Rotation curve")
        axs[1,2].set_xlabel("R [kpc]")
        axs[1,2].set_ylabel("v [arb]")

        axs[1,3].axis('off')  # leave last cell blank

        # ─── Row 2: 2-D FFT of S_T and its radial PSD ───
        fft_ST = np.fft.fftshift(np.fft.fft2(ST2))
        P2_ST  = np.abs(fft_ST)**2
        im5 = axs[2,0].imshow(np.log(P2_ST + 1e-6), origin='lower')
        axs[2,0].set_title("log-FFT  S_T")
        fig.colorbar(im5, ax=axs[2,0], fraction=0.046, pad=0.04)

        tbinT = np.bincount(Rk.ravel(), P2_ST.ravel())
        radial_ST = tbinT / (nr + 1e-6)
        axs[2,1].plot(radial_ST)
        axs[2,1].set_title("Radial PSD  S_T")

        axs[2,2].axis('off')
        axs[2,3].axis('off')

        # ─── finalise snapshot ───
        plt.tight_layout()
        fig.savefig(f"{output_dir}/snap_{idx:03d}.png",
                    dpi=120, bbox_inches='tight')
        plt.close(fig)

        print(f"[{step:>6}/{n_steps}]  snapshot {idx+1}/{len(snapshot_steps)} saved")

print("Simulation complete.")

