import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import math


def _center_unit(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    Mc = M - M.mean(axis=0, keepdims=True)
    n  = np.linalg.norm(Mc, axis=0, keepdims=True)
    n  = np.maximum(n, eps)
    return Mc / n

def _unit(M, eps=1e-12):
    n = np.linalg.norm(M, axis=0, keepdims=True)
    return M / np.maximum(n, eps)

def metricA_subspace_score(F: np.ndarray, G: np.ndarray) -> tuple[float, np.ndarray]:
    Fn = _center_unit(F)
    Gn = _center_unit(G)
    QF, _ = np.linalg.qr(Fn, mode='reduced')
    QG, _ = np.linalg.qr(Gn, mode='reduced')
    Svals = np.linalg.svd(QF.T @ QG, compute_uv=False)
    r_eff = Svals.shape[0]
    S = float(np.mean(Svals[:r_eff]**2))
    return S, Svals

def metricA_subspace_score_sym(F: np.ndarray, G: np.ndarray) -> tuple[float, np.ndarray]:
    Fn = _center_unit(F)
    Gn = _center_unit(G)

    QF, _ = np.linalg.qr(Fn, mode='reduced')
    QG, _ = np.linalg.qr(Gn, mode='reduced')

    r_eff = QF.shape[1]   # rank(F)
    s_eff = QG.shape[1]   # rank(G)

    Svals = np.linalg.svd(QF.T @ QG, compute_uv=False)

    frob_sq = float(np.sum(Svals**2))

    S_sym = frob_sq / np.sqrt(r_eff * s_eff)

    return S_sym, Svals


@torch.no_grad()
def vi_poisson(model, res: int = 256, outdir: str = "expert_vis"):

    os.makedirs(outdir, exist_ok=True)
    # exp  = model.experts[0].eval()
    exp  = model.eval()
    dev  = next(exp.parameters()).device

    x = torch.linspace(0., 1., res, device=dev).unsqueeze(-1)

    f1_vals = exp._eval_dim(x, 0).cpu().numpy()  
    f2_vals = exp._eval_dim(x, 1).cpu().numpy()  
    f3_vals = exp._eval_dim(x, 2).cpu().numpy()  
    f4_vals = exp._eval_dim(x, 3).cpu().numpy()  
    f5_vals = exp._eval_dim(x, 4).cpu().numpy() 
    

    xs = np.linspace(0, 1, res)
    target = np.sin(np.pi * xs)

    Fx1 = f1_vals
    Fx2 = f2_vals
    Fx3 = f3_vals
    Fx4 = f4_vals
    Fx5 = f5_vals

    Gx = target[:, None]

    Sx1, svals_x1 = metricA_subspace_score_sym(Fx1, Gx)
    Sx2, svals_x2 = metricA_subspace_score_sym(Fx2, Gx)
    Sx3, svals_x3 = metricA_subspace_score_sym(Fx3, Gx)
    Sx4, svals_x4 = metricA_subspace_score_sym(Fx4, Gx)
    Sx5, svals_x5 = metricA_subspace_score_sym(Fx5, Gx)

    S_all = 0.2 * (Sx1 + Sx2 + Sx3 + Sx4 + Sx5)

    print(f"[metric-A] S_avg={S_all:.4f}")
    return S_all

@torch.no_grad()
def vi_wave(model, res: int = 256, outdir: str = "expert_vis", c: int = 2, ep: int = 9999):

    os.makedirs(outdir, exist_ok=True)
    exp  = model.eval()
    dev  = next(exp.parameters()).device

    t = torch.linspace(0., 1., res, device=dev).unsqueeze(-1)           # (res,1)
    x = torch.linspace(0., 1, res, device=dev).unsqueeze(-1)

    ft_vals = exp._eval_dim(t, 0).cpu().numpy()                                 # (res,2)
    fx_vals = exp._eval_dim(x, 1).cpu().numpy()
    ft_vals = ft_vals.reshape(res, -1)    # (res, r_t)
    fx_vals = fx_vals.reshape(res, -1)

    xs = np.linspace(0, 1, res)
    ts = np.linspace(0, 1, res)
    c = c
    target_x = np.sin(np.pi*xs)  
    target_t = np.cos(ts * c * np.pi)

    Fx = fx_vals                          
    Ft = ft_vals                          
    Gx = target_x[:, None]                
    Gt = target_t[:, None]

    Sx, svals_x = metricA_subspace_score(Fx, Gx)
    St, svals_t = metricA_subspace_score(Ft, Gt)
    S_all = 0.5 * (Sx + St)

    print(f"[metric-A] S_avg={S_all:.4f}")
    return S_all


@torch.no_grad()
def vi_wave_2d(model, res: int = 256, outdir: str = "expert_vis", c: int = 2, ep: int = 9999):

    os.makedirs(outdir, exist_ok=True)
    exp  = model.eval()
    dev  = next(exp.parameters()).device

    t = torch.linspace(0., 1., res, device=dev).unsqueeze(-1)           # (res,1)
    x = torch.linspace(0., 1, res, device=dev).unsqueeze(-1)
    y = torch.linspace(0., 1, res, device=dev).unsqueeze(-1)

    ft_vals = exp._eval_dim(t, 0).cpu().numpy()                                 # (res,2)
    fx_vals = exp._eval_dim(x, 1).cpu().numpy()
    fy_vals = exp._eval_dim(y, 2).cpu().numpy()
    ft_vals = ft_vals.reshape(res, -1)    # (res, r_t)
    fx_vals = fx_vals.reshape(res, -1)
    fy_vals = fy_vals.reshape(res, -1)

    xs = np.linspace(0, 1, res)
    ts = np.linspace(0, 1, res)
    c = c
    target_x = np.sin(np.pi*xs)  
    target_t = np.cos(np.sqrt(2) * ts * c * np.pi)

    Fx = fx_vals   
    Fy = fy_vals                       
    Ft = ft_vals                          
    Gx = target_x[:, None]                
    Gt = target_t[:, None]

    Sx, svals_x = metricA_subspace_score(Fx, Gx)
    Sy, svals_y = metricA_subspace_score(Fy, Gx)
    St, svals_t = metricA_subspace_score(Ft, Gt)
    S_all = (Sx + St + Sy) / 3.0

    print(f"[metric-A] S_avg={S_all:.4f}")
    return S_all


X_MIN, X_MAX = 0.0, 1.0
T_MIN, T_MAX = 0.0, 1.0
L = 1.0
DIM = 2

# -----------------------------
# PDE parameters
# -----------------------------
V_ADV = 1.0      # advection speed v
D_DIFF = 0.01    # diffusion coefficient D

LAMBDA0 = 0.02
DLAMBDA = 20.0
ALPHA = 50.0
X1, X2 = 0.45, 0.55   # transition locations for lambda(x)

X1_WIN = 0.35
X2_WIN = 0.65
ALPHA_WIN = 30.0      # window sharpness
A_HI = 0.25           # amplitude of local high-frequency component
K_HI = 12

# Exact-solution temporal parameters
GAMMA = 0.2
OMEGA = 2.0 * math.pi

# --- new: x-dependent temporal frequency (piecewise) ---
OMEGA1 = 2.0 * math.pi * 1.0     # left region frequency
OMEGA2 = 2.0 * math.pi * 4.0     # right region frequency
X_IFACE = 0.5
K_IFACE = 5000.0  # sharpness of the transition (larger => sharper)

# u_eq parameters
BETA = 40.0

U_L, U_R = 2.0, 1.0       # left/right states
EPS = 0.02                # interface thickness (smaller -> sharper)
X0 = 0.35                 # initial interface location
C_FRONT = V_ADV

@torch.no_grad()
def vi_ADR(model, res: int = 256, outdir: str = "expert_vis", ep: int = 9999):

    os.makedirs(outdir, exist_ok=True)
    VI = 0
    for i in range(2):
        exp  = model.experts[i].eval()
        dev  = next(exp.parameters()).device

        t = torch.linspace(0., 1., res, device=dev).unsqueeze(-1)           # (res,1)
        x = torch.linspace(0., 1, res, device=dev).unsqueeze(-1)

        Ft = exp._eval_dim(t, 0).cpu().numpy()  
        Fx = exp._eval_dim(x, 1).cpu().numpy()   

        ft_vals = Ft.reshape(res, -1)    # (res, r_t)
        fx_vals = Fx.reshape(res, -1)

        xs = np.linspace(0, 1, res).reshape(-1, 1)
        ts = np.linspace(0, 1, res).reshape(-1, 1)
        
        s = 1.0 / (1.0 + np.exp(-K_IFACE * (xs - X_IFACE)))
        w = 0.5 * (np.tanh(ALPHA_WIN * (xs - X1_WIN)) - np.tanh(ALPHA_WIN * (xs - X2_WIN)))  # (res,1)
        A = (np.sin(np.pi * xs)
            + 0.3 * np.sin(3.0 * np.pi * xs)
            + 0.1 * np.sin(5.0 * np.pi * xs)
            + A_HI * w * np.sin(2.0 * np.pi * K_HI * xs))
        f1 = np.exp(-GAMMA * ts) * np.cos(OMEGA1 * ts)
        f2 = np.exp(-GAMMA * ts) * np.cos(OMEGA2 * ts)

        g1 = A * (1.0 - s)
        g2 = A * s

        Gt = np.concatenate([f1, f2], axis=1)
        Gx = np.concatenate([g1, g2], axis=1)

        St, svals_t = metricA_subspace_score(Ft, Gt)
        Sx, svals_x = metricA_subspace_score(Fx, Gx)
        
        S_all = 0.5 * (Sx + St)
        VI = VI+S_all

        print(f"[metric-A] S_avg={S_all:.4f}")

    return VI / 2

@torch.no_grad()
def vi_Unsteady_ADR(model, res: int = 256, outdir: str = "expert_vis", ep: int = 9999):

    os.makedirs(outdir, exist_ok=True)
    VI = []
    for i in range(2):
        exp  = model.experts[i].eval()
        dev  = next(exp.parameters()).device

        t = torch.linspace(0., 1., res, device=dev).unsqueeze(-1)           # (res,1)
        x = torch.linspace(0., 1, res, device=dev).unsqueeze(-1)

        Ft = exp._eval_dim(t, 0).cpu().numpy()  
        Fx = exp._eval_dim(x, 1).cpu().numpy()   

        xs = np.linspace(0, 1, res)[:, None]
        ts = np.linspace(0, 1, res)[:, None]
        
        A_t = np.exp(-GAMMA * ts) * np.cos(OMEGA * ts)
        phi_x = np.tanh((xs - X0) / EPS)

        Gt = A_t            # (res,1)
        Gx = phi_x

        St, svals_t = metricA_subspace_score(Ft, Gt)
        Sx, svals_x = metricA_subspace_score(Fx, Gx)
        
        S_all = 0.5 * (Sx + St)
        VI.append(S_all)

        print(f"[metric-A] S_avg={S_all:.4f}")

    return VI