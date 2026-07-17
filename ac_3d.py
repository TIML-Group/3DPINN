import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, LBFGS
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from model import MLP, DomainMoE
import time
import random
from numpy.fft import fft, ifft
import json


T = 1.0
t_eval = T
a, b = -1.0, 1.0
P = b - a
gamma1 = 1e-3
gamma2 = 5.0

device = "cuda" if torch.cuda.is_available() else "cpu"

def set_model_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def u0(x):
    return x**2 * np.cos(np.pi * x)

def solve_ac_reference(N=1024, T=1.0, dt=1/200,
                          gamma1=1e-3, gamma2=5.0):
    """
    Solve: u_t = gamma1 u_xx - gamma2 (u^3 - u)
         = gamma1 u_xx + gamma2 (u - u^3)
    periodic on [-1,1).
    """
    a, b = -1.0, 1.0
    L = b - a
    dx = L / N
    x = a + dx * np.arange(N)  # periodic grid, no endpoint duplication

    # physical wave numbers
    kx = 2*np.pi * np.fft.fftfreq(N, d=dx)   # cycles -> radians
    k2 = kx**2

    nsteps = int(round(T / dt))
    tgrid = dt * np.arange(nsteps + 1)

    u = np.zeros((nsteps + 1, N), dtype=np.float64)
    u[0, :] = u0(x)

    for n in range(nsteps):
        un = u[n, :]
        unhat = fft(un)
        un3hat = fft(un**3)

        rhs_hat = unhat + dt * gamma2 * (unhat - un3hat)
        denom = 1.0 + dt * gamma1 * k2
        uhat_next = rhs_hat / denom

        u[n+1, :] = np.real(ifft(uhat_next))

        # optional: early NaN check
        if not np.isfinite(u[n+1, :]).all():
            raise FloatingPointError(f"Non-finite at step {n+1}, t={tgrid[n+1]}")

    return tgrid, x, u


def u0_torch(x):
    return (x**2) * torch.cos(math.pi * x)

def psi(t, x, C):
    return torch.exp(-C * t) * u0_torch(x)

def phi(t, x):
    return t


class AC(nn.Module):
    def __init__(self, 
                 num_experts=3,
                 expert_hidden=32, expert_rank=16,
                 router_hidden=64, router_depth=2):
        super().__init__()
        self.moe = DomainMoE(
            in_features=2,
            num_experts=num_experts,
            expert_hidden=expert_hidden,
            expert_rank=expert_rank,
            router_hidden=router_hidden,
            router_depth=router_depth,
        )

    def set_reference(self, t_ref, x_ref, u_ref, C=1.0):
        # store on CPU for numpy interpolation
        self._ref = {
            "t": t_ref.detach().cpu().numpy().astype(np.float64),              # (nt+1,)
            "x": x_ref.detach().cpu().numpy().reshape(-1).astype(np.float64),  # (N,)
            "u": u_ref.detach().cpu().numpy().astype(np.float64),              # (nt+1,N)
            "C": float(C),
        }

    @torch.no_grad()
    def predict_on_grid(self, Nt=200, Nx=200, C=None):
        self.eval()
        device = next(self.parameters()).device

        # grid in (t,x)
        t = torch.linspace(0.0, T, Nt, device=device).view(-1, 1)    # (Nt,1)
        x = torch.linspace(a, b, Nx, device=device).view(-1, 1)      # (Nx,1)

        TT_torch = t.repeat(1, Nx)                                   # (Nt,Nx)
        XX_torch = x.t().repeat(Nt, 1)                               # (Nt,Nx)

        t_flat = TT_torch.reshape(-1, 1)                             # (Nt*Nx,1)
        x_flat = XX_torch.reshape(-1, 1)                             # (Nt*Nx,1)

        # choose C
        if C is None:
            if hasattr(self, "_ref") and ("C" in self._ref):
                C_use = self._ref["C"]
            else:
                C_use = 1.0
        else:
            C_use = float(C)

        # forward
        # u_tilde uses psi+phi*u_nn; u_nn may be moe or not
        u_pred = self.u_tilde(t_flat, x_flat, C_use).view(Nt, Nx)    # torch (Nt,Nx)
        U = u_pred.detach().cpu().numpy()

        # gates (optional)
        G = None
        try:
            out = self.u_nn(t_flat, x_flat, return_gates=True)
            if isinstance(out, tuple) and len(out) == 2:
                _, gates = out                           # (Nt*Nx,E)
                G = gates.view(Nt, Nx, -1).detach().cpu().numpy()
        except TypeError:
            # u_nn doesn't support return_gates -> ignore
            pass

        TT = TT_torch.detach().cpu().numpy()
        XX = XX_torch.detach().cpu().numpy()

        # --- reference interpolation ---
        if not hasattr(self, "_ref"):
            raise RuntimeError("No reference found. Call model.set_reference(t_ref, x_ref, u_ref, C=...) before visualize().")

        tref = self._ref["t"]          # (nt+1,)
        xref = self._ref["x"]          # (N,)
        uref = self._ref["u"]          # (nt+1,N)

        # time interpolation indices for each grid t
        t_grid = np.linspace(0.0, T, Nt)
        # clamp to range
        t_grid = np.clip(t_grid, tref[0], tref[-1])

        # find left indices
        idx = np.searchsorted(tref, t_grid, side="right") - 1
        idx = np.clip(idx, 0, len(tref) - 2)
        idx2 = idx + 1

        t0 = tref[idx]
        t1 = tref[idx2]
        w = (t_grid - t0) / (t1 - t0 + 1e-14)    # (Nt,)

        # space interpolation for each time-slice:
        # reference x is periodic on [-1,1). Our plot x includes endpoint b=1.
        # We'll map x_grid into [-1,1) by wrapping b -> a.
        x_grid = np.linspace(a, b, Nx)
        x_wrap = x_grid.copy()
        x_wrap[x_wrap >= b] = a  # send endpoint to a for periodic consistency

        U_true = np.zeros((Nt, Nx), dtype=np.float64)
        for it in range(Nt):
            u_left = uref[idx[it], :]
            u_right = uref[idx2[it], :]
            u_t = (1.0 - w[it]) * u_left + w[it] * u_right  # (N,)

            # numpy.interp expects increasing x; xref is increasing on [-1,1)
            # periodic endpoint handled by x_wrap
            U_true[it, :] = np.interp(x_wrap, xref, u_t)

        # rel L2
        num = np.linalg.norm(U - U_true)
        den = np.linalg.norm(U_true) + 1e-14
        error = float(num / den)

        self.train()
        return TT, XX, U, U_true, G, error

    @torch.no_grad()
    def visualize(self, ep=0, out_dir="AC", Nt=200, Nx=400):
        os.makedirs(out_dir, exist_ok=True)

        T, X, U, U_true, G, error = self.predict_on_grid(Nt=Nt, Nx=Nx)
        error_map = np.abs(U_true - U)

        FONT_TITLE = 22
        FONT_LABEL = 24
        FONT_TICK = 20
        FONT_CBAR = 18

        fig1, ax1 = plt.subplots(figsize=(8, 6.5)) 
        im1 = ax1.pcolormesh(T, X, U, cmap='rainbow', shading='auto')
        ax1.set_xlabel('$t$', fontsize=FONT_LABEL, labelpad=10)
        ax1.set_ylabel('$x$', fontsize=FONT_LABEL, labelpad=10)
        ax1.set_title('Predicted $u(t,x)$', fontsize=FONT_TITLE, pad=15)
        ax1.tick_params(axis='both', labelsize=FONT_TICK)
        cbar1 = fig1.colorbar(im1, ax=ax1)
        cbar1.ax.tick_params(labelsize=FONT_CBAR)

        plt.tight_layout()
        save_name1 = os.path.join(out_dir, f"AC_Prediction_ep{ep:05d}.png")
        plt.savefig(save_name1, dpi=300, bbox_inches='tight')
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(8, 6.5))
        im2 = ax2.pcolormesh(T, X, error_map, cmap='inferno', shading='auto')
        ax2.set_xlabel('$t$', fontsize=FONT_LABEL, labelpad=10)
        ax2.set_ylabel('$x$', fontsize=FONT_LABEL, labelpad=10)
        ax2.set_title('Absolute Error', fontsize=FONT_TITLE, pad=15)
        ax2.tick_params(axis='both', labelsize=FONT_TICK)
        cbar2 = fig2.colorbar(im2, ax=ax2)
        cbar2.ax.tick_params(labelsize=FONT_CBAR)

        plt.tight_layout()
        error_png = os.path.join(out_dir, f"AC_Absolute_Error_ep{ep:05d}.png")
        plt.savefig(error_png, dpi=300, bbox_inches='tight')
        plt.close(fig2)

        # print("U_true min/max:", float(U_true.min()), float(U_true.max()))
        # print(f"  [viz @ ep {ep}] Rel L2 Error on grid: {error:.4e}. Saving plots...")

        # --- 3-panel plot: True / Pred / abs error ---
        # fig = plt.figure(figsize=(15, 4.5))

        # plt.subplot(1, 3, 1)
        # plt.pcolormesh(TT, XX, U_true, cmap="rainbow", shading="auto")
        # plt.colorbar()
        # plt.xlabel("t", fontsize=14, labelpad=6)
        # plt.ylabel("x", fontsize=14, labelpad=6)
        # plt.tick_params(axis="both", labelsize=12)
        # plt.title("True u", fontsize=14, pad=6)

        if G is None:
            return

        E = G.shape[-1]
        for i in range(E):
            fig = plt.figure(figsize=(8, 6.5))
            plt.pcolormesh(T, X, G[:, :, i], cmap="hot", shading="auto", vmin=0, vmax=1)
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=FONT_CBAR)

            plt.xlabel("t", fontsize=FONT_LABEL, labelpad=10)
            plt.ylabel("x", fontsize=FONT_LABEL, labelpad=10)
            plt.tick_params(axis="both", labelsize=FONT_TICK)
            plt.title(f"Expert {i+1} Gate Weight", fontsize=FONT_TITLE, pad=15)
            plt.tight_layout()

            gate_png = os.path.join(out_dir, f"gate_ep{ep:05d}_expert{i+1:02d}.png")
            plt.savefig(gate_png, dpi=300)
            plt.close(fig)

    def u_nn(self, t, x, return_gates: bool = False):
        xin = torch.cat([t, x], dim=1)   # (N,2)
        u, gates = self.moe(xin)
        if return_gates:
            return u, gates
        return u  # (N,1)

    def u_tilde(self, t, x, C):
        # HC ansatz only needs u (gates are auxiliary)
        u = self.u_nn(t, x, return_gates=False)
        return psi(t, x, C) + phi(t, x) * u

def pde_residual(model: AC, t, x, C):
    t_ = t.detach().clone().requires_grad_(True)
    x_ = x.detach().clone().requires_grad_(True)
    u = model.u_tilde(t_, x_, C)

    u_t = torch.autograd.grad(u, t_, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x_, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    r = u_t - gamma1 * u_xx + gamma2 * (u**3 - u)
    return r

def sample_uniform(n, low, high, device):
    return (low + (high - low) * torch.rand(n, 1, device=device))

def build_training_points(N_f=16384, N_ic=128, N_bc=2048, device="cuda"):
    # interior: t in (0,T], x in [a,b]
    t_eps = 0.00
    t_f = sample_uniform(N_f, t_eps, T, device)
    x_f = sample_uniform(N_f, a, b, device)

    # IC points at t=0
    t_ic = torch.zeros(N_ic, 1, device=device)
    x_ic = sample_uniform(N_ic, a, b, device)
    u_ic = u0_torch(x_ic)

    # BC points: periodic -> x=a and x=b (same t)
    t_bc = sample_uniform(N_bc, 0.0, T, device)
    x_a = torch.full((N_bc, 1), a, device=device)
    # dx = (b - a) / 2048
    x_b = torch.full((N_bc, 1), b, device=device)

    return (t_f, x_f), (t_ic, x_ic, u_ic), (t_bc, x_a, x_b)

def train_case1(t_ref, x_ref, u_ref, device="cuda", C=1.0,
               adam_steps=50_000, lbfgs_steps=5_000, expert_rank=16,
               use_soft_icbc=False):
    model = AC(
        num_experts=3,
        expert_hidden=64,
        expert_rank=expert_rank,
        router_hidden=64,
        router_depth=2
    ).to(device)
    model.set_reference(t_ref, x_ref, u_ref, C=C)

    (t_f, x_f), (t_ic, x_ic, u_ic), (t_bc, x_a, x_b) = build_training_points(device=device)

    idx_T = torch.argmin(torch.abs(t_ref - T)).item()

    x_eval = x_ref.to(device)              # (nx,1)
    u_ref_T = u_ref[idx_T].to(device)

    opt = Adam(model.parameters(), lr=1e-3)

    def loss_fn():
        r = pde_residual(model, t_f, x_f, C)
        loss = torch.mean(r**2)
        # print(loss.item())

        if use_soft_icbc:
            # soft IC
            u0_pred = model.u_tilde(t_ic, x_ic, C)
            loss_ic = torch.mean((u0_pred - u_ic)**2)
            # soft periodic BC (value match)
            x_a.requires_grad_(True)
            x_b.requires_grad_(True)
            ua = model.u_tilde(t_bc, x_a, C)
            ub = model.u_tilde(t_bc, x_b, C)
            uxa = torch.autograd.grad(ua, x_a, torch.ones_like(ua), create_graph=True)[0]
            uxb = torch.autograd.grad(ub, x_b, torch.ones_like(ub), create_graph=True)[0]
            loss_bc_val = torch.mean((ua - ub)**2)
            loss_bc_der = torch.mean((uxa - uxb)**2)
            # loss = loss
            loss = loss + 0.01*loss_bc_val + 0.001*loss_bc_der

            # loss = loss + loss_ic + loss_bc
        return loss

    # Adam stage

    history = {
            "iter": [],
            "loss": [],
            "loss_f": [],
            "loss_bc": [],
            "error": [],
            "elapsed_time": []
        }
    start_ts = time.time()
    for it in range(1, adam_steps + 1):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()

        if it % 100 == 0:
            # rel_err = eval_rel_l2_at_T(model, x_eval, u_ref_T, C)
            rel_err = eval_rel_l2(model, t_ref, x_ref, u_ref, C)
            history["iter"].append(it)
            history["error"].append(rel_err)
            history["elapsed_time"].append(time.time() - start_ts)

        if it % 1000 == 0:
            # rel_err = eval_rel_l2_at_T(model, x_eval, u_ref_T, C)
            rel_err = eval_rel_l2(model, t_ref, x_ref, u_ref, C)
            # print("shift-check rel:", torch.norm(u_ref_T - torch.roll(u_ref_T, shifts=1, dims=0)) / torch.norm(u_ref_T))
            print(
                  f"[Adam {it:6d}] "
                  f"loss={loss.item():.3e} | "
                  f"relL2(T)={rel_err:.3e}"
            )
        # if it % 5000 == 0:
        #     model.visualize(ep=it, out_dir="transport_ac", Nt=200, Nx=400)

    # LBFGS stage
    opt2 = LBFGS(
            model.parameters(),
            lr=0.5,
            max_iter=20000,
            max_eval=20000,
            history_size=100,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

    state = {"ncall": 0}
    def closure():
        opt2.zero_grad()
        loss = loss_fn()
        loss.backward()
        state["ncall"] += 1

        if state["ncall"] % 100 == 0 or state["ncall"] == 1:
            with torch.no_grad():
                rel_err = eval_rel_l2(model, t_ref, x_ref, u_ref, C)
            history["iter"].append(state["ncall"] + adam_steps)
            history["error"].append(rel_err)
            history["elapsed_time"].append(time.time() - start_ts)

        if state["ncall"] % 1000 == 0 or state["ncall"] == 1:
            with torch.no_grad():
                rel_err = eval_rel_l2(model, t_ref, x_ref, u_ref, C)
            print(f"[LBFGS {state['ncall']:5d}] loss={loss.item():.3e} | relL2={rel_err:.3e}")
        
        # if state["ncall"] % 2000 == 0:
        #     model.visualize(ep=state["ncall"] + adam_steps, out_dir="transport_ac")

        return loss

    print("\n--- Starting LBFGS Stage ---")
    opt2.step(closure)
    
    final_loss = closure().item() 
    return model, history

@torch.no_grad()

def eval_rel_l2(model, t_ref, x_ref, u_ref, C):

    device = next(model.parameters()).device
    ntp1, nx = u_ref.shape

    x = x_ref.to(device)  # (nx, 1) assumed
    preds = []

    for idx in range(ntp1):
        t0 = t_ref[idx].item()
        t = torch.full((nx, 1), t0, device=device)
        up = model.u_tilde(t, x, C).view(-1)  # (nx,)
        preds.append(up)

    u_pred = torch.stack(preds, dim=0)  # (nt+1, nx)
    u_ref = u_ref.to(device)

    rel = torch.linalg.norm(u_pred - u_ref) / torch.linalg.norm(u_ref)

    return rel.item()


if __name__ == "__main__":

    num_experts = 2
    # r_values = [1, 4, 8, 16, 32]
    expert_rank = 32

    model_seeds = [1234]

    all_results = {}
    ms = 1234
    for ms in model_seeds:
    # for expert_rank in r_values:
        print(f"\n=== Training model with seed {ms} and expert rank {expert_rank} ===")
        set_model_seed(ms)

        t_ref, x_ref, u_ref = solve_ac_reference(N=2048, dt=1/200, gamma1=1e-3, gamma2=5.0)
        t_ref = torch.from_numpy(t_ref).float().to(device)          # (nt+1,)
        x_ref = torch.from_numpy(x_ref).float().to(device).view(-1, 1)  # (N,1)
        u_ref = torch.from_numpy(u_ref).float().to(device)           # (nt+1, N)

        # train
        start = time.time()
        model, history = train_case1(t_ref, x_ref, u_ref, device=device, C=1.0,
                            adam_steps=10000, lbfgs_steps=10000, expert_rank=expert_rank,
                            use_soft_icbc=True)

        elapsed = time.time() - start
        print(f"Total time: {time.time() - start:.2f}s")

        total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_all       = sum(p.numel() for p in model.parameters())
        print(f"Trainable: {total_trainable:,}  |  All: {total_all:,}")

        model.set_reference(t_ref, x_ref, u_ref, C=1.0)
        model.visualize(ep=0, out_dir="transport_ac", Nt=200, Nx=400)

        # eval
        errs = eval_rel_l2(model, t_ref, x_ref, u_ref, C=1.0)
        print("Rel L2 errors:", errs)

        # result_key = f"AC_seed_{ms}"
        # all_results[result_key] = {
        #         "r": expert_rank,
        #         "K": num_experts, 
        #         "history": history,
        #         "final_error": errs,
        #         "train_time_sec": elapsed,
        #         "params": total_all
        #     }

        # file_path = f"training_results_AC_seed.json"
        # with open(file_path, 'w') as f:
        #     json.dump(all_results, f, indent=4)
