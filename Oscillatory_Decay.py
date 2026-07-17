import math
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from model import DomainMoE
import torch.nn.functional as F
import json
import random
from utils import ADR_slice_sampler, infer_on_coords, visualize_solution_2d, visualize_solution_single, save_expert_gates
from collections import defaultdict
from vi import vi_Unsteady_ADR

def set_model_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# -----------------------------
# Domain
# -----------------------------
X_MIN, X_MAX = 0.0, 1.0
T_MIN, T_MAX = 0.0, 1.0
L = 1.0
DIM = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
EPS = 0.01                # interface thickness (smaller -> sharper)
X0 = 0.35                 # initial interface location
C_FRONT = V_ADV

def omega_x(x: torch.Tensor) -> torch.Tensor:
    s = torch.sigmoid(K_IFACE * (x - X_IFACE))   # ~0 on left, ~1 on right
    return OMEGA1 * (1.0 - s) + OMEGA2 * s

def lambda_x(x: torch.Tensor) -> torch.Tensor:
    """
    Spatially varying reaction rate:
      lambda(x) = lambda0 + dlambda * [tanh(alpha*(x-x1)) - tanh(alpha*(x-x2))]/2
    """
    return LAMBDA0 + 0.5 * DLAMBDA * (torch.tanh(ALPHA * (x - X1)) - torch.tanh(ALPHA * (x - X2)))

def u_eq_x(x: torch.Tensor) -> torch.Tensor:
    """Equilibrium profile u_eq(x) = 0.5*(1 + tanh(beta*(x-0.5)))."""
    return 0.5 * (1.0 + torch.tanh(BETA * (x - 0.5)))

def smooth_window(x: torch.Tensor, x1=X1_WIN, x2=X2_WIN, alpha=ALPHA_WIN) -> torch.Tensor:
    return 0.5 * (torch.tanh(alpha * (x - x1)) - torch.tanh(alpha * (x - x2)))

def X_spatial(x: torch.Tensor) -> torch.Tensor:
    """
    Spatial factor X(x): global low-frequency + localized high-frequency in [x1,x2].
    Still smooth and fully separable with T(t).
    """
    # normalize x to [0,1] for frequency control
    xi = (x - X_MIN) / (X_MAX - X_MIN)  # in [0,1]

    # base (your original-ish multi-sine low frequency)
    X_base = (
        torch.sin(math.pi * xi)
        + 0.3 * torch.sin(3.0 * math.pi * xi)
        + 0.1 * torch.sin(5.0 * math.pi * xi)
    )

    # localized high-frequency
    w = smooth_window(x)  # (N,1)
    X_hi = A_HI * w * torch.sin(2.0 * math.pi * K_HI * xi)

    return X_base + X_hi

def u_exact(coords: torch.Tensor) -> torch.Tensor:
    """
    Exact solution (CP-separable form):
        u(t,x) = m - a * A(t) * tanh((x - X0)/EPS)
    where A(t) depends only on t.
    """
    t = coords[:, 0:1]
    x = coords[:, 1:2]

    m = 0.5 * (U_L + U_R)
    a = 0.5 * (U_L - U_R)

    # time-only amplitude (use existing constants GAMMA, OMEGA)
    A_t = torch.exp(-GAMMA * t) * torch.cos(OMEGA * t)

    # space-only front
    z = (x - X0) / EPS
    phi_x = torch.tanh(z)

    return m - a * A_t * phi_x

def source_S(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Return S(t,x) such that u_exact satisfies:
      u_t + V_ADV u_x - D_DIFF u_xx - lam(x)*(u-ueq(x)) = S(t,x)
    with CP-separable u_exact:
      u(t,x) = m - a * A(t) * tanh((x - X0)/EPS).
    """
    m = 0.5 * (U_L + U_R)
    a = 0.5 * (U_L - U_R)

    # A(t) and A'(t)
    exp_term = torch.exp(-GAMMA * t)
    cos_term = torch.cos(OMEGA * t)
    sin_term = torch.sin(OMEGA * t)

    A_t = exp_term * cos_term
    A_t_prime = exp_term * (-GAMMA * cos_term - OMEGA * sin_term)

    # space front phi(x) = tanh(z), z=(x-X0)/EPS
    z = (x - X0) / EPS
    phi = torch.tanh(z)
    sech2 = 1.0 - phi**2

    # u and derivatives
    u = m - a * A_t * phi
    u_t = -a * A_t_prime * phi
    u_x = -a * A_t * (1.0 / EPS) * sech2
    u_xx = 2.0 * a * A_t * (1.0 / (EPS**2)) * sech2 * phi

    lam = lambda_x(x)
    ueq = u_eq_x(x)

    return u_t + V_ADV * u_x - D_DIFF * u_xx - lam * (u - ueq)

def u_exact_x_at_boundary(t: torch.Tensor, x_boundary: float) -> torch.Tensor:
    """
    Compute u_x(t, x_boundary) from u_exact using autograd (stable, no manual algebra).
    Args:
        t: (N,1) tensor
        x_boundary: float, e.g. X_MAX
    Returns:
        ux: (N,1) tensor
    """
    x = torch.full_like(t, float(x_boundary)).requires_grad_(True)  # (N,1)
    coords = torch.cat([t, x], dim=1)                                # (N,2)
    u = u_exact(coords)                                              # (N,1)
    ux = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u),
        create_graph=False, retain_graph=False
    )[0]
    return ux

def sample_interior(N: int) -> torch.Tensor:
    """Uniform interior samples in (t,x)."""
    t = (T_MAX - T_MIN) * torch.rand(N, 1, device=device) + T_MIN
    x = (X_MAX - X_MIN) * torch.rand(N, 1, device=device) + X_MIN
    return torch.cat([t, x], dim=1)

def sample_ic(N: int) -> torch.Tensor:
    """Initial-condition samples at t=T_MIN."""
    t = torch.full((N, 1), T_MIN, device=device)
    x = (X_MAX - X_MIN) * torch.rand(N, 1, device=device) + X_MIN
    return torch.cat([t, x], dim=1)

def sample_bc_left_dirichlet(N: int) -> torch.Tensor:
    """
    Left boundary samples at x = X_MIN.
    For advection v>0, x=0 is inflow; Dirichlet BC is natural and stabilizes training.
    """
    t = (T_MAX - T_MIN) * torch.rand(N, 1, device=device) + T_MIN
    x = torch.full((N, 1), X_MIN, device=device)
    return torch.cat([t, x], dim=1)

def sample_bc_right_neumann(N: int) -> torch.Tensor:
    """Right boundary samples at x = X_MAX for Neumann BC."""
    t = (T_MAX - T_MIN) * torch.rand(N, 1, device=device) + T_MIN
    x = torch.full((N, 1), X_MAX, device=device)
    return torch.cat([t, x], dim=1)

class PINN_Unsteady_ADR:
    def __init__(self, pinn_model: nn.Module, fixed_test_coords=None):
        self.pinn = pinn_model.to(device)

        # Loss weights (tuneable)
        self.w_pde = 1.0
        self.w_ic = 100.0
        self.w_bc_left = 10.0
        self.w_bc_right = 50.0 

        # Sample sizes
        self.N_f = 8192
        self.N_ic = 4096
        self.N_bcL = 2048
        self.N_bcR = 2048

        self.coords_test = fixed_test_coords.detach()

        # Adam + scheduler
        self.opt = torch.optim.Adam(self.pinn.parameters(), lr=1e-3)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=20000, eta_min=1e-6
        )

        # LBFGS (optional fine-tuning stage)
        self.opt_lbfgs = torch.optim.LBFGS(
            self.pinn.parameters(),
            lr=0.5,
            max_iter=20000,
            max_eval=20000,
            history_size=100,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

    def _normalize(self, coords: torch.Tensor) -> torch.Tensor:
        """Normalize (t,x) to [-1,1]^2."""
        t = coords[:, 0:1]
        x = coords[:, 1:2]
        t_n = 2.0 * (t - T_MIN) / (T_MAX - T_MIN) - 1.0
        x_n = 2.0 * (x - X_MIN) / (X_MAX - X_MIN) - 1.0
        return torch.cat([t_n, x_n], dim=1)

    def pde_residual(self, coords_f: torch.Tensor) -> torch.Tensor:
        """
        PDE:
          u_t + v u_x = D u_xx + lambda(x)(u - u_eq(x)) + S(x,t)
        Residual:
          r = u_t + v u_x - D u_xx - lambda(x)(u - u_eq) - S
        """
        coords_req = coords_f.clone().detach().requires_grad_(True)
        u_pred, _ = self.pinn(self._normalize(coords_req))

        grads = torch.autograd.grad(
            u_pred, coords_req, grad_outputs=torch.ones_like(u_pred),
            create_graph=True, retain_graph=True
        )[0]
        u_t = grads[:, 0:1]
        u_x = grads[:, 1:2]

        u_x_grads = torch.autograd.grad(
            u_x, coords_req, grad_outputs=torch.ones_like(u_x),
            create_graph=True, retain_graph=True
        )[0]
        u_xx = u_x_grads[:, 1:2]

        x = coords_req[:, 1:2]
        lam = lambda_x(x)
        ueq = u_eq_x(x)
        t_req = coords_req[:, 0:1].detach()
        x_req = coords_req[:, 1:2].detach()
        S = source_S(t_req, x_req)

        r = u_t + V_ADV * u_x - D_DIFF * u_xx - lam * (u_pred - ueq) - S
        return r, u_pred

    def neumann_right_loss(self, coords_bcR: torch.Tensor) -> torch.Tensor:
        """
        Scheme A: enforce u_x(X_MAX,t) = u_exact_x(X_MAX,t), i.e. consistent Neumann BC.
        """
        coords_req = coords_bcR.clone().detach().requires_grad_(True)
        u_pred, _ = self.pinn(self._normalize(coords_req))
        grads = torch.autograd.grad(
            u_pred, coords_req, grad_outputs=torch.ones_like(u_pred),
            create_graph=True, retain_graph=True
        )[0]
        ux_pred = grads[:, 1:2]

        t = coords_bcR[:, 0:1]
        ux_true = u_exact_x_at_boundary(t, X_MAX)  # exact derivative at x=X_MAX
        return F.mse_loss(ux_pred, ux_true)

    def dirichlet_left_loss(self, coords_bcL: torch.Tensor) -> torch.Tensor:
        """Enforce u(X_MIN,t) = u_exact(X_MIN,t)."""
        u_pred, _ = self.pinn(self._normalize(coords_bcL))
        u_true = u_exact(coords_bcL)
        return F.mse_loss(u_pred, u_true)

    def ic_loss(self, coords_ic: torch.Tensor) -> torch.Tensor:
        """Enforce u(x,0) = X(x) (since T(0)=1)."""
        u_pred, _ = self.pinn(self._normalize(coords_ic))
        u_true = u_exact(coords_ic)
        return F.mse_loss(u_pred, u_true)

    def _total_loss_from_coords(
        self,
        coords_f: torch.Tensor,
        coords_ic: torch.Tensor,
        coords_bcL: torch.Tensor,
        coords_bcR: torch.Tensor,
    ):
        """Compute total loss and its components given fixed coordinate batches."""
        r, u_pred= self.pde_residual(coords_f)

        loss_pde = torch.mean(r ** 2)
        loss_ic = self.ic_loss(coords_ic)
        loss_bcL = self.dirichlet_left_loss(coords_bcL)
        loss_bcR = self.neumann_right_loss(coords_bcR)

        loss = (
            self.w_pde * loss_pde
            + self.w_ic * loss_ic
            + self.w_bc_left * loss_bcL
            + self.w_bc_right * loss_bcR
        )
        u_true = u_exact(coords_f)
        error = torch.linalg.norm(u_pred - u_true) / torch.linalg.norm(u_true)
        return loss, (loss_pde, loss_ic, loss_bcL, loss_bcR, error)

    @torch.no_grad()
    def rel_error_on_grid(self, Nt=200, Nx=400) -> float:
        """Compute relative L2 error on a fixed grid for monitoring."""
        t = torch.linspace(T_MIN, T_MAX, Nt, device=device).unsqueeze(1)
        x = torch.linspace(X_MIN, X_MAX, Nx, device=device).unsqueeze(1)
        Tm, Xm = torch.meshgrid(t.squeeze(1), x.squeeze(1), indexing="ij")
        coords = torch.stack([Tm.reshape(-1), Xm.reshape(-1)], dim=1)
        u_pred, _ = self.pinn(self._normalize(coords))
        u_true = u_exact(coords)
        return (torch.linalg.norm(u_pred - u_true) / torch.linalg.norm(u_true)).item()
        # return (torch.norm(u_pred - u_true) / (torch.norm(u_true) + 1e-12)).item()

    @torch.no_grad()
    def evaluate_rel_l2(self):
        self.pinn.eval()
        coords = self.coords_test
        coords_norm = self._normalize(coords)
        u_pred, _ = self.pinn(coords_norm)
        u_true = u_exact(coords)
        err = torch.linalg.norm(u_pred - u_true) / torch.linalg.norm(u_true)
        return float(err.item())

    @torch.no_grad()
    def visualize(self, ep=0, out_dir="unsteady_ADR", Nt=200, Nx=400, device="cuda:4"):

        os.makedirs(out_dir, exist_ok=True)
        t_vals, x_vals, coords = ADR_slice_sampler()
        u_pred, gates = infer_on_coords(self.pinn, coords, normalize_fn=self._normalize)
        G = gates.reshape(201, 101, gates.shape[-1])

        u_pred = u_pred.reshape(201, 101)
        coords = torch.from_numpy(coords).float().to(device)
        u_true = u_exact(coords).reshape(201, 101)
        u_true = u_true.cpu().numpy()
        error = np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)
        png = os.path.join(out_dir, f"Unsteady_ADR_slice_ep{ep:05d}.png")
         # visualize_solution_single(t_vals, x_vals, u_true, png)
        save_expert_gates(t_vals, x_vals, G, out_dir, ep)
        if ep == 99999:
            data_save_path = os.path.join(out_dir, f"final_data_ep{ep:05d}.npz")
            save_dict = {
                "t_grid": t_vals,           
                "x_grid": x_vals,           
                "u_pred": u_pred,           
                "u_true": u_true,           
                "gates": G,                 
                "error": error,             
                "epoch": ep                 
            }

            np.savez(data_save_path, **save_dict)
            print(f"  [Final Data] All plotting data saved to {data_save_path}")


    def train(
        self,
        epochs_adam: int = 15000,
        log_every: int = 500,
        lbfgs_grad_clip=1.0,
        lbfgs_log_every: int = 1000,
        lbfgs_refresh_data_every: int = 0,
        use_lbfgs: bool = True
    ):
        
        history = {
            "iter": [],
            "loss": [],
            "loss_f": [],
            "loss_bc": [],
            "error": [],
            "elapsed_time": []
        }

        # -------------------------
        # Stage 1: Adam
        # -------------------------
        self.pinn.train()
        start_ts = time.time()
        for ep in range(1, epochs_adam + 1):
            self.opt.zero_grad(set_to_none=True)

            coords_f = sample_interior(self.N_f)
            coords_ic = sample_ic(self.N_ic)
            coords_bcL = sample_bc_left_dirichlet(self.N_bcL)
            coords_bcR = sample_bc_right_neumann(self.N_bcR)

            loss, parts = self._total_loss_from_coords(coords_f, coords_ic, coords_bcL, coords_bcR)
            loss_pde, loss_ic, loss_bcL, loss_bcR, error = parts
            loss.backward()
            self.opt.step()
            self.sched.step()

            if ep % 100 == 0:
                history["iter"].append(ep)
                history["error"].append(error.item())
                history["elapsed_time"].append(time.time() - start_ts)


            if ep % log_every == 0 or ep == 1:
                # rel = self.rel_error_on_grid()
                lr = self.opt.param_groups[0]["lr"]
                # loss_pde, loss_ic, loss_bcL, loss_bcR = parts
                print(
                    f"[Adam {ep:6d}] loss={loss.item():.3e} | "
                    f"pde={loss_pde.item():.3e} ic={loss_ic.item():.3e} "
                    f"bcL={loss_bcL.item():.3e} bcR={loss_bcR.item():.3e} | "
                    f"error={error.item():.3e} lr={lr:.2e}"
                )
            # if ep % 2000 == 0:
            #     self.visualize(ep=ep)
                # vi = vi_Unsteady_ADR(model=self.pinn, res=256)

        if not use_lbfgs:
            return

        self.pinn.train()

        # Fix batches for LBFGS (stable objective)
        coords_f = sample_interior(self.N_f)
        coords_ic = sample_ic(self.N_ic)
        coords_bcL = sample_bc_left_dirichlet(self.N_bcL)
        coords_bcR = sample_bc_right_neumann(self.N_bcR)

        state = {"ncall": 0}

        def closure():
            self.opt_lbfgs.zero_grad(set_to_none=True)

            # Optionally refresh batches (usually keep 0 for stability)
            if lbfgs_refresh_data_every and state["ncall"] % lbfgs_refresh_data_every == 0 and state["ncall"] > 0:
                nonlocal coords_f, coords_ic, coords_bcL, coords_bcR
                coords_f = sample_interior(self.N_f)
                coords_ic = sample_ic(self.N_ic)
                coords_bcL = sample_bc_left_dirichlet(self.N_bcL)
                coords_bcR = sample_bc_right_neumann(self.N_bcR)

            loss, parts = self._total_loss_from_coords(coords_f, coords_ic, coords_bcL, coords_bcR)
            loss_pde, loss_ic, loss_bcL, loss_bcR, error = parts
            loss.backward()

            state["ncall"] += 1
            if state["ncall"] % 100 == 0:
                history["iter"].append(state["ncall"]+epochs_adam)
                history["elapsed_time"].append(time.time() - start_ts)
                history["error"].append(error.item())
                
            if state["ncall"] % lbfgs_log_every == 0 or state["ncall"] == 1:
                # with torch.no_grad():
                #     rel = self.rel_error_on_grid()
                # loss_pde, loss_ic, loss_bcL, loss_bcR = parts
                print(
                    f"[LBFGS {state['ncall']:5d}] loss={loss.item():.3e} | "
                    f"pde={loss_pde.item():.3e} ic={loss_ic.item():.3e} "
                    f"bcL={loss_bcL.item():.3e} bcR={loss_bcR.item():.3e} | "
                    f"error={error.item():.3e}"
                )

            return loss

        # Run LBFGS step (internally performs up to max_iter closure evaluations)
        self.opt_lbfgs.step(closure)

        final_error = self.evaluate_rel_l2()
        print(f"\n--- Optimization Finished ---")
        print(f"Final L2 Relative Error: {final_error:.4e}")
        return history, final_error, state["ncall"]


if __name__ == "__main__":
    # Instantiate your DomainMoE (the NAMExpert MoE you posted)
    results = []

    set_model_seed(1234) 
    print("Generating global test set (20,000 points)...")
    global_test_coords = sample_interior(20000).detach()

    model_seeds = [1234]
    # ms = 1234
    # r_values = [16]
    all_results = {}
    expert_rank = 4
    num_experts = 3
    for ms in model_seeds:
    # for expert_rank in r_values:
        print(f"\n{'='*50}")
        print(f"           TRAINING MODEL WITH seeds = {ms}           ")
        # print(f"           TRAINING MODEL WITH expert_rank = {expert_rank}           ")
        print(f"{'='*50}\n")
        set_model_seed(ms)

        pinn = DomainMoE(
            in_features=DIM,
            num_experts=num_experts,
            expert_hidden=64,
            expert_rank=expert_rank,
            router_hidden=64,
            router_depth=4
        ).to(device)


        model = PINN_Unsteady_ADR(pinn, global_test_coords)
        total_trainable = sum(p.numel() for p in model.pinn.parameters() if p.requires_grad)
        total_all       = sum(p.numel() for p in model.pinn.parameters())
        print(f"Trainable: {total_trainable:,}  |  All: {total_all:,}")

        start = time.time()
        history, final_error, steps = model.train(epochs_adam=10000, log_every=1000)
        elapsed = time.time() - start
        print(f"Total time: {time.time() - start:.2f}s")
        print("Final rel error:", final_error)

        # final_vi = vi_Unsteady_ADR(model=model.pinn, res = 256)

        # result_key = f"steady_ADR_seed_{ms}"
        # all_results[result_key] = {
        #         "r": expert_rank,
        #         "K": num_experts,
        #         "history": history,
        #         "final_error": final_error,
        #         "final_vi": final_vi,
        #         "train_time_sec": elapsed,
        #         "params": total_all
        #     }

        # file_path = f"training_results_ADR(OD)_r={expert_rank}.json"
        # with open(file_path, 'w') as f:
        #     json.dump(all_results, f, indent=4)

        model.visualize(ep=99999)

