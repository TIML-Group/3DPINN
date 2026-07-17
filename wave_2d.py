import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time
import math
import os
from model import Expert
from vi import vi_wave_2d
from utils import wave_slice_sampler_2d, infer_on_coords, visualize_solution_2d
import json
import random
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

np.random.seed(1234)
def set_model_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

C_SPEED = 2.0
DIM = 3
X_MIN, X_MAX = 0.0, 1.0
T_MIN, T_MAX = 0.0, 1.0

def true_u_torch(coords: torch.Tensor, c: float) -> torch.Tensor:
    t = coords[:, 0:1]
    x = coords[:, 1:2]
    y = coords[:, 2:3]
    return torch.sin(math.pi * x) * torch.cos(math.sqrt(2) * c * math.pi * t) * torch.sin(math.pi * y)

def sample_interior(N, device=device) -> torch.Tensor:
    t = torch.rand(N, 1, device=device) * (T_MAX - T_MIN) + T_MIN
    x = torch.rand(N, 1, device=device) * (X_MAX - X_MIN) + X_MIN
    y = torch.rand(N, 1, device=device) * (X_MAX - X_MIN) + X_MIN
    return torch.cat([t, x, y], dim=-1)

def sample_ic(N: int, device=device) -> torch.Tensor:
    t = torch.zeros(N, 1, device=device)
    x = torch.rand(N, 1, device=device) * (X_MAX - X_MIN) + X_MIN
    y = torch.rand(N, 1, device=device) * (X_MAX - X_MIN) + X_MIN
    return torch.cat([t, x, y], dim=-1)

def sample_bc(N: int, device=device) -> torch.Tensor:
    t = torch.rand(N, 1, device=device) * (T_MAX - T_MIN) + T_MIN
    x0 = torch.full_like(t, X_MIN)
    y0 = torch.full_like(t, X_MIN)
    x1 = torch.full_like(t, X_MAX)
    y1 = torch.full_like(t, X_MAX)
    coords0 = torch.cat([t, x0, y0], dim=-1)
    coords1 = torch.cat([t, x1, y1], dim=-1)
    coords2 = torch.cat([t, x0, y1], dim=-1)
    coords3 = torch.cat([t, x1, y0], dim=-1)
    return coords0, coords1, coords2, coords3

class Wave:
    def __init__(self,
                 c=C_SPEED,
                 expert_hidden=32,
                 expert_rank=16,
                 fixed_test_coords=None):
        self.c = c

        self.w_f = 1.0
        self.w_ic_u = 100.0
        self.w_ic_ut = 100.0
        self.w_bc = 100.0

        self.N_interior = 8192
        self.N_ic = 1024
        self.N_bc = 1024

        self.coords_test = fixed_test_coords.detach()


        self.pinn = Expert(hidden=expert_hidden, r=expert_rank, dim=DIM).to(device)

        self.optimizer_adam = torch.optim.Adam(self.pinn.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_adam, T_max=20000, eta_min=1e-6)
        self.optimizer_lbfgs = torch.optim.LBFGS(
            self.pinn.parameters(), max_iter=20000, 
            tolerance_grad=1e-9, tolerance_change=1e-12,
            history_size=100, line_search_fn="strong_wolfe"
        )

    @staticmethod
    def _autograd_grads(u, coords):
        grad_outputs = torch.ones_like(u)
        du_dcoords = torch.autograd.grad(u, coords, grad_outputs=grad_outputs, create_graph=True)[0]
        u_t = du_dcoords[:, 0:1]
        u_x = du_dcoords[:, 1:2]
        u_y = du_dcoords[:, 2:3]
        u_tt = torch.autograd.grad(u_t, coords, grad_outputs=grad_outputs, create_graph=True)[0][:, 0:1]
        u_xx = torch.autograd.grad(u_x, coords, grad_outputs=grad_outputs, create_graph=True)[0][:, 1:2]
        u_yy = torch.autograd.grad(u_y, coords, grad_outputs=grad_outputs, create_graph=True)[0][:, 2:3]
        return u_t, u_x, u_y, u_tt, u_xx, u_yy

    def pde_residual(self, coords):
        coords_grad = coords.clone().detach().requires_grad_(True)
        u_pred = self.pinn(coords_grad)
        _, _, _, u_tt, u_xx, u_yy = self._autograd_grads(u_pred, coords_grad)
        residual = u_tt - (self.c**2) * (u_xx + u_yy)
        return residual, u_pred

    def loss_func(self, coords_f=None, coords_ic=None, periodic_pairs=None):
        if coords_f is None:
            coords_f = sample_interior(self.N_interior, device=device)
        if coords_ic is None:
            coords_ic = sample_ic(self.N_ic, device=device)
        if periodic_pairs is None:
            c0, c1, c2, c3 = sample_bc(self.N_bc, device=device)
        else:
            c0, c1, c2, c3 = periodic_pairs

        self.pinn.train()

        resid, u_for_err_calc = self.pde_residual(coords_f)
        loss_f = F.mse_loss(resid, torch.zeros_like(resid))

        coords_ic_grad = coords_ic.clone().detach().requires_grad_(True)   
        u_ic_pred = self.pinn(coords_ic_grad)
        u_t_ic_pred, _, _, _, _, _ = self._autograd_grads(u_ic_pred, coords_ic_grad)

        u_ic_true = torch.sin(math.pi * coords_ic[:, 1:2]) * torch.sin(math.pi * coords_ic[:, 2:3])
        loss_ic_u = F.mse_loss(u_ic_pred, u_ic_true)

        loss_ic_ut = F.mse_loss(u_t_ic_pred, torch.zeros_like(u_t_ic_pred))

        coords_bc = torch.cat([c0, c1, c2, c3], dim=0)
        u_bc_pred = self.pinn(coords_bc)
        loss_bc = F.mse_loss(u_bc_pred, torch.zeros_like(u_bc_pred))

        loss = self.w_f * loss_f + self.w_ic_u * loss_ic_u + self.w_ic_ut * loss_ic_ut + self.w_bc * loss_bc
        
        true = true_u_torch(coords_f, self.c)
        error = torch.norm(u_for_err_calc - true, p=2) / torch.norm(true, p=2)
        
        return loss, loss_f, loss_ic_u, loss_bc, error

    def train(self, n_epochs_adam=10000, viz_every=1000):
        print("--- Starting Adam Optimization ---")
        for ep in range(1, n_epochs_adam + 1):
            self.optimizer_adam.zero_grad()
            loss, loss_f, loss_ic, loss_per, error = self.loss_func()
            loss.backward()
            self.optimizer_adam.step()
            self.scheduler.step()

            if ep % 2000 == 0:
                print(f"[Adam {ep:05d}] loss={float(loss):.4e} "
                      f"(ic_u={float(loss_ic):.2e}, bc={float(loss_per):.2e}, error={error:.2e})") 

            # if viz_every and ep % viz_every == 0 and ep > 0:
            #     vi = vi_wave_2d(model=self.pinn, res = 256, c=C_SPEED, ep=ep)
            #     self.visualize(ep)

        print("\n--- Starting L-BFGS Optimization ---")
        self.pinn.train()
        
        # 1. Create a fixed set of points for the deterministic loss
        coords_f_lbfgs = sample_interior(20000, device=device) # Use more points for L-BFGS
        coords_ic_lbfgs = sample_ic(5000, device=device)
        periodic_pairs_lbfgs = sample_bc(5000, device=device)
        
        self.lbfgs_iter = 0
        def closure():
            self.optimizer_lbfgs.zero_grad()
            loss, loss_f, loss_ic, loss_per, error = self.loss_func(
                coords_f=coords_f_lbfgs,
                coords_ic=coords_ic_lbfgs,
                periodic_pairs=periodic_pairs_lbfgs
            )
            # Use final PDE weight for L-BFGS
            # loss = self.w_f_final * loss_f + self.w_ic * loss_ic + self.w_per * loss_per
            loss.backward()
            
            self.lbfgs_iter += 1
            if self.lbfgs_iter % 100 == 0:
                print(f'[L-BFGS {self.lbfgs_iter:05d}] loss={loss.item():.4e} (L2_err={error.item():.2e})')
            return loss
        
        # 2. Run the optimizer
        self.optimizer_lbfgs.step(closure)

        final_error = self.evaluate_rel_l2()
        print(f"\n--- Optimization Finished ---")
        print(f"Final L2 Relative Error: {final_error:.4e}")
        return final_error

    @torch.no_grad()
    def evaluate_rel_l2(self):
        self.pinn.eval()
        coords = self.coords_test
        # coords_norm = self._normalize(coords)
        u_pred = self.pinn(coords)
        u_true = true_u_torch(coords, self.c)
        err = torch.linalg.norm(u_pred - u_true) / torch.linalg.norm(u_true)
        return float(err.item())

    @torch.no_grad()
    def visualize(self, ep=0, out_dir="wave_viz"):
        os.makedirs(out_dir, exist_ok=True)
        t_vals, x_vals, y_vals, coords = wave_slice_sampler_2d()
        u_pred, _ = infer_on_coords(self.pinn, coords, normalize_fn=None)

        u_pred = u_pred.reshape(201, 101)
        u_true = np.sin(math.pi * x_vals) * np.cos(math.sqrt(2) * self.c * math.pi * t_vals) * np.sin(math.pi * y_vals)
        error = np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)

        png = os.path.join(out_dir, f"wave_2d_slice_ep{ep:05d}.png")
        visualize_solution_2d(t_vals, x_vals, u_pred, u_true, error, png, labels=("t", "x"), title_prefix="Wave")


if __name__ == '__main__':
    expert_hidden = 64
    expert_rank = 1

    results = [] 

    set_model_seed(1234) 
    print("Generating global test set (20,000 points)...")
    global_test_coords = sample_interior(20000, device=device).detach()

    model_seeds = [1234, 42, 2, 4, 2025]
    for ms in model_seeds:
        print(f"\n{'='*50}")
        print(f"           TRAINING MODEL WITH seeds = {ms}           ")
        print(f"{'='*50}\n")
        set_model_seed(ms)

        model = Wave(
            c=C_SPEED,
            expert_hidden=expert_hidden,
            expert_rank=expert_rank,
            fixed_test_coords=global_test_coords
        )
    
        start = time.time()
        final_error = model.train(n_epochs_adam=10000, viz_every=5000)
        elapsed = time.time() - start
        print(f"Training time: {time.time()-start:.2f}s")
        final_vi = vi_wave_2d(model=model.pinn, res = 256, c=C_SPEED)

        results.append({
            "seed": ms,
            "final_rel_l2": final_error,
            "final_vi": final_vi,
            "train_time_sec": elapsed,
            "expert_hidden": expert_hidden,
            "expert_rank": expert_rank,
            "c": C_SPEED,
        })

        os.makedirs("logs", exist_ok=True)
        fname = f"logs/wave_results_2D_c{C_SPEED}.json"
        with open(fname, "w") as f:
            json.dump(results, f, indent=2)

      #   model.visualize(ep=99999, out_dir="wave_viz_final")