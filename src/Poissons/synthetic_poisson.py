import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import os
import random
from model import Expert, DomainMoE, Unshared_Expert, MLP
from utils import visualize_slice_single_5d, save_expert_gates_5d
import json
import random
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def set_model_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

DIM = 5
X_MIN, X_MAX = 0.0, 1.0
PI = math.pi
EPSILON = 0.05  

def true_u_torch(coords: torch.Tensor) -> torch.Tensor:

    z = (torch.sum(coords, dim=1, keepdim=True) - 2.5) / EPSILON
    return torch.tanh(z)

def source_term_f(coords: torch.Tensor) -> torch.Tensor:

    z = (torch.sum(coords, dim=1, keepdim=True) - 2.5) / EPSILON
    sech_z = 1.0 / torch.cosh(z)
    tanh_z = torch.tanh(z)
    
    laplacian = -(10.0 / (EPSILON**2)) * (sech_z**2) * tanh_z
    u3 = tanh_z**3
    
    return -laplacian + u3

@torch.no_grad()
def sample_interior(N: int, device=device) -> torch.Tensor:
    return torch.rand(N, DIM, device=device) * (X_MAX - X_MIN) + X_MIN

@torch.no_grad()
def sample_boundary(N: int, device=device) -> torch.Tensor:
    pts = torch.rand(N, DIM, device=device)
    face_dim = torch.randint(low=0, high=DIM, size=(N,), device=device)
    face_side = torch.randint(low=0, high=2, size=(N,), device=device)  # 0 -> 0.0, 1 -> 1.0
    idx = torch.arange(N, device=device)
    pts[idx, face_dim] = face_side.float()
    return pts

class Poisson:
    def __init__(self, expert_hidden=32, expert_rank=16, fixed_test_coords=None):

        self.w_bc = 5000.0
        self.w_f_initial = 0.01
        self.w_f_final   = 1.0

        self.N_interior = 8192
        self.N_bc = 2048  

        self.coords_test = fixed_test_coords.detach()

        self.pinn = DomainMoE(in_features=DIM,
                              num_experts=2, expert_hidden=expert_hidden, expert_rank=expert_rank,
                              router_hidden=64, router_depth=4).to(device)
                                                           
        self.optimizer_adam = torch.optim.Adam(self.pinn.parameters(), lr=5e-4)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_adam, T_max=20000, eta_min=1e-6)
        
        self.optimizer_lbfgs = torch.optim.LBFGS(
            self.pinn.parameters(), max_iter=20000, 
            tolerance_grad=1e-9, tolerance_change=1e-12,
            history_size=100, line_search_fn="strong_wolfe"
        )

    def _normalize(self, coords):
        return 2.0 * (coords - X_MIN) / (X_MAX - X_MIN) - 1.0

    def load_from_checkpoint(self, path):
        ckpt = torch.load(path, map_location=device)
        sd   = ckpt["pinn_state_dict"]
        missing, unexpected = self.pinn.load_state_dict(sd, strict=False)
        print("Loaded from", path)
        print("  missing keys   :", missing)
        print("  unexpected keys:", unexpected)

    @staticmethod
    def _autograd_grads(u, coords):
        return torch.autograd.grad(u, coords, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    def pde_residual(self, coords):

        coords_with_grad = coords.clone().detach().requires_grad_(True)
        coords_norm = self._normalize(coords_with_grad)
        u_pred, _ = self.pinn(coords_norm) 

        du = self._autograd_grads(u_pred, coords_with_grad)   # (N,5)

        lap_terms = []
        for d in range(DIM):
            dudx_d = torch.autograd.grad(du[:, d:d+1], coords_with_grad,
                                         grad_outputs=torch.ones_like(du[:, d:d+1]),
                                         create_graph=True)[0][:, d:d+1]
            lap_terms.append(dudx_d)
        lap = sum(lap_terms)                                   # (N,1)
        u_pred_cubed = u_pred ** 3
        f_rhs = source_term_f(coords_with_grad)
        
        # -\Delta u + u^3 - f = 0
        resid = -lap + u_pred_cubed - f_rhs
        
        return resid, u_pred

    def loss_func(self, coords_f=None, coords_ic=None, periodic_pairs=None):

        if coords_f is None:
            coords_f = sample_interior(self.N_interior, device=device)
        if coords_ic is None:
            coords_ic = sample_boundary(self.N_bc, device=device)

        # PDE Residual
        resid, u_on_f_pts = self.pde_residual(coords_f)
        loss_f = F.mse_loss(resid, torch.zeros_like(resid))

        coords_bc_norm = self._normalize(coords_ic)
        u_bc_pred, _ = self.pinn(coords_bc_norm)
        # u_bc_pred = self.pinn(coords_bc_norm)
        
        with torch.no_grad():
            u_bc_true = true_u_torch(coords_ic)
            
        loss_bc = F.mse_loss(u_bc_pred, u_bc_true)

        with torch.no_grad():
            u_true_f = true_u_torch(coords_f)
            error = torch.norm(u_on_f_pts - u_true_f) / torch.norm(u_true_f)

        return loss_f, loss_bc, error  

    def train(self, n_epochs_adam=10000, viz_every=2000, ckpt_dir="ckpts_8d"):
        os.makedirs(ckpt_dir, exist_ok=True)
        history = {
            "iter": [],
            "loss": [],
            "loss_f": [],
            "loss_bc": [],
            "error": [],
            "elapsed_time": []
        }
        print("--- Starting Adam Optimization ---")
        anneal_epochs = n_epochs_adam * 0.75

        start_ts = time.time()
        for ep in range(1, n_epochs_adam + 1):
            self.pinn.train()
            # w_f = self.w_f_initial + (self.w_f_final - self.w_f_initial) * min(ep / anneal_epochs, 1.0)
            w_f = self.w_f_final

            self.optimizer_adam.zero_grad()
            loss_f, loss_bc, error = self.loss_func()
            loss = w_f * loss_f + self.w_bc * loss_bc
            loss.backward()
            self.optimizer_adam.step()
            self.scheduler.step()

            if ep % 100 == 0:
                history["iter"].append(ep)
                history["loss"].append(loss.item())
                history["loss_f"].append(loss_f.item())
                history["loss_bc"].append(loss_bc.item())
                history["error"].append(error.item())
                history["elapsed_time"].append(time.time() - start_ts)

            if ep % 1000 == 0:
                print(f"{ep:05d} training time: {time.time()-start_ts:.2f}s")
                
                print(f"[Adam {ep:05d}] loss={loss.item():.4e} w_f={w_f:.2f} "
                      f"(f={loss_f.item():.2e}, bc={loss_bc.item():.2e}, L2_err={error.item():.2e})")

            if ep % 1000 == 0:
                self.visualize_gates(epoch_name=f"{ep:05d}")

        print("\n--- Starting L-BFGS Optimization ---")
        self.pinn.train()
        coords_f_lbfgs = sample_interior(20000, device=device)
        coords_bc_lbfgs = sample_boundary(5000, device=device)

        self.lbfgs_iter = 0
        def closure():
            self.optimizer_lbfgs.zero_grad()
            loss_f, loss_bc, error = self.loss_func(
                coords_f=coords_f_lbfgs,
                coords_ic=coords_bc_lbfgs,
                periodic_pairs=None
            )
            loss = self.w_f_final * loss_f + self.w_bc * loss_bc
            loss.backward()
            self.lbfgs_iter += 1
            if self.lbfgs_iter % 100 == 0:
                history["iter"].append(self.lbfgs_iter+n_epochs_adam)
                history["elapsed_time"].append(time.time() - start_ts)
                history["loss"].append(loss.item())
                history["error"].append(error.item())
                print(f'[L-BFGS {self.lbfgs_iter:05d}] loss={loss.item():.4e} (L2_err={error.item():.2e})')
            return loss
        
        self.optimizer_lbfgs.step(closure)
        
        final_error = self.evaluate_rel_l2(20000)
        print(f"\n--- Optimization Finished ---")
        print(f"Final L2 Relative Error: {final_error:.4e}")

        return history, final_error

    @torch.no_grad()
    def visualize_slice(self, epoch_name="final", grid_res=200, save_dir="plots"):
        self.pinn.eval()
        os.makedirs(save_dir, exist_ok=True)

        x1_range = np.linspace(X_MIN, X_MAX, grid_res)
        x2_range = np.linspace(X_MIN, X_MAX, grid_res)
        X1, X2 = np.meshgrid(x1_range, x2_range)

        N_pts = grid_res * grid_res
        coords_np = np.zeros((N_pts, DIM))
        coords_np[:, 0] = X1.ravel()
        coords_np[:, 1] = X2.ravel()
        coords_np[:, 2:] = 0.5  

        coords_torch = torch.tensor(coords_np, dtype=torch.float32, device=device)

        coords_norm = self._normalize(coords_torch)
        u_pred_torch, _ = self.pinn(coords_norm)
        # u_pred_torch = self.pinn(coords_norm)
        u_true_torch_val = true_u_torch(coords_torch)
        
        error_torch = torch.abs(u_true_torch_val - u_pred_torch)

        U_pred = u_pred_torch.cpu().numpy().reshape(grid_res, grid_res)
        U_true = u_true_torch_val.cpu().numpy().reshape(grid_res, grid_res)
        Error = error_torch.cpu().numpy().reshape(grid_res, grid_res)

        vmin_u = min(U_true.min(), U_pred.min())
        vmax_u = max(U_true.max(), U_pred.max())

        visualize_slice_single_5d(
            X1, X2, U_true, 
            out_path=os.path.join(save_dir, f"exact_ep_{epoch_name}.png"),
            title="Exact Solution $u(x_1,x_2,0.5^3)$", 
            cmap="jet", vmin=vmin_u, vmax=vmax_u
        )

    @torch.no_grad()
    def visualize_gates(self, epoch_name="final", grid_res=200, save_dir="plots"):

        self.pinn.eval()
        os.makedirs(save_dir, exist_ok=True)

        x1_range = np.linspace(X_MIN, X_MAX, grid_res)
        x2_range = np.linspace(X_MIN, X_MAX, grid_res)
        X1, X2 = np.meshgrid(x1_range, x2_range)

        N_pts = grid_res * grid_res
        coords_np = np.zeros((N_pts, DIM))
        coords_np[:, 0] = X1.ravel()
        coords_np[:, 1] = X2.ravel()
        coords_np[:, 2:] = 0.5  

        coords_torch = torch.tensor(coords_np, dtype=torch.float32, device=device)
        coords_norm = self._normalize(coords_torch)

        moe_out = self.pinn(coords_norm)
        if isinstance(moe_out, tuple):
            gates = moe_out[1]  
        else:
            raise ValueError("Model output is not a tuple. Make sure DomainMoE returns gates.")

        gates_np = gates.cpu().numpy()
        G = gates_np.reshape(grid_res, grid_res, -1)
        save_expert_gates_5d(X1, X2, G, save_dir, epoch_name)

    @torch.no_grad()
    def evaluate_rel_l2(self, N=50000):
        self.pinn.eval()
        coords = self.coords_test
        coords_norm = self._normalize(coords)
        u_pred, _ = self.pinn(coords_norm)
        # u_pred = self.pinn(coords_norm)
        u_true = true_u_torch(coords)
        err = torch.linalg.norm(u_pred - u_true) / torch.linalg.norm(u_true)
        return float(err.item())

    def save_checkpoint(self, path, epoch=None):
        torch.save(
            {
                "epoch": epoch,
                "pinn_state_dict": self.pinn.state_dict(),
                "optimizer_state_dict": self.optimizer_adam.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "expert_rank": expert_rank,
                "dim": DIM,
            },
            path
        )

if __name__ == '__main__':
    expert_hidden = 64

    configs = {
        "Expert (Ours)": {"type": "Expert", "hidden": 64, "r": 1},
        "Unshared_Expert": {"type": "Unshared_Expert", "hidden": 64, "r": 1},
        "MLP (Baseline)": {"type": "MLP", "hidden": 64, "depth": 6}
    }

    set_model_seed(1234) 
    print("Generating global test set (20,000 points)...")
    global_test_coords = sample_interior(20000, device=device).detach()

    r_values = [5]
    all_results = {}
    ms = 1234
    
    for expert_rank in r_values:
        print(f"\n{'='*50}")
        print(f"           TRAINING MODEL WITH expert_rank = {expert_rank}           ")
        print(f"{'='*50}\n")
        set_model_seed(ms)

        model = Poisson(expert_hidden=expert_hidden, expert_rank=expert_rank, fixed_test_coords=global_test_coords)
        model.pinn.to(device)

        total_trainable = sum(p.numel() for p in model.pinn.parameters() if p.requires_grad)
        total_all       = sum(p.numel() for p in model.pinn.parameters())
        print(f"Trainable: {total_trainable:,}  |  All: {total_all:,}")

        start = time.time()
        history, final_error = model.train(n_epochs_adam=10000, viz_every=2000)
        elapsed = time.time() - start
        print(f"Total training time: {elapsed:.2f}s")
        final_vi = vi_poisson(model=model.pinn, res=256)


        model.visualize_gates(epoch_name="final")
        # model.visualize_slice(epoch_name="final")

        # result_key = f"5dNonlinearPoisson_MLP"
        # all_results[result_key] = {
        #         "r": expert_rank,
        #         "history": history,
        #         "final_error": final_error,
        #         "train_time_sec": elapsed,
        #         "params": total_all
        #     }

        # file_path = "training_results_5d_nonlinear_nomoe.json"
        # with open(file_path, 'w') as f:
        #     json.dump(all_results, f, indent=4)