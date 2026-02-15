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
from vi import vi_poisson
from inference import poisson5d_slice_sampler, infer_on_coords, visualize_solution_2d, poisson10d_slice_sampler
import json
import random
from collections import defaultdict

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
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

DIM = 10
X_MIN, X_MAX = 0.0, 1.0
PI = math.pi

def true_u_torch(coords: torch.Tensor) -> torch.Tensor:
    return torch.prod(torch.sin(PI * coords), dim=1, keepdim=True)

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

        # self.pinn = DomainMoE(in_features=DIM,
        #                       num_experts=num_experts, expert_hidden=expert_hidden, expert_rank=expert_rank,
        #                       router_hidden=router_hidden, router_depth=router_depth).to(device)
        self.pinn = Expert(hidden=expert_hidden, r=expert_rank, dim=DIM).to(device)
        # self.pinn = Unshared_Expert(hidden=expert_hidden, r=expert_rank, dim=DIM).to(device)
        # self.pinn = MLP(in_features = DIM, hidden=64, num_experts=1, depth=6).to(device)

                                                                                    
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
        # u_pred, _ = self.pinn(coords_norm)                    # (N,1)
        u_pred = self.pinn(coords_norm) 

        du = self._autograd_grads(u_pred, coords_with_grad)   # (N,5)

        lap_terms = []
        for d in range(DIM):
            dudx_d = torch.autograd.grad(du[:, d:d+1], coords_with_grad,
                                         grad_outputs=torch.ones_like(du[:, d:d+1]),
                                         create_graph=True)[0][:, d:d+1]
            lap_terms.append(dudx_d)
        lap = sum(lap_terms)                                   # (N,1)

        f_rhs = DIM*(PI**2) * torch.prod(torch.sin(PI*coords_with_grad), dim=1, keepdim=True)
        resid = -lap - f_rhs
        return resid, u_pred

    def loss_func(self, coords_f=None, coords_ic=None, periodic_pairs=None):

        if coords_f is None:
            coords_f = sample_interior(self.N_interior, device=device)
        if coords_ic is None:
            coords_ic = sample_boundary(self.N_bc, device=device)

        # PDE Residual
        resid, u_on_f_pts = self.pde_residual(coords_f)
        loss_f = F.mse_loss(resid, torch.zeros_like(resid))

        # Dirichlet: u=0
        coords_bc_norm = self._normalize(coords_ic)
        # u_bc_pred, _ = self.pinn(coords_bc_norm)
        u_bc_pred = self.pinn(coords_bc_norm)
        loss_bc = F.mse_loss(u_bc_pred, torch.zeros_like(u_bc_pred))

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

            # if viz_every and ep % viz_every == 0 and ep > 0:
                # visualize_expert_curves(model=self.pinn, expert_idx=0, res=256)
                # vi = vi_poisson(model=self.pinn, res=256)
                # self.visualize(ep)

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

        # final_ckpt = os.path.join(ckpt_dir, f"poisson{DIM}D_{expert_rank}r_final.pt")
        # self.save_checkpoint(final_ckpt, epoch="final")
        # print(f"[Checkpoint] Final model saved to {final_ckpt}")
        return history, final_error

    @torch.no_grad()
    def evaluate_rel_l2(self, N=50000):
        self.pinn.eval()
        # coords = sample_interior(N, device=device)
        coords = self.coords_test
        coords_norm = self._normalize(coords)
        # u_pred, _ = self.pinn(coords_norm)
        u_pred = self.pinn(coords_norm)
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

    @torch.no_grad()
    def visualize(self, ep=0, out_dir="poisson10d_viz"):
        os.makedirs(out_dir, exist_ok=True)
        X1, X2, coords = poisson10d_slice_sampler(fixed_val=0.5)
        u_pred, _ = infer_on_coords(self.pinn, coords, self._normalize)

        u_pred = u_pred.reshape(X1.shape)
        u_true = np.prod(np.sin(np.pi * coords).reshape(*X1.shape, -1), axis=-1)
        error = np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)

        png = os.path.join(out_dir, f"poisson5d_slice_ep{ep:05d}.png")
        visualize_solution_2d(X1, X2, u_pred, u_true, error, png, labels=("x1", "x2"), title_prefix="Poisson")

if __name__ == '__main__':
    expert_hidden = 64
    # expert_rank = 5

    # configs = {
    #     "Expert (Ours)": {"type": "Expert", "hidden": 64, "r": 1},
    #     "Unshared_Expert": {"type": "Unshared_Expert", "hidden": 64, "r": 1},
    #     "MLP (Baseline)": {"type": "MLP", "hidden": 64, "depth": 6}
    # }


    set_model_seed(1234) 
    print("Generating global test set (20,000 points)...")
    global_test_coords = sample_interior(20000, device=device).detach()

    r_values = [1, 2, 5, 10, 16]
    all_results = {}
    # model_seeds = [1234, 42, 2, 4, 2025]
    results = []
    ms = 1234
    # for ms in model_seeds:
    for expert_rank in r_values:
        print(f"\n{'='*50}")
        print(f"           TRAINING MODEL WITH expert_rank = {expert_rank}           ")
        print(f"{'='*50}\n")
        set_model_seed(ms)

        model = Poisson(expert_hidden=expert_hidden, expert_rank=expert_rank, fixed_test_coords=global_test_coords)
        model.pinn.to(device)  # DomainMoE

        ## fine-tune ###
        # model.pinn.load_state_dict(
        #     torch.load("/scratch/shuyuan/random/ckpts_5d/poisson5D_5r_final.pt", map_location=device)["pinn_state_dict"],
        #     strict=True
        # )
        # print("[Info] Loaded 5D checkpoint, start fine-tuning on 10D")

        # model.optimizer_adam = torch.optim.Adam(
        #     model.pinn.parameters(), lr=1e-4   
        # )
        # model.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     model.optimizer_adam, T_max=15000, eta_min=1e-6
        # )

        # model ##

        total_trainable = sum(p.numel() for p in model.pinn.parameters() if p.requires_grad)
        total_all       = sum(p.numel() for p in model.pinn.parameters())
        print(f"Trainable: {total_trainable:,}  |  All: {total_all:,}")

        start = time.time()
        history, final_error = model.train(n_epochs_adam=10000, viz_every=2000)
        elapsed = time.time() - start
        print(f"Total training time: {time.time()-start:.2f}s")

        result_key = f"10dPoisson_r_{expert_rank}"
        all_results[result_key] = {
                "r": expert_rank,
                "history": history,
                "final_error": final_error,
                "train_time_sec": elapsed,
                "params": total_all
            }

        file_path = "training_results_10d_r.json"
        with open(file_path, 'w') as f:
            json.dump(all_results, f, indent=4)
        # model.visualize(ep=99999)  # Final visualization
        # final_vi = vi_poisson(model=model.pinn, res=256)

        # results.append({
        #     "seed": ms,
        #     'dim': DIM,
        #     "final_rel_l2": final_error,
        #     "final_vi": final_vi,
        #     "train_time_sec": elapsed,
        #     "expert_hidden": expert_hidden,
        #     "expert_rank": expert_rank,
        # })

        # os.makedirs("logs", exist_ok=True)
        # fname = f"logs/poisson_results_{DIM}D.json"
        # with open(fname, "w") as f:
        #     json.dump(results, f, indent=2)

