import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import time
import math
from model import DomainMoE
import os
import json
import random
from collections import defaultdict

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DIM =2

# data_seed = 1234
# np.random.seed(data_seed)

# def set_model_seed(seed):
#     # np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
    
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

class Burger:
    def __init__(self, X_u_train, u_train, X_f_train, lb, ub, nu, num_experts, expert_hidden, router_hidden, router_depth, r, coords_eval, u_star):
        self.lb = torch.tensor(lb, dtype=torch.float32).to(device)
        self.ub = torch.tensor(ub, dtype=torch.float32).to(device)
        
        self.X_u_train = torch.tensor(X_u_train, dtype=torch.float32, requires_grad=True).to(device)
        self.u_train = torch.tensor(u_train, dtype=torch.float32).to(device)
        self.X_f_train = torch.tensor(X_f_train, dtype=torch.float32, requires_grad=True).to(device)
        
        xcol = self.X_u_train[:, 0]
        tcol = self.X_u_train[:, 1]
        self.mask_ic  = (tcol == 0.0)
        self.mask_bcl = (xcol == -1.0)
        self.mask_bcr = (xcol ==  1.0)

        self.w_ic  = 10.0
        self.w_bcl = 1.0
        self.w_bcr = 1.0
        self.w_f   = 1.0

        self.nu = nu
        self.r = r  # Store r value
        # self.pinn = DomainMoE(num_experts=num_experts, r=self.r).to(device)
        self.pinn = DomainMoE(in_features=DIM,
                              num_experts=num_experts, expert_hidden=expert_hidden, expert_rank=r,
                              router_hidden=router_hidden, router_depth=router_depth).to(device)

        
        self.optimizer_adam = torch.optim.Adam(self.pinn.parameters(), lr=2e-3, weight_decay=1e-6)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_adam, T_max=20000, eta_min=1e-6)
        
        self.optimizer_lbfgs = torch.optim.LBFGS(
            self.pinn.parameters(), max_iter=50000, max_eval=None, 
            tolerance_grad=1e-9, tolerance_change=1.0 * np.finfo(float).eps,
            history_size=50, line_search_fn="strong_wolfe"
        )
        self.iter = 0
        self.err_steps = []
        self.err_values = []
        self._global_step = 0

        self.coords_eval = coords_eval
        self.u_star = u_star

    def _rel_l2_on(self, coords, u_star):
        self.pinn.eval()
        u_pred_eval, _, _ = self.predict(coords)
        error_u = np.linalg.norm(u_star - u_pred_eval, 2) / np.linalg.norm(u_star, 2)
        return float(error_u)

    def _log_error(self):
        err = self._rel_l2_on(self.coords_eval, self.u_star)
        self.err_steps.append(self._global_step)
        self.err_values.append(err)
        return err

    def net_u(self, x, t):
        X = torch.cat([x, t], dim=1)
        X_normalized = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        u, _ = self.pinn(X_normalized)
        return u

    def net_f(self, x, t):
        u = self.net_u(x, t)
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        f = u_t + u * u_x - self.nu * u_xx
        return f

    def loss_func(self):
        self.pinn.train() 
        u_pred = self.net_u(self.X_u_train[:, 0:1], self.X_u_train[:, 1:2])
        f_pred = self.net_f(self.X_f_train[:, 0:1], self.X_f_train[:, 1:2])
        loss_f = torch.mean(f_pred ** 2)

        def safe_mse(mask):
            if mask.sum() == 0:
                return torch.tensor(0.0, device=device)
            diff = (self.u_train[mask] - u_pred[mask])**2
            return diff.mean()

        loss_ic  = safe_mse(self.mask_ic)
        loss_bcl = safe_mse(self.mask_bcl)
        loss_bcr = safe_mse(self.mask_bcr)
        
        total_loss = self.w_ic*loss_ic + self.w_bcl*loss_bcl + self.w_bcr*loss_bcr + self.w_f*loss_f
        return total_loss

    def train(self, n_epochs_adam, X_star, u_star, ckpt_dir="ckpt_burgers"):
        os.makedirs(ckpt_dir, exist_ok=True)
        history = {
            "iter": [],
            "error": [],
            "elapsed_time": []
        }
        print("--- Starting Adam Optimization ---")
        self._global_step = 0
        start_ts = time.time()
        for epoch in range(n_epochs_adam):
            self.pinn.train()
            self.optimizer_adam.zero_grad()
            loss = self.loss_func()
            loss.backward()
            self.optimizer_adam.step()
            self.scheduler.step()
            self._global_step += 1
            if epoch % 100 == 0:
                history["iter"].append(epoch)
                history["error"].append(self._rel_l2_on(X_star, u_star))
                history["elapsed_time"].append(time.time() - start_ts)
                # self._log_error()

            if (epoch + 1) % 1000 == 0:
                lr = self.optimizer_adam.param_groups[0]['lr']
                error_u = self._rel_l2_on(X_star, u_star)
                print(f'Epoch {epoch:05d} => Loss: {loss.item():.4e}, L2 Error: {error_u:.4e}, LR: {lr:.4e}')

        print(f"Adam training time: {time.time()-t_adam_3d:.2f}s")
        
        print("\n--- Starting L-BFGS Optimization ---")
        self.lbfgs_iter = 0
        self.pinn.train()
        def closure():
            self.optimizer_lbfgs.zero_grad()
            loss = self.loss_func()
            loss.backward()
            self.lbfgs_iter += 1
            self._global_step += 1
            if self.lbfgs_iter % 100 == 0:
                # self._log_error()
                history["iter"].append(self.lbfgs_iter)
                history["error"].append(self._rel_l2_on(X_star, u_star))
                history["elapsed_time"].append(time.time() - start_ts)
                print(f'Iter {self.lbfgs_iter:05d} => Loss: {loss.item():.4e}')
            return loss
        self.optimizer_lbfgs.step(closure)
        error_u = self._rel_l2_on(X_star, u_star)
        print(f'Final L2 Error: {error_u:.4e}')

        # final_ckpt = os.path.join(ckpt_dir, f"poisson{DIM}D_final.pt")
        # self.save_checkpoint(final_ckpt, epoch="final")
        # print(f"[Checkpoint] Final model saved to {final_ckpt}")
        
        # Use self.r to create a dynamic label for the plot legend
        return history
    
    def predict(self, X_star):
        self.pinn.eval()
        with torch.no_grad():
            X_star_tensor = torch.tensor(X_star, dtype=torch.float32).to(device)
            X_normalized = 2.0 * (X_star_tensor - self.lb) / (self.ub - self.lb) - 1.0
            u_star, gates_star = self.pinn(X_normalized)
            
        X_star_tensor.requires_grad_(True)
        f_star = self.net_f(X_star_tensor[:, 0:1], X_star_tensor[:, 1:2])
        
        return u_star.detach().cpu().numpy(), f_star.detach().cpu().numpy(), gates_star.detach().cpu().numpy()

    def save_checkpoint(self, path, epoch=None):
        torch.save(
            {
                "epoch": epoch,
                "pinn_state_dict": self.pinn.state_dict(),
                "optimizer_state_dict": self.optimizer_adam.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "expert_rank": self.r,
                "dim": DIM,
            },
            path
        )

if __name__ == '__main__':
    nu = 0.01 / np.pi
    N_u = 900
    N_f = 10000
    n_epochs_adam = 10000

    # --- Data loading and preparation (same for all runs) ---
    data = scipy.io.loadmat('burgers_shock.mat')
    t = data['t'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    Exact = np.real(data['usol']).T
    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact.flatten()[:, None]
    lb = X_star.min(0)
    ub = X_star.max(0)

    X_ic  = np.hstack((x, np.zeros_like(x)));    u_ic  = -np.sin(np.pi * x)
    X_bcl = np.hstack((-np.ones_like(t), t));    u_bcl = np.zeros_like(t)
    X_bcr = np.hstack(( np.ones_like(t), t));    u_bcr = np.zeros_like(t)
  
    alpha = 0.00
    sigma_ic = alpha * np.std(u_ic)
    sigma_bc = alpha * np.std(u_bcl)

    u_ic  = u_ic  + sigma_ic * np.random.randn(*u_ic.shape)
    u_bcl = u_bcl + sigma_bc * np.random.randn(*u_bcl.shape)
    u_bcr = u_bcr + sigma_bc * np.random.randn(*u_bcr.shape)

    N_ic  = min(300, X_ic.shape[0])
    N_bcl = min(300, X_bcl.shape[0])
    N_bcr = min(300, X_bcr.shape[0])

    idx_ic  = np.random.choice(X_ic.shape[0],  N_ic,  replace=False)
    idx_bcl = np.random.choice(X_bcl.shape[0], N_bcl, replace=False)
    idx_bcr = np.random.choice(X_bcr.shape[0], N_bcr, replace=False)

    X_u_train = np.vstack([X_ic[idx_ic],  X_bcl[idx_bcl],  X_bcr[idx_bcr]])
    u_train   = np.vstack([u_ic[idx_ic],  u_bcl[idx_bcl],  u_bcr[idx_bcr]])

    N_u = min(N_u, X_u_train.shape[0]) 
    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    X_u_train = X_u_train[idx, :]
    u_train = u_train[idx, :]
    X_f_train = lb + (ub - lb) * np.random.rand(N_f, 2)
    
    # --- Loop over r values, train models, and collect histories ---
    r_values = [1, 4, 8, 16, 32]
    # r_val = 16
    all_histories = []
    results = []

    num_experts = 2
    expert_hidden = 64
    router_hidden = 64
    router_depth = 4

    # model_seeds = [1234, 42, 4, 2, 2025]
    all_results = {}
    # model_seeds = [1234]
    ms = 1234
    # for ms in model_seeds:
    for r_val in r_values:
        print(f"\n{'='*50}")
        # print(f"           TRAINING MODEL WITH seeds = {ms}           ")
        print(f"           TRAINING MODEL WITH r = {r_val}           ")
        print(f"{'='*50}\n")
        set_model_seed(ms)

        ### training ###
        model_3d = Burger(
            X_u_train, u_train, X_f_train, lb, ub, nu, 
            num_experts=num_experts, expert_hidden=expert_hidden, router_hidden=router_hidden, router_depth=router_depth,
            r=r_val, coords_eval=X_star, u_star=u_star
        )
        model_3d.pinn.to(device)
        
        total_trainable = sum(p.numel() for p in model_3d.pinn.parameters() if p.requires_grad)
        total_all       = sum(p.numel() for p in model_3d.pinn.parameters())
        print(f"Model for r={r_val} has {total_trainable:,} trainable parameters.")
        
        # Train and store the history
        start_3d = time.time()
        t_adam_3d = time.time()
        history = model_3d.train(n_epochs_adam, X_star, u_star)
        elapsed = time.time() - start_3d
        print(f"Total training time: {time.time()-start_3d:.2f}s")

        u_pred, _, gates_pred = model_3d.predict(X_star)
        error_u_final = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
        print(f'Final Relative L2 error (rel2): {error_u_final:.4e}')

        result_key = f"burgers_r_{r_val}"
        all_results[result_key] = {
                "r": r_val,
                "history": history,
                "final_error": error_u_final,
                "train_time_sec": elapsed,
                "params": total_all
            }

        file_path = "training_results_burgers_r.json"
        with open(file_path, 'w') as f:
            json.dump(all_results, f, indent=4)

        # results.append({
        #     "seed": ms,
        #     'dim': DIM,
        #     "final_rel_l2": error_u_final,
        #     "train_time_sec": elapsed,
        #     "expert_hidden": expert_hidden,
        #     "expert_rank": r_val,
        # })

        # os.makedirs("logs", exist_ok=True)
        # fname = f"logs/burger_results_0.02noise.json"
        # with open(fname, "w") as f:
        #     json.dump(results, f, indent=2)
        # all_histories.append(history)

        ### End training ###

        ### checkpoint ###
        # checkpoint = torch.load("/scratch/shuyuan/random/ckpt_burgers/burgers_final.pt", map_location='cpu')
        # rank = checkpoint['expert_rank']
        # dim = checkpoint['dim']

        # model_3d = Burger(
        #     X_u_train, u_train, X_f_train, lb, ub, nu, 
        #     num_experts=num_experts, expert_hidden=expert_hidden, router_hidden=router_hidden, router_depth=router_depth,
        #     r=r_val, coords_eval=X_star, u_star=u_star
        # )
        # model_3d.pinn.load_state_dict(checkpoint["pinn_state_dict"])
        # model_3d.pinn.eval()
        # u_pred, _, gates_pred = model_3d.predict(X_star)

        # plot_results(X_star, u_pred, Exact, gates_pred, X, T, ms="main")

        # U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
        # fig = plt.figure(figsize=(18, 5))
        # plt.subplot(1, 3, 1)
        # plt.pcolormesh(T, X, Exact, cmap='rainbow', shading='auto')
        # plt.colorbar(); plt.xlabel('t'); plt.ylabel('x'); plt.title('Exact u(t,x)')
        # plt.subplot(1, 3, 2)
        # plt.pcolormesh(T, X, U_pred, cmap='rainbow', shading='auto')
        # plt.colorbar(); plt.xlabel('t'); plt.ylabel('x'); plt.title('Predicted u(t,x)')
        # plt.subplot(1, 3, 3)
        # plt.pcolormesh(T, X, np.abs(Exact - U_pred), cmap='jet', shading='auto')
        # plt.colorbar(); plt.xlabel('t'); plt.ylabel('x'); plt.title('Absolute Error')
        # plt.tight_layout()
        # plt.savefig(f'Burgers_Solution_{ms}.png', dpi=300)

        # fig, axes = plt.subplots(1, num_experts, figsize=(16, 5))

        # for i in range(num_experts):
        #     gate_i_grid = griddata(X_star, gates_pred[:, i], (X, T), method='cubic')
        #     im = axes[i].pcolormesh(T, X, gate_i_grid, cmap='hot', shading='auto', vmin=0, vmax=1)
        #     axes[i].set_xlabel('t', fontsize=18, labelpad=6); axes[i].set_ylabel('x', fontsize=18, labelpad=6)
        #     axes[i].tick_params(axis='both', labelsize=14)
        #     axes[i].set_title(f'Expert {i} Gate Weight', fontsize=16, pad=6)
        #     fig.colorbar(im, ax=axes[i])
        # plt.tight_layout()
        # plt.savefig(f'burgers_domain_{ms}.png', dpi=300)

