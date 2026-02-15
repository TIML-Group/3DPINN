import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import os

def poisson5d_slice_sampler(
    res1=161, res2=161, fixed_val=0.5,
    x_min=0.0, x_max=1.0, dim=5
):
    x1 = np.linspace(x_min, x_max, res1)
    x2 = np.linspace(x_min, x_max, res2)
    X1, X2 = np.meshgrid(x1, x2, indexing="ij")

    n = X1.size
    fixed_block = np.full((n, dim - 2), fixed_val)

    coords = np.concatenate(
        [np.stack([X1.ravel(), X2.ravel()], axis=-1), fixed_block],
        axis=-1
    )

    return X1, X2, coords

def poisson10d_slice_sampler(
    res1=161, res2=161, fixed_val=0.5,
    x_min=0.0, x_max=1.0, dim=10
):
    x1 = np.linspace(x_min, x_max, res1)
    x2 = np.linspace(x_min, x_max, res2)
    X1, X2 = np.meshgrid(x1, x2, indexing="ij")

    n = X1.size
    fixed_block = np.full((n, dim - 2), fixed_val)

    coords = np.concatenate(
        [np.stack([X1.ravel(), X2.ravel()], axis=-1), fixed_block],
        axis=-1
    )

    return X1, X2, coords

def wave_slice_sampler(
    res_t=201, res_x=101, 
    min=0.0, max=1.0
):
    t_vals = np.linspace(min, max, res_t)
    x_vals = np.linspace(min, max, res_x)
    grid_t, grid_x = np.meshgrid(t_vals, x_vals, indexing="ij")
    coords = np.stack([grid_t.ravel(), grid_x.ravel()], axis=-1)

    return grid_t, grid_x, coords

def wave_slice_sampler_2d(
    res_t=201, res_x=101, 
    min=0.0, max=1.0
):
    t_vals = np.linspace(min, max, res_t)
    x_vals = np.linspace(min, max, res_x)
    y_vals = np.linspace(min, max, res_x)
    grid_t, grid_x, grid_y = np.meshgrid(t_vals, x_vals, y_vals, indexing="ij")
    coords = np.stack([grid_t.ravel(), grid_x.ravel(), grid_y.ravel()], axis=-1)

    return grid_t, grid_x, grid_y, coords

def ADR_slice_sampler(
    res_t=201, res_x=101, 
    min=0.0, max=1.0
):
    t_vals = np.linspace(min, max, res_t)
    x_vals = np.linspace(min, max, res_x)
    grid_t, grid_x = np.meshgrid(t_vals, x_vals, indexing="ij")
    coords = np.stack([grid_t.ravel(), grid_x.ravel()], axis=-1)

    return grid_t, grid_x, coords

@torch.no_grad()
def infer_on_coords(model, coords, normalize_fn=None, device="cuda:7"):
    model.eval()
    coords_t = torch.from_numpy(coords).float().to(device)
    if normalize_fn is not None:
        coords_t = normalize_fn(coords_t)
    u_pred, gates = model(coords_t)
    return (
        u_pred.detach().cpu().numpy(),
        None if gates is None else gates.detach().cpu().numpy()
    )

def visualize_solution_2d(
    X, Y, U_pred, U_true, error, out_path,
    labels=("x", "y"), title_prefix=""
):
    fig = plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.pcolormesh(X, Y, U_true, shading="auto")
    plt.colorbar(); plt.title(f"{title_prefix}Exact")

    plt.subplot(1, 3, 2)
    plt.pcolormesh(X, Y, U_pred, shading="auto")
    plt.colorbar(); plt.title(f"{title_prefix}Pred")

    plt.subplot(1, 3, 3)
    plt.pcolormesh(X, Y, np.abs(U_pred - U_true), shading="auto")
    plt.colorbar(); plt.title(f"L2 Rel: {error:.2e}")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

def save_expert_gates(TT, XX, G, out_dir, ep):
    FONT_TITLE = 22
    FONT_LABEL = 24
    FONT_TICK = 20
    FONT_CBAR = 18

    E = G.shape[-1]
    
    for i in range(E):

        fig = plt.figure(figsize=(8, 6.5))

        plt.pcolormesh(TT, XX, G[:, :, i], cmap="hot", shading="auto", vmin=0, vmax=1)

        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=FONT_CBAR)

        plt.xlabel("t", fontsize=FONT_LABEL, labelpad=10)
        plt.ylabel("x", fontsize=FONT_LABEL, labelpad=10)
        plt.tick_params(axis="both", labelsize=FONT_TICK)
        
        plt.title(f"Expert {i+1} Gate Weight", fontsize=FONT_TITLE, pad=15)
        
        plt.tight_layout()
        
        gate_png = os.path.join(out_dir, f"gate_ep{ep:05d}_expert{i+1:02d}.png")
        plt.savefig(gate_png, dpi=300, bbox_inches='tight') 
        plt.close(fig)
        
    print(f"  -> Saved {E} expert gate plots to {out_dir}")


def visualize_solution_single(X, Y, U_true, out_path, labels=("t", "x")):
    LABEL_SIZE = 24    
    TICK_SIZE = 20     
    CB_TICK_SIZE = 18  

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    im = ax.pcolormesh(X, Y, U_true, shading="auto", cmap="viridis")

    ax.set_xlabel(labels[0], fontsize=LABEL_SIZE, labelpad=10)
    ax.set_ylabel(labels[1], fontsize=LABEL_SIZE, labelpad=10)

    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE, length=6, width=1.5)

    cbar = fig.colorbar(im, ax=ax, pad=0.03, fraction=0.046)
    cbar.ax.tick_params(labelsize=CB_TICK_SIZE)

    plt.title("Exact Solution", fontsize=LABEL_SIZE) 

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=600)
    # plt.savefig(out_path.replace(".png", ".pdf"), bbox_inches='tight')
    plt.close(fig)
    print(f"Single 2D plot saved to {out_path}")

