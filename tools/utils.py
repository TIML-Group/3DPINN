import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import os

def draw_interfaces(ax, color='white'):
    """Helper to draw the 3-subdomain interfaces."""
    ax.axvline(0.0, color=color, lw=2.0, ls='--', alpha=0.85)   
    ax.plot([0, 1], [0, 0], color=color, lw=2.0, ls='--', alpha=0.85)

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

def AC_slice_sampler(
    res_t=201, res_x=201, 
    min=-1.0, max=1.0
):
    t_vals = np.linspace(min, max, res_t)
    x_vals = np.linspace(min, max, res_x)
    grid_t, grid_x = np.meshgrid(t_vals, x_vals, indexing="ij")
    coords = np.stack([grid_t.ravel(), grid_x.ravel()], axis=-1)

    return grid_t, grid_x, coords

@torch.no_grad()
def infer_on_coords(model, coords, normalize_fn=None, device="cuda:4"):
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
    FONT_TITLE = 24
    FONT_LABEL = 22
    FONT_TICK = 20
    FONT_CBAR = 18

    E = G.shape[-1]

    # Create a figure with 1x2 subplot layout
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), dpi=300)
    axes = axes.flatten()  # Flatten to 1D array for easy indexing

    # Plot each expert gate
    for i in range(E):
        ax = axes[i]
        im = ax.pcolormesh(TT, XX, G[:, :, i], cmap="hot", shading="auto", vmin=0, vmax=1)

        # Add colorbar to each subplot
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=FONT_CBAR)

        # Set labels
        ax.set_xlabel("t", fontsize=FONT_LABEL, labelpad=8)
        # Only show y-label for first subplot
        if i == 0:
            ax.set_ylabel("x", fontsize=FONT_LABEL, labelpad=8)
        else:
            # Completely remove y-axis for second subplot
            ax.yaxis.set_visible(False)
        ax.tick_params(axis="both", labelsize=FONT_TICK)

        # Set title
        ax.set_title(f"Expert {i+1}", fontsize=FONT_TITLE, pad=12)

        # Draw interfaces
        # draw_interfaces(ax, color='black')

    # Hide any unused subplots (if E < 2)
    for i in range(E, len(axes)):
        axes[i].set_visible(False)

    # Add a main title for the entire figure
    # fig.suptitle(f"Expert Gates - Episode {ep:05d}", fontsize=28, y=0.98)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.08, left=0.08, right=0.96, wspace=0.05)

    # Save the combined figure
    gate_png = os.path.join(out_dir, f"gate_ep{ep:05d}_combined.png")
    plt.savefig(gate_png, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"  -> Saved combined expert gate plot to {out_dir}")


def visualize_solution_single(X, Y, U_true, out_path, labels=("t", "x")):
    LABEL_SIZE = 24    
    TICK_SIZE = 20     
    CB_TICK_SIZE = 18  

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    im = ax.pcolormesh(X, Y, U_true, shading="auto", cmap="viridis")

    # ax.set_xlabel(labels[0], fontsize=LABEL_SIZE, labelpad=10)
    # ax.set_ylabel(labels[1], fontsize=LABEL_SIZE, labelpad=10)
    ax.set_xlabel("x1", fontsize=LABEL_SIZE, labelpad=10)
    ax.set_ylabel("x2", fontsize=LABEL_SIZE, labelpad=10)

    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE, length=6, width=1.5)

    cbar = fig.colorbar(im, ax=ax, pad=0.03, fraction=0.046)
    cbar.ax.tick_params(labelsize=CB_TICK_SIZE)

    plt.title("Exact Poisson", fontsize=LABEL_SIZE) 
    line_color = 'black' 
    draw_interfaces(ax, color=line_color)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=600)
    # plt.savefig(out_path.replace(".png", ".pdf"), bbox_inches='tight')
    plt.close(fig)
    print(f"Single 2D plot saved to {out_path}")


def visualize_slice_single_5d(X1, X2, Field, out_path, title, cmap="jet", vmin=None, vmax=None):

    LABEL_SIZE = 24    
    TICK_SIZE = 20     
    CB_TICK_SIZE = 18  

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # 替换 contourf 为 pcolormesh
    im = ax.pcolormesh(X1, X2, Field, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xlabel("$x_1$", fontsize=LABEL_SIZE, labelpad=10)
    ax.set_ylabel("$x_2$", fontsize=LABEL_SIZE, labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE, length=6, width=1.5)

    cbar = fig.colorbar(im, ax=ax, pad=0.03, fraction=0.046)
    cbar.ax.tick_params(labelsize=CB_TICK_SIZE)

    ax.set_title(title, fontsize=LABEL_SIZE, pad=15) 
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

def save_expert_gates_5d(X1, X2, G, out_dir, epoch_name):
    FONT_TITLE = 24
    FONT_LABEL = 22
    FONT_TICK = 20
    FONT_CBAR = 18

    E = G.shape[-1]

    # Create a figure with 1x2 subplot layout
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), dpi=300)
    axes = axes.flatten()  # Flatten to 1D array for easy indexing

    # Plot each expert gate
    for i in range(E):
        ax = axes[i]
        im = ax.pcolormesh(X1, X2, G[:, :, i], cmap="hot", shading="auto", vmin=0, vmax=1)

        # Add colorbar to each subplot
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=FONT_CBAR)

        # Set labels
        ax.set_xlabel("$x_1$", fontsize=FONT_LABEL, labelpad=8)

        # Only show y-label for first subplot
        if i == 0:
            ax.set_ylabel("x2", fontsize=FONT_LABEL, labelpad=8)
        else:
            # Completely remove y-axis for second subplot
            ax.yaxis.set_visible(False)
        ax.tick_params(axis="both", labelsize=FONT_TICK)

        # Set title
        ax.set_title(f"Expert {i+1}", fontsize=FONT_TITLE, pad=12)

        # Draw interfaces
        # draw_interfaces(ax, color='black')

    # Hide any unused subplots (if E < 2)
    for i in range(E, len(axes)):
        axes[i].set_visible(False)

    # Add a main title for the entire figure
    # fig.suptitle(f"Expert Gates - {epoch_name}", fontsize=28, y=0.98)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.08, left=0.08, right=0.96, wspace=0.05)

    # Save the combined figure
    gate_png = os.path.join(out_dir, f"gate_ep{epoch_name}_combined.png")
    plt.savefig(gate_png, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"  -> Saved expert gates to {out_dir}")

if __name__ == "__main__":
    data = np.load("/random/unsteady_ADR/final_data_ep99999.npz")
    print(data.files)
    t_vals = data["t_grid"]
    x_vals = data["x_grid"]
    u_pred = data["u_pred"]
    u_true = data["u_true"]
    G = data["gates"]
    error = data["error"]

    print("Relative error:", error)
    out_dir="unsteady_ADR"
    ep = 99999
    save_expert_gates(t_vals, x_vals, G, out_dir, ep)
    png = os.path.join(out_dir, f"Unsteady_ADR_slice_ep{ep:05d}.png")
    visualize_solution_single(t_vals, x_vals, u_true, png)

