import json
import matplotlib.pyplot as plt
import numpy as np
import os

# FILES_CONFIG = {
#     "Shared MLP (Ours)":    "/scratch/shuyuan/random/training_results_our.json",
#     "Unshared MLP":  "/scratch/shuyuan/random/training_results_unshared.json",
#     "PINNs (Baseline)":   "/scratch/shuyuan/random/training_results_mlp.json"
# }

# STYLES = {
#     "Shared MLP (Ours)":   {"color": "#E31A1C", "ls": "-",  "lw": 3.0, "zorder": 10},
#     "Unshared MLP": {"color": "#1F78B4", "ls": "--", "lw": 2.5, "zorder": 5},
#     "PINNs (Baseline)":  {"color": "#333333", "ls": ":",  "lw": 2.5, "zorder": 1}
# }

# FILES_CONFIG = {
#     "10d_Finetune":    "/scratch/shuyuan/random/training_results_10d_finetune.json",
#     "10d_scratch":  "/scratch/shuyuan/random/training_results_10d_scratch.json",
#     "15d_Finetune":   "/scratch/shuyuan/random/training_results_15d_finetune.json"
# }

# STYLES = {
#     "10d_Finetune":   {"color": "#E31A1C", "ls": "-",  "lw": 3.0, "zorder": 10},
#     "10d_scratch": {"color": "#1F78B4", "ls": "--", "lw": 2.5, "zorder": 5},
#     "15d_Finetune":  {"color": "#333333", "ls": ":",  "lw": 2.5, "zorder": 1}
# }

FILES_CONFIG = {
    "Shared MLP (Ours)":    "/scratch/shuyuan/random/training_results_10d_scratch.json",
    "PINNs (Baseline)":  "/scratch/shuyuan/random/training_results_10d_mlp.json",
}

STYLES = {
    "Shared MLP (Ours)":   {"color": "#E31A1C", "ls": "-",  "lw": 3.0, "zorder": 10},
    "PINNs (Baseline)": {"color": "#1F78B4", "ls": "--", "lw": 2.5, "zorder": 5},
}

LABEL_SIZE = 22    
TICK_SIZE = 18     
LEGEND_SIZE = 18   


def load_all_data():
    all_raw_data = {}
    max_iters_list = []

    for label, filepath in FILES_CONFIG.items():
        if not os.path.exists(filepath):
            print(f"No {filepath}")
            continue
            
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        core_data = data[list(data.keys())[0]]
        history = core_data["history"]
        
        iters = np.array(history["iter"])
        errors = np.array(history["error"])

        min_len = min(len(iters), len(errors))
        iters = iters[:min_len]
        errors = errors[:min_len]
        
        all_raw_data[label] = (iters, errors)
        max_iters_list.append(iters[-1])

    shortest_limit = min(max_iters_list) if max_iters_list else 0
    print(f"truncated steps: {shortest_limit}")
    
    return all_raw_data, shortest_limit

def load_all_time_data():
    all_raw_data = {}
    end_times = []

    for label, filepath in FILES_CONFIG.items():
        if not os.path.exists(filepath):
            print(f"No {filepath}")
            continue
            
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        core_data = data[list(data.keys())[0]]
        history = core_data["history"]
        
        times = np.array(history["elapsed_time"])
        errors = np.array(history["error"])
        
        min_len = min(len(times), len(errors))
        times = times[:min_len]
        errors = errors[:min_len]
        
        all_raw_data[label] = (times, errors)
        if len(times) > 0:
            end_times.append(times[-1]) 

    shortest_time = min(end_times) if end_times else 0
    print(f"truncated time: {shortest_time:.2f}s")
    
    return all_raw_data, shortest_time

def plot_final():
    raw_data, limit = load_all_data()
    
    plt.figure(figsize=(7, 6))
    
    for label, (x, y) in raw_data.items():
        mask = x <= limit
        x_final = x[mask]
        y_final = y[mask]
        
        style = STYLES.get(label, {})
        
        plt.semilogy(x_final, y_final, label=label, 
                     color=style.get("color"), 
                     linestyle=style.get("ls"), 
                     linewidth=style.get("lw"),
                     zorder=style.get("zorder"),
                     alpha=0.9)

    plt.xlabel("Epochs", fontsize=LABEL_SIZE, labelpad=10)
    plt.ylabel("Relative $L^2$ Error", fontsize=LABEL_SIZE, labelpad=10)

    plt.xticks(fontsize=TICK_SIZE)
    plt.yticks(fontsize=TICK_SIZE)

    plt.margins(x=0.05)

    plt.grid(True, which="both", ls="-", alpha=0.15)

    plt.legend(fontsize=LEGEND_SIZE, loc='upper right', frameon=False)

    plt.tight_layout()

#     plt.savefig("error_vs_epoch_truncated.pdf", bbox_inches='tight', dpi=600)
    plt.savefig("error_vs_epoch_truncated.png", bbox_inches='tight', dpi=300)
    print("保存成功: error_vs_epoch_truncated.png")

def plot_time_efficiency():
    raw_data, limit_time = load_all_time_data()
    
    plt.figure(figsize=(7, 6))
    
    for label, (t, e) in raw_data.items():
        mask = t <= limit_time
        t_final = t[mask]
        e_final = e[mask]
        
        style = STYLES.get(label, {})
        
        plt.semilogy(t_final, e_final, label=label, 
                     color=style.get("color"), 
                     linestyle=style.get("ls"), 
                     linewidth=style.get("lw"),
                     zorder=style.get("zorder"),
                     alpha=0.9)

    plt.xlabel("Time (seconds)", fontsize=LABEL_SIZE, labelpad=10)
    plt.ylabel("Relative $L^2$ Error", fontsize=LABEL_SIZE, labelpad=10)

    plt.xticks(fontsize=TICK_SIZE)
    plt.yticks(fontsize=TICK_SIZE)

    plt.margins(x=0.05)

    plt.grid(True, which="both", ls="-", alpha=0.15)

    plt.legend(fontsize=LEGEND_SIZE, loc='upper right', frameon=False)

    plt.tight_layout()

#     plt.savefig("error_vs_time_truncated.pdf", bbox_inches='tight', dpi=600)
    plt.savefig("error_vs_time_truncated.png", bbox_inches='tight', dpi=300)
    print("保存成功: error_vs_time_truncated.png")

def plot_r_comparison(json_path="training_results_burgers_all_r.json", 
                      save_prefix="burgers_comparison", 
                      target_rs=None, 
                      x_mode="epoch"):  
    
    # 字体设置
    LABEL_SIZE = 22    
    TICK_SIZE = 18     
    LEGEND_SIZE = 16  

    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, 'r') as f:
        all_results = json.load(f)

    plot_data = []
    max_vals_list = [] 

    sorted_keys = sorted(all_results.keys(), key=lambda x: int(x.split('_')[-1]))

    for key in sorted_keys:
        res = all_results[key]
        r_val = res.get("r_value", key.split('_')[-1])
        r_val_int = int(r_val)

        if target_rs is not None and r_val_int not in target_rs:
            continue

        history = res["history"]

        if x_mode == "time":
            if "elapsed_time" not in history:
                print(f"Warning: No 'time' data for r={r_val}. Skipping.")
                continue
            x_vals = np.array(history["elapsed_time"])
            x_label = "Wall-clock Time (s)"
        else:
            x_vals = np.array(history["iter"])
            x_label = "Epochs"
        # -------------------------------------

        errors = np.array(history["error"])
        
        # 对齐长度（防止记录时长度不一致）
        min_len = min(len(x_vals), len(errors))
        x_vals, errors = x_vals[:min_len], errors[:min_len]
        
        plot_data.append({
            "label": f"Rank $r={r_val}$",
            "x": x_vals,
            "y": errors
        })
        max_vals_list.append(x_vals[-1])

    if not plot_data:
        print(f"No data to plot for mode={x_mode} with targets={target_rs}")
        return

    # limit = min(max_vals_list) 
    limit = max(max_vals_list) 

    plt.figure(figsize=(7, 6))
    
    styles = [
        # {"color": "#E31A1C", "ls": "-"},  
        {"color": "#1F78B4", "ls": "-"}, 
        {"color": "#33A02C", "ls": "-"}, 
        {"color": "#FF7F00", "ls": "-"},  
        {"color": "#6A3D9A", "ls": "-"}   
    ]

    for i, item in enumerate(plot_data):
        mask = item["x"] <= limit
        x_final = item["x"][mask]
        y_final = item["y"][mask]
        
        style = styles[i % len(styles)]
        
        plt.semilogy(x_final, y_final, 
                     label=item["label"], 
                     color=style["color"], 
                     linestyle=style["ls"], 
                     linewidth=3.0, 
                     alpha=0.9)

    plt.xlabel(x_label, fontsize=LABEL_SIZE, labelpad=10) # 自动设置 x 轴标签
    plt.ylabel("Relative $L^2$ Error", fontsize=LABEL_SIZE, labelpad=10)
    
    plt.xticks(fontsize=TICK_SIZE-2)
    plt.yticks(fontsize=TICK_SIZE)
    plt.margins(x=0.05)
    plt.grid(True, which="both", ls="-", alpha=0.15)
    plt.legend(fontsize=LEGEND_SIZE, loc='upper right', frameon=False)
    plt.tight_layout()

    save_name = f"{save_prefix}_{x_mode}.png"
    plt.savefig(save_name, bbox_inches='tight', dpi=300)
    print(f"Saved {x_mode} comparison to {save_name}")

if __name__ == "__main__":
      path = "/scratch/shuyuan/random/training_results_burgers_r.json"
      plot_r_comparison(json_path=path, save_prefix="burgers_r_comp",target_rs=[4, 8, 16, 32], x_mode="time")
    #   plot_time_efficiency()
    #   plot_final()