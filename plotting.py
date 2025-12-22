import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from utils import params_to_mat


def plot_settings(w0, w1, pi0, pi1, s):
    # Helper to convert to numpy
    def to_np(x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
    
    # 1. Prepare Data
    w0_np = to_np(w0).flatten()
    w1_np = to_np(w1).flatten()
    pi0_np = to_np(pi0).reshape(1, -1)
    pi1_np = to_np(pi1).reshape(1, -1)
    s_np = to_np(s).reshape(1, -1)

    # Calculate N from parameter length: len = N^2 - N
    # N^2 - N - len = 0  =>  N = (1 + sqrt(1 + 4*len)) / 2
    param_len = len(w0_np)
    n = int((1 + np.sqrt(1 + 4 * param_len)) / 2)

    # Reshape weights to matrices
    W0_mat = params_to_mat(w0_np, n)
    W1_mat = params_to_mat(w1_np, n)

    # Annotate if dimensions are small
    annot = n <= 5
    
    # 2. Setup Layout
    # Grid: 
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.3, 2, 0.5], hspace=0.4)
    
    ax_s = fig.add_subplot(gs[0, :])
    ax_w0 = fig.add_subplot(gs[1, 0])
    ax_w1 = fig.add_subplot(gs[1, 1])
    ax_p0 = fig.add_subplot(gs[2, 0])
    ax_p1 = fig.add_subplot(gs[2, 1])

    # 3. Plotting

    # -- Stimulus --
    sns.heatmap(s_np, ax=ax_s, cbar=False, annot=annot, cmap='Greens', square=True,
                yticklabels=False, fmt=".2f", linewidths=0.5, linecolor='gray')
    ax_s.set_title('Stimulus $s$', fontweight='bold', fontsize=14)

    # -- Weights (Matrices) --
    sns.heatmap(W0_mat, ax=ax_w0, cbar=False, annot=annot, cmap='bwr', center=0, square=True, fmt=".2f", linewidths=0.5, linecolor='gray')
    ax_w0.set_title('Initial Weights $w_0$', fontweight='bold', fontsize=12)
    ax_w0.set_xlabel('Post-synaptic')
    ax_w0.set_ylabel('Pre-synaptic')

    sns.heatmap(W1_mat, ax=ax_w1, cbar=False, annot=annot, cmap='bwr', center=0, square=True, fmt=".2f", linewidths=0.5, linecolor='gray' )
    ax_w1.set_title('Final Weights $w_1$', fontweight='bold', fontsize=12)
    ax_w1.set_xlabel('Post-synaptic')
    ax_w1.set_ylabel('Pre-synaptic')

    # -- Functional Outputs --
    sns.heatmap(pi0_np, ax=ax_p0, cbar=False, annot=False, cmap='Purples', square=False,
                yticklabels=False, linewidths=0.1, linecolor='gray')
    ax_p0.set_title('Initial Output $\\phi_0$', fontweight='bold', fontsize=12)

    sns.heatmap(pi1_np, ax=ax_p1, cbar=False, annot=False, cmap='Purples', square=False,
                yticklabels=False, linewidths=0.1, linecolor='gray')
    ax_p1.set_title('Final Output $\\phi_1$', fontweight='bold', fontsize=12)

    # remove ticks from all axes
    for ax in [ax_s, ax_w0, ax_w1, ax_p0, ax_p1]:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_trajectories(solutions, func_embeddings, w0, w1, title='Optimal Trajectories'):
    """
    Plots structural (weight) and functional (MDS) trajectories for a list of Solution objects.
    
    Args:
        solutions: List of Solution objects.
        func_embeddings: List of numpy arrays (T, 2) corresponding to the MDS embedding of each solution.
        w0, w1: Start and End weights (tensors or numpy arrays) for the naive baseline.
    """
    n_cols = len(solutions)
    # squeeze=False ensures axs is always 2D [rows, cols] even if n_cols=1
    fig, axs = plt.subplots(2, n_cols, figsize=(5 * n_cols, 10), squeeze=False)

    # Convert baselines to numpy once for efficiency
    w0_np = w0.detach().cpu().numpy() if isinstance(w0, torch.Tensor) else w0
    w1_np = w1.detach().cpu().numpy() if isinstance(w1, torch.Tensor) else w1

    for i, (sol, embedding) in enumerate(zip(solutions, func_embeddings)):
        
        # --- Extract Data from Solution Object ---
        t = sol.t
        w_traj = sol.w   # Shape (D, T)
        alpha = sol.alpha
        ls_val = sol.L_s
        lf_val = sol.L_f
        
        # --- Row 0: Structural Space (Weights) ---
        # Calculate naive linear interpolation for comparison
        naive_traj = np.linspace(w0_np, w1_np, len(t)).T
        
        for j in range(w_traj.shape[0]):
            # Plot optimized path
            axs[0, i].plot(t, w_traj[j, :], linewidth=2, label=f'$w_{j}$')
            # Plot naive straight line path
            axs[0, i].plot(t, naive_traj[j, :], 'k:', alpha=0.5, label='Naive' if j == 0 else "")
        
        # Title with Alpha and Structural Cost
        axs[0, i].set_title(f'$\\alpha = {alpha}$\nStructural Space [$L_{{struct}}={ls_val:.2f}$]', fontsize=14)
        axs[0, i].set_xlabel('Time')
        if i == 0: axs[0, i].set_ylabel('Weight Value')
        
        # Clean up ticks
        axs[0, i].set_xticks([0, 0.5, 1])
        
        # Only show legend for small networks to avoid clutter
        if w_traj.shape[0] <= 5:
            axs[0, i].legend(fontsize='small')
        
        # --- Row 1: Functional Space (MDS) ---
        # Scatter colored by time
        sc = axs[1, i].scatter(embedding[:, 0], embedding[:, 1], c=t, cmap='jet', s=20, alpha=0.8)
        # Add a faint dashed line connecting the points to show flow
        axs[1, i].plot(embedding[:, 0], embedding[:, 1], 'k--', alpha=0.2, linewidth=1)
        
        # Mark Start and End points explicitly
        axs[1, i].scatter(embedding[0, 0], embedding[0, 1], marker='*', s=200, color='blue', label='Start', edgecolors='k')
        axs[1, i].scatter(embedding[-1, 0], embedding[-1, 1], marker='*', s=200, color='red', label='End', edgecolors='k')
        
        axs[1, i].set_title(f'Functional Space (MDS) [$L_{{func}}={lf_val:.2f}$]', fontsize=14)
        axs[1, i].set_xlabel('Dim 1')
        if i == 0: axs[1, i].set_ylabel('Dim 2')
        
        # Remove ticks for MDS as absolute coordinates are arbitrary
        axs[1, i].set_xticks([])
        axs[1, i].set_yticks([])

    plt.suptitle(title, fontsize=20, y=1.02)
    plt.tight_layout()
    plt.show()