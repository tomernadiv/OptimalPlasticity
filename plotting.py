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
                yticklabels=False)
    ax_s.set_title('Stimulus $s$', fontweight='bold', fontsize=14)

    # -- Weights (Matrices) --
    sns.heatmap(W0_mat, ax=ax_w0, cbar=False, annot=annot, cmap='bwr', center=0, square=True, fmt=".2f")
    ax_w0.set_title('Initial Weights $w_0$', fontweight='bold', fontsize=12)
    ax_w0.set_xlabel('Post-synaptic')
    ax_w0.set_ylabel('Pre-synaptic')

    sns.heatmap(W1_mat, ax=ax_w1, cbar=False, annot=annot, cmap='bwr', center=0, square=True, fmt=".2f")
    ax_w1.set_title('Final Weights $w_1$', fontweight='bold', fontsize=12)
    ax_w1.set_xlabel('Post-synaptic')
    ax_w1.set_ylabel('Pre-synaptic')

    # -- Functional Outputs --
    sns.heatmap(pi0_np, ax=ax_p0, cbar=False, annot=annot, cmap='Purples', square=True, fmt=".2f",
                yticklabels=False)
    ax_p0.set_title('Initial Output $\\phi_0$', fontweight='bold', fontsize=12)

    sns.heatmap(pi1_np, ax=ax_p1, cbar=False, annot=annot, cmap='Purples', square=True, fmt=".2f",
                yticklabels=False)
    ax_p1.set_title('Final Output $\\phi_1$', fontweight='bold', fontsize=12)

    # remove ticks from all axes
    for ax in [ax_s, ax_w0, ax_w1, ax_p0, ax_p1]:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()

def plot_trajectories(sols, func_embeddings, w0, w1, title='Optimal Trajectories'):
    n_cols = len(sols)
    # squeeze=False ensures axs is always 2D [rows, cols]
    fig, axs = plt.subplots(2, n_cols, figsize=(5*n_cols, 12), squeeze=False)

    for i, ((_, w_traj, alpha), (_, embedding, alpha_f)) in enumerate(zip(sols, func_embeddings)):
        
        # --- Row 0: Structural Space (Weights) ---
        t = np.linspace(0, 1, w_traj.shape[1])
        w0_np = w0.detach().cpu().numpy() if isinstance(w0, torch.Tensor) else w0
        w1_np = w1.detach().cpu().numpy() if isinstance(w1, torch.Tensor) else w1
        
        naive_traj = np.linspace(w0_np, w1_np, w_traj.shape[1]).T
        
        for j in range(w_traj.shape[0]):
            axs[0,i].plot(t, w_traj[j, :], label=f'$w_{j}$')
            axs[0,i].plot(t, naive_traj[j, :], 'k--', label='Naive' if j==0 else "")
            
        axs[0,i].scatter(t[0], w_traj[j, 0], marker='*', s=500, color='blue')
        axs[0,i].scatter(t[-1], w_traj[j, -1], marker='*', s=500, color='red')
        
        # COMBINED TITLE: Alpha (Column Header) + Space Name (Plot Title)
        axs[0,i].set_title(f'alpha = {alpha}\n\nStructural Space', fontsize=16)
        axs[0,i].set_xlabel('Time')
        axs[0,i].set_ylabel('Weights')
        axs[0,i].set_xticks([])
        axs[0,i].set_yticks([])

        if w0.shape[0] <= 5:
            axs[0,i].legend()
        
        # --- Row 1: Functional Space (MDS) ---
        axs[1,i].scatter(embedding[:,0], embedding[:,1], c=t, cmap='jet')
        axs[1,i].plot(embedding[:,0], embedding[:,1], 'k--', alpha=0.3)  
        axs[1,i].scatter(embedding[0,0], embedding[0,1], marker='*', s=500, color='blue', label='Start')
        axs[1,i].scatter(embedding[-1,0], embedding[-1,1], marker='*', s=500, color='red', label='End')
        
        axs[1,i].set_title('Functional Space (MDS)', fontsize=16)
        axs[1,i].set_xlabel('Dim 1')
        axs[1,i].set_ylabel('Dim 2')
        axs[1,i].set_xticks([])
        axs[1,i].set_yticks([])

    plt.suptitle(title, fontsize=24)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()