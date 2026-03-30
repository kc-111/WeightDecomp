"""
Linear model: W_eff = W + BC
Compare fixed B vs trainable B, analyze B dynamics.

Key questions:
1. Does trainable B beat fixed B?
2. Does trainable B collapse (columns align)?
3. What directions does B evolve toward?
4. How does scaling of B affect things?

Usage: python test_linear_trainable_B.py
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(42)

# ============================================================
# Problem setup
# ============================================================

m = 64
n = 64
r = 8
num_samples = 512
lr = 0.01
num_steps = 2000

# Ill-conditioned data (condition number ~1000)
singular_values = torch.logspace(0, -3, n)
U_data = torch.linalg.qr(torch.randn(n, n))[0]
sqrt_cov = U_data @ torch.diag(singular_values)
X = torch.randn(num_samples, n) @ sqrt_cov.T
W_star = torch.randn(m, n) * 0.1
Y = X @ W_star.T


def compute_loss(W_eff):
    return 0.5 * ((X @ W_eff.T - Y) ** 2).mean()


def get_gradient(W_eff):
    residual = X @ W_eff.T - Y
    return residual.T @ X / num_samples


# ============================================================
# Training functions
# ============================================================

def train_baseline(W_init, lr, num_steps):
    W = W_init.clone()
    losses = []
    for step in range(num_steps):
        loss = compute_loss(W)
        losses.append(loss.item())
        G = get_gradient(W)
        W = W - lr * G
    return losses, None


def train_BC(W_init, B_init, lr, num_steps, B_trainable=True,
             B_scale=1.0, reset_interval=None):
    """
    Train W_eff = W + BC.
    Track B dynamics: singular values of B, alignment with G, effective rank.
    """
    W = W_init.clone()
    B = B_init.clone() * B_scale
    C = torch.zeros(r, n)

    losses = []
    # Track B dynamics
    B_effective_rank = []    # effective rank of BB^T
    B_col_cosines = []       # avg pairwise cosine between columns of B (collapse measure)
    B_alignment_top = []     # alignment of B with top singular vectors of G
    B_alignment_bot = []     # alignment of B with bottom singular vectors of G
    precond_cond_number = [] # condition number of (I + BB^T)

    for step in range(num_steps):
        W_eff = W + B @ C
        loss = compute_loss(W_eff)
        losses.append(loss.item())

        G = get_gradient(W_eff)

        # --- Analysis every 50 steps ---
        if step % 50 == 0:
            # Effective rank of BB^T (via entropy of normalized eigenvalues)
            eigvals = torch.linalg.eigvalsh(B @ B.T)
            eigvals = eigvals.clamp(min=1e-12)
            p = eigvals / eigvals.sum()
            eff_rank = torch.exp(-torch.sum(p * torch.log(p))).item()
            B_effective_rank.append(eff_rank)

            # Column collapse: average pairwise cosine similarity
            B_normed = B / (B.norm(dim=0, keepdim=True) + 1e-12)
            cosine_matrix = B_normed.T @ B_normed
            mask = ~torch.eye(r, dtype=bool)
            avg_cosine = cosine_matrix[mask].abs().mean().item()
            B_col_cosines.append(avg_cosine)

            # Alignment with gradient singular vectors
            if torch.isfinite(G).all() and torch.isfinite(B).all():
                U_G, S_G, Vt_G = torch.linalg.svd(G, full_matrices=False)
                U_G_top = U_G[:, :r]
                U_G_bot = U_G[:, -r:]
                top_align = (U_G_top.T @ B).norm() ** 2 / (B.norm() ** 2 + 1e-12)
                bot_align = (U_G_bot.T @ B).norm() ** 2 / (B.norm() ** 2 + 1e-12)
                B_alignment_top.append(top_align.item())
                B_alignment_bot.append(bot_align.item())

                P_eigs = torch.linalg.eigvalsh(torch.eye(m) + B @ B.T)
                precond_cond_number.append((P_eigs[-1] / P_eigs[0]).item())
            else:
                B_alignment_top.append(float('nan'))
                B_alignment_bot.append(float('nan'))
                precond_cond_number.append(float('nan'))

        # --- Updates ---
        if not torch.isfinite(G).all():
            # Diverged, fill remaining with last loss
            losses.extend([losses[-1]] * (num_steps - step - 1))
            break

        W = W - lr * G
        C = C - lr * (B.T @ G)
        if B_trainable:
            B = B - lr * (G @ C.T)

        # Periodic reset
        if reset_interval and (step + 1) % reset_interval == 0:
            W = W + B @ C
            C = torch.zeros(r, n)

    diagnostics = {
        'effective_rank': B_effective_rank,
        'col_cosines': B_col_cosines,
        'alignment_top': B_alignment_top,
        'alignment_bot': B_alignment_bot,
        'precond_cond': precond_cond_number,
    }
    return losses, diagnostics


# ============================================================
# Run experiments
# ============================================================

W_init = torch.randn(m, n) * 0.5
B_init = torch.randn(m, r)

scales = [0.1, 0.3, 0.5, 1.0]

print("Running baseline...")
losses_base, _ = train_baseline(W_init, lr, num_steps)

results = {}
for trainable in [False, True]:
    for scale in scales:
        tag = f"{'trainable' if trainable else 'fixed'}_scale{scale}"
        print(f"Running: {tag}...")
        losses, diag = train_BC(W_init, B_init, lr, num_steps,
                                B_trainable=trainable, B_scale=scale)
        results[tag] = (losses, diag)

# Also run trainable with reset
print("Running: trainable with reset...")
losses_reset, diag_reset = train_BC(W_init, B_init, lr, num_steps,
                                     B_trainable=True, B_scale=1.0,
                                     reset_interval=200)
results['trainable_reset'] = (losses_reset, diag_reset)


# ============================================================
# Plot results
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# --- Plot 1: Loss curves, fixed B at different scales ---
ax = axes[0, 0]
ax.semilogy(losses_base, 'k-', linewidth=2, label='Baseline')
for scale in scales:
    tag = f'fixed_scale{scale}'
    ax.semilogy(results[tag][0], alpha=0.8, label=f'Fixed B, scale={scale}')
ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.set_title('Fixed B: Effect of Scale')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# --- Plot 2: Loss curves, trainable B at different scales ---
ax = axes[0, 1]
ax.semilogy(losses_base, 'k-', linewidth=2, label='Baseline')
for scale in scales:
    tag = f'trainable_scale{scale}'
    ax.semilogy(results[tag][0], alpha=0.8, label=f'Trainable B, scale={scale}')
ax.semilogy(results['trainable_reset'][0], '--', color='tab:red',
            alpha=0.8, label='Trainable B + reset')
ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.set_title('Trainable B: Effect of Scale')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# --- Plot 3: Best fixed vs best trainable ---
ax = axes[0, 2]
ax.semilogy(losses_base, 'k-', linewidth=2, label='Baseline')
best_fixed = min(scales, key=lambda s: results[f'fixed_scale{s}'][0][-1])
best_train = min(scales, key=lambda s: results[f'trainable_scale{s}'][0][-1])
ax.semilogy(results[f'fixed_scale{best_fixed}'][0],
            label=f'Best fixed (scale={best_fixed})', linewidth=2)
ax.semilogy(results[f'trainable_scale{best_train}'][0],
            label=f'Best trainable (scale={best_train})', linewidth=2)
ax.semilogy(results['trainable_reset'][0], '--',
            label='Trainable + reset', linewidth=2)
ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.set_title('Best Fixed vs Best Trainable')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Plot 4: Column collapse (pairwise cosine) ---
ax = axes[1, 0]
analysis_steps = list(range(0, num_steps, 50))
for scale in [0.5, 1.0]:
    tag_f = f'fixed_scale{scale}'
    tag_t = f'trainable_scale{scale}'
    ax.plot(analysis_steps, results[tag_f][1]['col_cosines'],
            '--', alpha=0.8, label=f'Fixed, s={scale}')
    ax.plot(analysis_steps, results[tag_t][1]['col_cosines'],
            '-', alpha=0.8, label=f'Trainable, s={scale}')
ax.set_xlabel('Step')
ax.set_ylabel('Avg |cosine| between B columns')
ax.set_title('Column Collapse of B')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# --- Plot 5: Effective rank of BB^T ---
ax = axes[1, 1]
for scale in [0.5, 1.0]:
    tag_f = f'fixed_scale{scale}'
    tag_t = f'trainable_scale{scale}'
    ax.plot(analysis_steps, results[tag_f][1]['effective_rank'],
            '--', alpha=0.8, label=f'Fixed, s={scale}')
    ax.plot(analysis_steps, results[tag_t][1]['effective_rank'],
            '-', alpha=0.8, label=f'Trainable, s={scale}')
if diag_reset:
    ax.plot(analysis_steps, diag_reset['effective_rank'],
            '-', color='tab:red', alpha=0.8, label='Trainable+reset')
ax.set_xlabel('Step')
ax.set_ylabel('Effective rank of BB^T')
ax.set_title('Effective Rank of Preconditioner')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# --- Plot 6: Alignment of B with gradient singular vectors ---
ax = axes[1, 2]
for scale in [1.0]:
    tag_t = f'trainable_scale{scale}'
    ax.plot(analysis_steps, results[tag_t][1]['alignment_top'],
            '-', color='tab:red', label=f'Trainable: align w/ top SV of G')
    ax.plot(analysis_steps, results[tag_t][1]['alignment_bot'],
            '-', color='tab:blue', label=f'Trainable: align w/ bottom SV of G')
    tag_f = f'fixed_scale{scale}'
    ax.plot(analysis_steps, results[tag_f][1]['alignment_top'],
            '--', color='tab:red', alpha=0.5, label=f'Fixed: align w/ top SV of G')
    ax.plot(analysis_steps, results[tag_f][1]['alignment_bot'],
            '--', color='tab:blue', alpha=0.5, label=f'Fixed: align w/ bottom SV of G')
ax.set_xlabel('Step')
ax.set_ylabel('Fractional alignment')
ax.set_title('B Alignment with Gradient Subspaces')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('linear_trainable_B_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Summary table
# ============================================================

print("\n" + "=" * 70)
print(f"{'Condition':<35s} {'Final Loss':>12s} {'Speedup vs Base':>15s}")
print("=" * 70)
print(f"{'Baseline':<35s} {losses_base[-1]:>12.6f} {'1.00x':>15s}")
for tag, (losses, diag) in sorted(results.items()):
    # Compute steps to reach baseline's final loss
    target = losses_base[-1]
    steps_to_target = num_steps
    for i, l in enumerate(losses):
        if l <= target:
            steps_to_target = i
            break
    speedup = num_steps / steps_to_target if steps_to_target > 0 else float('inf')
    print(f"{tag:<35s} {losses[-1]:>12.6f} {speedup:>14.1f}x")
print("=" * 70)