"""
Two-layer MLP fitting linear data with stability-constrained parameterization.

Data: Y = X @ W_star.T  (linear)
Model: Y_hat = W2 @ ReLU(X @ W1.T)  (nonlinear)

W1, W2 are parameterized via structured factorizations inspired by
Stability-guaranteed Generalized Lotka-Volterra (StableGLV) models.
All modes guarantee Re(eigenvalues) < 0 (Hurwitz stability).

Modes:
  'VL'       : Volterra-Lyapunov diagonal stability.
               W = D * (-L*L^T - eps*I + K)
               D = diag(d), d_i > 0.

  'hurwitz'  : General Hurwitz stability via full PD scaling.
               W = D * (-L*L^T - eps*I + K)
               D = C*C^T + eps*I.

  'diag_dom' : Diagonal dominance via Gershgorin discs.
               Off-diag free; diag = -(row_abs_sum + softplus(margin)).

Overparameterization via n_components K: each mode sums K independent
structured terms, preserving the stability guarantee.

Key questions:
1. How does stability-constrained parameterization affect convergence?
2. Does overparameterization (K > 1) help?
3. How do VL, hurwitz, and diag_dom compare?

Usage: python stable_param_linear_test.py
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(42)

# ============================================================
# Problem setup (same as linear_test.py)
# ============================================================

d_in = 128    # input dim
d_hid = 128   # hidden dim
d_out = 128   # output dim
num_samples = 1024
num_steps = 2000

# Ill-conditioned data (condition number ~1e5)
singular_values = torch.logspace(0, -5, d_in)
U_data = torch.linalg.qr(torch.randn(d_in, d_in))[0]
sqrt_cov = U_data @ torch.diag(singular_values)
X = torch.randn(num_samples, d_in) @ sqrt_cov.T
W_star = torch.randn(d_out, d_in) * 0.1
Y = X @ W_star.T  # linear data


def compute_loss(Y_hat, Y):
    return 0.5 * ((Y_hat - Y) ** 2).mean()


# ============================================================
# Structured parameterization builders
# ============================================================

def sftb(x):
    """Softplus with beta=10."""
    return F.softplus(10.0 * x) / 10.0


def flat_to_tril(flat, N):
    """Differentiably place flat vector [N*(N+1)/2] into lower-triangular NxN matrix."""
    idx = torch.tril_indices(N, N, device=flat.device)
    flat_pos = idx[0] * N + idx[1]
    result = torch.zeros(N * N, dtype=flat.dtype, device=flat.device)
    result = result.scatter(0, flat_pos, flat)
    return result.reshape(N, N)


def flat_to_strict_tril(flat, N):
    """Differentiably place flat vector [N*(N-1)/2] into strict lower-triangular NxN matrix."""
    idx = torch.tril_indices(N, N, offset=-1, device=flat.device)
    flat_pos = idx[0] * N + idx[1]
    result = torch.zeros(N * N, dtype=flat.dtype, device=flat.device)
    result = result.scatter(0, flat_pos, flat)
    return result.reshape(N, N)


def build_W_VL(L_flat, M_flat, D_raw, eps_raw, N):
    """VL mode: W = diag(softplus(d)) * (-LLT - eps*I + Skew).

    When n_components > 1, L_flat is [K, L_size] and LLT = sum_k L_k L_k^T.
    """
    eps = sftb(eps_raw) + 5e-4
    eye = torch.eye(N, dtype=L_flat.dtype, device=L_flat.device)

    if L_flat.dim() == 1:
        L = flat_to_tril(L_flat, N)
        LLT = L @ L.T
        M = flat_to_strict_tril(M_flat, N)
        Skew = M - M.T
        D_diag = sftb(D_raw)
    else:
        K = L_flat.shape[0]
        LLT = torch.zeros(N, N, dtype=L_flat.dtype, device=L_flat.device)
        Skew = torch.zeros(N, N, dtype=M_flat.dtype, device=M_flat.device)
        for c in range(K):
            L = flat_to_tril(L_flat[c], N)
            LLT = LLT + L @ L.T
            M = flat_to_strict_tril(M_flat[c], N)
            Skew = Skew + (M - M.T)
        D_diag = sftb(D_raw).sum(dim=0)

    inner = -LLT - eps * eye + Skew
    return D_diag.unsqueeze(1) * inner


def build_W_hurwitz(L_flat, M_flat, D_raw, eps_raw, N):
    """Hurwitz mode: W = (C@C^T + eps*I) * (-LLT - eps*I + Skew).

    D_raw parameterizes lower-triangular C; D = C@C^T + eps*I is full PD.
    """
    eps = sftb(eps_raw) + 5e-4
    eye = torch.eye(N, dtype=L_flat.dtype, device=L_flat.device)

    if L_flat.dim() == 1:
        L = flat_to_tril(L_flat, N)
        LLT = L @ L.T
        M = flat_to_strict_tril(M_flat, N)
        Skew = M - M.T
        C = flat_to_tril(D_raw, N)
        D = C @ C.T + eps * eye
    else:
        K = L_flat.shape[0]
        LLT = torch.zeros(N, N, dtype=L_flat.dtype, device=L_flat.device)
        Skew = torch.zeros(N, N, dtype=M_flat.dtype, device=M_flat.device)
        D = eps * eye
        for c in range(K):
            L = flat_to_tril(L_flat[c], N)
            LLT = LLT + L @ L.T
            M = flat_to_strict_tril(M_flat[c], N)
            Skew = Skew + (M - M.T)
            C = flat_to_tril(D_raw[c], N)
            D = D + C @ C.T

    inner = -LLT - eps * eye + Skew
    return D @ inner


def build_W_diag_dom(R_raw, N):
    """Diag-dom mode: off-diag free, diag = -(row_abs_sum + softplus(margin)).

    Sum of diag-dominant matrices with negative diag is diag-dominant.
    """
    eye = torch.eye(N, dtype=R_raw.dtype, device=R_raw.device)

    if R_raw.dim() == 2:
        R = R_raw * (1.0 - eye)
        row_abs_sum = R.abs().sum(dim=1)
        margin = sftb(torch.diag(R_raw))
        return R + torch.diag(-(row_abs_sum + margin))
    else:
        K = R_raw.shape[0]
        W = torch.zeros(N, N, dtype=R_raw.dtype, device=R_raw.device)
        for c in range(K):
            R = R_raw[c] * (1.0 - eye)
            row_abs_sum = R.abs().sum(dim=1)
            margin = sftb(torch.diag(R_raw[c]))
            W = W + R + torch.diag(-(row_abs_sum + margin))
        return W


def build_W(params, mode, N):
    """Dispatch to the appropriate builder."""
    if mode == 'diag_dom':
        return build_W_diag_dom(params['R_raw'], N)
    elif mode == 'VL':
        return build_W_VL(params['L_flat'], params['M_flat'],
                          params['D_raw'], params['eps_raw'], N)
    else:
        return build_W_hurwitz(params['L_flat'], params['M_flat'],
                               params['D_raw'], params['eps_raw'], N)


def init_params(mode, N, n_components, scale=0.1):
    """Initialize raw parameters for a given mode and n_components K."""
    L_size = N * (N + 1) // 2
    M_size = N * (N - 1) // 2
    comp_scale = scale / np.sqrt(n_components)

    if mode == 'diag_dom':
        if n_components == 1:
            R = torch.randn(N, N) * scale
        else:
            R = torch.randn(n_components, N, N) * comp_scale
        return {'R_raw': torch.nn.Parameter(R)}

    if n_components == 1:
        L = torch.randn(L_size) * scale
        M = torch.randn(M_size) * scale
        D_size = N if mode == 'VL' else L_size
        D = torch.randn(D_size) * scale
    else:
        L = torch.randn(n_components, L_size) * comp_scale
        M = torch.randn(n_components, M_size) * comp_scale
        D_size = N if mode == 'VL' else L_size
        D = torch.randn(n_components, D_size) * comp_scale

    return {
        'L_flat': torch.nn.Parameter(L),
        'M_flat': torch.nn.Parameter(M),
        'D_raw': torch.nn.Parameter(D),
        'eps_raw': torch.nn.Parameter(torch.tensor(0.0)),
    }


# ============================================================
# Training functions
# ============================================================

def train_baseline(W1_init, W2_init, lr, num_steps):
    W1 = torch.nn.Parameter(W1_init.clone())
    W2 = torch.nn.Parameter(W2_init.clone())
    optimizer = torch.optim.Adam([W1, W2], lr=lr)
    losses = []
    for step in range(num_steps):
        optimizer.zero_grad()
        Y_hat = torch.relu(X @ W1.T) @ W2.T
        loss = compute_loss(Y_hat, Y)
        losses.append(loss.item())
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in [W1, W2]):
            losses.extend([losses[-1]] * (num_steps - step - 1))
            break
        optimizer.step()
    return losses


def train_stable(mode, n_components, lr, num_steps):
    """Train with stability-constrained parameterization of W."""
    params1 = init_params(mode, d_hid, n_components)
    params2 = init_params(mode, d_out, n_components)
    all_params = list(params1.values()) + list(params2.values())
    optimizer = torch.optim.Adam(all_params, lr=lr)

    losses = []
    for step in range(num_steps):
        optimizer.zero_grad()
        W1 = build_W(params1, mode, d_hid)
        W2 = build_W(params2, mode, d_out)
        Y_hat = torch.relu(X @ W1.T) @ W2.T
        loss = compute_loss(Y_hat, Y)
        losses.append(loss.item())
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in all_params):
            losses.extend([losses[-1]] * (num_steps - step - 1))
            break
        optimizer.step()

    return losses, params1, params2


def get_diagnostics(mode, n_components, lr, num_steps):
    """Run training and return end-of-training diagnostics for W1."""
    losses, params1, _ = train_stable(mode, n_components, lr, num_steps)

    with torch.no_grad():
        W1 = build_W(params1, mode, d_hid)

        # Spectral abscissa: max Re(lambda) -- should be < 0
        eigvals = torch.linalg.eigvals(W1)
        spectral_abscissa = eigvals.real.max().item()

        # Singular value analysis
        svs = torch.linalg.svdvals(W1)
        sv_cond = (svs[0] / svs[-1].clamp(min=1e-12)).item()

        # Effective rank (entropy of normalized SVs)
        p = svs / svs.sum()
        p = p.clamp(min=1e-12)
        eff_rank = torch.exp(-torch.sum(p * torch.log(p))).item()

        # Symmetric vs skew-symmetric decomposition
        sym_part = (W1 + W1.T) / 2
        skew_part = (W1 - W1.T) / 2
        sym_norm = sym_part.norm().item()
        skew_norm = skew_part.norm().item()

    return {
        'spectral_abscissa': spectral_abscissa,
        'sv_condition': sv_cond,
        'effective_rank': eff_rank,
        'sym_norm': sym_norm,
        'skew_norm': skew_norm,
        'final_loss': losses[-1],
    }


# ============================================================
# Run experiments: sweep mode x K, tune lr for each
# ============================================================

W1_init = torch.randn(d_hid, d_in) * 0.5
W2_init = torch.randn(d_out, d_hid) * 0.5

modes = ['VL', 'hurwitz', 'diag_dom']
K_values = [1, 2, 4, 8]
lr_candidates = [100.0, 10.0, 1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003]

# Baseline: tune lr
print("Tuning baseline lr...")
best_base_lr, best_base_loss = None, float('inf')
for lr in lr_candidates:
    losses = train_baseline(W1_init, W2_init, lr, num_steps)
    if losses[-1] < best_base_loss:
        best_base_loss = losses[-1]
        best_base_lr = lr
print(f"  Best lr={best_base_lr}, final loss={best_base_loss:.6f}")
losses_base = train_baseline(W1_init, W2_init, best_base_lr, num_steps)

# For each mode x K: tune lr, then run final training + diagnostics
results = {}
for mode in modes:
    results[mode] = {}
    for K in K_values:
        best_lr, best_loss = None, float('inf')
        print(f"Tuning lr for {mode} K={K}...")
        for lr in lr_candidates:
            losses = train_stable(mode, K, lr, num_steps)[0]
            if losses[-1] < best_loss:
                best_loss = losses[-1]
                best_lr = lr
        print(f"  Best lr={best_lr}, final loss={best_loss:.6f}")
        losses = train_stable(mode, K, best_lr, num_steps)[0]
        diag = get_diagnostics(mode, K, best_lr, num_steps)
        results[mode][K] = (losses, diag, best_lr)


# ============================================================
# Plot results
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
mode_colors = {'VL': 'tab:blue', 'hurwitz': 'tab:red', 'diag_dom': 'tab:green'}
K_labels = [str(K) for K in K_values]
x_pos = np.arange(len(K_values))
width = 0.25

# --- Plot 1: Mode comparison at K=1 ---
ax = axes[0, 0]
ax.semilogy(losses_base, 'k-', linewidth=2, label=f'Baseline (lr={best_base_lr})')
for mode in modes:
    losses, diag, lr = results[mode][1]
    ax.semilogy(losses, color=mode_colors[mode], alpha=0.8,
                label=f'{mode} K=1 (lr={lr})')
ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.set_title('Mode Comparison at K=1')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# --- Plot 2: All modes x K ---
ax = axes[0, 1]
linestyles = ['-', '--', '-.', ':']
ax.semilogy(losses_base, 'k-', linewidth=2, label='Baseline')
for mode in modes:
    for j, K in enumerate(K_values):
        losses, diag, lr = results[mode][K]
        ax.semilogy(losses, color=mode_colors[mode],
                    alpha=0.3 + 0.7 * (j + 1) / len(K_values),
                    linestyle=linestyles[j],
                    label=f'{mode} K={K}')
ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.set_title('All Modes x K')
ax.legend(fontsize=4, ncol=3)
ax.grid(True, alpha=0.3)

# --- Plot 3: Final loss vs K (grouped bar) ---
ax = axes[0, 2]
for i, mode in enumerate(modes):
    final_losses = [results[mode][K][1]['final_loss'] for K in K_values]
    ax.bar(x_pos + i * width - width, final_losses, width,
           label=mode, color=mode_colors[mode])
ax.axhline(losses_base[-1], color='k', linestyle='--', label='Baseline')
ax.set_xticks(x_pos)
ax.set_xticklabels(K_labels)
ax.set_xlabel('n_components K')
ax.set_ylabel('Final Loss')
ax.set_yscale('log')
ax.set_title('Final Loss vs K')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3, axis='y')

# --- Plot 4: Spectral abscissa (max real eigenvalue) ---
ax = axes[1, 0]
for i, mode in enumerate(modes):
    spec_abs = [results[mode][K][1]['spectral_abscissa'] for K in K_values]
    ax.bar(x_pos + i * width - width, spec_abs, width,
           label=mode, color=mode_colors[mode])
ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(K_labels)
ax.set_xlabel('n_components K')
ax.set_ylabel('max Re(lambda) of W1')
ax.set_title('Spectral Abscissa (should be < 0)')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3, axis='y')

# --- Plot 5: Effective rank of W1 ---
ax = axes[1, 1]
for i, mode in enumerate(modes):
    eff_ranks = [results[mode][K][1]['effective_rank'] for K in K_values]
    ax.bar(x_pos + i * width - width, eff_ranks, width,
           label=mode, color=mode_colors[mode])
ax.set_xticks(x_pos)
ax.set_xticklabels(K_labels)
ax.set_xlabel('n_components K')
ax.set_ylabel('Effective rank of W1')
ax.set_title('Effective Rank of Learned W1')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3, axis='y')

# --- Plot 6: SV condition number ---
ax = axes[1, 2]
for i, mode in enumerate(modes):
    sv_conds = [results[mode][K][1]['sv_condition'] for K in K_values]
    ax.bar(x_pos + i * width - width, sv_conds, width,
           label=mode, color=mode_colors[mode])
ax.set_xticks(x_pos)
ax.set_xticklabels(K_labels)
ax.set_xlabel('n_components K')
ax.set_ylabel('sigma_max / sigma_min')
ax.set_yscale('log')
ax.set_title('SV Condition Number of W1')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('stable_param_linear_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Summary table
# ============================================================

print("\n" + "=" * 90)
print(f"{'Condition':<30s} {'Best LR':>8s} {'Final Loss':>12s} {'Spec.Abs.':>10s} {'Speedup':>10s}")
print("=" * 90)
print(f"{'Baseline':<30s} {best_base_lr:>8.4f} {losses_base[-1]:>12.6f} {'N/A':>10s} {'1.00x':>10s}")
for mode in modes:
    for K in K_values:
        losses, diag, lr = results[mode][K]
        target = losses_base[-1]
        steps_to_target = num_steps
        for i, l in enumerate(losses):
            if l <= target:
                steps_to_target = i
                break
        speedup = num_steps / steps_to_target if steps_to_target > 0 else float('inf')
        label = f"{mode} K={K}"
        print(f"{label:<30s} {lr:>8.4f} {diag['final_loss']:>12.6f} "
              f"{diag['spectral_abscissa']:>10.4f} {speedup:>9.1f}x")
print("=" * 90)
