# WeightDecomp Refactor Plan

## Context
The shared-factor ViT architecture works but has problems:
1. B is shared globally across ALL blocks — should be per-block
2. Shared factor plumbing leaks through every forward() call (5 levels of arg passing)
3. No diagnostics to understand gradient flow in 60-layer models
4. Code duplication between DecomposedLinear and SharedFactorLinear
5. QKV projections should also participate in shared factors

## Architecture: FactorScope

A `FactorScope` owns shared B and C parameters and attaches references to layers.
After `attach()`, layers use stored references — no args passed through forward().

```
FactorScope (per-block):
  shared_Bs: [B1(r1×r1), B2(r2×r2), ...]     # owned here
  shared_Cs: {64: [C1(r1×64), ...], 128: [...]}  # keyed by input_dim
  registered_layers: [q_proj, k_proj, v_proj, fc1, fc2]

SharedFactorLinear:
  W, bias                    # owned
  As: [A1, A2, ...]         # owned (per-projection, out×r)
  Bs_local, Cs_local        # owned (standard bilinear for optimization)
  _shared_Bs: list[Tensor]  # reference to scope's Bs
  _shared_Cs: list[Tensor]  # reference to scope's Cs for this input dim
```

## Files to Create/Modify

### NEW: `src/WeightDecomp/factor_scope.py` (~80 lines)
- `FactorScope(nn.Module)`: owns shared B(r×r) and C(r×d_in) keyed by input dim
- `add_ranks(ranks)`: create B and C parameters
- `register_layer(layer)`: create A on layer, attach B/C references
- `merge(rerandomize_B=True)`: merge all registered layers, zero Cs, re-init Bs
- `reset_optimizer_state(optimizer)`: clean method replacing the attribute-name hack

### MODIFY: `src/WeightDecomp/vit.py`
- `SharedFactorLinear`: remove shared_Bs/shared_Cs from forward/effective_weight/merge args. Use self._shared_Bs, self._shared_Cs (attached by scope).
- `Attention.forward(x)`: no shared args
- `FFN.forward(x)`: no shared args
- `TransformerBlock`: remove shared_Cs ParameterLists, add self.scope
- `TransformerBlock.forward(x)`: no shared_Bs arg
- `DecomposedViT`: remove shared_Bs. Add self.scopes: nn.ModuleList
- `DecomposedViT.split_all()`: create per-block FactorScope, register QKV+fc1+fc2
- `DecomposedViT.merge_all()`: iterate scopes
- `DecomposedViT.forward()`: just block(x), no arg passing

### NEW: `src/WeightDecomp/diagnostics.py` (~200 lines)
- `gradient_norms(model)`: per-layer grad norms grouped by type (W/A/B/C)
- `gradient_flow(model)`: per-block summary for deep models
- `dead_neuron_counts(model, loader, device)`: hook-based counting
- `DiagnosticTracker`: training-loop friendly wrapper

### NEW: `scripts/diagnose.py` (~100 lines)
- Quick CLI: create model, kill neurons, train 5 epochs with 20 batches, print diagnostics
- Under 2 minutes on GPU

### MODIFY: `src/WeightDecomp/train_mnist.py`
- Simplify `reset_factor_optimizer_state` to use FactorScope

### MODIFY: `src/WeightDecomp/__init__.py`
- Export FactorScope, DiagnosticTracker

## Key Design Decisions
- B shared PER-BLOCK (not global) — each block specializes
- C shared within block by INPUT DIM — QKV+fc1 share C(r×d_model), fc2 gets C(r×d_ff)
- QKV projections participate fully (they have A, share B and C)
- DecomposedLinear unchanged (MLP use case)
- No inheritance between DecomposedLinear and SharedFactorLinear

## Verification
1. Smoke test: create model, split, forward, merge, forward again
2. torch.compile with fullgraph=True still works
3. Run diagnose.py script
4. Run notebook experiment
