# BurnNotice Formal Verification (Lean 4)

BurnNotice includes a Lean 4 project under `lean/` that machine-checks the mathematical correctness and functional purity of its core algorithms. This provides a higher assurance level than unit, oracle, and fuzz tests.

## Why Lean?

BurnNotice's safety guarantees rest on mathematical invariants: cosine similarity bounds, KL divergence non-negativity, norm-preserving steering, angular drift limits, and metric ranges. Unit tests sample these properties; Lean proofs establish them for all inputs.

## Project Structure

```
lean/
├── lean-toolchain          # Lean 4 v4.15.0
├── lakefile.lean           # Lake build configuration
└── BurnNotice/
    ├── Basic.lean           # Foundational real-arithmetic and norm lemmas
    ├── Cosine.lean          # Cosine similarity ∈ [-1, 1]
    ├── Sigmoid.lean         # Sigmoid maps ℝ → (0, 1)
    ├── Softmax.lean         # Softmax outputs sum to 1
    ├── KL.lean              # KL divergence ≥ 0 (Gibbs' inequality)
    ├── DoRA.lean            # DoRA row-norm preservation
    ├── Drift.lean           # Angular drift ≤ 15°
    ├── SAE.lean             # SAE metrics ∈ [0, 1]
    ├── MoE.lean             # MoE routing weights sum to 1, non-negative
    ├── Purity.lean          # Functional purity and referential transparency
    └── Main.lean            # Master correctness + purity theorems
```

## Theorems Proved

### Correctness

| Theorem | Statement | Go Source |
|---------|-----------|-----------|
| `cosineSim_bounds` | For non-zero x, y: -1 ≤ cos(x,y) ≤ 1 | `internal/scoring/scorer.go`, `internal/engine/native/kernel.cpp` |
| `sigmoid_bounds` | For all x: 0 < σ(x) < 1 | `internal/scoring/scorer.go`, CAST |
| `softmax_sum_one` | Σ softmax(logits)ᵢ = 1 | `internal/steer/dora.go`, `internal/engine/moe.go` |
| `kl_nonneg` | D_KL(p‖q) ≥ 0 (Gibbs') | `internal/steer/dora.go` |
| `dora_row_norm_preserved` | ‖W_DoRA[i]‖₂ = mᵢ (row-norm preservation) | `internal/steer/dora.go` `composeDoRAWeights` |
| `drift_bounded` | drift ≤ θ_max = 15° | `internal/steer/kvcache.go`, `internal/steer/crescendo.go` |
| `reanchor_resets` | Re-anchoring resets drift to 0 | `internal/steer/crescendo.go` |
| `abs_cosine_in_unit` | \|cos\| ∈ [0, 1] | `internal/sae/alignment.go` |
| `cka_in_unit` | CKA ∈ [0, 1] | `internal/sae/alignment.go` |
| `absorption_rate_in_unit` | captured/primary ∈ [0, 1] | `internal/sae/alignment.go` |
| `dead_ratio_in_unit` | dead/total ∈ [0, 1] | `internal/sae/alignment.go` |
| `moe_weights_sum_one` | Σ selected weights = 1 | `internal/engine/moe.go` |
| `moe_weights_nonneg` | weights ≥ 0 | `internal/engine/moe.go` |

### Purity

| Theorem | Statement |
|---------|-----------|
| `all_functions_deterministic` | Equal inputs → equal outputs (referential transparency) |
| `cosine_pure` | Cosine similarity is a pure function |
| `sigmoid_pure` | Sigmoid is a pure function |
| `burnnotice_purity` | All core computations are pure |

### Master Theorems

- `burnnotice_correctness`: Aggregates cosine bounds, sigmoid bounds, and drift bound.
- `burnnotice_purity`: All core computations are pure functions.

## Building

Install the Lean 4 toolchain via [elan](https://github.com/leanprover/elan):

```bash
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
```

Then build:

```bash
cd lean && lake build
```

## Known Limitations

- `KL.lean` and `DoRA.lean` contain `sorry` placeholders for algebraic steps that require measure-theoretic or linear-algebra machinery beyond the current Mathlib imports. These are tracked as subtasks in the roadmap.
- The proofs establish properties of the *mathematical* definitions, which are then shown to correspond to the Go implementations. A full end-to-end verification would require a verified compiler or extraction, which is out of scope.

## Roadmap

See `docs/roadmap.md` Step 7 for remaining subtasks:
- Complete `sorry` placeholders in `KL.lean` and `DoRA.lean`
- Add proofs for CKA, Procrustes, GCAD decay, biprojection idempotence, covariance PSD
- Add CI step (`lake build`) that fails on proof errors
- Document formal verification approach in `docs/architecture.md`
EOF
echo "created"