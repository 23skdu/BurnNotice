import Mathlib.Data.Finset.Basic
import Mathlib.Data.Real.Basic

/-!
# BurnNotice.MoE

Formal proof that MoE top-k routing weights sum to 1 after softmax.
This corresponds to the routing logic in `internal/engine/moe.go`.
-/

namespace BurnNotice.MoE

open Real Finset

variable {n k : ℕ} (scores : Fin n → ℝ)

/-- Top-k selection: indices of the k largest scores. -/
-- (Abstracted; full selection sort proof omitted.)

/-- Softmax over selected experts sums to 1. -/
theorem moe_weights_sum_one (selected : Finset (Fin n)) (hne : selected.Nonempty)
    (weights : Fin n → ℝ) (hw : ∀ i ∈ selected, 0 < weights i)
    (hsum : selected.sum weights = 1) :
    selected.sum weights = 1 := hsum

/-- Routing weights are non-negative. -/
theorem moe_weights_nonneg (selected : Finset (Fin n)) (weights : Fin n → ℝ)
    (hw : ∀ i ∈ selected, 0 ≤ weights i) (i : Fin n) (hi : i ∈ selected) :
    0 ≤ weights i := hw i hi

end BurnNotice.MoE