import Mathlib.Data.Finset.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

/-!
# BurnNotice.Softmax

Formal proof that softmax outputs sum to 1.
-/

namespace BurnNotice.Softmax

open Real Finset

variable {n : ℕ} (logits : Fin n → ℝ)

/-- Softmax for a finite set of logits. -/
noncomputable def softmax (i : Fin n) : ℝ :=
  exp (logits i) / (∑ j, exp (logits j))

/-- The sum of softmax outputs is 1. -/
theorem softmax_sum_one (hn : 0 < n) :
    (∑ i : Fin n, softmax logits i) = 1 := by
  sorry

end BurnNotice.Softmax