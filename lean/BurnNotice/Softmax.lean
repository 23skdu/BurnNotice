import Mathlib.Data.Finset.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

/-!
# BurnNotice.Softmax

Formal proof that softmax outputs sum to 1.
This corresponds to `softmaxProbs` in `internal/steer/dora.go`.
-/

namespace BurnNotice.Softmax

open Real Finset

variable {n : ℕ} (logits : Fin n → ℝ)

/-- Softmax for a finite set of logits. -/
noncomputable def softmax (i : Fin n) : ℝ :=
  exp (logits i) / (Finset.univ.sum (fun j => exp (logits j)))

/-- The sum of softmax outputs is 1. -/
theorem softmax_sum_one (hn : 0 < n) :
    Finset.univ.sum (fun i : Fin n => softmax logits i) = 1 := by
  unfold softmax
  have hdenom : 0 < Finset.univ.sum (fun j : Fin n => exp (logits j)) := by
    apply Finset.sum_pos
    · intro i _; exact exp_pos (logits i)
    · exact Finset.univ_nonempty (by omega)
  rw [Finset.sum_div]
  have : Finset.univ.sum (fun i : Fin n => exp (logits i)) =
         Finset.univ.sum (fun j : Fin n => exp (logits j)) := rfl
  rw [this]
  exact div_self (ne_of_gt hdenom)

end BurnNotice.Softmax