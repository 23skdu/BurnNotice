import Mathlib.Data.Finset.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Algebra.BigOperators.Ring
import Mathlib.Data.Fintype.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# BurnNotice.Softmax

Formal proof that softmax outputs sum to 1.
-/

namespace BurnNotice.Softmax

open Real Finset

variable {n : ℕ} (logits : Fin n → ℝ)

/-- Softmax for a finite set of logits. -/
noncomputable def softmax (i : Fin n) : ℝ :=
  exp (logits i) / (∑ j ∈ univ, exp (logits j))

/-- The sum of softmax outputs is 1. -/
theorem softmax_sum_one (hn : 0 < n) :
    (∑ i ∈ univ, softmax logits i) = 1 := by
  unfold softmax
  have hdenom : 0 < (∑ j ∈ univ, exp (logits j)) := by
    have h0 : (⟨0, hn⟩ : Fin n) ∈ univ := Finset.mem_univ _
    have hterm : 0 < exp (logits ⟨0, hn⟩) := exp_pos (logits ⟨0, hn⟩)
    have hall : ∀ j ∈ univ, 0 ≤ exp (logits j) := fun j _ => le_of_lt (exp_pos (logits j))
    calc 0 < exp (logits ⟨0, hn⟩) := hterm
      _ ≤ ∑ j ∈ univ, exp (logits j) := Finset.single_le_sum hall h0
  have hdiv : (∑ i ∈ univ, exp (logits i) / (∑ j ∈ univ, exp (logits j))) =
      (∑ i ∈ univ, exp (logits i)) / (∑ j ∈ univ, exp (logits j)) := by
    rw [← Finset.sum_div univ (fun i => exp (logits i)) (∑ j ∈ univ, exp (logits j))]
  rw [hdiv]
  exact div_self (ne_of_gt hdenom)

end BurnNotice.Softmax