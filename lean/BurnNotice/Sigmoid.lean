import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Data.Real.Basic

/-!
# BurnNotice.Sigmoid

Formal proof that the sigmoid function maps ℝ to (0, 1).
This corresponds to the sigmoid used in `internal/scoring/scorer.go`
and `internal/engine/native/kernel.cpp`.
-/

namespace BurnNotice.Sigmoid

open Real

/-- Sigmoid function. -/
noncomputable def sigmoid (x : ℝ) : ℝ := 1 / (1 + exp (-x))

/-- Sigmoid is always positive. -/
theorem sigmoid_pos (x : ℝ) : 0 < sigmoid x := by
  unfold sigmoid
  have h : 0 < 1 + exp (-x) := by
    have := exp_pos (-x)
    linarith
  exact div_pos (by norm_num) h

/-- Sigmoid is always less than 1. -/
theorem sigmoid_lt_one (x : ℝ) : sigmoid x < 1 := by
  unfold sigmoid
  have h : 0 < 1 + exp (-x) := by
    have := exp_pos (-x)
    linarith
  rw [div_lt_one h]
  linarith [exp_pos (-x)]

/-- Sigmoid is in (0, 1). -/
theorem sigmoid_bounds (x : ℝ) : 0 < sigmoid x ∧ sigmoid x < 1 :=
  ⟨sigmoid_pos x, sigmoid_lt_one x⟩

end BurnNotice.Sigmoid