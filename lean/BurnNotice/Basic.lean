import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Data.Real.Sqrt

/-!
# BurnNotice.Basic

Foundational lemmas.
-/

namespace BurnNotice.Basic

/-- The square of a real number is non-negative. -/
theorem sq_nonneg (x : ℝ) : 0 ≤ x ^ 2 := by
  nlinarith

/-- Absolute value is non-negative. -/
theorem abs_nonneg' (x : ℝ) : 0 ≤ |x| := by
  positivity

/-- Exponential is always positive. -/
theorem exp_pos (x : ℝ) : 0 < Real.exp x := by
  exact Real.exp_pos x

/-- Logarithm of 1 is 0. -/
theorem log_one : Real.log 1 = 0 := by
  exact Real.log_one

end BurnNotice.Basic