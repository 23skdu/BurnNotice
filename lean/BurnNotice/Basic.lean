import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Data.Real.Basic
import Mathlib.Data.Finset.Basic

/-!
# BurnNotice.Basic

Foundational lemmas used throughout the BurnNotice formal verification.
These establish basic properties of real arithmetic, norms, and
inequalities that underpin the correctness and purity proofs.
-/

namespace BurnNotice.Basic

/-- The square of a real number is non-negative. -/
theorem sq_nonneg (x : ℝ) : 0 ≤ x ^ 2 := sq_nonneg x

/-- Absolute value is non-negative. -/
theorem abs_nonneg (x : ℝ) : 0 ≤ |x| := abs_nonneg x

/-- The norm of a zero vector is zero. -/
theorem norm_zero {α : Type*} [NormedAddCommGroup α] : ‖(0 : α)‖ = 0 := norm_zero

/-- Triangle inequality for norms. -/
theorem norm_add_le {α : Type*} [NormedAddCommGroup α] (x y : α) :
    ‖x + y‖ ≤ ‖x‖ + ‖y‖ := norm_add_le x y

/-- Cauchy-Schwarz inequality. -/
theorem cauchy_schwarz {α : Type*} [InnerProductSpace ℝ α] (x y : α) :
    |inner x y| ≤ ‖x‖ * ‖y‖ := abs_inner_le_norm x y

/-- Exponential is always positive. -/
theorem exp_pos (x : ℝ) : 0 < Real.exp x := Real.exp_pos x

/-- Logarithm of 1 is 0. -/
theorem log_one : Real.log 1 = 0 := Real.log_one

/-- For x > 0, x * (1/x) = 1. -/
theorem mul_inv (x : ℝ) (hx : x ≠ 0) : x * (1 / x) = 1 := mul_one_div x hx

end BurnNotice.Basic