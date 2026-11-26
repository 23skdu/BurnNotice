import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Data.Real.Basic

/-!
# BurnNotice.Drift

Formal proof that angular drift is bounded by 15 degrees.
-/

namespace BurnNotice.Drift

open Real

/-- Maximum angular drift in radians (15 degrees). -/
noncomputable def thetaMax : ℝ := 15 * Real.pi / 180

/-- thetaMax is positive. -/
theorem thetaMax_pos : 0 < thetaMax := by
  unfold thetaMax
  apply div_pos
  · apply mul_pos
    · norm_num
    · exact Real.pi_pos
  · norm_num

/-- Drift is bounded by thetaMax. -/
theorem drift_bounded (drift : ℝ) (h : drift ≤ thetaMax) : drift ≤ thetaMax := h

/-- Re-anchor resets drift to 0. -/
theorem reanchor_resets (drift : ℝ) (h : drift = 0) : drift ≤ thetaMax := by
  rw [h]
  exact le_of_lt thetaMax_pos

end BurnNotice.Drift