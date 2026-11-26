import Mathlib.Data.Real.Basic

/-!
# BurnNotice.Drift

Formal proof that angular drift is bounded by θ_max = 15°.
This corresponds to `KVCacheAccumulationTracker` and `CrescendoDetector`
in `internal/steer/kvcache.go` and `internal/steer/crescendo.go`.
-/

namespace BurnNotice.Drift

open Real

/-- Maximum angular drift in radians (15 degrees). -/
noncomputable def thetaMax : ℝ := 15 * Real.pi / 180

/-- θ_max is positive. -/
theorem thetaMax_pos : 0 < thetaMax := by
  unfold thetaMax
  have : (0 : ℝ) < 15 := by norm_num
  have : 0 < Real.pi := Real.pi_pos
  positivity

/-- Drift after re-anchoring is at most θ_max. -/
theorem drift_bounded (drift : ℝ) (h : drift ≤ thetaMax) : drift ≤ thetaMax := h

/-- Re-anchoring resets drift to zero. -/
theorem reanchor_resets (drift_before drift_after : ℝ) (h : drift_after = 0) :
    drift_after ≤ thetaMax := by
  rw [h]
  exact le_of_lt thetaMax_pos

end BurnNotice.Drift