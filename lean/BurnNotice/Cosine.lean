import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Data.Real.Basic

/-!
# BurnNotice.Cosine

Formal proof that cosine similarity is bounded in [-1, 1].
-/

namespace BurnNotice.Cosine

open Real

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- Cosine similarity between two non-zero vectors. -/
noncomputable def cosineSim (x y : V) : ℝ :=
  inner x y / (‖x‖ * ‖y‖)

/-- Cosine similarity is bounded in [-1, 1]. -/
theorem cosineSim_bounds (x y : V) (hx : x ≠ 0) (hy : y ≠ 0) :
    -1 ≤ cosineSim x y ∧ cosineSim x y ≤ 1 := by
  constructor
  · -- Lower bound: -1 ≤ inner x y / (‖x‖ * ‖y‖)
    have hcs := abs_real_inner_le_norm x y
    have hnorm : 0 < ‖x‖ * ‖y‖ := by
      apply mul_pos (norm_pos_iff.mpr hx) (norm_pos_iff.mpr hy)
    have : - (‖x‖ * ‖y‖) ≤ inner x y := by
      have := (abs_le.mp hcs).1
      linarith
    unfold cosineSim
    rw [le_div_iff₀ hnorm]
    linarith
  · -- Upper bound: inner x y / (‖x‖ * ‖y‖) ≤ 1
    have hcs := abs_real_inner_le_norm x y
    have hnorm : 0 < ‖x‖ * ‖y‖ := by
      apply mul_pos (norm_pos_iff.mpr hx) (norm_pos_iff.mpr hy)
    have : inner x y ≤ ‖x‖ * ‖y‖ := by
      have := (abs_le.mp hcs).2
      linarith
    unfold cosineSim
    rw [div_le_iff₀ hnorm]
    linarith

end BurnNotice.Cosine