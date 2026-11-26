import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Data.Real.Basic

/-!
# BurnNotice.Cosine

Formal proof that cosine similarity is bounded in [-1, 1].
This corresponds to `vecCosine` / `CosineSimilarity` in
`internal/scoring/scorer.go` and `internal/engine/native/`.
-/

namespace BurnNotice.Cosine

open Real

variable {α : Type*} [InnerProductSpace ℝ α]

/-- Cosine similarity between two non-zero vectors. -/
noncomputable def cosineSim (x y : α) : ℝ :=
  inner x y / (‖x‖ * ‖y‖)

/-- Cosine similarity is bounded below by -1. -/
theorem cosineSim_ge_neg_one (x y : α) (hx : x ≠ 0) (hy : y ≠ 0) :
    -1 ≤ cosineSim x y := by
  have hcs := abs_inner_le_norm x y
  have hnorm : 0 < ‖x‖ * ‖y‖ := by
    apply mul_pos (norm_pos_iff.mpr hx) (norm_pos_iff.mpr hy)
  have : - (‖x‖ * ‖y‖) ≤ inner x y := by
    have := (abs_le.mp hcs).1
    linarith
  unfold cosineSim
  rw [ge_iff_le, le_div_iff₀ hnorm]
  linarith

/-- Cosine similarity is bounded above by 1. -/
theorem cosineSim_le_one (x y : α) (hx : x ≠ 0) (hy : y ≠ 0) :
    cosineSim x y ≤ 1 := by
  have hcs := abs_inner_le_norm x y
  have hnorm : 0 < ‖x‖ * ‖y‖ := by
    apply mul_pos (norm_pos_iff.mpr hx) (norm_pos_iff.mpr hy)
  have : inner x y ≤ ‖x‖ * ‖y‖ := by
    have := (abs_le.mp hcs).2
    linarith
  unfold cosineSim
  rw [div_le_iff₀ hnorm]
  linarith

/-- Cosine similarity is in [-1, 1]. -/
theorem cosineSim_bounds (x y : α) (hx : x ≠ 0) (hy : y ≠ 0) :
    -1 ≤ cosineSim x y ∧ cosineSim x y ≤ 1 :=
  ⟨cosineSim_ge_neg_one x y hx hy, cosineSim_le_one x y hx hy⟩

end BurnNotice.Cosine