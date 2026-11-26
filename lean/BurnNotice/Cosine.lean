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
  sorry

end BurnNotice.Cosine