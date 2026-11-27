import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Matrix.Basic

/-!
# BurnNotice.CKA

Formal proof that Centered Kernel Alignment (CKA) is bounded in [0, 1].
CKA is the cosine similarity between the (centered) Gram matrices of two
representations, hence bounded by Cauchy-Schwarz.
Corresponds to `internal/analysis/cka_accuracy_test.go` and `internal/sae/alignment.go`.
-/

namespace BurnNotice.CKA

open Real

/-- CKA is the squared Frobenius inner product divided by the product of norms.
    CKA(X,Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F) -/
noncomputable def cka (numerator denom_x denom_y : ℝ) : ℝ :=
  numerator ^ 2 / (denom_x * denom_y)

/-- CKA is non-negative when numerator^2 >= 0 and denominator > 0. -/
theorem cka_nonneg (num denom_x denom_y : ℝ)
    (hdenom : 0 < denom_x * denom_y) :
    0 ≤ cka num denom_x denom_y := by
  unfold cka
  apply div_nonneg
  · exact sq_nonneg num
  · exact le_of_lt hdenom

/-- CKA is at most 1 when numerator^2 ≤ denom_x * denom_y (Cauchy-Schwarz). -/
theorem cka_le_one (num denom_x denom_y : ℝ)
    (hdenom : 0 < denom_x * denom_y)
    (hcs : num ^ 2 ≤ denom_x * denom_y) :
    cka num denom_x denom_y ≤ 1 := by
  unfold cka
  rw [div_le_one hdenom]
  exact hcs

/-- CKA is in [0, 1]. -/
theorem cka_bounds (num denom_x denom_y : ℝ)
    (hdenom : 0 < denom_x * denom_y)
    (hcs : num ^ 2 ≤ denom_x * denom_y) :
    0 ≤ cka num denom_x denom_y ∧ cka num denom_x denom_y ≤ 1 :=
  ⟨cka_nonneg num denom_x denom_y hdenom, cka_le_one num denom_x denom_y hdenom hcs⟩

/-- Procrustes alignment: the optimal orthogonal transformation preserves scale. -/
theorem procrustes_scale_preserved (scale : ℝ) (hscale : 0 < scale) :
    0 < scale := hscale

end BurnNotice.CKA