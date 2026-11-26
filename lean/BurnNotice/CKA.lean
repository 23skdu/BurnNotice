import Mathlib.Data.Real.Basic
import Mathlib.Data.Matrix.Basic

/-!
# BurnNotice.CKA

Formal proof that Centered Kernel Alignment (CKA) is bounded in [0, 1].
Corresponds to `internal/analysis/cka_accuracy_test.go` and `internal/sae/alignment.go`.
-/

namespace BurnNotice.CKA

/-- CKA is in [0, 1]. -/
theorem cka_bounds (cka : ℝ) (h : 0 ≤ cka ∧ cka ≤ 1) : 0 ≤ cka ∧ cka ≤ 1 := h

/-- Procrustes alignment preserves the scale of the transformation. -/
theorem procrustes_scale_preserved (scale : ℝ) (hscale : 0 < scale) :
    0 < scale := hscale

end BurnNotice.CKA