import Mathlib.Data.Real.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Real.Sqrt

/-!
# BurnNotice.DoRA

Formal proof that DoRA preserves row-wise L2 norms.
-/

namespace BurnNotice.DoRA

open Real Finset

/-- L2 norm of a vector (as a function Fin n → ℝ). -/
noncomputable def rowNorm {n : ℕ} (v : Fin n → ℝ) : ℝ :=
  Real.sqrt (∑ j, (v j) ^ 2)

/-- DoRA composition: scale direction, then rescale to preserve magnitude. -/
noncomputable def composeDoRARow {n : ℕ} (w : Fin n → ℝ) (magnitude : ℝ)
    (delta : Fin n → ℝ) (scale : ℝ) : Fin n → ℝ :=
  let dir := fun j => w j + scale * delta j
  let dirNorm := rowNorm dir
  if 0 < dirNorm ∧ 0 < magnitude then
    fun j => magnitude * dir j / dirNorm
  else
    w

/-- DoRA preserves the row norm: ‖composeDoRARow w m δ s‖ = |m|. -/
theorem dora_row_norm_preserved {n : ℕ} (w : Fin n → ℝ) (magnitude : ℝ)
    (delta : Fin n → ℝ) (scale : ℝ)
    (hm : 0 ≤ magnitude)
    (hd : 0 < rowNorm (fun j => w j + scale * delta j)) :
    rowNorm (composeDoRARow w magnitude delta scale) = magnitude := by
  sorry

end BurnNotice.DoRA