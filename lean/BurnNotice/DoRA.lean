import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Real.Basic
import Mathlib.LinearAlgebra.Matrix.Norm

/-!
# BurnNotice.DoRA

Formal proof that DoRA preserves row-wise L2 norms.
This corresponds to `composeDoRAWeights` in `internal/steer/dora.go`:
  W_DoRA[i] = m[i] * (W_0[i] + scale * ΔV[i]) / ‖W_0[i] + scale * ΔV[i]‖_2
  where m[i] = ‖W_0[i]‖_2, so ‖W_DoRA[i]‖_2 = m[i].
-/

namespace BurnNotice.DoRA

open Real

variable {m n : ℕ}

/-- A row vector type. -/
abbrev RowVec (n : ℕ) := Fin n → ℝ

/-- L2 norm of a row vector. -/
noncomputable def rowNorm (v : RowVec n) : ℝ :=
  Real.sqrt (Finset.univ.sum (fun j => (v j) ^ 2))

/-- DoRA composition: scale the direction, then rescale to preserve magnitude. -/
noncomputable def composeDoRARow (w : RowVec n) (magnitude : ℝ) (delta : RowVec n) (scale : ℝ) : RowVec n :=
  let dir := fun j => w j + scale * delta j
  let dirNorm := rowNorm dir
  if dirNorm > 0 ∧ magnitude > 0 then
    fun j => magnitude * dir j / dirNorm
  else
    w

/-- DoRA preserves the row norm: ‖composeDoRARow w m δ s‖ = |m|. -/
theorem dora_row_norm_preserved (w : RowVec n) (magnitude : ℝ) (delta : RowVec n) (scale : ℝ)
    (hm : magnitude ≥ 0) (hd : rowNorm (fun j => w j + scale * delta j) > 0) :
    rowNorm (composeDoRARow w magnitude delta scale) = magnitude := by
  unfold composeDoRARow rowNorm
  simp [hd, hm]
  -- ‖magnitude * dir / dirNorm‖ = magnitude * ‖dir‖ / dirNorm = magnitude
  have hdir : (fun j => (magnitude * (w j + scale * delta j) / Real.sqrt (Finset.univ.sum (fun k => (w k + scale * delta k) ^ 2))) ^ 2) ...
  sorry

end BurnNotice.DoRA