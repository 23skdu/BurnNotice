import Mathlib.Data.Real.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Algebra.BigOperators.Ring

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

/-- DoRA preserves the row norm. -/
theorem dora_row_norm_preserved {n : ℕ} (w : Fin n → ℝ) (magnitude : ℝ)
    (delta : Fin n → ℝ) (scale : ℝ)
    (hm : 0 < magnitude)
    (hd : 0 < rowNorm (fun j => w j + scale * delta j)) :
    rowNorm (composeDoRARow w magnitude delta scale) = magnitude := by
  unfold composeDoRARow
  rw [if_pos ⟨hd, hm⟩]
  let dir := fun j => w j + scale * delta j
  let dirNorm := rowNorm dir
  have hdir : 0 < dirNorm := hd
  change rowNorm (fun j => magnitude * dir j / dirNorm) = magnitude
  unfold rowNorm
  -- Show sqrt(Σ (magnitude * dir_j / dirNorm)^2) = magnitude
  have hsum : (∑ j, (magnitude * dir j / dirNorm) ^ 2) = magnitude ^ 2 := by
    have h1 : (∑ j, (magnitude * dir j / dirNorm) ^ 2) =
        (magnitude / dirNorm) ^ 2 * (∑ j, (dir j) ^ 2) := by
      have : (∑ j, (magnitude * dir j / dirNorm) ^ 2) =
          (∑ j, (magnitude / dirNorm) ^ 2 * (dir j) ^ 2) := by
        congr 1
        ext j
        ring
      rw [this, ← mul_sum]
    rw [h1]
    have h2 : (magnitude / dirNorm) ^ 2 * (∑ j, (dir j) ^ 2) = magnitude ^ 2 := by
      have hdir_sq : (∑ j, (dir j) ^ 2) = dirNorm ^ 2 := by
        unfold dirNorm rowNorm
        rw [Real.sq_sqrt (by positivity : 0 ≤ ∑ j, (dir j) ^ 2)]
      rw [hdir_sq]
      field_simp [hdir.ne']
    exact h2
  rw [hsum]
  exact Real.sqrt_sq (le_of_lt hm)

end BurnNotice.DoRA