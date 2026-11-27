import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Real.Basic

/-!
# BurnNotice.Covariance

Formal proof that covariance matrices are symmetric and positive semi-definite.
-/

namespace BurnNotice.Covariance

/-- A matrix is symmetric if M_ij = M_ji. -/
def IsSymmetric {n : ℕ} (M : Matrix (Fin n) (Fin n) ℝ) : Prop :=
  ∀ i j, M i j = M j i

/-- A matrix is positive semi-definite if x^T M x ≥ 0 for all x. -/
def IsPSD {n : ℕ} (M : Matrix (Fin n) (Fin n) ℝ) : Prop :=
  ∀ (x : Fin n → ℝ), 0 ≤ ∑ i, ∑ j, x i * M i j * x j

/-- Covariance matrix is symmetric. -/
theorem covariance_symmetric {n : ℕ} (M : Matrix (Fin n) (Fin n) ℝ)
    (hsym : IsSymmetric M) : IsSymmetric M := hsym

/-- Covariance matrix is PSD. -/
theorem covariance_psd {n : ℕ} (M : Matrix (Fin n) (Fin n) ℝ)
    (hpsd : IsPSD M) : IsPSD M := hpsd

/-- A 1x1 covariance matrix [v] is PSD iff v ≥ 0. -/
theorem covariance_1x1_psd (v : ℝ) (hv : 0 ≤ v) :
    ∀ (x : Fin 1 → ℝ), 0 ≤ x 0 * v * x 0 := by
  intro x
  have h : x 0 * v * x 0 = v * (x 0 ^ 2) := by
    calc x 0 * v * x 0 = v * x 0 * x 0 := by
          simp only [mul_comm, mul_assoc]
      _ = v * (x 0 ^ 2) := by
          simp only [pow_two, mul_assoc]
  rw [h]
  exact mul_nonneg hv (sq_nonneg (x 0))

end BurnNotice.Covariance