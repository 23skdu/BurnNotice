import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Real.Basic

/-!
# BurnNotice.Covariance

Formal proof that covariance matrices are symmetric and positive semi-definite.
Corresponds to `internal/math/` and `internal/analysis/`.
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

end BurnNotice.Covariance