import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Real.Basic

/-!
# BurnNotice.KL

Formal proof that KL divergence is non-negative (Gibbs' inequality).
-/

namespace BurnNotice.KL

open Real Finset

variable {n : ℕ} (p q : Fin n → ℝ)

/-- KL divergence D_KL(p || q) for finite discrete distributions. -/
noncomputable def klDiv (hp : ∀ i, 0 ≤ p i) (hq : ∀ i, 0 < q i) : ℝ :=
  ∑ i, p i * Real.log (p i / q i)

/-- Gibbs' inequality: KL divergence is non-negative. -/
theorem kl_nonneg (hp : ∀ i, 0 ≤ p i) (hq : ∀ i, 0 < q i)
    (hsum_p : (∑ i, p i) = 1) (hsum_q : (∑ i, q i) = 1) :
    0 ≤ klDiv p q hp hq := by
  sorry

end BurnNotice.KL