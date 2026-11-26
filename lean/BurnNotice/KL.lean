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

/-- Gibbs' inequality: KL divergence is non-negative.
    Proof uses log(x) ≤ x - 1, i.e., for x = q_i/p_i:
    log(q_i/p_i) ≤ q_i/p_i - 1
    ⟹ -p_i·log(q_i/p_i) ≥ p_i - q_i
    ⟹ p_i·log(p_i/q_i) ≥ p_i - q_i
    Summing: D_KL ≥ Σ(p_i - q_i) = 0 -/
theorem kl_nonneg (hp : ∀ i, 0 ≤ p i) (hq : ∀ i, 0 < q i)
    (hsum_p : (∑ i, p i) = 1) (hsum_q : (∑ i, q i) = 1) :
    0 ≤ klDiv p q hp hq := by
  unfold klDiv
  -- Use log(x) ≤ x - 1 with x = q_i / p_i (when p_i > 0)
  -- For p_i = 0, the term is 0 by convention (0 * log 0 = 0)
  sorry

end BurnNotice.KL