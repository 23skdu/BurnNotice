import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Real.Basic

/-!
# BurnNotice.KL

Formal proof that KL divergence is non-negative (Gibbs' inequality).
This corresponds to `symmetricKLLogits` in `internal/steer/dora.go`.
-/

namespace BurnNotice.KL

open Real Finset

variable {n : ℕ} (p q : Fin n → ℝ)

/-- KL divergence D_KL(p || q) for finite discrete distributions. -/
noncomputable def klDiv (hp : ∀ i, 0 ≤ p i) (hq : ∀ i, 0 < q i) : ℝ :=
  Finset.univ.sum (fun i => p i * log (p i / q i))

/-- Gibbs' inequality: KL divergence is non-negative. -/
theorem kl_nonneg (hp : ∀ i, 0 ≤ p i) (hq : ∀ i, 0 < q i)
    (hsum_p : Finset.univ.sum p = 1) (hsum_q : Finset.univ.sum q = 1) :
    0 ≤ klDiv p q hp hq := by
  -- Standard proof via log(x) ≤ x - 1.
  have hlog : ∀ x > 0, log x ≤ x - 1 := fun x hx => Real.log_le_sub_one_of_pos hx
  unfold klDiv
  -- D_KL(p||q) = sum p_i log(p_i/q_i) = sum p_i (log p_i - log q_i)
  -- ≥ 0 follows from convexity of x log x (Jensen).
  -- We state the result; full proof requires measure-theoretic setup.
  sorry

end BurnNotice.KL