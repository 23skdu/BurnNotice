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

/-- Helper: for p_i ≥ 0 and q_i > 0, p_i * log(p_i/q_i) ≥ p_i - q_i. -/
theorem kl_term_bound (i : Fin n) (hp : ∀ i, 0 ≤ p i) (hq : ∀ i, 0 < q i) :
    p i * Real.log (p i / q i) ≥ p i - q i := by
  by_cases hpi : p i = 0
  · -- p_i = 0: term is 0, need 0 ≥ -q_i (true since q_i > 0)
    rw [hpi]
    simp [Real.log_zero]
    linarith [hq i]
  · -- p_i > 0: use log(x) ≤ x - 1 with x = q_i / p_i
    have hpi_pos : 0 < p i := lt_of_le_of_ne (hp i) (Ne.symm hpi)
    have hq_pos : 0 < q i := hq i
    have hratio : 0 < q i / p i := div_pos hq_pos hpi_pos
    have hlog := Real.log_le_sub_one_of_pos hratio
    -- log(q_i/p_i) ≤ q_i/p_i - 1
    -- ⟹ -p_i * log(q_i/p_i) ≥ p_i - q_i
    -- ⟹ p_i * log(p_i/q_i) ≥ p_i - q_i
    have h1 : Real.log (p i / q i) = - Real.log (q i / p i) := by
      rw [Real.log_div (ne_of_gt hpi_pos) (ne_of_gt hq_pos)]
      rw [Real.log_div (ne_of_gt hq_pos) (ne_of_gt hpi_pos)]
      ring
    rw [h1]
    have h2 : -p i * Real.log (q i / p i) ≥ p i - q i := by
      have := mul_le_mul_of_nonpos_left hlog (le_of_lt (neg_lt_zero.mpr hpi_pos))
      simp [mul_sub, mul_div_cancel₀ _ (ne_of_gt hpi_pos)] at this
      linarith
    linarith

/-- Gibbs' inequality: KL divergence is non-negative. -/
theorem kl_nonneg (hp : ∀ i, 0 ≤ p i) (hq : ∀ i, 0 < q i)
    (hsum_p : (∑ i, p i) = 1) (hsum_q : (∑ i, q i) = 1) :
    0 ≤ klDiv p q hp hq := by
  unfold klDiv
  -- Each term ≥ p_i - q_i, so sum ≥ Σ(p_i - q_i) = 0
  have hterm : ∀ i, p i * Real.log (p i / q i) ≥ p i - q i :=
    fun i => kl_term_bound p q i hp hq
  have hsum : (∑ i, (p i - q i)) = 0 := by
    rw [Finset.sum_sub_distrib]
    rw [hsum_p, hsum_q]
    ring
  -- Sum of terms ≥ sum of (p_i - q_i) = 0
  have : (∑ i, p i * Real.log (p i / q i)) ≥ (∑ i, (p i - q i)) := by
    apply Finset.sum_le_sum
    intro i _
    exact hterm i
  linarith [hsum, this]

end BurnNotice.KL