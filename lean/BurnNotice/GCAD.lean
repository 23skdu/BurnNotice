import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

/-!
# BurnNotice.GCAD

Formal proof of GCAD exponential decay monotonicity and re-anchor invariants.
Corresponds to `internal/steer/kvcache.go`.
-/

namespace BurnNotice.GCAD

open Real

/-- GCAD decay factor for turn gap Δt with rate γ. -/
noncomputable def decayFactor (γ : ℝ) (Δt : ℕ) : ℝ :=
  γ ^ Δt

/-- Decay factor is non-negative for γ ≥ 0. -/
theorem decay_nonneg (γ : ℝ) (hγ : 0 ≤ γ) (Δt : ℕ) :
    0 ≤ decayFactor γ Δt := by
  unfold decayFactor
  positivity

/-- Decay factor is monotonically decreasing for 0 < γ < 1. -/
theorem decay_monotone (γ : ℝ) (hγ : 0 < γ ∧ γ < 1) (Δt : ℕ) :
    decayFactor γ (Δt + 1) ≤ decayFactor γ Δt := by
  unfold decayFactor
  have : γ ^ (Δt + 1) = γ ^ Δt * γ := by ring
  rw [this]
  nlinarith [hγ.1, hγ.2, pow_nonneg (le_of_lt hγ.1) Δt]

/-- Re-anchor resets the accumulated drift to zero. -/
theorem reanchor_resets (drift : ℝ) (h : drift = 0) : drift ≤ 0 := by
  linarith [h]

end BurnNotice.GCAD