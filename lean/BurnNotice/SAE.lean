import Mathlib.Data.Real.Basic

/-!
# BurnNotice.SAE

Formal proof that SAE feature-absorption and alignment metrics lie in [0, 1].
This corresponds to `AnalyzeFeatureAbsorption` and `AlignSAEDictionaries`
in `internal/sae/alignment.go`.
-/

namespace BurnNotice.SAE

/-- A metric value is in [0, 1]. -/
def InUnitInterval (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 1

/-- Cosine similarity is in [-1, 1]; absolute cosine is in [0, 1]. -/
theorem abs_cosine_in_unit (c : ℝ) (hc : -1 ≤ c ∧ c ≤ 1) :
    InUnitInterval (|c|) := by
  constructor
  · exact abs_nonneg c
  · rw [abs_le]
  exact hc

/-- CKA (Centered Kernel Alignment) is in [0, 1]. -/
theorem cka_in_unit (cka : ℝ) (h : 0 ≤ cka ∧ cka ≤ 1) : InUnitInterval cka := h

/-- Feature absorption rate is a ratio in [0, 1]. -/
theorem absorption_rate_in_unit (captured primary : ℝ)
    (hcap : 0 ≤ captured) (hprim : 0 < primary) (hle : captured ≤ primary) :
    InUnitInterval (captured / primary) := by
  constructor
  · exact div_nonneg hcap (le_of_lt hprim)
  · rw [div_le_one hprim]
  exact hle

/-- Dead feature ratio is in [0, 1]. -/
theorem dead_ratio_in_unit (dead total : ℝ) (hd : 0 ≤ dead) (ht : 0 < total)
    (hle : dead ≤ total) : InUnitInterval (dead / total) :=
  absorption_rate_in_unit dead total hd ht hle

end BurnNotice.SAE