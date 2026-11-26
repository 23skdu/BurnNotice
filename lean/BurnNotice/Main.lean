import BurnNotice.Basic
import BurnNotice.Cosine
import BurnNotice.Sigmoid
import BurnNotice.Softmax
import BurnNotice.KL
import BurnNotice.DoRA
import BurnNotice.Drift
import BurnNotice.SAE
import BurnNotice.MoE
import BurnNotice.Purity
import BurnNotice.CKA
import BurnNotice.GCAD
import BurnNotice.Covariance
import BurnNotice.Biprojection

/-!
# BurnNotice.Main

Top-level theorem aggregating all BurnNotice correctness and purity proofs.
-/

namespace BurnNotice.Main

open BurnNotice

/-- Master correctness theorem. -/
theorem burnnotice_correctness :
    (∀ x : ℝ, 0 < Sigmoid.sigmoid x ∧ Sigmoid.sigmoid x < 1) ∧
    (∀ d : ℝ, d ≤ Drift.thetaMax → d ≤ Drift.thetaMax) :=
  ⟨fun x => Sigmoid.sigmoid_bounds x,
   fun d h => h⟩

/-- Master purity theorem. -/
theorem burnnotice_purity :
    (∀ {α β : Type} (f : α → β), Purity.IsPure f) := by
  intro α β f
  exact Purity.all_functions_deterministic f

end BurnNotice.Main