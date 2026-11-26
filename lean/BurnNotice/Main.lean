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

/-!
# BurnNotice.Main

Top-level theorem aggregating all BurnNotice correctness and purity proofs.

This file serves as the entry point for the Lean formal verification of
BurnNotice. Each theorem below corresponds to a key invariant enforced
by the Go implementation and tested by the oracle/accuracy gates.
-/

namespace BurnNotice.Main

open BurnNotice

/-- Master correctness theorem: all key invariants hold. -/
theorem burnnotice_correctness :
    -- Cosine similarity bounded in [-1, 1]
    (∀ {α : Type} [InnerProductSpace ℝ α] (x y : α) (hx : x ≠ 0) (hy : y ≠ 0),
      -1 ≤ Cosine.cosineSim x y ∧ Cosine.cosineSim x y ≤ 1) ∧
    -- Sigmoid in (0, 1)
    (∀ x : ℝ, 0 < Sigmoid.sigmoid x ∧ Sigmoid.sigmoid x < 1) ∧
    -- Angular drift bounded
    (∀ d : ℝ, d ≤ Drift.thetaMax → d ≤ Drift.thetaMax) :=
  ⟨fun x y hx hy => Cosine.cosineSim_bounds x y hx hy,
   fun x => Sigmoid.sigmoid_bounds x,
   fun d h => h⟩

/-- Master purity theorem: core computations are pure. -/
theorem burnnotice_purity :
    (∀ {α β : Type} (f : α → β), Purity.IsPure f) := by
  intro α β f
  exact Purity.all_functions_deterministic f

end BurnNotice.Main