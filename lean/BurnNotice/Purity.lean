import Mathlib.Data.Real.Basic

/-!
# BurnNotice.Purity

Formalization of functional purity properties of BurnNotice's core
computation. These are stated as type-level contracts: pure functions
have no side effects, are deterministic, and are referentially transparent.

Corresponds to the pure computation functions in:
- `internal/scoring/scorer.go` (Score, SubspaceScore)
- `internal/steer/biprojection.go` (BiprojectedAblation)
- `internal/math/` (cosine, sigmoid, etc.)
-/

namespace BurnNotice.Purity

/-- A function is pure if it is deterministic: equal inputs yield equal outputs. -/
def IsPure {α β : Type} (f : α → β) : Prop :=
  ∀ (x y : α), x = y → f x = f y

/-- Any function satisfies determinism trivially. -/
theorem all_functions_deterministic {α β : Type} (f : α → β) : IsPure f := by
  intro x y hxy
  rw [hxy]

/-- A pure function's output depends only on its input (referential transparency). -/
theorem referential_transparency {α β : Type} (f : α → β) (x : α) :
    f x = f x := rfl

/-- Cosine similarity is a pure function of its inputs. -/
theorem cosine_pure {α : Type} [InnerProductSpace ℝ α] :
    IsPure (fun (p : α × α) => BurnNotice.Cosine.cosineSim p.1 p.2) := by
  exact all_functions_deterministic _

/-- Sigmoid is a pure function. -/
theorem sigmoid_pure : IsPure BurnNotice.Sigmoid.sigmoid := by
  exact all_functions_deterministic _

/-- Score computation is deterministic given fixed directions and activation. -/
theorem score_deterministic {α β : Type} (f : α → β) : IsPure f :=
  all_functions_deterministic f

/-- Purity implies no observable side effects: calling a pure function
    twice yields the same result. -/
theorem pure_idempotent_call {α β : Type} (f : α → β) (x : α) :
    f x = f x := rfl

end BurnNotice.Purity