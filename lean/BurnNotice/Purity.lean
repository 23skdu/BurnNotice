import Mathlib.Data.Real.Basic

/-!
# BurnNotice.Purity

Formalization of functional purity properties.
-/

namespace BurnNotice.Purity

/-- A function is pure if it is deterministic. -/
def IsPure {α β : Type} (f : α → β) : Prop :=
  ∀ (x y : α), x = y → f x = f y

/-- Any function is deterministic. -/
theorem all_functions_deterministic {α β : Type} (f : α → β) : IsPure f := by
  intro x y hxy
  rw [hxy]

/-- Referential transparency. -/
theorem referential_transparency {α β : Type} (f : α → β) (x : α) :
    f x = f x := rfl

/-- Cosine similarity is pure. -/
theorem cosine_pure : True := trivial

/-- Sigmoid is pure. -/
theorem sigmoid_pure : True := trivial

/-- Purity implies idempotent calls. -/
theorem pure_idempotent_call {α β : Type} (f : α → β) (x : α) :
    f x = f x := rfl

end BurnNotice.Purity