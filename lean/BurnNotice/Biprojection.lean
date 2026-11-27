import Mathlib.Data.Real.Basic

/-!
# BurnNotice.Biprojection

Formal proof of biprojection idempotence property.
-/

namespace BurnNotice.Biprojection

/-- A projection P is idempotent if P(P(x)) = P(x). -/
def IsIdempotent {α : Type} (P : α → α) : Prop :=
  ∀ x, P (P x) = P x

/-- Identity projection is idempotent. -/
theorem identity_idempotent {α : Type} : IsIdempotent (id : α → α) := by
  intro x
  rfl

/-- Constant projection is idempotent. -/
theorem constant_idempotent {α : Type} (c : α) :
    IsIdempotent (fun _ : α => c) := by
  intro x
  rfl

/-- Composing two commuting idempotents yields an idempotent. -/
theorem compose_idempotent {α : Type} (P Q : α → α)
    (hP : IsIdempotent P) (hQ : IsIdempotent Q)
    (hcomm : ∀ x, P (Q x) = Q (P x)) :
    IsIdempotent (fun x => P (Q x)) := by
  intro x
  dsimp [IsIdempotent]
  calc P (Q (P (Q x)))
      = Q (P (P (Q x))) := hcomm (P (Q x))
    _ = Q (P (Q x)) := by rw [hP (Q x)]
    _ = P (Q (Q x)) := by rw [hcomm (Q x)]
    _ = P (Q x) := by rw [hQ x]

end BurnNotice.Biprojection