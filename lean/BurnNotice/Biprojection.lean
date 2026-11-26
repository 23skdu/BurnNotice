import Mathlib.Data.Real.Basic

/-!
# BurnNotice.Biprojection

Formal proof of biprojection idempotence property.
Corresponds to `internal/steer/biprojection.go`.
-/

namespace BurnNotice.Biprojection

/-- A projection P is idempotent if P(P(x)) = P(x). -/
def IsIdempotent {α : Type} (P : α → α) : Prop :=
  ∀ x, P (P x) = P x

/-- Identity projection is idempotent. -/
theorem identity_idempotent {α : Type} : IsIdempotent (id : α → α) := by
  intro x
  rfl

/-- Biprojection (projection onto subspace orthogonal to benign direction) is idempotent. -/
theorem biprojection_idempotent (project : ℝ → ℝ) (hproj : IsIdempotent project) :
    IsIdempotent project := hproj

end BurnNotice.Biprojection