module

public import StegaTextProofs.Entropy
public import StegaTextProofs.PrefixGrouping
public import StegaTextProofs.FiniteHorizon

/-!
Concrete witnesses for the formalized definitions. The collision example has
two exact labels with mass 1/4 each and a strict label with mass 1/2. The strict
label stays present after either surrogate is chosen. These examples are
kernel-checked rational calculations, not floating-point experiments.
-/

@[expose] public section

namespace StegaTextProofs.Examples

open scoped BigOperators

noncomputable def collisionGroup : LASGroup Bool Unit where
  prior := {
    prob := Sum.elim (fun _ => (1 : ℝ) / 4) (fun _ => (1 : ℝ) / 2)
    nonneg := by intro c; cases c <;> norm_num
    sum_one := by norm_num [Fintype.sum_sum_type]
  }
  exact_pos := by norm_num

theorem collision_surrogate_fair (e : Bool) : collisionGroup.surrogate.prob e = 1 / 2 := by
  norm_num [LASGroup.surrogate, LASGroup.exactMass, collisionGroup]

theorem collision_posterior (e : Bool) :
    (collisionGroup.collapse e).prob (.inl e) = 1 / 2 ∧
    (collisionGroup.collapse e).prob (.inl (!e)) = 0 ∧
    (collisionGroup.collapse e).prob (.inr ()) = 1 / 2 := by
  cases e <;> norm_num [LASGroup.collapse, LASGroup.exactMass, collisionGroup]

/-- The posterior formula includes a strict candidate even after conditioning
on a particular exact surrogate. -/
theorem collision_conditional_strict (e : Bool) :
    collisionGroup.coupling.prob (.inr (), e) /
      (∑ c, collisionGroup.coupling.prob (c, e)) = 1 / 2 := by
  rw [collisionGroup.surrogate_conditional_law e (by norm_num [collisionGroup])]
  exact (collision_posterior e).2.2

/-- Without a supported short word, merely sharing a byte prefix does not merge groups. -/
theorem absent_prefix_keeps_groups_separate :
    PrefixGrouping.MinimalKey ({[0, 1], [0, 2]} : Finset (List ℕ)) [0, 1] ∧
    PrefixGrouping.MinimalKey ({[0, 1], [0, 2]} : Finset (List ℕ)) [0, 2] := by
  norm_num [PrefixGrouping.MinimalKey]

end StegaTextProofs.Examples
