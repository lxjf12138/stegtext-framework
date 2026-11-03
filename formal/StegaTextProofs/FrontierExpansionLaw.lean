module

public import StegaTextProofs.PositiveSupport
public import StegaTextProofs.FrontierTransitions

/-!
The active frontier atom is expanded by one token draw; retained atoms are
carried unchanged. This is a normalized finite-law construction, including
zero active mass. Removing zero weights and selecting native prefix groups
then gives a concrete LAS one-step disintegration.
-/

@[expose] public section

namespace StegaTextProofs

open scoped BigOperators

namespace FiniteLaw

variable {α β γ : Type*} [Fintype α] [Fintype β] [Fintype γ]

theorem mix_assoc (p : FiniteLaw α) (q : α → FiniteLaw β) (r : β → FiniteLaw γ) :
    (p.mix q).mix r = p.mix (fun a => (q a).mix r) := by
  apply ext_prob
  intro c
  exact p.expect_mix q (fun b => (r b).prob c)

end FiniteLaw

namespace FrontierExpansion

attribute [local instance] Classical.propDecidable

variable {Token Retained Output Byte : Type*}
  [Fintype Token] [Fintype Retained] [Fintype Output]

noncomputable def expanded (frontier : FiniteLaw (Unit ⊕ Retained)) (tokens : FiniteLaw Token) :
    FiniteLaw (Token ⊕ Retained) :=
  frontier.mix (Sum.elim (fun _ => tokens.mix (fun t => FiniteLaw.point (.inl t)))
    (fun r => FiniteLaw.point (.inr r)))

theorem expanded_expect (frontier : FiniteLaw (Unit ⊕ Retained)) (tokens : FiniteLaw Token)
    (f : Token ⊕ Retained → ℝ) :
    (expanded frontier tokens).expect f =
      frontier.prob (.inl ()) * tokens.expect (fun t => f (.inl t)) +
        ∑ r, frontier.prob (.inr r) * f (.inr r) := by
  rw [expanded, FiniteLaw.expect_mix]
  have hi (c : Unit ⊕ Retained) :
      (Sum.elim (fun _ => tokens.mix (fun t => FiniteLaw.point (.inl t)))
        (fun r => FiniteLaw.point (.inr r)) c).expect f =
      Sum.elim (fun _ => tokens.expect (fun t => f (.inl t))) (fun r => f (.inr r)) c := by
    cases c <;> simp only [Sum.elim_inl, Sum.elim_inr, FiniteLaw.expect_mix, FiniteLaw.expect_point]
  simp only [hi]
  simp [FiniteLaw.expect, Fintype.sum_sum_type]

theorem fresh_probability (frontier : FiniteLaw (Unit ⊕ Retained)) (tokens : FiniteLaw Token)
    (t : Token) :
    (expanded frontier tokens).prob (.inl t) = frontier.prob (.inl ()) * tokens.prob t := by
  classical
  simp [expanded, FiniteLaw.mix, FiniteLaw.point, Fintype.sum_sum_type]

theorem retained_probability (frontier : FiniteLaw (Unit ⊕ Retained)) (tokens : FiniteLaw Token)
    (r : Retained) : (expanded frontier tokens).prob (.inr r) = frontier.prob (.inr r) := by
  classical
  simp [expanded, FiniteLaw.mix, FiniteLaw.point, Fintype.sum_sum_type]

theorem expanded_channel (frontier : FiniteLaw (Unit ⊕ Retained)) (tokens : FiniteLaw Token)
    (fresh : Token → FiniteLaw Output) (retained : Retained → FiniteLaw Output) :
    (expanded frontier tokens).mix (Sum.elim fresh retained) =
      frontier.mix (Sum.elim (fun _ => tokens.mix fresh) retained) := by
  rw [expanded, FiniteLaw.mix_assoc]
  congr 1
  funext c
  cases c with
  | inl u =>
    simp only [Sum.elim_inl, FiniteLaw.mix_assoc, FiniteLaw.point_mix]
  | inr r => simp only [Sum.elim_inr, FiniteLaw.point_mix]

/-- Constructed group laws reproduce the original active/retained path belief. -/
theorem grouped_expansion_channel (frontier : FiniteLaw (Unit ⊕ Retained))
    (tokens : FiniteLaw Token) (residual : Token ⊕ Retained → List Byte)
    (fresh : Token → FiniteLaw Output) (retained : Retained → FiniteLaw Output) :
    let pool := PrefixPool.ofLaw (expanded frontier tokens) residual
    pool.groups.mix (fun g => (pool.selected g).prior.mix (fun c =>
      Sum.elim fresh retained (pool.candidate g c).val)) =
        frontier.mix (Sum.elim (fun _ => tokens.mix fresh) retained) := by
  exact (PrefixPool.ofLaw_disintegrate (expanded frontier tokens) residual
    (Sum.elim fresh retained)).trans (expanded_channel frontier tokens fresh retained)

end FrontierExpansion

end StegaTextProofs
