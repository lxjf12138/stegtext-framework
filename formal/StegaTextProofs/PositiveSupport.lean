module

public import StegaTextProofs.PrefixLASStep

/-!
Zero-weight candidates are removed by a constructed positive support law.
All expectations and continuation mixtures are preserved. This permits the
prefix-pool constructor to start from any finite probability law.
-/

@[expose] public section

namespace StegaTextProofs

open scoped BigOperators
attribute [local instance] Classical.propDecidable

namespace FiniteLaw

variable {α β : Type*} [Fintype α] [Fintype β]

abbrev Positive (p : FiniteLaw α) := {a : α // 0 < p.prob a}

noncomputable def supportLaw (p : FiniteLaw α) : FiniteLaw p.Positive where
  prob a := p.prob a.val
  nonneg a := p.nonneg a.val
  sum_one := by
    have hz : ∑ a : {a : α // ¬ 0 < p.prob a}, p.prob a.val = 0 := by
      apply Finset.sum_eq_zero
      intro a _
      exact le_antisymm (le_of_not_gt a.property) (p.nonneg a.val)
    have h := Fintype.sum_subtype_add_sum_subtype (fun a => 0 < p.prob a) p.prob
    rw [hz, add_zero] at h
    exact h.trans p.sum_one

theorem supportLaw_positive (p : FiniteLaw α) (a : p.Positive) :
    0 < p.supportLaw.prob a := a.property

theorem supportLaw_expect (p : FiniteLaw α) (f : α → ℝ) :
    p.supportLaw.expect (fun a => f a.val) = p.expect f := by
  have hz : ∑ a : {a : α // ¬ 0 < p.prob a}, p.prob a.val * f a.val = 0 := by
    apply Finset.sum_eq_zero
    intro a _
    rw [le_antisymm (le_of_not_gt a.property) (p.nonneg a.val), zero_mul]
  have h := Fintype.sum_subtype_add_sum_subtype (fun a => 0 < p.prob a)
    (fun a => p.prob a * f a)
  simpa only [expect, supportLaw, hz, add_zero] using h

theorem supportLaw_mix (p : FiniteLaw α) (channel : α → FiniteLaw β) :
    p.supportLaw.mix (fun a => channel a.val) = p.mix channel := by
  apply ext_prob
  intro y
  exact p.supportLaw_expect (fun a => (channel a).prob y)

end FiniteLaw

namespace PrefixPool

variable {Candidate Byte Output : Type*} [Fintype Candidate] [Fintype Output]

noncomputable def ofLaw (p : FiniteLaw Candidate) (residual : Candidate → List Byte) :
    PrefixPool p.Positive Byte where
  prior := p.supportLaw
  positive := p.supportLaw_positive
  residual c := residual c.val

theorem ofLaw_disintegrate (p : FiniteLaw Candidate) (residual : Candidate → List Byte)
    (channel : Candidate → FiniteLaw Output) :
    (ofLaw p residual).groups.mix (fun g => ((ofLaw p residual).selected g).prior.mix
      (fun c => channel ((ofLaw p residual).candidate g c).val)) = p.mix channel := by
  rw [(ofLaw p residual).disintegrate_channel (fun c => channel c.val)]
  exact p.supportLaw_mix channel

end PrefixPool

end StegaTextProofs
