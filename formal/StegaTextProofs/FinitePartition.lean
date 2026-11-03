module

public import StegaTextProofs.TerminalCollapse

/-!
Finite partition disintegration, constructed from candidate weights.
Groups have their actual fibre masses; positive groups have normalized
conditional laws. Averaging those laws recovers the original law and every
continuation channel. No disintegration identity is supplied as a premise.
-/

@[expose] public section

namespace StegaTextProofs

open scoped BigOperators
attribute [local instance] Classical.propDecidable

namespace FinitePartition

variable {Candidate Label Output : Type*}
  [Fintype Candidate] [Fintype Label] [Fintype Output]

abbrev Fiber (key : Candidate → Label) (g : Label) := {c : Candidate // key c = g}

noncomputable instance (priority := 2000) fiberFintype (key : Candidate → Label) (g : Label) :
    Fintype (Fiber key g) := Fintype.ofFinite _

theorem sum_fibers (key : Candidate → Label) (f : Candidate → ℝ) :
    (∑ g, ∑ c : Fiber key g, f c.val) = ∑ c, f c := by
  exact (Fintype.sum_sigma (fun x : Sigma (Fiber key) => f x.2.val)).symm.trans
    ((Equiv.sigmaFiberEquiv key).sum_comp f)

noncomputable def mass (p : FiniteLaw Candidate) (key : Candidate → Label) (g : Label) : ℝ :=
  ∑ c : Fiber key g, p.prob c.val

omit [Fintype Label] in
theorem mass_nonneg (p : FiniteLaw Candidate) (key : Candidate → Label) (g : Label) :
    0 ≤ mass p key g := Finset.sum_nonneg (fun c _ => p.nonneg c.val)

omit [Fintype Label] in
theorem mass_positive_of_member (p : FiniteLaw Candidate) (key : Candidate → Label)
    (g : Label) (c : Candidate) (hc : key c = g) (hp : 0 < p.prob c) :
    0 < mass p key g := by
  have hle : p.prob c ≤ mass p key g := Finset.single_le_sum
    (fun (a : Fiber key g) _ => p.nonneg a.val) (Finset.mem_univ ⟨c, hc⟩)
  exact hp.trans_le hle

noncomputable def groups (p : FiniteLaw Candidate) (key : Candidate → Label) : FiniteLaw Label where
  prob := mass p key
  nonneg := mass_nonneg p key
  sum_one := (sum_fibers key p.prob).trans p.sum_one

noncomputable def conditional (p : FiniteLaw Candidate) (key : Candidate → Label)
    (g : Label) (hg : 0 < mass p key g) : FiniteLaw (Fiber key g) where
  prob c := p.prob c.val / mass p key g
  nonneg c := div_nonneg (p.nonneg c.val) (le_of_lt hg)
  sum_one := by
    rw [← Finset.sum_div]
    exact div_self (ne_of_gt hg)

theorem disintegrate_expect (p : FiniteLaw Candidate) (key : Candidate → Label)
    (hpositive : ∀ g, 0 < mass p key g) (f : Candidate → ℝ) :
    (groups p key).expect (fun g => (conditional p key g (hpositive g)).expect
      (fun c => f c.val)) = p.expect f := by
  simp only [FiniteLaw.expect, groups, conditional, Finset.mul_sum]
  calc
    _ = ∑ g, ∑ c : Fiber key g, p.prob c.val * f c.val := by
      apply Finset.sum_congr rfl
      intro g _
      apply Finset.sum_congr rfl
      intro c _
      field_simp [ne_of_gt (hpositive g)]
    _ = _ := sum_fibers key (fun c => p.prob c * f c)

theorem disintegrate_channel (p : FiniteLaw Candidate) (key : Candidate → Label)
    (hpositive : ∀ g, 0 < mass p key g) (channel : Candidate → FiniteLaw Output) :
    (groups p key).mix (fun g => (conditional p key g (hpositive g)).mix
      (fun c => channel c.val)) = p.mix channel := by
  apply FiniteLaw.ext_prob
  intro y
  exact disintegrate_expect p key hpositive (fun c => (channel c).prob y)

end FinitePartition

namespace FiniteLaw

variable {α β γ : Type*} [Fintype α] [Fintype β] [Fintype γ]

noncomputable def reindex (p : FiniteLaw α) (e : β ≃ α) : FiniteLaw β where
  prob b := p.prob (e b)
  nonneg b := p.nonneg (e b)
  sum_one := (e.sum_comp p.prob).trans p.sum_one

theorem reindex_expect (p : FiniteLaw α) (e : β ≃ α) (f : α → ℝ) :
    (p.reindex e).expect (fun b => f (e b)) = p.expect f :=
  e.sum_comp (fun a => p.prob a * f a)

theorem reindex_mix (p : FiniteLaw α) (e : β ≃ α) (channel : α → FiniteLaw γ) :
    (p.reindex e).mix (fun b => channel (e b)) = p.mix channel := by
  apply ext_prob
  intro y
  exact p.reindex_expect e (fun a => (channel a).prob y)

end FiniteLaw

end StegaTextProofs
