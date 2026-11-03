module

public import StegaTextProofs.FiniteEntropyChannel
public import StegaTextProofs.PositiveSupport
public import StegaTextProofs.LASReachability

/-! Conditional laws on positive labels, with zero-weight labels removed explicitly. -/

@[expose] public section

namespace StegaTextProofs.FiniteLaw

open scoped BigOperators
attribute [local instance] Classical.propDecidable
variable {α β : Type*} [Fintype α] [Fintype β]

theorem map_positive_iff (p : FiniteLaw α) (f : α → β) (b : β) :
    0 < (p.map f).prob b ↔ ∃ a, 0 < p.prob a ∧ f a = b := by
  rw [map, mix_positive_iff]
  constructor
  · rintro ⟨a, ha, hb⟩
    refine ⟨a, ha, ?_⟩
    by_contra hn
    simp [point, Ne.symm hn] at hb
  · rintro ⟨a, ha, rfl⟩
    exact ⟨a, ha, by simp [point]⟩

theorem supportLaw_entropyNat (p : FiniteLaw α) : p.supportLaw.entropyNat = p.entropyNat := by
  have he : p.supportLaw.map Subtype.val = p := by
    apply ext_prob
    intro a
    change (p.supportLaw.mix (fun c => point c.val)).prob a = p.prob a
    rw [p.supportLaw_mix (fun c => point c)]
    simp [mix, point]
  exact (p.supportLaw.entropyNat_map_of_injective_support Subtype.val
    (fun _ _ _ _ h => Subtype.ext h)).symm.trans (congrArg FiniteLaw.entropyNat he)

noncomputable def fiberLaw (p : FiniteLaw α) (f : α → β) (b : β)
    (hb : 0 < (p.map f).prob b) : FiniteLaw α where
  prob a := (if f a = b then p.prob a else 0) / (p.map f).prob b
  nonneg a := div_nonneg (by split_ifs; exact p.nonneg a; exact le_rfl) (le_of_lt hb)
  sum_one := by
    rw [← Finset.sum_div, ← p.map_prob f b]
    exact div_self (ne_of_gt hb)

theorem fiberLaw_positive (p : FiniteLaw α) (f : α → β) (b : β)
    (hb : 0 < (p.map f).prob b) (a : α) (ha : 0 < (p.fiberLaw f b hb).prob a) :
    f a = b ∧ 0 < p.prob a := by
  by_cases he : f a = b
  · refine ⟨he, ?_⟩
    change 0 < (if f a = b then p.prob a else 0) / (p.map f).prob b at ha
    rw [ite_eq_left he, div_pos_iff_of_pos_right hb] at ha
    exact ha
  · simp [fiberLaw, he] at ha

theorem fiberLaw_scaled_expect (p : FiniteLaw α) (f : α → β) (b : β)
    (hb : 0 < (p.map f).prob b) (h : α → ℝ) :
    (p.map f).prob b * (p.fiberLaw f b hb).expect h =
      ∑ a, if f a = b then p.prob a * h a else 0 := by
  simp only [expect, fiberLaw, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro a _
  split_ifs
  · field_simp
  · simp

theorem disintegrate_fiber_expect (p : FiniteLaw α) (f : α → β) (h : α → ℝ) :
    (p.map f).supportLaw.expect (fun b => (p.fiberLaw f b.val b.property).expect h) =
      p.expect h := by
  change (∑ b : (p.map f).Positive, (p.map f).prob b.val *
    (p.fiberLaw f b.val b.property).expect h) = _
  simp only [p.fiberLaw_scaled_expect]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro a _
  by_cases ha : 0 < p.prob a
  · have hb := (p.map_positive_iff f (f a)).mpr ⟨a, ha, rfl⟩
    let b : (p.map f).Positive := ⟨f a, hb⟩
    rw [Finset.sum_eq_single b]
    · simp [b]
    · intro c _ hcb
      have hn : f a ≠ c.val := fun he => hcb (Subtype.ext he.symm)
      simp only [ite_eq_right hn]
    · simp
  · have hz : p.prob a = 0 := le_antisymm (le_of_not_gt ha) (p.nonneg a)
    simp [hz]

end StegaTextProofs.FiniteLaw
