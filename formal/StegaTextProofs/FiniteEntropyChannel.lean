module

public import StegaTextProofs.WeightedEntropy
public import StegaTextProofs.FinitePartition

/-! Finite dependent joint laws, Shannon conditional entropy and lossless readout. -/

@[expose] public section

namespace StegaTextProofs.FiniteLaw

open scoped BigOperators
attribute [local instance] Classical.propDecidable
variable {α β : Type*} [Fintype α] [Fintype β]

noncomputable def joint {κ : α → Type*} [∀ a, Fintype (κ a)]
    (p : FiniteLaw α) (q : ∀ a, FiniteLaw (κ a)) : FiniteLaw ((a : α) × κ a) where
  prob x := p.prob x.1 * (q x.1).prob x.2
  nonneg x := mul_nonneg (p.nonneg _) ((q x.1).nonneg _)
  sum_one := by
    rw [Fintype.sum_sigma]
    calc
      _ = ∑ a, p.prob a := by
        apply Finset.sum_congr rfl
        intro a _
        change (∑ b, p.prob a * (q a).prob b) = p.prob a
        rw [← Finset.mul_sum, (q a).sum_one, mul_one]
      _ = 1 := p.sum_one

theorem joint_expect {κ : α → Type*} [∀ a, Fintype (κ a)]
    (p : FiniteLaw α) (q : ∀ a, FiniteLaw (κ a)) (f : ((a : α) × κ a) → ℝ) :
    (p.joint q).expect f = p.expect (fun a => (q a).expect (fun b => f ⟨a, b⟩)) := by
  simp only [expect, joint, Fintype.sum_sigma, Finset.mul_sum, mul_assoc]

theorem joint_entropyNat {κ : α → Type*} [∀ a, Fintype (κ a)]
    (p : FiniteLaw α) (q : ∀ a, FiniteLaw (κ a)) :
    (p.joint q).entropyNat = p.entropyNat + p.expect (fun a => (q a).entropyNat) := by
  have h := weightedEntropy_sigma_chain (p.joint q).prob
  rw [weightedEntropy_law] at h
  have hm : (fun a => ∑ b, (p.joint q).prob ⟨a, b⟩) = p.prob := by
    funext a
    change (∑ b, p.prob a * (q a).prob b) = p.prob a
    rw [← Finset.mul_sum, (q a).sum_one, mul_one]
  have hc : ∀ a, weightedEntropy (fun b => (p.joint q).prob ⟨a, b⟩) =
      p.prob a * (q a).entropyNat := by
    intro a
    change weightedEntropy (fun b => p.prob a * (q a).prob b) = _
    rw [weightedEntropy_smul, weightedEntropy_law]
  simpa only [hm, hc, weightedEntropy_law, expect] using h

noncomputable def conditionalEntropyNat (p : FiniteLaw α) (q : α → FiniteLaw β) : ℝ :=
  (p.joint q).entropyNat - p.entropyNat

theorem conditionalEntropyNat_eq_expect (p : FiniteLaw α) (q : α → FiniteLaw β) :
    p.conditionalEntropyNat q = p.expect (fun a => (q a).entropyNat) := by
  rw [conditionalEntropyNat, joint_entropyNat]
  ring

theorem entropyNat_point (a : α) : (point a).entropyNat = 0 := by
  classical
  simp [entropyNat, point, apply_ite]

noncomputable def map (p : FiniteLaw α) (f : α → β) : FiniteLaw β :=
  p.mix (fun a => point (f a))

theorem map_expect (p : FiniteLaw α) (f : α → β) (g : β → ℝ) :
    (p.map f).expect g = p.expect (fun a => g (f a)) := by
  simp only [map, expect_mix, expect_point]

theorem expect_comm (p : FiniteLaw α) (q : FiniteLaw β) (f : α → β → ℝ) :
    p.expect (fun a => q.expect (f a)) = q.expect (fun b => p.expect (fun a => f a b)) := by
  simp only [expect, Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro b _
  apply Finset.sum_congr rfl
  intro a _
  ring

theorem map_prob (p : FiniteLaw α) (f : α → β) (b : β) :
    (p.map f).prob b = ∑ a, if f a = b then p.prob a else 0 := by
  classical
  simp [map, mix, point, eq_comm, mul_ite]

/-- A deterministic readout loses no entropy when it is injective on positive support. -/
theorem entropyNat_map_of_injective_support (p : FiniteLaw α) (f : α → β)
    (hinj : ∀ a b, 0 < p.prob a → 0 < p.prob b → f a = f b → a = b) :
    (p.map f).entropyNat = p.entropyNat := by
  classical
  have hfiber : ∀ b, weightedEntropy (fun a => if f a = b then p.prob a else 0) = 0 := by
    intro b
    by_cases hex : ∃ a, 0 < p.prob a ∧ f a = b
    · obtain ⟨a, ha, hab⟩ := hex
      apply weightedEntropy_single a
      intro c hca
      by_cases hcb : f c = b
      · rw [ite_eq_left hcb]
        apply le_antisymm _ (p.nonneg c)
        apply le_of_not_gt
        intro hc
        exact hca (hinj c a hc ha (hcb.trans hab.symm))
      · exact ite_eq_right hcb
    · have hz : ∀ a, (if f a = b then p.prob a else 0) = 0 := by
        intro a
        split_ifs with hab
        · exact le_antisymm (le_of_not_gt (fun ha => hex ⟨a, ha, hab⟩)) (p.nonneg a)
        · rfl
      simp only [hz, weightedEntropy_zero]
  have hsum : ∀ b, Real.negMulLog ((p.map f).prob b) =
      ∑ a, if f a = b then Real.negMulLog (p.prob a) else 0 := by
    intro b
    have h := hfiber b
    rw [weightedEntropy, sub_eq_zero] at h
    rw [map_prob, ← h]
    apply Finset.sum_congr rfl
    intro a _
    split_ifs <;> simp
  simp only [entropyNat, hsum]
  rw [Finset.sum_comm]
  simp

end StegaTextProofs.FiniteLaw
