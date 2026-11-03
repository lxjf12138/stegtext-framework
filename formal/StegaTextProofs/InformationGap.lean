module

public import StegaTextProofs.FiniteConditionalLaw
public import StegaTextProofs.SourceCursors

/-! Finite channel information and convexity in the channel, in nats. -/

@[expose] public section

namespace StegaTextProofs

open scoped BigOperators

namespace FiniteLaw

variable {α β γ : Type*} [Fintype α] [Fintype β] [Fintype γ]

noncomputable def informationNat (p : FiniteLaw α) (q : α → FiniteLaw β) : ℝ :=
  (p.mix q).entropyNat - p.expect (fun a => (q a).entropyNat)

theorem informationNat_nonneg (p : FiniteLaw α) (q : α → FiniteLaw β) :
    0 ≤ p.informationNat q := sub_nonneg.mpr (p.entropyNat_mixture_concave q)

theorem informationNat_le_entropy (p : FiniteLaw α) (q : α → FiniteLaw β) :
    p.informationNat q ≤ p.entropyNat := by
  have := p.entropyNat_mix_le q
  unfold informationNat
  linarith

theorem mix_const (p : FiniteLaw α) (q : FiniteLaw β) : p.mix (fun _ => q) = q := by
  apply ext_prob
  intro b
  exact p.expect_const (q.prob b)

@[simp] theorem informationNat_const (p : FiniteLaw α) (q : FiniteLaw β) :
    p.informationNat (fun _ => q) = 0 := by
  simp [informationNat, mix_const, expect_const]

theorem informationNat_posterior (p : FiniteLaw α) (q : α → FiniteLaw β) :
    p.informationNat q = p.entropyNat -
      ∑ b, weightedEntropy (fun a => p.prob a * (q a).prob b) := by
  let w : α × β → ℝ := fun ab => p.prob ab.1 * (q ab.1).prob ab.2
  have hr := weightedEntropy_prod_chain w
  have hc := weightedEntropy_prod_chain (fun ba : β × α => w (ba.2, ba.1))
  have hs : weightedEntropy (fun ba : β × α => w (ba.2, ba.1)) = weightedEntropy w := by
    simp only [weightedEntropy, Fintype.sum_prod_type]
    rw [Finset.sum_comm (f := fun b a => Real.negMulLog (w (a, b))),
      Finset.sum_comm (f := fun b a => w (a, b))]
  have hr' : weightedEntropy w = p.entropyNat + p.expect (fun a => (q a).entropyNat) := by
    simpa only [w, ← Finset.mul_sum, sum_one, mul_one, weightedEntropy_law,
      weightedEntropy_smul, expect] using hr
  rw [hs] at hc
  change weightedEntropy w = weightedEntropy (p.mix q).prob + _ at hc
  rw [weightedEntropy_law] at hc
  unfold informationNat
  dsimp [w] at hc
  linarith

/-- Mixing channels independently of the input cannot increase information. -/
theorem informationNat_mix_channel_le (p : FiniteLaw α) (r : FiniteLaw γ)
    (q : γ → α → FiniteLaw β) :
    p.informationNat (fun a => r.mix (fun c => q c a)) ≤
      r.expect (fun c => p.informationNat (q c)) := by
  simp only [informationNat_posterior]
  have h : r.expect (fun c => ∑ b, weightedEntropy (fun a => p.prob a * (q c a).prob b)) ≤
      ∑ b, weightedEntropy (fun a => p.prob a * (r.mix (fun c => q c a)).prob b) := by
    simp only [expect, Finset.mul_sum]
    rw [Finset.sum_comm]
    apply Finset.sum_le_sum
    intro b _
    have h := weightedEntropy_expect_le r (fun c a => p.prob a * (q c a).prob b)
      (fun c a => mul_nonneg (p.nonneg a) ((q c a).nonneg b))
    simpa only [expect, mix, Finset.mul_sum, mul_left_comm] using h
  rw [expect_sub, expect_const]
  linarith

theorem mix_map (p : FiniteLaw α) (q : α → FiniteLaw β) (f : β → γ) :
    (p.mix q).map f = p.mix (fun a => (q a).map f) := by
  apply ext_expect
  intro g
  simp only [map_expect, expect_mix]

/-- Data processing for a stochastic post-processing channel. -/
theorem informationNat_data_processing (p : FiniteLaw α) (q : α → FiniteLaw β)
    (r : β → FiniteLaw γ) :
    p.informationNat (fun a => (q a).mix r) ≤ p.informationNat q := by
  rw [informationNat_posterior, informationNat_posterior]
  have h : (∑ b, weightedEntropy (fun a => p.prob a * (q a).prob b)) ≤
      ∑ c, weightedEntropy (fun a => p.prob a * ((q a).mix r).prob c) := by
    have hc : ∀ c, (∑ b, (r b).prob c * weightedEntropy (fun a => p.prob a * (q a).prob b)) ≤
        weightedEntropy (fun a => p.prob a * ((q a).mix r).prob c) := by
      intro c
      have h := weightedEntropy_sum_le
        (fun b a => (r b).prob c * (p.prob a * (q a).prob b))
        (fun b a => mul_nonneg ((r b).nonneg c) (mul_nonneg (p.nonneg a) ((q a).nonneg b)))
      simp only [weightedEntropy_smul] at h
      simpa only [mix, Finset.mul_sum, mul_comm, mul_left_comm, mul_assoc] using h
    have hh := Finset.sum_le_sum (fun c (_ : c ∈ Finset.univ) => hc c)
    rw [Finset.sum_comm] at hh
    simpa only [← Finset.sum_mul, sum_one, one_mul] using hh
  linarith

/-- In particular, full future token information upper-bounds deterministic visible readout. -/
theorem informationNat_map_le (p : FiniteLaw α) (q : α → FiniteLaw β) (f : β → γ) :
    p.informationNat (fun a => (q a).map f) ≤ p.informationNat q :=
  p.informationNat_data_processing q (fun b => point (f b))

theorem mix_eq_joint_map (p : FiniteLaw α) (q : α → FiniteLaw β) :
    p.mix q = (p.joint q).map (fun x => x.2) := by
  apply ext_expect
  intro f
  simp only [expect_mix, map_expect, joint_expect]

/-- A visible output identifying the mixture component yields equality in the chain bound. -/
theorem entropyNat_mix_eq_of_disjoint (p : FiniteLaw α) (q : α → FiniteLaw β)
    (hd : ∀ a c b, 0 < (q a).prob b → 0 < (q c).prob b → a = c) :
    (p.mix q).entropyNat = p.entropyNat + p.expect (fun a => (q a).entropyNat) := by
  rw [mix_eq_joint_map, entropyNat_map_of_injective_support, joint_entropyNat]
  intro x y hx hy he
  have hc := hd x.1 y.1 x.2
    (pos_of_mul_pos_right hx (p.nonneg _))
    (he ▸ pos_of_mul_pos_right hy (p.nonneg _))
  exact Sigma.ext hc (heq_of_eq he)

end FiniteLaw
end StegaTextProofs
