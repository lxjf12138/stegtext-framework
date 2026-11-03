module

public import StegaTextProofs.Entropy

/-!
Unnormalised entropy of a nonnegative finite weight vector. This is the
perspective of Shannon entropy, and is the contribution of one decision-tree
node. Natural logarithms are used throughout this module.
-/

@[expose] public section

namespace StegaTextProofs

open scoped BigOperators

variable {α β : Type*} [Fintype α] [Fintype β]

noncomputable def weightedEntropy (w : α → ℝ) : ℝ :=
  (∑ a, Real.negMulLog (w a)) - Real.negMulLog (∑ a, w a)

@[simp] theorem weightedEntropy_zero : weightedEntropy (fun _ : α => 0) = 0 := by
  simp [weightedEntropy]

/-- Homogeneity is algebraic, including zero mass. -/
theorem weightedEntropy_smul (c : ℝ) (w : α → ℝ) :
    weightedEntropy (fun a => c * w a) = c * weightedEntropy w := by
  simp only [weightedEntropy, Real.negMulLog_mul, Finset.sum_add_distrib,
    ← Finset.sum_mul, ← Finset.mul_sum]
  ring

theorem weightedEntropy_law (p : FiniteLaw α) :
    weightedEntropy p.prob = p.entropyNat := by
  simp [weightedEntropy, FiniteLaw.entropyNat, p.sum_one]

/-- A vector supported at one coordinate carries no decision entropy. -/
theorem weightedEntropy_single (b : α) (w : α → ℝ)
    (hsupport : ∀ a, a ≠ b → w a = 0) : weightedEntropy w = 0 := by
  classical
  have hw : w = fun a => if a = b then w b else 0 := by
    funext a
    split_ifs with h
    · subst a; rfl
    · exact hsupport a h
  rw [hw]
  simp [weightedEntropy, apply_ite]

namespace FiniteLaw

theorem prob_le_one (p : FiniteLaw α) (a : α) : p.prob a ≤ 1 := by
  calc
    p.prob a ≤ ∑ b, p.prob b :=
      Finset.single_le_sum (fun b _ => p.nonneg b) (Finset.mem_univ a)
    _ = 1 := p.sum_one

theorem entropyNat_nonneg (p : FiniteLaw α) : 0 ≤ p.entropyNat := by
  exact Finset.sum_nonneg fun a _ => Real.negMulLog_nonneg (p.nonneg a) (p.prob_le_one a)

/-- Normalisation is used only when the total weight is strictly positive. -/
noncomputable def ofWeights (w : α → ℝ) (hw : ∀ a, 0 ≤ w a)
    (hm : 0 < ∑ a, w a) : FiniteLaw α where
  prob a := w a / ∑ b, w b
  nonneg a := div_nonneg (hw a) (le_of_lt hm)
  sum_one := by rw [← Finset.sum_div]; exact div_self (ne_of_gt hm)

end FiniteLaw

theorem weightedEntropy_normalize (w : α → ℝ) (hw : ∀ a, 0 ≤ w a)
    (hm : 0 < ∑ a, w a) :
    weightedEntropy w = (∑ a, w a) * (FiniteLaw.ofWeights w hw hm).entropyNat := by
  rw [← weightedEntropy_law, ← weightedEntropy_smul]
  congr 1
  funext a
  exact (mul_div_cancel₀ (w a) (ne_of_gt hm)).symm

theorem weightedEntropy_nonneg (w : α → ℝ) (hw : ∀ a, 0 ≤ w a) :
    0 ≤ weightedEntropy w := by
  have hm : 0 ≤ ∑ a, w a := Finset.sum_nonneg fun a _ => hw a
  rcases eq_or_lt_of_le hm with hm | hm
  · have hz : ∀ a, w a = 0 := by
      intro a
      have ha : w a ≤ ∑ b, w b :=
        Finset.single_le_sum (fun b _ => hw b) (Finset.mem_univ a)
      linarith [hw a]
    rw [show w = (fun _ => 0) from funext hz, weightedEntropy_zero]
  · rw [weightedEntropy_normalize w hw hm]
    exact mul_nonneg (le_of_lt hm) (FiniteLaw.entropyNat_nonneg _)

theorem weights_zero_of_sum_zero (w : α → ℝ) (hw : ∀ a, 0 ≤ w a)
    (hm : ∑ a, w a = 0) (a : α) : w a = 0 := by
  have ha := Finset.single_le_sum (fun b _ => hw b) (Finset.mem_univ a)
  exact le_antisymm (hm ▸ ha) (hw a)

/-- Entropy concavity with arbitrary nonnegative outer masses. -/
theorem weightedEntropy_sum_laws (m : α → ℝ) (hm : ∀ a, 0 ≤ m a)
    (q : α → FiniteLaw β) :
    (∑ a, m a * (q a).entropyNat) ≤
      weightedEntropy (fun b => ∑ a, m a * (q a).prob b) := by
  have htotal : 0 ≤ ∑ a, m a := Finset.sum_nonneg fun a _ => hm a
  rcases eq_or_lt_of_le htotal with hzero | hpos
  · have hz := weights_zero_of_sum_zero m hm hzero.symm
    simp [hz, weightedEntropy]
  · let p := FiniteLaw.ofWeights m hm hpos
    have hid : (fun b => ∑ a, m a * (q a).prob b) =
        (fun b => (∑ a, m a) * (p.mix q).prob b) := by
      funext b
      simp only [FiniteLaw.mix, Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro a _
      dsimp [p, FiniteLaw.ofWeights]
      field_simp
    rw [hid, weightedEntropy_smul, weightedEntropy_law]
    calc
      (∑ a, m a * (q a).entropyNat) =
          (∑ a, m a) * p.expect (fun a => (q a).entropyNat) := by
        simp only [FiniteLaw.expect, Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro a _
        dsimp [p, FiniteLaw.ofWeights]
        field_simp
      _ ≤ (∑ a, m a) * (p.mix q).entropyNat :=
        mul_le_mul_of_nonneg_left (p.entropyNat_mixture_concave q) (le_of_lt hpos)

/-- The entropy perspective is superadditive on nonnegative vectors. -/
theorem weightedEntropy_sum_le (w : α → β → ℝ) (hw : ∀ a b, 0 ≤ w a b) :
    (∑ a, weightedEntropy (w a)) ≤ weightedEntropy (fun b => ∑ a, w a b) := by
  classical
  by_cases hn : Nonempty β
  · let m : α → ℝ := fun a => ∑ b, w a b
    have hm : ∀ a, 0 ≤ m a := fun a => Finset.sum_nonneg fun b _ => hw a b
    let q : α → FiniteLaw β := fun a =>
      if h : 0 < m a then FiniteLaw.ofWeights (w a) (hw a) h
      else FiniteLaw.point (Classical.choice hn)
    have hfactor : ∀ a b, w a b = m a * (q a).prob b := by
      intro a b
      dsimp [q]
      split_ifs with h
      · dsimp [FiniteLaw.ofWeights]
        exact (mul_div_cancel₀ (w a b) (ne_of_gt h)).symm
      · have hz : m a = 0 := le_antisymm (le_of_not_gt h) (hm a)
        rw [hz, zero_mul]
        exact weights_zero_of_sum_zero (w a) (hw a) hz b
    have hcost : ∀ a, weightedEntropy (w a) = m a * (q a).entropyNat := by
      intro a
      rw [show w a = (fun b => m a * (q a).prob b) from funext (hfactor a),
        weightedEntropy_smul, weightedEntropy_law]
    simpa only [hcost, ← hfactor] using weightedEntropy_sum_laws m hm q
  · have : IsEmpty β := not_nonempty_iff.mp hn
    simp [weightedEntropy]

/-- Jensen's inequality for node contributions, even when row masses differ. -/
theorem weightedEntropy_expect_le (p : FiniteLaw α) (w : α → β → ℝ)
    (hw : ∀ a b, 0 ≤ w a b) :
    p.expect (fun a => weightedEntropy (w a)) ≤
      weightedEntropy (fun b => p.expect (fun a => w a b)) := by
  simpa only [FiniteLaw.expect, weightedEntropy_smul] using
    weightedEntropy_sum_le (fun a b => p.prob a * w a b)
      (fun a b => mul_nonneg (p.nonneg a) (hw a b))

/-- The entropy chain rule for an arbitrary finite dependent partition. -/
theorem weightedEntropy_sigma_chain {ι : Type*} [Fintype ι]
    {κ : ι → Type*} [∀ i, Fintype (κ i)] (w : (i : ι) × κ i → ℝ) :
    weightedEntropy w = weightedEntropy (fun i => ∑ k, w ⟨i, k⟩) +
      ∑ i, weightedEntropy (fun k => w ⟨i, k⟩) := by
  simp only [weightedEntropy, Fintype.sum_sigma, Finset.sum_sub_distrib]
  ring

theorem weightedEntropy_prod_chain (w : α × β → ℝ) :
    weightedEntropy w = weightedEntropy (fun a => ∑ b, w (a, b)) +
      ∑ a, weightedEntropy (fun b => w (a, b)) := by
  simp only [weightedEntropy, Fintype.sum_prod_type, Finset.sum_sub_distrib]
  ring

/-- Revealing a mixture component costs at most its entropy. -/
theorem FiniteLaw.entropyNat_mix_le (p : FiniteLaw α) (q : α → FiniteLaw β) :
    (p.mix q).entropyNat ≤ p.entropyNat + p.expect (fun a => (q a).entropyNat) := by
  let joint : α × β → ℝ := fun ab => p.prob ab.1 * (q ab.1).prob ab.2
  have hrow := weightedEntropy_prod_chain joint
  have hcol := weightedEntropy_prod_chain (fun ba : β × α => joint (ba.2, ba.1))
  have hswap : weightedEntropy (fun ba : β × α => joint (ba.2, ba.1)) =
      weightedEntropy joint := by
    simp only [weightedEntropy, Fintype.sum_prod_type]
    rw [Finset.sum_comm (f := fun b a => Real.negMulLog (joint (a, b))),
      Finset.sum_comm (f := fun b a => joint (a, b))]
  have hnonneg : 0 ≤ ∑ b, weightedEntropy (fun a => joint (a, b)) :=
    Finset.sum_nonneg fun b _ => weightedEntropy_nonneg _
      (fun a => mul_nonneg (p.nonneg a) ((q a).nonneg b))
  have hrows : weightedEntropy joint = p.entropyNat +
      p.expect (fun a => (q a).entropyNat) := by
    simpa only [joint, ← Finset.mul_sum, FiniteLaw.sum_one, mul_one,
      weightedEntropy_law, weightedEntropy_smul, FiniteLaw.expect] using hrow
  have hcols : weightedEntropy joint = (p.mix q).entropyNat +
      ∑ b, weightedEntropy (fun a => joint (a, b)) := by
    rw [hswap] at hcol
    change _ = weightedEntropy (p.mix q).prob + _ at hcol
    simpa only [weightedEntropy_law] using hcol
  linarith

end StegaTextProofs
