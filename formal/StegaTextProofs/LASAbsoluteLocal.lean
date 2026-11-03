module

public import StegaTextProofs.InformationGap
public import StegaTextProofs.CollapsedFrontier

/-! Absolute information lost at an exact collapse; arbitrary full-future channel. -/

@[expose] public section

namespace StegaTextProofs.LASGroup

open scoped BigOperators
variable {Exact Strict Output : Type*} [Fintype Exact] [Fintype Strict] [Fintype Output]

theorem collapse_channel_mean (g : LASGroup Exact Strict)
    (channel : Exact ⊕ Strict → FiniteLaw Output) :
    g.surrogate.mix (fun e => (g.collapse e).mix channel) = g.prior.mix channel := by
  apply FiniteLaw.ext_expect
  intro f
  simp only [FiniteLaw.expect_mix]
  exact g.las_collapse_unbiased (fun c => (channel c).expect f)

noncomputable def absoluteGapNat (g : LASGroup Exact Strict)
    (channel : Exact ⊕ Strict → FiniteLaw Output) : ℝ :=
  (g.prior.mix channel).entropyNat -
    g.surrogate.expect (fun e => ((g.collapse e).mix channel).entropyNat)

theorem absoluteGapNat_eq_information (g : LASGroup Exact Strict)
    (channel : Exact ⊕ Strict → FiniteLaw Output) :
    g.absoluteGapNat channel =
      g.surrogate.informationNat (fun e => (g.collapse e).mix channel) := by
  rw [FiniteLaw.informationNat, g.collapse_channel_mean]
  rfl

theorem absoluteGapNat_nonneg (g : LASGroup Exact Strict)
    (channel : Exact ⊕ Strict → FiniteLaw Output) : 0 ≤ g.absoluteGapNat channel := by
  rw [g.absoluteGapNat_eq_information]
  exact FiniteLaw.informationNat_nonneg _ _

/-- The exact mass multiplies the predictive information among exact histories. -/
theorem absoluteGapNat_le_exact_information (g : LASGroup Exact Strict)
    (channel : Exact ⊕ Strict → FiniteLaw Output) :
    g.absoluteGapNat channel ≤ g.exactMass *
      g.surrogate.informationNat (fun e => channel (.inl e)) := by
  let q : Unit ⊕ Strict → Exact → FiniteLaw Output :=
    Sum.elim (fun _ e => channel (.inl e)) (fun r _ => channel (.inr r))
  have h := g.surrogate.informationNat_mix_channel_le g.nextFrontier q
  have hl : (fun e => g.nextFrontier.mix (fun j => q j e)) =
      (fun e => (g.collapse e).mix channel) := by
    funext e
    have hq : (fun j => q j e) =
        Sum.elim (fun _ : Unit => channel (.inl e)) (fun r => channel (.inr r)) := by
      funext j
      cases j <;> rfl
    rw [hq, g.nextFrontier_channel]
  rw [hl, ← g.absoluteGapNat_eq_information] at h
  simpa [q, FiniteLaw.expect, nextFrontier, Fintype.sum_sum_type] using h

theorem absoluteGapNat_le_exact_entropy (g : LASGroup Exact Strict)
    (channel : Exact ⊕ Strict → FiniteLaw Output) :
    g.absoluteGapNat channel ≤ g.exactMass * g.surrogate.entropyNat :=
  (g.absoluteGapNat_le_exact_information channel).trans
    (mul_le_mul_of_nonneg_left (FiniteLaw.informationNat_le_entropy _ _)
      (le_of_lt g.exactMass_pos))

/-- Identical exact futures suffice for zero loss, irrespective of the strict futures. -/
theorem absoluteGapNat_zero_of_exact_constant (g : LASGroup Exact Strict)
    (channel : Exact ⊕ Strict → FiniteLaw Output) (q : FiniteLaw Output)
    (hc : ∀ e, channel (.inl e) = q) : g.absoluteGapNat channel = 0 := by
  have h := g.absoluteGapNat_le_exact_information channel
  simp only [hc, FiniteLaw.informationNat_const, mul_zero] at h
  exact le_antisymm h (g.absoluteGapNat_nonneg channel)

end StegaTextProofs.LASGroup
