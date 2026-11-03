module

public import StegaTextProofs.FiniteLawMeasure

/-!
Conditional Shannon entropy of finite outputs given an arbitrary sigma-algebra.
For independent raw tapes, conditioning on the full first tape integrates only
the second tape. Neither raw space is required to be finite or countable.
-/

@[expose] public section

namespace StegaTextProofs

open MeasureTheory
open scoped BigOperators
attribute [local instance] Classical.propDecidable

variable {Ω Ξ : Type*} [mΩ : MeasurableSpace Ω] [MeasurableSpace Ξ]
variable (μ : Measure Ω) (ν : Measure Ξ) [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]

theorem condExp_fst_prod (f : Ω × Ξ → ℝ) (hf : StronglyMeasurable f)
    (hi : Integrable f (μ.prod ν)) :
    (μ.prod ν)[f | mΩ.comap Prod.fst] =ᵐ[μ.prod ν]
      fun z => ∫ u, f (z.1, u) ∂ν := by
  refine (ae_eq_condExp_of_forall_setIntegral_eq measurable_fst.comap_le hi
    (fun s _ _ => (hi.integral_prod_left.comp_fst ν).integrableOn) ?_ ?_).symm
  · rintro s ⟨t, ht, rfl⟩ _
    have htprod : Prod.fst ⁻¹' t = t ×ˢ (Set.univ : Set Ξ) := by ext z; simp
    rw [htprod]
    rw [setIntegral_prod _ (hi.integral_prod_left.comp_fst ν).integrableOn,
      setIntegral_prod _ hi.integrableOn]
    simp
  · exact (hf.integral_prod_right'.comp_measurable (comap_measurable Prod.fst)).aestronglyMeasurable

noncomputable def conditionalEntropy {A : Type*} [Fintype A]
    (ρ : Measure Ω) (m : MeasurableSpace Ω) (Y : Ω → A) : ℝ :=
  (∫ ω, ∑ a, Real.negMulLog (ρ[fun x => if Y x = a then (1 : ℝ) else 0 | m] ω) ∂ρ) /
    Real.log 2

variable {R U A : Type*} [Fintype R] [Fintype U] [Fintype A]
variable [MeasurableSpace R] [MeasurableSingletonClass R]
variable [MeasurableSpace U] [MeasurableSingletonClass U]

theorem finite_channel_condProb
    (readR : Ω → R) (hr : Measurable readR) (readU : Ξ → U) (hu : Measurable readU)
    (p : FiniteLaw U) (hp : ν.map readU = p.toMeasure) (f : R → U → A) (a : A) :
    (μ.prod ν)[fun z => if f (readR z.1) (readU z.2) = a then (1 : ℝ) else 0 |
      mΩ.comap Prod.fst] =ᵐ[μ.prod ν]
        fun z => (p.map (f (readR z.1))).prob a := by
  have hm : Measurable (fun z : Ω × Ξ => (readR z.1, readU z.2)) :=
    (hr.comp measurable_fst).prodMk (hu.comp measurable_snd)
  have hs : StronglyMeasurable (fun ru : R × U => if f ru.1 ru.2 = a then (1 : ℝ) else 0) :=
    .of_discrete
  have hi := FiniteLaw.integrable_finite_read (μ.prod ν) _ hm
    (fun ru : R × U => if f ru.1 ru.2 = a then (1 : ℝ) else 0)
  refine (condExp_fst_prod μ ν _ (hs.comp_measurable hm) hi).trans ?_
  apply Filter.Eventually.of_forall
  intro z
  change (∫ u, if f (readR z.1) (readU u) = a then (1 : ℝ) else 0 ∂ν) = _
  rw [← integral_map hu.aemeasurable (f := fun u => if f (readR z.1) u = a then (1 : ℝ) else 0)
    (.of_discrete), hp, FiniteLaw.integral_toMeasure]
  simp [FiniteLaw.expect, FiniteLaw.map_prob, mul_ite]

theorem conditionalEntropy_finite_channel
    (readR : Ω → R) (hr : Measurable readR) (readU : Ξ → U) (hu : Measurable readU)
    (p : FiniteLaw U) (hp : ν.map readU = p.toMeasure) (f : R → U → A) :
    conditionalEntropy (μ.prod ν) (mΩ.comap Prod.fst)
      (fun z => f (readR z.1) (readU z.2)) =
        (FiniteLaw.ofMeasure (μ.map readR)).conditionalEntropyNat (fun r => p.map (f r)) /
          Real.log 2 := by
  rw [conditionalEntropy, FiniteLaw.conditionalEntropyNat_eq_expect]
  congr 1
  have he : (fun z => ∑ a, Real.negMulLog
      ((μ.prod ν)[fun x => if f (readR x.1) (readU x.2) = a then (1 : ℝ) else 0 |
        mΩ.comap Prod.fst] z)) =ᵐ[μ.prod ν]
      fun z => (p.map (f (readR z.1))).entropyNat := by
    filter_upwards [ae_all_iff.mpr (fun a => finite_channel_condProb μ ν readR hr readU hu p hp f a)]
      with z hz
    simp only [hz, FiniteLaw.entropyNat]
  rw [integral_congr_ae he,
    integral_fun_fst (μ := μ) (ν := ν) (fun r => (p.map (f (readR r))).entropyNat)]
  simp only [probReal_univ, one_smul]
  exact FiniteLaw.integral_finite_read μ readR hr (fun r => (p.map (f r)).entropyNat)

end StegaTextProofs
