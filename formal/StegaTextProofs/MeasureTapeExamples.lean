module

public import StegaTextProofs.ProtocolMeasureTapes
public import StegaTextProofs.SyncTapeExamples

/-! Exact continuous-coin and arbitrary unused-tape examples. -/

@[expose] public section

namespace StegaTextProofs.MeasureTapeExamples

open MeasureTheory Set
open SyncTapeExamples
attribute [local instance] Classical.propDecidable

noncomputable def uniform01 : Measure ℝ := volume.restrict (Ioc 0 1)

instance uniform01_probability : IsProbabilityMeasure uniform01 := by
  constructor
  norm_num [uniform01]

noncomputable def halfCoin (x : ℝ) : Bool := if x ≤ 1 / 2 then true else false

theorem halfCoin_measurable : Measurable halfCoin :=
  Measurable.ite measurableSet_Iic measurable_const measurable_const

theorem halfCoin_law : uniform01.map halfCoin = fair.toMeasure := by
  apply Measure.ext_of_singleton
  intro b
  rw [Measure.map_apply halfCoin_measurable (measurableSet_singleton b)]
  change uniform01 (halfCoin ⁻¹' {b}) = fair.toPMF.toMeasure {b}
  rw [PMF.toMeasure_apply_singleton _ _ (measurableSet_singleton b)]
  cases b with
  | false =>
    have he : halfCoin ⁻¹' {false} = Ioi (1 / 2 : ℝ) := by
      ext x
      simp [halfCoin]
    rw [he, uniform01, Measure.restrict_apply measurableSet_Ioi, inter_comm, Ioc_inter_Ioi]
    norm_num [FiniteLaw.toPMF, fair]
  | true =>
    have he : halfCoin ⁻¹' {true} = Iic (1 / 2 : ℝ) := by
      ext x
      simp [halfCoin]
    rw [he, uniform01, Measure.restrict_apply measurableSet_Iic, inter_comm, Ioc_inter_Iic]
    norm_num [FiniteLaw.toPMF, fair]

theorem fair_xor (r : Bool) : fair.map (fun b => Bool.xor r b) = fair := by
  apply FiniteLaw.ext_prob
  intro b
  cases r <;> cases b <;> norm_num [FiniteLaw.map_prob, fair]

/-- Two independent continuous uniforms, with the entire first real number conditioned upon. -/
theorem continuous_xor_conditionalEntropy :
    conditionalEntropy (uniform01.prod uniform01)
      ((inferInstance : MeasurableSpace ℝ).comap Prod.fst)
      (fun z => Bool.xor (halfCoin z.1) (halfCoin z.2)) = 1 := by
  rw [conditionalEntropy_finite_channel uniform01 uniform01 halfCoin halfCoin_measurable
    halfCoin halfCoin_measurable fair halfCoin_law Bool.xor]
  simp_rw [fair_xor]
  rw [FiniteLaw.conditionalEntropyNat_eq_expect]
  simp only [fair_entropyNat, FiniteLaw.expect_const]
  exact div_self (ne_of_gt (Real.log_pos (by norm_num)))

/-- Arbitrary extra independent coordinates are retained in the conditioned full raw tape. -/
theorem extra_raw_coordinates_entropy {Byte Ω Ξ : Type*} [MeasurableSpace Ω] [MeasurableSpace Ξ]
    (tree : ProtocolTree Byte) (μ : Measure Ω) (ν : Measure Ξ)
    [IsProbabilityMeasure μ] [IsProbabilityMeasure ν] :
    tree.measureConditionalEntropy (tree.tapeLaw.toMeasure.prod μ)
      (tree.payloadLaw.toMeasure.prod ν) Prod.fst Prod.fst = tree.observedConditionalEntropy := by
  exact tree.measureConditionalEntropy_eq_observed _ _ Prod.fst measurable_fst
    Prod.fst measurable_fst (by simp) (by simp)

end StegaTextProofs.MeasureTapeExamples
