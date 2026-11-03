module

public import StegaTextProofs.MeasureConditionalEntropy
public import StegaTextProofs.ProtocolTapeRealization

/-! Execute finite protocols on arbitrary independent measurable raw tape spaces. -/

@[expose] public section

namespace StegaTextProofs.ProtocolTree

open MeasureTheory
universe u v
attribute [local instance] Classical.propDecidable
variable {Byte : Type v}

instance tapeMeasurableSpace (tree : ProtocolTree.{u} Byte) : MeasurableSpace tree.Tape := ⊤
instance tapeMeasurableSingletonClass (tree : ProtocolTree.{u} Byte) :
    MeasurableSingletonClass tree.Tape := ⟨fun _ => trivial⟩

instance payloadTapeMeasurableSpace (tree : ProtocolTree.{u} Byte) :
    MeasurableSpace tree.PayloadTape := ⊤
instance payloadTapeMeasurableSingletonClass (tree : ProtocolTree.{u} Byte) :
    MeasurableSingletonClass tree.PayloadTape := ⟨fun _ => trivial⟩

theorem payload_readout_law (tree : ProtocolTree.{u} Byte) (r : tree.Tape) :
    tree.payloadLaw.map (fun z => tree.readout (tree.evaluate r z)) = tree.visibleGivenLaw r := by
  classical
  have h : ∀ f : tree.Visible → ℝ,
      (tree.payloadLaw.map (fun z => tree.readout (tree.evaluate r z))).expect f =
        (tree.visibleGivenLaw r).expect f := by
    intro f
    rw [FiniteLaw.map_expect, tree.evaluate_expect r (fun l => f (tree.readout l))]
    exact ((tree.givenLaw r).map_expect tree.readout f).symm
  apply FiniteLaw.ext_prob
  intro y
  simpa [FiniteLaw.expect] using h (fun z => if z = y then (1 : ℝ) else 0)

theorem averaged_givenLaw (tree : ProtocolTree.{u} Byte) :
    tree.tapeLaw.mix tree.givenLaw = tree.pathLaw := by
  apply FiniteLaw.ext_prob
  intro l
  have h := tree.averaged_given_expect (fun k => if k = l then (1 : ℝ) else 0)
  simpa [FiniteLaw.mix, FiniteLaw.expect, mul_ite] using h

variable {Ω Ξ : Type*} [mΩ : MeasurableSpace Ω] [MeasurableSpace Ξ]

noncomputable def measureConditionalEntropy (tree : ProtocolTree.{u} Byte)
    (μ : Measure Ω) (ν : Measure Ξ) (readR : Ω → tree.Tape) (readU : Ξ → tree.PayloadTape) : ℝ :=
  conditionalEntropy (μ.prod ν) (mΩ.comap Prod.fst)
    (fun z => tree.readout (tree.evaluate (readR z.1) (readU z.2)))

variable (tree : ProtocolTree.{u} Byte) (μ : Measure Ω) (ν : Measure Ξ)
variable [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]

theorem measureConditionalEntropy_eq_finite
    (readR : Ω → tree.Tape) (hr : Measurable readR)
    (readU : Ξ → tree.PayloadTape) (hu : Measurable readU)
    (hp : ν.map readU = tree.payloadLaw.toMeasure) :
    tree.measureConditionalEntropy μ ν readR readU =
      tree.rawConditionalEntropy (FiniteLaw.ofMeasure (μ.map readR)) id := by
  rw [measureConditionalEntropy, conditionalEntropy_finite_channel μ ν readR hr readU hu
    tree.payloadLaw hp (fun r z => tree.readout (tree.evaluate r z))]
  simp_rw [tree.payload_readout_law]
  rw [rawConditionalEntropy, tree.rawObservedLaw_eq_joint]
  rfl

theorem measureConditionalEntropy_eq_integral
    (readR : Ω → tree.Tape) (hr : Measurable readR)
    (readU : Ξ → tree.PayloadTape) (hu : Measurable readU)
    (hp : ν.map readU = tree.payloadLaw.toMeasure) :
    tree.measureConditionalEntropy μ ν readR readU =
      (∫ r, (tree.visibleGivenLaw (readR r)).entropyNat ∂μ) / Real.log 2 := by
  rw [tree.measureConditionalEntropy_eq_finite μ ν readR hr readU hu hp,
    rawConditionalEntropy, tree.rawObservedLaw_eq_joint]
  change (FiniteLaw.ofMeasure (μ.map readR)).conditionalEntropyNat tree.visibleGivenLaw /
    Real.log 2 = _
  rw [FiniteLaw.conditionalEntropyNat_eq_expect, FiniteLaw.ofMeasure_expect,
    integral_map hr.aemeasurable (f := fun r => (tree.visibleGivenLaw r).entropyNat) (.of_discrete)]

/-- Full raw-tape conditioning equals the finite payload budget once the trace law is correct. -/
theorem measureConditionalEntropy_eq_budget (hd : tree.Decodable)
    (readR : Ω → tree.Tape) (hr : Measurable readR)
    (readU : Ξ → tree.PayloadTape) (hu : Measurable readU)
    (hp : ν.map readU = tree.payloadLaw.toMeasure)
    (hc : (FiniteLaw.ofMeasure (μ.map readR)).mix tree.givenLaw = tree.pathLaw) :
    tree.measureConditionalEntropy μ ν readR readU = tree.budgetNat / Real.log 2 := by
  rw [tree.measureConditionalEntropy_eq_finite μ ν readR hr readU hu hp]
  exact tree.rawConditionalEntropy_eq_budget hd _ id hc

theorem measureConditionalEntropy_eq_observed
    (readR : Ω → tree.Tape) (hr : Measurable readR)
    (readU : Ξ → tree.PayloadTape) (hu : Measurable readU)
    (hs : μ.map readR = tree.tapeLaw.toMeasure)
    (hp : ν.map readU = tree.payloadLaw.toMeasure) :
    tree.measureConditionalEntropy μ ν readR readU = tree.observedConditionalEntropy := by
  rw [tree.measureConditionalEntropy_eq_finite μ ν readR hr readU hu hp]
  simp only [hs, FiniteLaw.ofMeasure_toMeasure]
  rw [rawConditionalEntropy, tree.rawObservedLaw_eq_joint,
    observedConditionalEntropy, tree.observedLaw_eq_joint]
  rfl

theorem measureConditionalEntropy_finite_raw
    {Raw : Type*} [Fintype Raw] [MeasurableSpace Raw] [MeasurableSingletonClass Raw]
    (p : FiniteLaw Raw) (read : Raw → tree.Tape) :
    tree.measureConditionalEntropy p.toMeasure tree.payloadLaw.toMeasure read id =
      tree.rawConditionalEntropy p read := by
  rw [tree.measureConditionalEntropy_eq_integral p.toMeasure tree.payloadLaw.toMeasure
    read (measurable_of_finite _) id measurable_id (by simp), FiniteLaw.integral_toMeasure,
    rawConditionalEntropy, tree.rawObservedLaw_eq_joint]
  change p.expect (fun r => (tree.visibleGivenLaw (read r)).entropyNat) / Real.log 2 =
    p.conditionalEntropyNat (fun r => tree.visibleGivenLaw (read r)) / Real.log 2
  rw [FiniteLaw.conditionalEntropyNat_eq_expect]

theorem measure_tapes_output_expect
    (readR : Ω → tree.Tape) (hr : Measurable readR)
    (readU : Ξ → tree.PayloadTape) (hu : Measurable readU)
    (hp : ν.map readU = tree.payloadLaw.toMeasure)
    (hc : (FiniteLaw.ofMeasure (μ.map readR)).mix tree.givenLaw = tree.pathLaw)
    (f : List Byte → ℝ) :
    (∫ z, f (tree.output (tree.evaluate (readR z.1) (readU z.2))) ∂μ.prod ν) =
      tree.outputExpect f := by
  have hm : Measurable (fun z : Ω × Ξ => (readR z.1, readU z.2)) :=
    (hr.comp measurable_fst).prodMk (hu.comp measurable_snd)
  have hi := FiniteLaw.integrable_finite_read (μ.prod ν) _ hm
    (fun rz : tree.Tape × tree.PayloadTape => f (tree.output (tree.evaluate rz.1 rz.2)))
  rw [integral_prod _ hi]
  have hinner : ∀ r, (∫ u, f (tree.output (tree.evaluate (readR r) (readU u))) ∂ν) =
      (tree.givenLaw (readR r)).expect (fun l => f (tree.output l)) := by
    intro r
    rw [← integral_map hu.aemeasurable
      (f := fun u => f (tree.output (tree.evaluate (readR r) u))) (.of_discrete), hp,
      FiniteLaw.integral_toMeasure, tree.evaluate_expect (readR r) (fun l => f (tree.output l))]
  simp_rw [hinner]
  rw [FiniteLaw.integral_finite_read μ readR hr
    (fun r => (tree.givenLaw r).expect (fun l => f (tree.output l))),
    ← FiniteLaw.expect_mix, hc]
  rfl

end StegaTextProofs.ProtocolTree
