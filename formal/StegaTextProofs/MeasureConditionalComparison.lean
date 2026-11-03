module

public import StegaTextProofs.ProtocolMeasureTapes
public import StegaTextProofs.FiniteConditionalComparison

/-! The finite-source comparison with arbitrary measurable independent raw tape spaces. -/

@[expose] public section

namespace StegaTextProofs.TokenSource

open MeasureTheory
universe u v
variable {Token : Type u} {Byte : Type v} [Fintype Token]
variable {RS US RL UL : Type*}
variable [MeasurableSpace RS] [MeasurableSpace US] [MeasurableSpace RL] [MeasurableSpace UL]

/-- Both raw tape spaces may be continuous or infinite; output and source remain finite. -/
theorem finite_las_conditionalEntropy_ge_native_measures
    (word : Token → List Byte) (eos : Byte) (source : TokenSource Token)
    (hs : source.EOSSource word eos) (hne : ∀ t, word t ≠ [])
    (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (μS : Measure RS) (νS : Measure US) (μL : Measure RL) (νL : Measure UL)
    [IsProbabilityMeasure μS] [IsProbabilityMeasure νS]
    [IsProbabilityMeasure μL] [IsProbabilityMeasure νL]
    (readS : RS → (source.nativeProtocol word).Tape)
    (readPS : US → (source.nativeProtocol word).PayloadTape)
    (readL : RL → (source.lasProtocol word eos hs hne hiso).Tape)
    (readPL : UL → (source.lasProtocol word eos hs hne hiso).PayloadTape)
    (hreadS : Measurable readS) (hreadPS : Measurable readPS)
    (hreadL : Measurable readL) (hreadPL : Measurable readPL)
    (hpS : νS.map readPS = (source.nativeProtocol word).payloadLaw.toMeasure)
    (hpL : νL.map readPL = (source.lasProtocol word eos hs hne hiso).payloadLaw.toMeasure)
    (hcS : (FiniteLaw.ofMeasure (μS.map readS)).mix (source.nativeProtocol word).givenLaw =
      (source.nativeProtocol word).pathLaw)
    (hcL : (FiniteLaw.ofMeasure (μL.map readL)).mix (source.lasProtocol word eos hs hne hiso).givenLaw =
      (source.lasProtocol word eos hs hne hiso).pathLaw) :
    (source.nativeProtocol word).measureConditionalEntropy μS νS readS readPS ≤
      (source.lasProtocol word eos hs hne hiso).measureConditionalEntropy μL νL readL readPL := by
  rw [ProtocolTree.measureConditionalEntropy_eq_budget _ μS νS
      (source.nativeProtocol_decodable word) readS hreadS readPS hreadPS hpS hcS,
    (source.lasProtocol word eos hs hne hiso).measureConditionalEntropy_eq_budget μL νL
      (LASState.protocol_decodable hs hne hiso _ _) readL hreadL readPL hreadPL hpL hcL,
    source.nativeProtocol_budget word, lasProtocol, LASState.protocol_budget]
  exact source.native_le_concrete_las word eos hs hne hiso

/-- A simple sufficient sampling contract: both table pushforwards have the constructed laws. -/
theorem finite_las_conditionalEntropy_ge_native_measure_preserving
    (word : Token → List Byte) (eos : Byte) (source : TokenSource Token)
    (hs : source.EOSSource word eos) (hne : ∀ t, word t ≠ [])
    (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (μS : Measure RS) (νS : Measure US) (μL : Measure RL) (νL : Measure UL)
    [IsProbabilityMeasure μS] [IsProbabilityMeasure νS]
    [IsProbabilityMeasure μL] [IsProbabilityMeasure νL]
    (readS : RS → (source.nativeProtocol word).Tape)
    (readPS : US → (source.nativeProtocol word).PayloadTape)
    (readL : RL → (source.lasProtocol word eos hs hne hiso).Tape)
    (readPL : UL → (source.lasProtocol word eos hs hne hiso).PayloadTape)
    (hreadS : Measurable readS) (hreadPS : Measurable readPS)
    (hreadL : Measurable readL) (hreadPL : Measurable readPL)
    (hpS : νS.map readPS = (source.nativeProtocol word).payloadLaw.toMeasure)
    (hpL : νL.map readPL = (source.lasProtocol word eos hs hne hiso).payloadLaw.toMeasure)
    (hcS : μS.map readS = (source.nativeProtocol word).tapeLaw.toMeasure)
    (hcL : μL.map readL = (source.lasProtocol word eos hs hne hiso).tapeLaw.toMeasure) :
    (source.nativeProtocol word).measureConditionalEntropy μS νS readS readPS ≤
      (source.lasProtocol word eos hs hne hiso).measureConditionalEntropy μL νL readL readPL := by
  rw [ProtocolTree.measureConditionalEntropy_eq_observed _ μS νS readS hreadS readPS hreadPS hcS hpS,
    ProtocolTree.measureConditionalEntropy_eq_observed _ μL νL readL hreadL readPL hreadPL hcL hpL]
  exact source.finite_las_conditionalEntropy_ge_native word eos hs hne hiso

theorem native_measure_tapes_output
    (word : Token → List Byte) (source : TokenSource Token)
    (μ : Measure RS) (ν : Measure US) [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    (readR : RS → (source.nativeProtocol word).Tape) (hr : Measurable readR)
    (readU : US → (source.nativeProtocol word).PayloadTape) (hu : Measurable readU)
    (hp : ν.map readU = (source.nativeProtocol word).payloadLaw.toMeasure)
    (hc : (FiniteLaw.ofMeasure (μ.map readR)).mix (source.nativeProtocol word).givenLaw =
      (source.nativeProtocol word).pathLaw) (f : List Byte → ℝ) :
    (∫ z, f ((source.nativeProtocol word).output
      ((source.nativeProtocol word).evaluate (readR z.1) (readU z.2))) ∂μ.prod ν) =
        source.pathExpect (fun h => f (PrefixGrouping.render word h)) := by
  rw [ProtocolTree.measure_tapes_output_expect _ μ ν readR hr readU hu hp hc f]
  exact source.nativeProtocol_output_expect word f

theorem las_measure_tapes_output
    (word : Token → List Byte) (eos : Byte) (source : TokenSource Token)
    (hs : source.EOSSource word eos) (hne : ∀ t, word t ≠ [])
    (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (μ : Measure RL) (ν : Measure UL) [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    (readR : RL → (source.lasProtocol word eos hs hne hiso).Tape) (hr : Measurable readR)
    (readU : UL → (source.lasProtocol word eos hs hne hiso).PayloadTape) (hu : Measurable readU)
    (hp : ν.map readU = (source.lasProtocol word eos hs hne hiso).payloadLaw.toMeasure)
    (hc : (FiniteLaw.ofMeasure (μ.map readR)).mix (source.lasProtocol word eos hs hne hiso).givenLaw =
      (source.lasProtocol word eos hs hne hiso).pathLaw) (f : List Byte → ℝ) :
    (∫ z, f ((source.lasProtocol word eos hs hne hiso).output
      ((source.lasProtocol word eos hs hne hiso).evaluate (readR z.1) (readU z.2))) ∂μ.prod ν) =
        source.pathExpect (fun h => f (PrefixGrouping.render word h)) := by
  rw [ProtocolTree.measure_tapes_output_expect _ μ ν readR hr readU hu hp hc f]
  change (LASState.protocol hs hne hiso (source.maxRenderedLength word)
    (LASState.initial word eos source)).outputExpect f = _
  exact (LASState.protocol_output_expect hs hne hiso f (source.maxRenderedLength word)
    (LASState.initial word eos source)).trans
      (source.concrete_las_visible_law word eos hs hne hiso f)

end StegaTextProofs.TokenSource
