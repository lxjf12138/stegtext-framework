module

public import StegaTextProofs.LASProtocolTree
public import StegaTextProofs.NativeProtocolTree
public import StegaTextProofs.ProtocolTapeRealization

/-!
The finite-source conditional-entropy comparison, with both complete independent
tapes and final visible outputs explicitly constructed. The tape contains all
synchronization choices on unvisited branches, and no payload choices.
-/

@[expose] public section

namespace StegaTextProofs.TokenSource

universe u v
attribute [local instance] Classical.propDecidable
open PrefixGrouping
variable {Token : Type u} {Byte : Type v} [Fintype Token]

noncomputable def lasProtocol (word : Token → List Byte) (eos : Byte) (source : TokenSource Token)
    (hs : source.EOSSource word eos) (hne : ∀ t, word t ≠ [])
    (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t)) : ProtocolTree.{u} Byte :=
  LASState.protocol hs hne hiso (source.maxRenderedLength word) (LASState.initial word eos source)

theorem lasProtocol_conditionalEntropy (word : Token → List Byte) (eos : Byte) (source : TokenSource Token)
    (hs : source.EOSSource word eos) (hne : ∀ t, word t ≠ [])
    (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t)) :
    (source.lasProtocol word eos hs hne hiso).observedConditionalEntropy =
      source.lasValue word eos hs hne hiso := by
  rw [lasProtocol, ProtocolTree.observedConditionalEntropy_eq_budget _
    (LASState.protocol_decodable hs hne hiso _ _)]
  rw [LASState.protocol_budget]
  rfl

theorem nativeProtocol_observedEntropy (word : Token → List Byte) (source : TokenSource Token) :
    (source.nativeProtocol word).observedConditionalEntropy = source.nativeValue word := by
  rw [ProtocolTree.observedConditionalEntropy_eq_budget _ (source.nativeProtocol_decodable word),
    source.nativeProtocol_budget word]
  rfl

/-- H(Y_S | R_S) ≤ H(Y_L | R_L), for the explicitly constructed independent complete tapes. -/
theorem finite_las_conditionalEntropy_ge_native (word : Token → List Byte) (eos : Byte)
    (source : TokenSource Token) (hs : source.EOSSource word eos) (hne : ∀ t, word t ≠ [])
    (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t)) :
    (source.nativeProtocol word).observedConditionalEntropy ≤
      (source.lasProtocol word eos hs hne hiso).observedConditionalEntropy := by
  rw [source.nativeProtocol_observedEntropy word, source.lasProtocol_conditionalEntropy word eos hs hne hiso]
  exact source.native_le_concrete_las word eos hs hne hiso

theorem native_independent_tapes_output (word : Token → List Byte) (source : TokenSource Token)
    (f : List Byte → ℝ) :
    (source.nativeProtocol word).independentTapes.expect (fun rz =>
      f ((source.nativeProtocol word).output ((source.nativeProtocol word).evaluate rz.1 rz.2))) =
        source.pathExpect (fun h => f (render word h)) := by
  rw [(source.nativeProtocol word).independentTapes_expect
    (fun _ l => f ((source.nativeProtocol word).output l)),
    (source.nativeProtocol word).averaged_given_expect (fun l => f ((source.nativeProtocol word).output l))]
  exact source.nativeProtocol_output_expect word f

theorem las_independent_tapes_output (word : Token → List Byte) (eos : Byte) (source : TokenSource Token)
    (hs : source.EOSSource word eos) (hne : ∀ t, word t ≠ [])
    (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t)) (f : List Byte → ℝ) :
    let tree := source.lasProtocol word eos hs hne hiso
    tree.independentTapes.expect (fun rz => f (tree.output (tree.evaluate rz.1 rz.2))) =
      source.pathExpect (fun h => f (render word h)) := by
  dsimp only
  rw [(source.lasProtocol word eos hs hne hiso).independentTapes_expect
    (fun _ l => f ((source.lasProtocol word eos hs hne hiso).output l)),
    (source.lasProtocol word eos hs hne hiso).averaged_given_expect
      (fun l => f ((source.lasProtocol word eos hs hne hiso).output l))]
  change (LASState.protocol hs hne hiso (source.maxRenderedLength word)
    (LASState.initial word eos source)).outputExpect f = _
  have h := LASState.protocol_output_expect hs hne hiso f (source.maxRenderedLength word)
    (LASState.initial word eos source)
  exact h.trans (source.concrete_las_visible_law word eos hs hne hiso f)

/-- The same comparison for any finite raw tapes that generate the correct executed trace laws.
Their independence from payload is built into `rawObservedLaw`. -/
theorem finite_las_conditionalEntropy_ge_native_of_tapes
    {RawS RawL : Type*} [Fintype RawS] [Fintype RawL]
    (word : Token → List Byte) (eos : Byte) (source : TokenSource Token)
    (hs : source.EOSSource word eos) (hne : ∀ t, word t ≠ [])
    (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (pS : FiniteLaw RawS) (pL : FiniteLaw RawL)
    (readS : RawS → (source.nativeProtocol word).Tape)
    (readL : RawL → (source.lasProtocol word eos hs hne hiso).Tape)
    (hS : pS.mix (fun r => (source.nativeProtocol word).givenLaw (readS r)) =
      (source.nativeProtocol word).pathLaw)
    (hL : pL.mix (fun r => (source.lasProtocol word eos hs hne hiso).givenLaw (readL r)) =
      (source.lasProtocol word eos hs hne hiso).pathLaw) :
    (source.nativeProtocol word).rawConditionalEntropy pS readS ≤
      (source.lasProtocol word eos hs hne hiso).rawConditionalEntropy pL readL := by
  rw [ProtocolTree.rawConditionalEntropy_eq_budget _ (source.nativeProtocol_decodable word) pS readS hS,
    (source.lasProtocol word eos hs hne hiso).rawConditionalEntropy_eq_budget
      (LASState.protocol_decodable hs hne hiso _ _) pL readL hL,
    source.nativeProtocol_budget word, lasProtocol, LASState.protocol_budget]
  exact source.native_le_concrete_las word eos hs hne hiso

end StegaTextProofs.TokenSource
