module

public import StegaTextProofs.AdaptiveLASBudget
public import StegaTextProofs.LASProtocolTree
public import StegaTextProofs.FiniteConditionalComparison

/-! Replayable adaptive protocol trees, including the extra FullSync suffix. -/

@[expose] public section

namespace StegaTextProofs.TokenSource

universe u v
attribute [local instance] Classical.propDecidable
open PrefixGrouping
variable {Token : Type u} {Byte : Type v} [Fintype Token]

namespace LASState

variable {word : Token → List Byte} {eos : Byte} {source : TokenSource Token}
variable (hs : source.EOSSource word eos) (hne : ∀ t, word t ≠ [])
  (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
  (gate : AdaptiveGate word eos source)

noncomputable def adaptiveProtocol : ℕ → LASState word eos source → ProtocolTree.{u} Byte
  | 0, _ => .leaf
  | n + 1, s => if ha : source.cursorSource s.active = terminal then .leaf else
      .payload s.pool.groups (fun g => s.pool.residual g.val) (fun g =>
        if gate s g then
          .sync (s.pool.selected g).surrogate (fun _ => [])
            (fun e => adaptiveProtocol n (s.next ha hs hne hiso g e))
        else
          .sync (s.pool.selected g).prior
            (fun c => (s.pool.residual (s.pool.candidate g c)).drop
              (s.pool.residual g.val).length)
            (fun c => adaptiveProtocol n (s.fullSync g c)))

theorem adaptiveProtocol_decodable (n : ℕ) (s : LASState word eos source) :
    (adaptiveProtocol hs hne hiso gate n s).Decodable := by
  induction n generalizing s with
  | zero => trivial
  | succ n ih =>
    by_cases ht : source.cursorSource s.active = terminal
    · simp only [adaptiveProtocol, dite_eq_left ht, ProtocolTree.Decodable]
    · simp only [adaptiveProtocol, dite_eq_right ht, ProtocolTree.Decodable]
      refine ⟨s.pool.keys_prefix_free, ?_⟩
      intro g
      split
      · exact fun e => ih (s.next ht hs hne hiso g e)
      · exact fun c => ih (s.fullSync g c)

theorem adaptiveProtocol_budget (n : ℕ) (s : LASState word eos source) :
    (adaptiveProtocol hs hne hiso gate n s).budgetNat =
      adaptiveBudgetNat hs hne hiso gate n s := by
  induction n generalizing s with
  | zero => rfl
  | succ n ih =>
    by_cases ht : source.cursorSource s.active = terminal
    · simp only [adaptiveProtocol, adaptiveBudgetNat, adaptiveSum, dite_eq_left ht,
        ProtocolTree.budgetNat]
    · simp only [adaptiveProtocol, adaptiveBudgetNat, adaptiveSum, dite_eq_right ht,
        ProtocolTree.budgetNat, adaptiveStep]
      congr 1
      congr 1
      funext g
      unfold groupExpect
      split <;> simp only [ProtocolTree.budgetNat, ih, adaptiveBudgetNat]

theorem adaptiveProtocol_output_expect (f : List Byte → ℝ) (n : ℕ)
    (s : LASState word eos source) :
    (adaptiveProtocol hs hne hiso gate n s).outputExpect (fun w => f (s.emitted ++ w)) =
      adaptiveRun hs hne hiso gate (fun t => f t.emitted) n s := by
  induction n generalizing s with
  | zero => simp only [adaptiveProtocol, ProtocolTree.outputExpect_leaf, List.append_nil, adaptiveRun]
  | succ n ih =>
    by_cases ht : source.cursorSource s.active = terminal
    · simp only [adaptiveProtocol, dite_eq_left ht, ProtocolTree.outputExpect_leaf,
        List.append_nil, adaptiveRun]
    · simp only [adaptiveProtocol, dite_eq_right ht, ProtocolTree.outputExpect_payload,
        adaptiveRun, adaptiveStep]
      congr 1
      funext g
      unfold groupExpect
      split
      · simp only [ProtocolTree.outputExpect_sync, List.nil_append]
        congr 1
        funext e
        simpa only [next, List.append_assoc] using ih (s.next ht hs hne hiso g e)
      · simp only [ProtocolTree.outputExpect_sync]
        congr 1
        funext c
        have h := ih (s.fullSync g c)
        rw [s.fullSync_emitted ht, s.fullSync_residual_split g c] at h
        simpa only [List.append_assoc] using h

theorem adaptiveProtocol_conditional_entropy (n : ℕ) (s : LASState word eos source) :
    (adaptiveProtocol hs hne hiso gate n s).visibleConditionalEntropy =
      adaptiveBudgetNat hs hne hiso gate n s / Real.log 2 := by
  rw [ProtocolTree.visibleConditionalEntropy_eq_budget _
    (adaptiveProtocol_decodable hs hne hiso gate n s),
    adaptiveProtocol_budget hs hne hiso gate n s]

/-- Visible collisions do not prevent transcript recovery with the full sync tape. -/
theorem adaptiveProtocol_output_injective (n : ℕ) (s : LASState word eos source)
    (r : (adaptiveProtocol hs hne hiso gate n s).Tape)
    (a b : (adaptiveProtocol hs hne hiso gate n s).Leaves)
    (ha : 0 < ((adaptiveProtocol hs hne hiso gate n s).givenLaw r).prob a)
    (hb : 0 < ((adaptiveProtocol hs hne hiso gate n s).givenLaw r).prob b)
    (he : (adaptiveProtocol hs hne hiso gate n s).output a =
      (adaptiveProtocol hs hne hiso gate n s).output b) : a = b :=
  ProtocolTree.output_injective_given _ (adaptiveProtocol_decodable hs hne hiso gate n s)
    r a b ha hb he

end LASState

noncomputable def adaptiveProtocol (word : Token → List Byte) (eos : Byte)
    (source : TokenSource Token) (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (gate : LASState.AdaptiveGate word eos source) : ProtocolTree.{u} Byte :=
  LASState.adaptiveProtocol hs hne hiso gate (source.maxRenderedLength word)
    (LASState.initial word eos source)

theorem adaptiveProtocol_source_law (word : Token → List Byte) (eos : Byte)
    (source : TokenSource Token) (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (gate : LASState.AdaptiveGate word eos source) (f : List Byte → ℝ) :
    (source.adaptiveProtocol word eos hs hne hiso gate).outputExpect f =
      source.pathExpect (fun h => f (render word h)) := by
  exact (LASState.adaptiveProtocol_output_expect hs hne hiso gate f _
    (LASState.initial word eos source)).trans
      (source.concrete_adaptive_visible_law word eos hs hne hiso gate f)

theorem adaptiveProtocol_entropy_eq_value (word : Token → List Byte) (eos : Byte)
    (source : TokenSource Token) (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (gate : LASState.AdaptiveGate word eos source) :
    (source.adaptiveProtocol word eos hs hne hiso gate).visibleConditionalEntropy =
      source.adaptiveValue word eos hs hne hiso gate :=
  LASState.adaptiveProtocol_conditional_entropy hs hne hiso gate _ _

theorem adaptiveProtocol_observedEntropy (word : Token → List Byte) (eos : Byte)
    (source : TokenSource Token) (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (gate : LASState.AdaptiveGate word eos source) :
    (source.adaptiveProtocol word eos hs hne hiso gate).observedConditionalEntropy =
      source.adaptiveValue word eos hs hne hiso gate := by
  rw [adaptiveProtocol, ProtocolTree.observedConditionalEntropy_eq_budget _
    (LASState.adaptiveProtocol_decodable hs hne hiso gate _ _),
    LASState.adaptiveProtocol_budget]
  rfl

theorem adaptive_independent_tapes_output (word : Token → List Byte) (eos : Byte)
    (source : TokenSource Token) (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (gate : LASState.AdaptiveGate word eos source) (f : List Byte → ℝ) :
    let tree := source.adaptiveProtocol word eos hs hne hiso gate
    tree.independentTapes.expect (fun rz => f (tree.output (tree.evaluate rz.1 rz.2))) =
      source.pathExpect (fun h => f (render word h)) := by
  dsimp only
  rw [(source.adaptiveProtocol word eos hs hne hiso gate).independentTapes_expect
    (fun _ l => f ((source.adaptiveProtocol word eos hs hne hiso gate).output l)),
    (source.adaptiveProtocol word eos hs hne hiso gate).averaged_given_expect
      (fun l => f ((source.adaptiveProtocol word eos hs hne hiso gate).output l))]
  exact source.adaptiveProtocol_source_law word eos hs hne hiso gate f

/-- A conditional-entropy comparison with the actual adaptive cumulative loss. -/
theorem finite_adaptive_conditionalEntropy_loss_bound (word : Token → List Byte) (eos : Byte)
    (source : TokenSource Token) (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (gate : LASState.AdaptiveGate word eos source) :
    (source.nativeProtocol word).observedConditionalEntropy ≤
      (source.adaptiveProtocol word eos hs hne hiso gate).observedConditionalEntropy +
        LASState.adaptiveLossNat hs hne hiso gate (source.maxRenderedLength word)
          (LASState.initial word eos source) / Real.log 2 := by
  rw [source.nativeProtocol_observedEntropy word,
    source.adaptiveProtocol_observedEntropy word eos hs hne hiso gate]
  exact source.native_le_adaptive_add_loss word eos hs hne hiso gate

end StegaTextProofs.TokenSource
