module

public import StegaTextProofs.AdaptiveLASThreshold
public import StegaTextProofs.FiniteConditionalComparison

/-! The public parameterized-LAS interface.

The older `AdaptiveLAS*` modules already prove the results for an arbitrary
state/group gate.  This file gives that family the single LAS name used by the
paper, while retaining the older declarations as compatibility definitions.
`thresholdGate 0` is the always-retain (fixed LAS) setting.
-/

@[expose] public section

namespace StegaTextProofs.TokenSource

attribute [local instance] Classical.propDecidable
open PrefixGrouping
universe u v
variable {Token : Type u} {Byte : Type v} [Fintype Token]

namespace LASState

variable {word : Token → List Byte} {eos : Byte} {source : TokenSource Token}

/-- A parameterized LAS policy is a common gate over the selected group. -/
abbrev LASGate (word : Token → List Byte) (eos : Byte)
    (source : TokenSource Token) := AdaptiveGate word eos source

/-- The threshold gate exposed as the LAS parameter. -/
noncomputable def thresholdLASGate (tau : ℝ) : LASGate word eos source :=
  thresholdGate tau

/-- The zero-threshold LAS gate written without a threshold test. -/
noncomputable def alwaysLASGate : LASGate word eos source :=
  fun _ _ => true

/-- Zero threshold selects the retaining transition at every state/group. -/
theorem thresholdLASGate_zero (s : LASState word eos source) (g : s.pool.Group) :
    thresholdLASGate (word := word) (eos := eos) (source := source) 0 s g = true := by
  exact thresholdGate_zero s g

theorem thresholdLASGate_zero_eq_always :
    thresholdLASGate (word := word) (eos := eos) (source := source) 0 =
      alwaysLASGate (word := word) (eos := eos) (source := source) := by
  funext s g
  exact thresholdLASGate_zero s g

variable (hs : source.EOSSource word eos) (hne : ∀ t, word t ≠ [])
  (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))

theorem adaptiveBudgetNat_alwaysLAS_eq_budgetNat (n : ℕ) (s : LASState word eos source) :
    adaptiveBudgetNat hs hne hiso
        (alwaysLASGate (word := word) (eos := eos) (source := source)) n s =
      budgetNat hs hne hiso n s := by
  induction n generalizing s with
  | zero => rfl
  | succ n ih =>
      by_cases ht : source.cursorSource s.active = terminal
      · simp [adaptiveBudgetNat, adaptiveSum, budgetNat, ht]
      · simp only [adaptiveBudgetNat, adaptiveSum, budgetNat, dite_eq_right ht]
        congr 1
        unfold adaptiveStep groupExpect alwaysLASGate
        simp only [↓reduceIte]
        congr 1
        funext g
        congr 1
        funext e
        exact ih (s.next ht hs hne hiso g e)

end LASState

/-- The ideal budget of parameterized LAS under an arbitrary common gate. -/
noncomputable def parameterizedLASValue (word : Token → List Byte) (eos : Byte)
    (source : TokenSource Token) (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (gate : LASState.LASGate word eos source) : ℝ :=
  adaptiveValue word eos source hs hne hiso gate

/-- The replayable protocol for parameterized LAS. -/
noncomputable def parameterizedLASProtocol (word : Token → List Byte) (eos : Byte)
    (source : TokenSource Token) (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (gate : LASState.LASGate word eos source) : ProtocolTree.{u} Byte :=
  adaptiveProtocol word eos source hs hne hiso gate

theorem parameterizedLASValue_zero_eq_lasValue (word : Token → List Byte) (eos : Byte)
    (source : TokenSource Token) (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t)) :
    parameterizedLASValue word eos source hs hne hiso
        (LASState.alwaysLASGate (word := word) (eos := eos) (source := source)) =
      source.lasValue word eos hs hne hiso := by
  unfold parameterizedLASValue adaptiveValue lasValue
  rw [LASState.adaptiveBudgetNat_alwaysLAS_eq_budgetNat hs hne hiso]

theorem parameterizedLAS_source_law (word : Token → List Byte) (eos : Byte)
    (source : TokenSource Token) (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (gate : LASState.LASGate word eos source) (f : List Byte → ℝ) :
    (parameterizedLASProtocol word eos source hs hne hiso gate).outputExpect f =
      source.pathExpect (fun h => f (render word h)) := by
  exact adaptiveProtocol_source_law word eos source hs hne hiso gate f

theorem parameterizedLAS_value_eq_budget (word : Token → List Byte) (eos : Byte)
    (source : TokenSource Token) (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (gate : LASState.LASGate word eos source) :
    (parameterizedLASProtocol word eos source hs hne hiso gate).observedConditionalEntropy =
      parameterizedLASValue word eos source hs hne hiso gate := by
  exact adaptiveProtocol_observedEntropy word eos source hs hne hiso gate

theorem native_le_parameterizedLAS_add_uncertainty (word : Token → List Byte)
    (eos : Byte) (source : TokenSource Token) (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (gate : LASState.LASGate word eos source) :
    source.nativeValue word ≤
      parameterizedLASValue word eos source hs hne hiso gate +
        LASState.adaptiveUncertaintyNat hs hne hiso gate
          (source.maxRenderedLength word) (LASState.initial word eos source) /
          Real.log 2 := by
  exact native_le_adaptive_add_uncertainty word eos source hs hne hiso gate

end StegaTextProofs.TokenSource

