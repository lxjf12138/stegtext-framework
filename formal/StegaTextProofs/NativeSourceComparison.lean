module

public import StegaTextProofs.TokenSource
public import StegaTextProofs.FiniteTermination
public import StegaTextProofs.TerminalCollapse
public import StegaTextProofs.VisibleBoundary

/-!
Combine the actual native source-tree root identity with the LAS selective
entropy inequality and finite termination. The LAS model and its visible
state invariants are explicit inputs. Constructing them for every source,
and identifying the entropy budget with H(Y | R), are separate obligations.
-/

@[expose] public section

namespace StegaTextProofs.TokenSource

variable {Token Byte State : Type*} [Fintype Token] [Fintype State]
variable {Group : State → Type*} [∀ s, Fintype (Group s)]
variable {Exact Strict : (s : State) → Group s → Type*}
  [∀ s k, Fintype (Exact s k)] [∀ s k, Fintype (Strict s k)]

/-- Native source budget versus LAS budget, with the remaining correction explicit. -/
theorem native_le_las_with_correction (word : Token → List Byte) (source : TokenSource Token)
    (model : LASDecisionModel source.decisionTree State Group Exact Strict)
    (initial : State) (hroot : model.belief initial = source.sourceLaw word) (n : ℕ) :
    source.nativeValueNat word ≤
      FiniteHorizon.budget model.kernel model.rewardNat n initial +
        FiniteHorizon.after model.kernel n model.potential initial := by
  have h := model.finite_selective_comparison n initial
  simpa only [LASDecisionModel.potential, hroot, source.sourceLaw_selective word] using h

/-- The finite native/LAS comparison in bits, with a proved rank bound. -/
theorem native_le_las_of_rank (word : Token → List Byte) (source : TokenSource Token)
    (model : LASDecisionModel source.decisionTree State Group Exact Strict)
    (initial : State) (hroot : model.belief initial = source.sourceLaw word)
    (terminal : State → Prop) (rank : State → ℕ)
    (hdecrease : ∀ s t, ¬ terminal s → 0 < (model.kernel s).prob t → rank t < rank s)
    (hclosed : ∀ s t, terminal s → 0 < (model.kernel s).prob t → terminal t)
    (hresolved : ∀ s, terminal s → ∃ l, model.belief s = FiniteLaw.point l) :
    source.nativeValue word ≤
      FiniteHorizon.budget model.kernel (fun s => (model.groups s).entropy)
        (rank initial) initial := by
  have hz : ∀ s, terminal s → model.potential s = 0 := by
    intro s hs
    obtain ⟨l, hl⟩ := hresolved s hs
    exact model.potential_zero_of_resolved s l hl
  have hstop := FiniteHorizon.after_zero_of_rank model.kernel terminal rank model.potential
    hdecrease hclosed hz (rank initial) initial le_rfl
  have h := model.terminating_selective_comparison_bits (rank initial) initial hstop
  simpa only [hroot, source.sourceLaw_selective_bits word] using h

/-- A bound obtained from full source outputs and nonempty visible keys.
No zero terminal expectation or root entropy identity is assumed. -/
theorem native_le_las_of_visible_progress (word : Token → List Byte)
    (source : TokenSource Token)
    (model : LASDecisionModel source.decisionTree State Group Exact Strict)
    (initial : State) (hroot : model.belief initial = source.sourceLaw word)
    (terminal : State → Prop) (emitted : State → List Byte)
    (hprefix : ∀ s l, 0 < (model.belief s).prob l →
      emitted s <+: PrefixGrouping.render word (source.tokenPath l))
    (hprogress : ∀ s t, ¬ terminal s → 0 < (model.kernel s).prob t →
      ∃ key : List Byte, key ≠ [] ∧ emitted t = emitted s ++ key)
    (hclosed : ∀ s t, terminal s → 0 < (model.kernel s).prob t → terminal t)
    (hresolved : ∀ s, terminal s → ∃ l, model.belief s = FiniteLaw.point l) :
    source.nativeValue word ≤
      FiniteHorizon.budget model.kernel (fun s => (model.groups s).entropy)
        (VisibleTermination.maxLength (fun l =>
          PrefixGrouping.render word (source.tokenPath l)) - (emitted initial).length) initial := by
  apply source.native_le_las_of_rank word model initial hroot terminal
    (fun s => VisibleTermination.maxLength (fun l =>
      PrefixGrouping.render word (source.tokenPath l)) - (emitted s).length) _ hclosed hresolved
  intro s t hs ht
  obtain ⟨key, hne, happend⟩ := hprogress s t hs ht
  exact VisibleTermination.rank_decreases model.belief _ emitted hprefix s t
    (VisibleTermination.length_progress_of_append emitted s t key hne happend)

end StegaTextProofs.TokenSource
