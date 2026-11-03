module

public import StegaTextProofs.NativeTreeSeparation
public import StegaTextProofs.NativeSourceComparison
public import StegaTextProofs.SourceTerminalPaths
public import StegaTextProofs.LASReachability

/-!
A source-aware LAS model specified by ordinary candidate/history/path
invariants. The native-tree separation certificate is constructed, rather
than included as a field. Probabilistic disintegration and posterior updates
remain explicit requirements on the represented transition system.
-/

@[expose] public section

namespace StegaTextProofs

open PrefixGrouping

variable {Token Byte : Type*} [Fintype Token]

structure NativeFrontierModel (word : Token → List Byte) (source : TokenSource Token)
    (State : Type*) [Fintype State]
    (Group : State → Type*) [∀ s, Fintype (Group s)]
    (Exact Strict : (s : State) → Group s → Type*)
    [∀ s k, Fintype (Exact s k)] [∀ s k, Fintype (Strict s k)] where
  belief : State → FiniteLaw source.decisionTree.Leaves
  groups : (s : State) → FiniteLaw (Group s)
  selected : (s : State) → (k : Group s) → LASGroup (Exact s k) (Strict s k)
  continuation : (s : State) → (k : Group s) →
    Exact s k ⊕ Strict s k → FiniteLaw source.decisionTree.Leaves
  next : (s : State) → (k : Group s) → Exact s k → State
  history : (s : State) → (k : Group s) → Exact s k ⊕ Strict s k → List Token
  disintegrate : ∀ s, (groups s).mix (fun k =>
    (selected s k).prior.mix (continuation s k)) = belief s
  posterior : ∀ s k e, belief (next s k e) =
    ((selected s k).collapse e).mix (continuation s k)
  antichain : ∀ s k, HistoryAntichain (history s k)
  boundary : ∀ s k e c, render word (history s k (.inl e)) <+: render word (history s k c)
  source_support : ∀ s k c l, (continuation s k c).prob l ≠ 0 →
    0 < (source.sourceLaw word).prob l
  future_prefix : ∀ s k c l, (continuation s k c).prob l ≠ 0 →
    history s k c <+: source.tokenPath l

namespace NativeFrontierModel

variable {word : Token → List Byte} {source : TokenSource Token}
variable {State : Type*} [Fintype State]
variable {Group : State → Type*} [∀ s, Fintype (Group s)]
variable {Exact Strict : (s : State) → Group s → Type*}
  [∀ s k, Fintype (Exact s k)] [∀ s k, Fintype (Strict s k)]

def toDecisionModel (frontier : NativeFrontierModel word source State Group Exact Strict) :
    LASDecisionModel source.decisionTree State Group Exact Strict where
  belief := frontier.belief
  groups := frontier.groups
  selected := frontier.selected
  continuation := frontier.continuation
  next := frontier.next
  disintegrate := frontier.disintegrate
  posterior := frontier.posterior
  separated := fun s k => source.native_tree_separated word [] (frontier.history s k)
    (fun c => (frontier.continuation s k c).prob) (frontier.antichain s k)
    (frontier.boundary s k) (frontier.source_support s k)
    (by simpa using frontier.future_prefix s k)

/-- Complete candidate histories imply resolved posteriors in terminal groups. -/
theorem terminal_next_resolved (frontier : NativeFrontierModel word source State Group Exact Strict)
    (s : State) (k : Group s) (e : Exact s k) (l : source.decisionTree.Leaves)
    (hl : 0 < (source.sourceLaw word).prob l)
    (hhistory : frontier.history s k (.inl e) = source.tokenPath l)
    (hstrict : ∀ r, (frontier.selected s k).prior.prob (.inr r) = 0) :
    frontier.belief (frontier.next s k e) = FiniteLaw.point l := by
  apply frontier.toDecisionModel.belief_next_of_terminal_group s k e l hstrict
  apply source.continuation_point_of_complete_path word l hl
    (frontier.continuation s k (.inl e)) (frontier.source_support s k (.inl e))
  simpa only [hhistory] using frontier.future_prefix s k (.inl e)

/-- Visible compatibility of the state belief follows from its candidate paths. -/
theorem belief_visible_prefix (frontier : NativeFrontierModel word source State Group Exact Strict)
    (emitted : State → List Byte)
    (hvisible : ∀ s k c, emitted s <+: render word (frontier.history s k c))
    (s : State) (l : source.decisionTree.Leaves) (hl : 0 < (frontier.belief s).prob l) :
    emitted s <+: render word (source.tokenPath l) := by
  rw [← frontier.disintegrate s] at hl
  obtain ⟨k, _, hk⟩ := (FiniteLaw.mix_positive_iff _ _ l).mp hl
  obtain ⟨c, _, hc⟩ := (FiniteLaw.mix_positive_iff _ _ l).mp hk
  exact (hvisible s k c).trans
    ((frontier.future_prefix s k c l (ne_of_gt hc)).flatMap word)

/-- Source/frontier invariants and visible progress imply the ideal budget comparison. -/
theorem native_le_las (frontier : NativeFrontierModel word source State Group Exact Strict)
    (initial : State) (hroot : frontier.belief initial = source.sourceLaw word)
    (terminal : State → Prop) (emitted : State → List Byte)
    (hprefix : ∀ s l, 0 < (frontier.belief s).prob l →
      emitted s <+: render word (source.tokenPath l))
    (hprogress : ∀ s t, ¬ terminal s → 0 < (frontier.toDecisionModel.kernel s).prob t →
      ∃ key : List Byte, key ≠ [] ∧ emitted t = emitted s ++ key)
    (hclosed : ∀ s t, terminal s → 0 < (frontier.toDecisionModel.kernel s).prob t → terminal t)
    (hresolved : ∀ s, terminal s → ∃ l, frontier.belief s = FiniteLaw.point l) :
    source.nativeValue word ≤
      FiniteHorizon.budget frontier.toDecisionModel.kernel (fun s => (frontier.groups s).entropy)
        (VisibleTermination.maxLength (fun l => render word (source.tokenPath l)) -
          (emitted initial).length) initial :=
  source.native_le_las_of_visible_progress word frontier.toDecisionModel initial hroot
    terminal emitted hprefix hprogress hclosed hresolved

/-- A version stated directly in terms of candidate histories and protocol updates. -/
theorem native_le_las_of_steps (frontier : NativeFrontierModel word source State Group Exact Strict)
    (initial : State) (hroot : frontier.belief initial = source.sourceLaw word)
    (terminal : State → Prop) (emitted : State → List Byte)
    (hvisible : ∀ s k c, emitted s <+: render word (frontier.history s k c))
    (hprogress : ∀ s k e, ¬ terminal s → 0 < (frontier.groups s).prob k →
      0 < (frontier.selected s k).surrogate.prob e →
      ∃ key : List Byte, key ≠ [] ∧ emitted (frontier.next s k e) = emitted s ++ key)
    (hclosed : ∀ s k e, terminal s → 0 < (frontier.groups s).prob k →
      0 < (frontier.selected s k).surrogate.prob e → terminal (frontier.next s k e))
    (hresolved : ∀ s, terminal s → ∃ l, frontier.belief s = FiniteLaw.point l) :
    source.nativeValue word ≤
      FiniteHorizon.budget frontier.toDecisionModel.kernel (fun s => (frontier.groups s).entropy)
        (VisibleTermination.maxLength (fun l => render word (source.tokenPath l)) -
          (emitted initial).length) initial := by
  apply frontier.native_le_las initial hroot terminal emitted
    (frontier.belief_visible_prefix emitted hvisible) _ _ hresolved
  · intro s t hs ht
    obtain ⟨k, e, hk, he, rfl⟩ := (frontier.toDecisionModel.kernel_positive_iff s t).mp ht
    exact hprogress s k e hs hk he
  · intro s t hs ht
    obtain ⟨k, e, hk, he, rfl⟩ := (frontier.toDecisionModel.kernel_positive_iff s t).mp ht
    exact hclosed s k e hs hk he

end NativeFrontierModel

end StegaTextProofs
