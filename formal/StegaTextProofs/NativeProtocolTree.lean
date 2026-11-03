module

public import StegaTextProofs.IndependentProtocolTapes
public import StegaTextProofs.ProtocolPrefixKeys
public import StegaTextProofs.TokenSource
public import StegaTextProofs.FiniteTermination
public import StegaTextProofs.VisibleBoundary

/-! Native SyncPool chooses a prefix group, synchronizes its token, and emits the full token. -/

@[expose] public section

namespace StegaTextProofs.TokenSource

universe u v
attribute [local instance] Classical.propDecidable
open PrefixGrouping
variable {Token : Type u} {Byte : Type v} [Fintype Token]

noncomputable def nativeProtocol (word : Token → List Byte) : TokenSource Token → ProtocolTree.{u} Byte
  | .terminal => .leaf
  | .step p next =>
      .payload (p.map (NativeGrouping.label word p)).supportLaw (fun g => word g.val)
        (fun g => .sync (p.fiberLaw (NativeGrouping.label word p) g.val g.property)
          (fun t => (word t).drop (word g.val).length) (fun t => nativeProtocol word (next t)))

theorem nativeProtocol_decodable (word : Token → List Byte) (source : TokenSource Token) :
    (source.nativeProtocol word).Decodable := by
  induction source with
  | terminal => trivial
  | step p next ih =>
    refine ⟨?_, fun _ t => ih t⟩
    intro g h hp
    exact Subtype.ext (NativeGrouping.positive_labels_prefix_free word p g.val h.val
      g.property h.property hp)

theorem nativeProtocol_budget (word : Token → List Byte) (source : TokenSource Token) :
    (source.nativeProtocol word).budgetNat = source.nativeValueNat word := by
  induction source with
  | terminal => rfl
  | step p next ih =>
    simp only [nativeProtocol, ProtocolTree.budgetNat, ih]
    rw [(p.map (NativeGrouping.label word p)).supportLaw_entropyNat,
      p.disintegrate_fiber_expect (NativeGrouping.label word p) (fun t => (next t).nativeValueNat word)]
    rfl

theorem nativeProtocol_conditional_entropy (word : Token → List Byte) (source : TokenSource Token) :
    (source.nativeProtocol word).visibleConditionalEntropy = source.nativeValue word := by
  rw [ProtocolTree.visibleConditionalEntropy_eq_budget _ (source.nativeProtocol_decodable word),
    source.nativeProtocol_budget word]
  rfl

theorem nativeProtocol_output_expect (word : Token → List Byte) (source : TokenSource Token)
    (f : List Byte → ℝ) :
    (source.nativeProtocol word).outputExpect f =
      source.pathExpect (fun h => f (render word h)) := by
  induction source generalizing f with
  | terminal => simp [nativeProtocol, ProtocolTree.outputExpect_leaf, pathExpect, render]
  | step p next ih =>
    simp only [nativeProtocol, ProtocolTree.outputExpect_payload, ProtocolTree.outputExpect_sync]
    have hchild : ∀ (g : (p.map (NativeGrouping.label word p)).Positive) t,
        ((next t).nativeProtocol word).outputExpect
        (fun w => f (word g.val ++ ((word t).drop (word g.val).length ++ w))) =
        (next t).pathExpect (fun h => f (word g.val ++
          ((word t).drop (word g.val).length ++ render word h))) := by
      intro g t
      exact ih t (fun w => f (word g.val ++ ((word t).drop (word g.val).length ++ w)))
    simp only [hchild]
    have hf : ∀ g : (p.map (NativeGrouping.label word p)).Positive,
        (p.fiberLaw (NativeGrouping.label word p) g.val g.property).expect
          (fun t => (next t).pathExpect (fun h => f (word g.val ++
            ((word t).drop (word g.val).length ++ render word h)))) =
        (p.fiberLaw (NativeGrouping.label word p) g.val g.property).expect
          (fun t => (next t).pathExpect (fun h => f (word t ++ render word h))) := by
      intro g
      unfold FiniteLaw.expect
      apply Finset.sum_congr rfl
      intro t _
      by_cases ht : 0 < (p.fiberLaw (NativeGrouping.label word p) g.val g.property).prob t
      · obtain ⟨tail, hw⟩ := NativeGrouping.conditional_key_prefix word p g t ht
        simp [← hw, List.append_assoc]
      · have hz := le_antisymm (le_of_not_gt ht)
          ((p.fiberLaw (NativeGrouping.label word p) g.val g.property).nonneg t)
        simp only [hz, zero_mul]
    simp only [hf]
    rw [p.disintegrate_fiber_expect (NativeGrouping.label word p)
      (fun t => (next t).pathExpect (fun h => f (word t ++ render word h)))]
    rfl

end StegaTextProofs.TokenSource
