module

public import StegaTextProofs.CollapsedFrontier
public import StegaTextProofs.NativeTreeSeparation

/-!
The selective-entropy step inequality for constructed prefix-pool groups.
The visible frontier supplies only histories, residuals and source-path
support. Group disintegration, positive exact mass, the selected frontier
boundary and the collapse posterior are constructed and proved here.
-/

@[expose] public section

namespace StegaTextProofs.PrefixPool

attribute [local instance] Classical.propDecidable
open PrefixGrouping

variable {Candidate Token Byte : Type*} [Fintype Candidate] [Fintype Token]

theorem candidate_injective (pool : PrefixPool Candidate Byte) (g : pool.Group) :
    Function.Injective (pool.candidate g) := by
  intro c d h
  exact (pool.splitEquiv g).injective (Subtype.ext h)

omit [Fintype Token] in
theorem selected_antichain (pool : PrefixPool Candidate Byte) (g : pool.Group)
    (history : Candidate → List Token)
    (hanti : ∀ c d, history c <+: history d → c = d) :
    HistoryAntichain (fun c => history (pool.candidate g c)) := by
  intro c d hp
  exact pool.candidate_injective g (hanti _ _ hp)

omit [Fintype Token] in
theorem selected_boundary (pool : PrefixPool Candidate Byte) (g : pool.Group)
    (word : Token → List Byte) (history : Candidate → List Token) (emitted : List Byte)
    (hrender : ∀ c, render word (history c) = emitted ++ pool.residual c) :
    ∀ e c, render word (history (pool.candidate g (.inl e))) <+:
      render word (history (pool.candidate g c)) := by
  intro e c
  rw [hrender, hrender]
  have he : pool.residual (pool.candidate g (.inl e)) = pool.residual g.val := e.property
  rw [he]
  exact (List.prefix_append_right_inj emitted).mpr
    (pool.group_key_prefix g (pool.splitEquiv g c))

noncomputable def posterior (pool : PrefixPool Candidate Byte) (source : TokenSource Token)
    (channel : Candidate → FiniteLaw source.decisionTree.Leaves)
    (g : pool.Group) (e : pool.Exact g) : FiniteLaw source.decisionTree.Leaves :=
  ((pool.selected g).collapse e).mix (fun c => channel (pool.candidate g c))

theorem expected_selective_after_collapse (pool : PrefixPool Candidate Byte)
    (word : Token → List Byte) (source : TokenSource Token)
    (history : Candidate → List Token) (emitted : List Byte)
    (channel : Candidate → FiniteLaw source.decisionTree.Leaves)
    (hanti : ∀ c d, history c <+: history d → c = d)
    (hrender : ∀ c, render word (history c) = emitted ++ pool.residual c)
    (hsupport : ∀ c l, (channel c).prob l ≠ 0 → 0 < (source.sourceLaw word).prob l)
    (hfuture : ∀ c l, (channel c).prob l ≠ 0 → history c <+: source.tokenPath l)
    (g : pool.Group) :
    (pool.selected g).surrogate.expect (fun e => source.decisionTree.selective true
      (pool.posterior source channel g e).prob) = source.decisionTree.selective true
        ((pool.selected g).prior.mix (fun c => channel (pool.candidate g c))).prob := by
  exact source.native_tree_collapse_invariant word (pool.selected g)
    (fun c => history (pool.candidate g c)) (fun c => (channel (pool.candidate g c)).prob)
    (pool.selected_antichain g history hanti)
    (pool.selected_boundary g word history emitted hrender)
    (fun c => hsupport (pool.candidate g c)) (fun c => hfuture (pool.candidate g c))

/-- A concrete LAS step, with neither separation nor disintegration assumed. -/
theorem selective_step_inequality (pool : PrefixPool Candidate Byte)
    (word : Token → List Byte) (source : TokenSource Token)
    (history : Candidate → List Token) (emitted : List Byte)
    (channel : Candidate → FiniteLaw source.decisionTree.Leaves)
    (hanti : ∀ c d, history c <+: history d → c = d)
    (hrender : ∀ c, render word (history c) = emitted ++ pool.residual c)
    (hsupport : ∀ c l, (channel c).prob l ≠ 0 → 0 < (source.sourceLaw word).prob l)
    (hfuture : ∀ c l, (channel c).prob l ≠ 0 → history c <+: source.tokenPath l) :
    source.decisionTree.selective true (pool.prior.mix channel).prob ≤
      pool.groups.entropyNat + pool.groups.expect (fun g => (pool.selected g).surrogate.expect
        (fun e => source.decisionTree.selective true (pool.posterior source channel g e).prob)) := by
  simp only [pool.expected_selective_after_collapse word source history emitted channel
    hanti hrender hsupport hfuture]
  have h := source.decisionTree.selective_mix_le pool.groups (fun g =>
    (pool.selected g).prior.mix (fun c => channel (pool.candidate g c)))
  rw [pool.disintegrate_channel channel] at h
  exact h

end StegaTextProofs.PrefixPool
