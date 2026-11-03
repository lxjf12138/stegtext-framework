module

public import StegaTextProofs.TokenSource
public import StegaTextProofs.TerminalCollapse

/-!
Positive source leaves form an antichain of complete labelled token paths.
Consequently a normalized continuation supported on source paths extending
an already completed history is a point mass. This supplies the terminal
continuation premise of exact-collapse resolution from path semantics.
-/

@[expose] public section

namespace StegaTextProofs.TokenSource

variable {Token Byte : Type*} [Fintype Token]

theorem complete_path_prefix_eq (word : Token → List Byte) (source : TokenSource Token)
    (l m : source.decisionTree.Leaves)
    (hl : 0 < (source.sourceLaw word).prob l)
    (hm : 0 < (source.sourceLaw word).prob m)
    (hp : source.tokenPath l <+: source.tokenPath m) : l = m := by
  induction source with
  | terminal => exact Subsingleton.elim _ _
  | step p next ih =>
    rcases l with ⟨g, t, l⟩
    rcases m with ⟨h, u, m⟩
    have hs := (sourceLaw_step_positive word p next g t l).mp hl
    have ht := (sourceLaw_step_positive word p next h u m).mp hm
    have hprefix := List.cons_prefix_cons.mp hp
    obtain rfl := hprefix.1
    have hgh : g = h := hs.1.symm.trans ht.1
    obtain rfl := hgh
    have hlm := ih t l m hs.2.2 ht.2.2 hprefix.2
    subst m
    rfl

theorem continuation_point_of_complete_path (word : Token → List Byte)
    (source : TokenSource Token) (l : source.decisionTree.Leaves)
    (hl : 0 < (source.sourceLaw word).prob l)
    (q : FiniteLaw source.decisionTree.Leaves)
    (hsupport : ∀ m, q.prob m ≠ 0 → 0 < (source.sourceLaw word).prob m)
    (hfuture : ∀ m, q.prob m ≠ 0 → source.tokenPath l <+: source.tokenPath m) :
    q = FiniteLaw.point l := by
  classical
  have hzero : ∀ m, m ≠ l → q.prob m = 0 := by
    intro m hml
    by_contra hm
    exact hml (source.complete_path_prefix_eq word l m hl (hsupport m hm)
      (hfuture m hm)).symm
  have hone : q.prob l = 1 := by
    have hsum : ∑ m, q.prob m = q.prob l :=
      Finset.sum_eq_single l (fun m _ hml => hzero m hml) (by simp)
    exact hsum.symm.trans q.sum_one
  apply FiniteLaw.ext_prob
  intro m
  by_cases hm : m = l
  · subst m
    simpa [FiniteLaw.point] using hone
  · simp [FiniteLaw.point, hm, hzero m hm]

end StegaTextProofs.TokenSource
