module

public import StegaTextProofs.TokenSource
public import StegaTextProofs.FrontierSeparation

/-!
Derive separation for the entire compiled native tree from source-path
support and visible frontier invariants. Recursive restrictions of a channel
need not be normalized. All positive support is that of the original source.
-/

@[expose] public section

namespace StegaTextProofs

open scoped BigOperators
open PrefixGrouping

namespace SourceFrontier

theorem history_after_node {Token : Type*} (history node : List Token)
    (t : Token) (tail : List Token) (hfuture : history <+: node ++ t :: tail)
    (hbefore : ¬ history <+: node) : ∃ rest, history = node ++ t :: rest := by
  have hn : node <+: node ++ t :: tail := List.prefix_append node (t :: tail)
  rcases List.prefix_or_prefix_of_prefix hfuture hn with hb | ha
  · exact False.elim (hbefore hb)
  · obtain ⟨suffix, hsuffix⟩ := ha
    cases suffix with
    | nil =>
      have heq : node = history := by simpa using hsuffix
      exact False.elim (hbefore (by rw [← heq]))
    | cons u rest =>
      have hh : history = node ++ u :: rest := hsuffix.symm
      rw [hh] at hfuture
      have hut : u = t :=
        (List.cons_prefix_cons.mp ((List.prefix_append_right_inj node).mp hfuture)).1
      subst u
      exact ⟨rest, hh⟩

end SourceFrontier

namespace TokenSource

variable {Token Byte Exact Strict : Type*} [Fintype Token]

/-- Root payload rows in the compiled source step. -/
noncomputable def rootRows (next : Token → TokenSource Token)
    (channel : Exact ⊕ Strict → (DecisionTree.grouped
      (fun t => (next t).decisionTree)).Leaves → ℝ) : Exact ⊕ Strict → Token → ℝ :=
  fun c g => ∑ l, channel c ⟨g, l⟩

theorem rootRows_nonzero (next : Token → TokenSource Token)
    (channel : Exact ⊕ Strict → (DecisionTree.grouped
      (fun t => (next t).decisionTree)).Leaves → ℝ)
    (c : Exact ⊕ Strict) (g : Token) (h : rootRows next channel c g ≠ 0) :
    ∃ t l, channel c ⟨g, ⟨t, l⟩⟩ ≠ 0 := by
  classical
  by_contra hn
  apply h
  apply Finset.sum_eq_zero
  intro l _
  exact not_ne_iff.mp (fun hl => hn ⟨l.1, l.2, hl⟩)

theorem native_root_separation (word : Token → List Byte) (p : FiniteLaw Token)
    (next : Token → TokenSource Token) (node : List Token)
    (history : Exact ⊕ Strict → List Token)
    (channel : Exact ⊕ Strict → (step p next).decisionTree.Leaves → ℝ)
    (hanti : HistoryAntichain history)
    (hboundary : ∀ e c, render word (history (.inl e)) <+: render word (history c))
    (hsupport : ∀ c l, channel c l ≠ 0 → 0 < ((step p next).sourceLaw word).prob l)
    (hfuture : ∀ c l, channel c l ≠ 0 →
      history c <+: node ++ (step p next).tokenPath l) :
    NodeSeparation (rootRows next channel) := by
  classical
  by_cases hbelow : ∃ e, history (.inl e) <+: node
  · obtain ⟨e, he⟩ := hbelow
    right; left
    refine ⟨e, ?_⟩
    intro c hc g
    by_contra hne
    obtain ⟨t, l, hl⟩ := rootRows_nonzero next channel c g hne
    have hp := hfuture c ⟨g, ⟨t, l⟩⟩ hl
    have hep : history (.inl e) <+: node ++ (step p next).tokenPath ⟨g, ⟨t, l⟩⟩ :=
      he.trans (List.prefix_append node _)
    rcases List.prefix_or_prefix_of_prefix hp hep with hce | hec
    · exact hc (hanti c (.inl e) hce)
    · exact hc (hanti (.inl e) c hec).symm
  · by_cases hexact : ∃ e g, rootRows next channel (.inl e) g ≠ 0
    · obtain ⟨e, g₀, he⟩ := hexact
      obtain ⟨t₀, l₀, hl₀⟩ := rootRows_nonzero next channel (.inl e) g₀ he
      have hs₀ := (sourceLaw_step_positive word p next g₀ t₀ l₀).mp
        (hsupport (.inl e) ⟨g₀, ⟨t₀, l₀⟩⟩ hl₀)
      obtain ⟨rest₀, hhist₀⟩ := SourceFrontier.history_after_node (history (.inl e)) node
        t₀ ((next t₀).tokenPath l₀) (hfuture (.inl e) ⟨g₀, ⟨t₀, l₀⟩⟩ hl₀)
        (fun hb => hbelow ⟨e, hb⟩)
      right; right
      refine ⟨g₀, ?_⟩
      intro c g hg
      by_contra hne
      obtain ⟨t, l, hl⟩ := rootRows_nonzero next channel c g hne
      have hs := (sourceLaw_step_positive word p next g t l).mp
        (hsupport c ⟨g, ⟨t, l⟩⟩ hl)
      have hbefore : ¬ history c <+: node := by
        intro hb
        have hn : node <+: history (.inl e) :=
          hhist₀ ▸ List.prefix_append node (t₀ :: rest₀)
        have hce := hanti c (.inl e) (hb.trans hn)
        subst c
        exact hbelow ⟨e, hb⟩
      obtain ⟨rest, hhist⟩ := SourceFrontier.history_after_node (history c) node t
        ((next t).tokenPath l) (hfuture c ⟨g, ⟨t, l⟩⟩ hl) hbefore
      have hvis := hboundary e c
      rw [hhist₀, hhist] at hvis
      have hkeys := native_group_fixed_at_history word (NativeGrouping.support word p)
        node t₀ t rest₀ rest (NativeGrouping.token_mem_support word p t₀ hs₀.2.1)
        (NativeGrouping.token_mem_support word p t hs.2.1) hvis
      have hlabels := (NativeGrouping.same_label_iff_same_key word p t₀ t
        hs₀.2.1 hs.2.1).mpr hkeys
      exact hg (hs.1.symm.trans (hlabels.symm.trans hs₀.1))
    · left
      intro e g
      exact not_ne_iff.mp (fun h => hexact ⟨e, g, h⟩)

/-- Every native payload node is separated by the same visible frontier.
The statement applies to unnormalized restrictions at descendant nodes. -/
theorem native_tree_separated (word : Token → List Byte) (source : TokenSource Token)
    (node : List Token) (history : Exact ⊕ Strict → List Token)
    (channel : Exact ⊕ Strict → source.decisionTree.Leaves → ℝ)
    (hanti : HistoryAntichain history)
    (hboundary : ∀ e c, render word (history (.inl e)) <+: render word (history c))
    (hsupport : ∀ c l, channel c l ≠ 0 → 0 < (source.sourceLaw word).prob l)
    (hfuture : ∀ c l, channel c l ≠ 0 → history c <+: node ++ source.tokenPath l) :
    source.decisionTree.Separated channel := by
  induction source generalizing node with
  | terminal => trivial
  | step p next ih =>
    constructor
    · intro _
      exact native_root_separation word p next node history channel hanti hboundary
        hsupport hfuture
    · intro g
      constructor
      · intro h
        cases h
      · intro t
        apply ih t (node ++ [t]) (fun c l => channel c ⟨g, ⟨t, l⟩⟩)
        · intro c l hl
          exact ((sourceLaw_step_positive word p next g t l).mp
            (hsupport c ⟨g, ⟨t, l⟩⟩ hl)).2.2
        · intro c l hl
          simpa only [tokenPath, List.append_assoc, List.singleton_append] using
            hfuture c ⟨g, ⟨t, l⟩⟩ hl

/-- Whole-tree entropy conservation now follows from source/frontier semantics. -/
theorem native_tree_collapse_invariant [Fintype Exact] [Fintype Strict]
    (word : Token → List Byte) (source : TokenSource Token)
    (g : LASGroup Exact Strict) (history : Exact ⊕ Strict → List Token)
    (channel : Exact ⊕ Strict → source.decisionTree.Leaves → ℝ)
    (hanti : HistoryAntichain history)
    (hboundary : ∀ e c, render word (history (.inl e)) <+: render word (history c))
    (hsupport : ∀ c l, channel c l ≠ 0 → 0 < (source.sourceLaw word).prob l)
    (hfuture : ∀ c l, channel c l ≠ 0 → history c <+: source.tokenPath l) :
    g.surrogate.expect (fun e => source.decisionTree.selective true
      (nodeWeights (g.collapse e) channel)) =
      source.decisionTree.selective true (nodeWeights g.prior channel) := by
  apply source.decisionTree.selective_collapse_invariant
  exact source.native_tree_separated word [] history channel hanti hboundary hsupport
    (by simpa using hfuture)

end TokenSource

end StegaTextProofs
