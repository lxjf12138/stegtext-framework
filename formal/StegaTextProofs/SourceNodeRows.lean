module

public import StegaTextProofs.FrontierSeparation

/-!
Actual source-node rows are probabilities of visiting a fixed native group
at a labelled history, under each candidate's complete-path continuation
law. Derive the row support criterion from prefix support of those laws.
-/

@[expose] public section

namespace StegaTextProofs

open scoped BigOperators
open PrefixGrouping

variable {Token Byte Candidate Outcome : Type*} [Fintype Outcome]

noncomputable def sourceNodeRows (support : Finset (List Byte))
    (channel : Candidate → FiniteLaw Outcome) (visit : Outcome → Option support) :
    Candidate → support → ℝ := by
  classical
  exact fun c b => ∑ ω, if visit ω = some b then (channel c).prob ω else 0

theorem sourceNodeRows_nonzero (support : Finset (List Byte))
    (channel : Candidate → FiniteLaw Outcome) (visit : Outcome → Option support)
    (c : Candidate) (b : support) (hne : sourceNodeRows support channel visit c b ≠ 0) :
    ∃ ω, (channel c).prob ω ≠ 0 ∧ visit ω = some b := by
  classical
  by_contra h
  apply hne
  apply Finset.sum_eq_zero
  intro ω _
  by_cases hv : visit ω = some b
  · have hp : (channel c).prob ω = 0 := by
      by_contra hp
      exact h ⟨ω, hp, hv⟩
    simp [hv, hp]
  · simp [hv]

/-- Only prefix-supported future paths are needed; their probabilities may
depend arbitrarily on the complete candidate history. -/
theorem sourceNodeRows_support (word : Token → List Byte)
    (support : Finset (List Byte)) (history : Candidate → List Token)
    (path : Outcome → List Token) (node : List Token)
    (channel : Candidate → FiniteLaw Outcome) (visit : Outcome → Option support)
    (hfuture : ∀ c ω, (channel c).prob ω ≠ 0 → history c <+: path ω)
    (hsupported : ∀ c t tail, history c = node ++ t :: tail → word t ∈ support)
    (hvisit : ∀ ω b, visit ω = some b →
      ∃ t tail, path ω = node ++ t :: tail ∧
        ∃ ht : word t ∈ support, b.val = key support (word t) ht) :
    ∀ c b, sourceNodeRows support channel visit c b ≠ 0 → history c <+: node ∨
      ∃ t tail, ∃ ht : history c = node ++ t :: tail,
        b.val = key support (word t) (hsupported c t tail ht) := by
  intro c b hne
  obtain ⟨ω, hp, hv⟩ := sourceNodeRows_nonzero support channel visit c b hne
  obtain ⟨t, tail, hpath, ht, hkey⟩ := hvisit ω b hv
  have hc := hfuture c ω hp
  have hn : node <+: path ω := hpath ▸ List.prefix_append node (t :: tail)
  rcases List.prefix_or_prefix_of_prefix hc hn with hbefore | hafter
  · exact Or.inl hbefore
  · obtain ⟨suffix, hsuffix⟩ := hafter
    cases suffix with
    | nil =>
      left
      have heq : node = history c := by simpa using hsuffix
      rw [← heq]
    | cons u rest =>
      have hhist : history c = node ++ u :: rest := hsuffix.symm
      rw [hhist, hpath] at hc
      have htoken : u = t :=
        (List.cons_prefix_cons.mp ((List.prefix_append_right_inj node).mp hc)).1
      subst u
      exact Or.inr ⟨t, rest, hhist, hkey⟩

/-- The node-entropy invariant now follows from complete labelled paths,
their continuation laws, the frontier antichain and the visible boundary. -/
theorem source_node_entropy_collapse_invariant {Exact Strict : Type*}
    [Fintype Exact] [Fintype Strict] (g : LASGroup Exact Strict)
    (word : Token → List Byte) (support : Finset (List Byte))
    (history : Exact ⊕ Strict → List Token) (path : Outcome → List Token)
    (node : List Token) (channel : Exact ⊕ Strict → FiniteLaw Outcome)
    (visit : Outcome → Option support)
    (hanti : HistoryAntichain history)
    (hboundary : ∀ e c, render word (history (.inl e)) <+: render word (history c))
    (hfuture : ∀ c ω, (channel c).prob ω ≠ 0 → history c <+: path ω)
    (hsupported : ∀ c t tail, history c = node ++ t :: tail → word t ∈ support)
    (hvisit : ∀ ω b, visit ω = some b →
      ∃ t tail, path ω = node ++ t :: tail ∧
        ∃ ht : word t ∈ support, b.val = key support (word t) ht) :
    g.surrogate.expect (fun e => weightedEntropy
      (nodeWeights (g.collapse e) (sourceNodeRows support channel visit))) =
    weightedEntropy (nodeWeights g.prior (sourceNodeRows support channel visit)) := by
  apply g.node_entropy_collapse_invariant
  apply node_separation_of_visible_frontier word support history node _ hanti hboundary hsupported
  exact sourceNodeRows_support word support history path node channel visit hfuture hsupported hvisit

end StegaTextProofs
