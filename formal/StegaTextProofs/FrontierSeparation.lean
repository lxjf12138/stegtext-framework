module

public import StegaTextProofs.VisibleBoundary
public import StegaTextProofs.LASNodeInvariant

/-!
Derive the entropy-node separation criterion from labelled-history antichains
and compatible visible boundaries. The row support assumption states the
ordinary semantics of a source node: a frontier history either precedes that
node or has already fixed the next native group there.
-/

@[expose] public section

namespace StegaTextProofs

open PrefixGrouping

variable {Token Byte Exact Strict : Type*}

/-- No two distinct frontier histories are ancestors of each other. -/
def HistoryAntichain (history : Exact ⊕ Strict → List Token) : Prop :=
  ∀ c d, history c <+: history d → c = d

/-- This is a source-tree support property. It makes no entropy or budget
assumption, and permits arbitrary history-dependent continuation laws. -/
theorem node_separation_of_visible_frontier
    (word : Token → List Byte) (support : Finset (List Byte))
    (history : Exact ⊕ Strict → List Token) (node : List Token)
    (row : Exact ⊕ Strict → support → ℝ)
    (hanti : HistoryAntichain history)
    (hboundary : ∀ e c, render word (history (.inl e)) <+: render word (history c))
    (hsupported : ∀ c t tail, history c = node ++ t :: tail → word t ∈ support)
    (hrow : ∀ c b, row c b ≠ 0 → history c <+: node ∨
      ∃ t tail, ∃ ht : history c = node ++ t :: tail,
        b.val = key support (word t) (hsupported c t tail ht)) :
    NodeSeparation row := by
  classical
  by_cases hbelow : ∃ e, history (.inl e) <+: node
  · obtain ⟨e, he⟩ := hbelow
    right; left
    refine ⟨e, ?_⟩
    intro c hc b
    by_contra hne
    rcases hrow c b hne with hp | ⟨t, tail, ht, _⟩
    · rcases List.prefix_or_prefix_of_prefix hp he with hce | hec
      · exact hc (hanti c (.inl e) hce)
      · exact hc (hanti (.inl e) c hec).symm
    · have hp : node <+: history c := ht ▸ List.prefix_append node (t :: tail)
      exact hc (hanti (.inl e) c (he.trans hp)).symm
  · by_cases habove : ∃ e, node <+: history (.inl e)
    · obtain ⟨e, tail, he⟩ := habove
      cases tail with
      | nil =>
        have heq : node = history (.inl e) := by simpa using he
        exact False.elim (hbelow ⟨e, by rw [← heq]⟩)
      | cons t tail =>
        have he' : history (.inl e) = node ++ t :: tail := he.symm
        let b₀ : support := ⟨key support (word t) (hsupported (.inl e) t tail he'),
          (key_spec support (word t) (hsupported (.inl e) t tail he')).1.1⟩
        right; right
        refine ⟨b₀, ?_⟩
        intro c b hb
        by_contra hne
        rcases hrow c b hne with hp | ⟨u, tailU, hu, hkey⟩
        · have hneprefix : node <+: history (.inl e) :=
            he' ▸ List.prefix_append node (t :: tail)
          have hce : c = .inl e := hanti c (.inl e) (hp.trans hneprefix)
          subst c
          exact hbelow ⟨e, hp⟩
        · apply hb
          apply Subtype.ext
          have hvis := hboundary e c
          rw [he', hu] at hvis
          have hkeys := native_group_fixed_at_history word support node t u tail tailU
            (hsupported (.inl e) t tail he') (hsupported c u tailU hu) hvis
          exact hkey.trans hkeys.symm
    · left
      intro e b
      by_contra hne
      rcases hrow (.inl e) b hne with hp | ⟨t, tail, ht, _⟩
      · exact hbelow ⟨e, hp⟩
      · exact habove ⟨e, ht ▸ List.prefix_append node (t :: tail)⟩

end StegaTextProofs
