module

public import StegaTextProofs.SourceCursorAdvance

/-!
Concrete fresh-only expansion on the source's finite cursor space. The
active atom advances once; every other atom stays fixed. Its positive
support is characterized exactly, legal source positions and antichains
are preserved, and the complete-path continuation mixture is unchanged.
-/

@[expose] public section

namespace StegaTextProofs.TokenSource

attribute [local instance] Classical.propDecidable
variable {Token Byte : Type*} [Fintype Token]

noncomputable def expandActive (source : TokenSource Token) (prior : FiniteLaw source.Cursor)
    (active : source.Cursor) : FiniteLaw source.Cursor :=
  prior.mix (fun c => if c = active then source.advance c else FiniteLaw.point c)

theorem expandActive_positive (source : TokenSource Token) (prior : FiniteLaw source.Cursor)
    (active d : source.Cursor) :
    0 < (source.expandActive prior active).prob d ↔
      (0 < prior.prob active ∧ 0 < (source.advance active).prob d) ∨
        (0 < prior.prob d ∧ d ≠ active) := by
  classical
  rw [expandActive, FiniteLaw.mix_positive_iff]
  constructor
  · rintro ⟨c, hc, hd⟩
    by_cases he : c = active
    · subst c
      exact Or.inl ⟨hc, by simpa using hd⟩
    · rw [ite_eq_right he] at hd
      have hdc : d = c := by
        by_contra hn
        simp [FiniteLaw.point, hn] at hd
      subst d
      exact Or.inr ⟨hc, he⟩
  · rintro (⟨hp, hd⟩ | ⟨hp, hn⟩)
    · exact ⟨active, hp, by simpa using hd⟩
    · exact ⟨d, hp, by simp [hn, FiniteLaw.point]⟩

theorem expandActive_continuation (word : Token → List Byte) (source : TokenSource Token)
    (prior : FiniteLaw source.Cursor) (active : source.Cursor) :
    (source.expandActive prior active).mix (source.cursorLaw word) =
      prior.mix (source.cursorLaw word) := by
  rw [expandActive, FiniteLaw.mix_assoc]
  congr 1
  funext c
  by_cases hc : c = active
  · simp only [hc, ↓reduceIte, source.advance_continuation]
  · simp only [hc, ↓reduceIte, FiniteLaw.point_mix]

theorem expandActive_legal (source : TokenSource Token) (prior : FiniteLaw source.Cursor)
    (active : source.Cursor)
    (hlegal : ∀ c, 0 < prior.prob c → source.LegalCursor c)
    (d : source.Cursor) (hd : 0 < (source.expandActive prior active).prob d) :
    source.LegalCursor d := by
  rcases (source.expandActive_positive prior active d).mp hd with ⟨ha, hd⟩ | ⟨hd, _⟩
  · exact source.advance_legal active d (hlegal active ha) hd
  · exact hlegal d hd

theorem expandActive_antichain (source : TokenSource Token) (prior : FiniteLaw source.Cursor)
    (active : source.Cursor) (hactive : source.cursorSource active ≠ terminal)
    (hanti : ∀ c d, 0 < prior.prob c → 0 < prior.prob d →
      source.cursorHistory c <+: source.cursorHistory d → c = d)
    (c d : source.Cursor)
    (hc : 0 < (source.expandActive prior active).prob c)
    (hd : 0 < (source.expandActive prior active).prob d)
    (hprefix : source.cursorHistory c <+: source.cursorHistory d) : c = d := by
  rcases (source.expandActive_positive prior active c).mp hc with ⟨ha, hc⟩ | ⟨hc, hca⟩
  · obtain ⟨t, ht⟩ := source.advance_history active c hactive hc
    rcases (source.expandActive_positive prior active d).mp hd with ⟨_, hd⟩ | ⟨hd, hda⟩
    · obtain ⟨u, hu⟩ := source.advance_history active d hactive hd
      apply source.cursorHistory_injective
      exact hprefix.eq_of_length_le (by simp [ht, hu])
    · have hap : source.cursorHistory active <+: source.cursorHistory c :=
        ht ▸ List.prefix_append _ [t]
      exact False.elim (hda (hanti active d ha hd (hap.trans hprefix)).symm)
  · rcases (source.expandActive_positive prior active d).mp hd with ⟨ha, hd⟩ | ⟨hd, _⟩
    · obtain ⟨t, ht⟩ := source.advance_history active d hactive hd
      have hap : source.cursorHistory active <+: source.cursorHistory d :=
        ht ▸ List.prefix_append _ [t]
      rcases List.prefix_or_prefix_of_prefix hprefix hap with hca' | hac
      · exact False.elim (hca (hanti c active hc ha hca'))
      · exact False.elim (hca (hanti active c ha hc hac).symm)
    · exact hanti c d hc hd hprefix

end StegaTextProofs.TokenSource
