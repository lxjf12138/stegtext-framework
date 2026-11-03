module

public import StegaTextProofs.SourceCursors

/-!
The source's own next-token sampler on its finite cursor space. Terminal
cursors self-loop. Advancing a legal nonterminal cursor follows a positive
source edge, appends one token, and preserves the continuation mixture.
-/

@[expose] public section

namespace StegaTextProofs.TokenSource

variable {Token Byte : Type*} [Fintype Token]

theorem cursorHistory_root (source : TokenSource Token) : source.cursorHistory source.rootCursor = [] := by
  cases source <;> rfl

noncomputable def advance : (source : TokenSource Token) → source.Cursor → FiniteLaw source.Cursor
  | .terminal, c => FiniteLaw.point c
  | .step p next, .inl _ => p.mix (fun t => FiniteLaw.point (.inr ⟨t, (next t).rootCursor⟩))
  | .step _ next, .inr ⟨t, c⟩ => ((next t).advance c).mix
      (fun d => FiniteLaw.point (.inr ⟨t, d⟩))

theorem advance_continuation (word : Token → List Byte) (source : TokenSource Token)
    (c : source.Cursor) :
    (source.advance c).mix (source.cursorLaw word) = source.cursorLaw word c := by
  induction source with
  | terminal => exact FiniteLaw.point_mix c _
  | step p next ih =>
    cases c with
    | inl _ =>
      simp only [advance, FiniteLaw.mix_assoc, FiniteLaw.point_mix]
      exact (cursorLaw_step word p next).symm
    | inr c =>
      rcases c with ⟨t, c⟩
      simp only [advance, FiniteLaw.mix_assoc, FiniteLaw.point_mix, cursorLaw]
      rw [← FiniteLaw.mix_assoc, ih]

theorem advance_legal (source : TokenSource Token) (c d : source.Cursor)
    (hc : source.LegalCursor c) (hd : 0 < (source.advance c).prob d) : source.LegalCursor d := by
  classical
  induction source with
  | terminal => trivial
  | step p next ih =>
    cases c with
    | inl _ =>
      obtain ⟨t, ht, heq⟩ := (FiniteLaw.mix_positive_iff _ _ d).mp hd
      have he : d = .inr ⟨t, (next t).rootCursor⟩ := by
        by_contra hn
        simp [FiniteLaw.point, hn] at heq
      subst d
      exact ⟨ht, (next t).rootCursor_legal⟩
    | inr c =>
      rcases c with ⟨t, c⟩
      obtain ⟨d', hd', heq⟩ := (FiniteLaw.mix_positive_iff _ _ d).mp hd
      have he : d = .inr ⟨t, d'⟩ := by
        by_contra hn
        simp [FiniteLaw.point, hn] at heq
      subst d
      exact ⟨hc.1, ih t c d' hc.2 hd'⟩

theorem advance_history (source : TokenSource Token) (c d : source.Cursor)
    (hc : source.cursorSource c ≠ terminal) (hd : 0 < (source.advance c).prob d) :
    ∃ t, source.cursorHistory d = source.cursorHistory c ++ [t] := by
  classical
  induction source with
  | terminal => exact False.elim (hc rfl)
  | step p next ih =>
    cases c with
    | inl _ =>
      obtain ⟨t, _, heq⟩ := (FiniteLaw.mix_positive_iff _ _ d).mp hd
      have he : d = .inr ⟨t, (next t).rootCursor⟩ := by
        by_contra hn
        simp [FiniteLaw.point, hn] at heq
      subst d
      exact ⟨t, by simp [cursorHistory, cursorHistory_root]⟩
    | inr c =>
      rcases c with ⟨t, c⟩
      obtain ⟨d', hd', heq⟩ := (FiniteLaw.mix_positive_iff _ _ d).mp hd
      have he : d = .inr ⟨t, d'⟩ := by
        by_contra hn
        simp [FiniteLaw.point, hn] at heq
      subst d
      obtain ⟨u, hu⟩ := ih t c d' hc hd'
      exact ⟨u, by simpa only [cursorHistory, List.cons_append] using congrArg (List.cons t) hu⟩

theorem advance_terminal (source : TokenSource Token) (c : source.Cursor)
    (hc : source.cursorSource c = terminal) : source.advance c = FiniteLaw.point c := by
  induction source with
  | terminal => rfl
  | step p next ih =>
    cases c with
    | inl _ => cases hc
    | inr c =>
      rcases c with ⟨t, c⟩
      simp only [advance, ih t c hc, FiniteLaw.point_mix]

end StegaTextProofs.TokenSource
