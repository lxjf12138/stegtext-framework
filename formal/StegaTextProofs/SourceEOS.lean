module

public import StegaTextProofs.SourceCursorAdvance
public import StegaTextProofs.EOSIsolation

/-! Positive source edges stop exactly on the reserved one-symbol EOS word. -/

@[expose] public section

namespace StegaTextProofs.TokenSource

variable {Token Byte : Type*} [Fintype Token]

def EOSSource (word : Token → List Byte) (eos : Byte) : TokenSource Token → Prop
  | .terminal => True
  | .step p next => ∀ t, 0 < p.prob t →
      ((next t = terminal) ↔ word t = [eos]) ∧ EOSSource word eos (next t)

theorem cursorSource_root (source : TokenSource Token) :
    source.cursorSource source.rootCursor = source := by
  cases source <;> rfl

theorem advance_eos (word : Token → List Byte) (eos : Byte) (source : TokenSource Token)
    (hs : source.EOSSource word eos) (c d : source.Cursor)
    (hlegal : source.LegalCursor c) (hc : source.cursorSource c ≠ terminal)
    (hd : 0 < (source.advance c).prob d) :
    ∃ t, source.cursorHistory d = source.cursorHistory c ++ [t] ∧
      (source.cursorSource d = terminal ↔ word t = [eos]) := by
  classical
  induction source with
  | terminal => exact False.elim (hc rfl)
  | step p next ih =>
    cases c with
    | inl _ =>
      obtain ⟨t, ht, heq⟩ := (FiniteLaw.mix_positive_iff _ _ d).mp hd
      have he : d = .inr ⟨t, (next t).rootCursor⟩ := by
        by_contra hn
        simp [FiniteLaw.point, hn] at heq
      subst d
      exact ⟨t, by simp [cursorHistory, cursorHistory_root],
        by simpa only [cursorSource, cursorSource_root] using (hs t ht).1⟩
    | inr c =>
      rcases c with ⟨t, c⟩
      obtain ⟨d', hd', heq⟩ := (FiniteLaw.mix_positive_iff _ _ d).mp hd
      have he : d = .inr ⟨t, d'⟩ := by
        by_contra hn
        simp [FiniteLaw.point, hn] at heq
      subst d
      obtain ⟨u, hu, heos⟩ := ih t (hs t hlegal.1).2 c d' hlegal.2 hc hd'
      refine ⟨u, ?_, heos⟩
      simpa only [cursorHistory, List.cons_append] using congrArg (List.cons t) hu

end StegaTextProofs.TokenSource
