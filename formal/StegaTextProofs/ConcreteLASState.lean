module

public import StegaTextProofs.ActiveLASStep
public import StegaTextProofs.SourceEOS
public import StegaTextProofs.PrefixPoolNextLaw

/-!
Concrete LAS states over a fixed source cursor space. All frontier and stopping
invariants are proved for the root and for every constructed group/surrogate update.
-/

@[expose] public section

namespace StegaTextProofs.TokenSource

attribute [local instance] Classical.propDecidable
open PrefixGrouping
variable {Token Byte : Type*} [Fintype Token]

structure LASState (word : Token → List Byte) (eos : Byte) (source : TokenSource Token) where
  prior : FiniteLaw source.Cursor
  active : source.Cursor
  emitted : List Byte
  active_pos : 0 < prior.prob active
  legal : ∀ c, 0 < prior.prob c → source.LegalCursor c
  antichain : ∀ c d, 0 < prior.prob c → 0 < prior.prob d →
    source.cursorHistory c <+: source.cursorHistory d → c = d
  active_render : render word (source.cursorHistory active) = emitted
  visible : ∀ c, 0 < prior.prob c → emitted <+: render word (source.cursorHistory c)
  strict_length : ∀ c, 0 < prior.prob c → c ≠ active →
    emitted.length < (render word (source.cursorHistory c)).length
  isolated : ∀ c, 0 < prior.prob c → c ≠ active →
    EOSIsolation.IsolatedWord eos ((render word (source.cursorHistory c)).drop emitted.length)
  retained_nonterminal : ∀ c, 0 < prior.prob c → c ≠ active → source.cursorSource c ≠ terminal
  terminal_point : source.cursorSource active = terminal → prior = FiniteLaw.point active

namespace LASState

variable {word : Token → List Byte} {eos : Byte} {source : TokenSource Token}

noncomputable def initial (word : Token → List Byte) (eos : Byte) (source : TokenSource Token) :
    LASState word eos source where
  prior := FiniteLaw.point source.rootCursor
  active := source.rootCursor
  emitted := []
  active_pos := by simp [FiniteLaw.point]
  legal := by
    intro c hc
    have he : c = source.rootCursor := by
      by_contra hn
      simp [FiniteLaw.point, hn] at hc
    subst c
    exact source.rootCursor_legal
  antichain := by
    intro c d hc hd _
    have h : ∀ c, 0 < (FiniteLaw.point source.rootCursor).prob c → c = source.rootCursor := by
      intro c hc
      by_contra hn
      simp [FiniteLaw.point, hn] at hc
    exact (h c hc).trans (h d hd).symm
  active_render := by simp [cursorHistory_root, render]
  visible := by simp
  strict_length := by intro c hc hn; simp [FiniteLaw.point, hn] at hc
  isolated := by intro c hc hn; simp [FiniteLaw.point, hn] at hc
  retained_nonterminal := by intro c hc hn; simp [FiniteLaw.point, hn] at hc
  terminal_point := fun _ => rfl

noncomputable def pool (s : LASState word eos source) :=
  source.activePool word s.prior s.active s.emitted

theorem pool_render (s : LASState word eos source) (ha : source.cursorSource s.active ≠ terminal)
    (c : (source.expandActive s.prior s.active).Positive) :
    render word (source.cursorHistory c.val) = s.emitted ++ s.pool.residual c :=
  source.activePool_render word s.prior s.active s.emitted ha s.active_render s.visible c

theorem pool_nonempty (s : LASState word eos source) (ha : source.cursorSource s.active ≠ terminal)
    (hword : ∀ t, word t ≠ []) (c : (source.expandActive s.prior s.active).Positive) :
    s.pool.residual c ≠ [] :=
  source.activePool_nonempty word s.prior s.active s.emitted ha s.active_render hword
    s.strict_length c

theorem fresh_residual (s : LASState word eos source)
    (c : (source.expandActive s.prior s.active).Positive) (t : Token)
    (ht : source.cursorHistory c.val = source.cursorHistory s.active ++ [t]) :
    s.pool.residual c = word t := by
  change (render word (source.cursorHistory c.val)).drop s.emitted.length = _
  have hr : render word (source.cursorHistory c.val) = s.emitted ++ word t := by
    simp only [ht, render, List.flatMap_append, List.flatMap_cons, List.flatMap_nil,
      List.append_nil]
    exact congrArg (fun xs => xs ++ word t) s.active_render
  simp [hr]

theorem pool_isolated (s : LASState word eos source) (ha : source.cursorSource s.active ≠ terminal)
    (hword : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (c : (source.expandActive s.prior s.active).Positive) :
    EOSIsolation.IsolatedWord eos (s.pool.residual c) := by
  rcases (source.expandActive_positive s.prior s.active c.val).mp c.property with
    ⟨_, hc⟩ | ⟨hc, hn⟩
  · obtain ⟨t, ht⟩ := source.advance_history s.active c.val ha hc
    rw [s.fresh_residual c t ht]
    exact hword t
  · exact s.isolated c.val hc hn

theorem pool_terminal (s : LASState word eos source) (ha : source.cursorSource s.active ≠ terminal)
    (hs : source.EOSSource word eos) (c : (source.expandActive s.prior s.active).Positive)
    (hc : source.cursorSource c.val = terminal) : s.pool.residual c = [eos] := by
  rcases (source.expandActive_positive s.prior s.active c.val).mp c.property with
    ⟨_, hd⟩ | ⟨hd, hn⟩
  · obtain ⟨t, ht, he⟩ := source.advance_eos word eos hs s.active c.val
      (s.legal s.active s.active_pos) ha hd
    exact (s.fresh_residual c t ht).trans (he.mp hc)
  · exact False.elim (s.retained_nonterminal c.val hd hn hc)

theorem strict_nonterminal (s : LASState word eos source)
    (ha : source.cursorSource s.active ≠ terminal) (hs : source.EOSSource word eos)
    (hword : ∀ t, word t ≠ []) (g : s.pool.Group) (r : s.pool.Strict g) :
    source.cursorSource r.val.val.val ≠ terminal := by
  intro ht
  have he := s.pool_terminal ha hs r.val.val ht
  have hp := s.pool.group_key_prefix g r.val
  have hn := List.length_pos_iff.mpr (s.pool_nonempty ha hword g.val)
  have hl : (s.pool.residual r.val.val).length ≤ (s.pool.residual g.val).length := by
    simp only [he, List.length_singleton]
    omega
  exact r.property (hp.eq_of_length_le hl).symm

theorem next_retained_render (s : LASState word eos source)
    (ha : source.cursorSource s.active ≠ terminal) (g : s.pool.Group)
    (e : s.pool.Exact g) (r : s.pool.Strict g) :
    render word (source.cursorHistory r.val.val.val) =
      (s.emitted ++ s.pool.residual g.val) ++ s.pool.nextResidual g r :=
  s.pool.next_retained_render word (fun c => source.cursorHistory c.val) s.emitted
    (s.pool_render ha) g e r

noncomputable def next (s : LASState word eos source)
    (ha : source.cursorSource s.active ≠ terminal) (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (g : s.pool.Group) (e : s.pool.Exact g) : LASState word eos source where
  prior := s.pool.nextLaw Subtype.val g e
  active := e.val.val.val
  emitted := s.emitted ++ s.pool.residual g.val
  active_pos := s.pool.nextLaw_active_positive Subtype.val g e
  legal := by
    intro c hc
    rcases s.pool.nextLaw_support Subtype.val g e c hc with he | ⟨r, he⟩
    · subst c
      exact source.expandActive_legal s.prior s.active s.legal _ e.val.val.property
    · subst c
      exact source.expandActive_legal s.prior s.active s.legal _ r.val.val.property
  antichain := by
    have hpos : ∀ c, 0 < (s.pool.nextLaw Subtype.val g e).prob c →
        0 < (source.expandActive s.prior s.active).prob c := by
      intro c hc
      rcases s.pool.nextLaw_support Subtype.val g e c hc with he | ⟨r, he⟩
      · exact he ▸ e.val.val.property
      · exact he ▸ r.val.val.property
    intro c d hc hd hp
    exact source.expandActive_antichain s.prior s.active ha s.antichain c d
      (hpos c hc) (hpos d hd) hp
  active_render := s.pool.next_active_render word (fun c => source.cursorHistory c.val)
    s.emitted (s.pool_render ha) g e
  visible := by
    intro c hc
    rcases s.pool.nextLaw_support Subtype.val g e c hc with he | ⟨r, he⟩
    · subst c
      rw [s.pool_render ha, e.property]
    · subst c
      rw [s.next_retained_render ha g e r]
      exact List.prefix_append _ _
  strict_length := by
    intro c hc hn
    rcases s.pool.nextLaw_support Subtype.val g e c hc with he | ⟨r, he⟩
    · exact False.elim (hn he)
    · subst c
      rw [s.next_retained_render ha g e r]
      simp only [List.length_append]
      exact Nat.lt_add_of_pos_right (List.length_pos_iff.mpr (s.pool.nextResidual_nonempty g r))
  isolated := by
    intro c hc hn
    rcases s.pool.nextLaw_support Subtype.val g e c hc with he | ⟨r, he⟩
    · exact False.elim (hn he)
    · subst c
      simpa only [s.next_retained_render ha g e r, List.drop_left] using
        s.pool.nextResidual_isolated eos (s.pool_isolated ha hiso) g r
  retained_nonterminal := by
    intro c hc hn
    rcases s.pool.nextLaw_support Subtype.val g e c hc with he | ⟨r, he⟩
    · exact False.elim (hn he)
    · subst c
      exact s.strict_nonterminal ha hs hne g r
  terminal_point := by
    intro ht
    have hg : s.pool.residual g.val = [eos] :=
      e.property.symm.trans (s.pool_terminal ha hs e.val.val ht)
    exact s.pool.nextLaw_eos Subtype.val eos (s.pool_isolated ha hiso) g hg e

theorem next_posterior (s : LASState word eos source)
    (ha : source.cursorSource s.active ≠ terminal) (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (g : s.pool.Group) (e : s.pool.Exact g) :
    (s.next ha hs hne hiso g e).prior.mix (source.cursorLaw word) =
      s.pool.posterior source (fun c => source.cursorLaw word c.val) g e :=
  s.pool.nextLaw_channel Subtype.val (source.cursorLaw word) g e

end LASState

end StegaTextProofs.TokenSource
