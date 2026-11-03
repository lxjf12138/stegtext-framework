module

public import StegaTextProofs.TokenSource
public import StegaTextProofs.FrontierExpansionLaw
public import StegaTextProofs.LASReachability

/-!
Finite, explicitly constructed source-node positions. Each cursor carries
its full token history, its remaining source and a continuation law in the
original leaf space. Legal cursors follow positive-probability edges.
-/

@[expose] public section

namespace StegaTextProofs

namespace FiniteLaw

variable {α : Type*} [Fintype α]

theorem ext_expect {p q : FiniteLaw α} (h : ∀ f, p.expect f = q.expect f) : p = q := by
  classical
  apply ext_prob
  intro a
  simpa [expect] using h (fun b => if b = a then 1 else 0)

end FiniteLaw

namespace TokenSource

universe u
variable {Token : Type u} {Byte : Type*} [Fintype Token]

@[reducible] def Cursor : TokenSource Token → Type u
  | .terminal => PUnit
  | .step _ next => PUnit ⊕ ((t : Token) × Cursor (next t))

noncomputable instance cursorFintype (source : TokenSource Token) : Fintype source.Cursor :=
  match source with
  | .terminal => inferInstanceAs (Fintype PUnit.{u + 1})
  | .step _ next =>
    letI := fun t => cursorFintype (next t)
    inferInstanceAs (Fintype (PUnit ⊕ ((t : Token) × Cursor (next t))))

def rootCursor : (source : TokenSource Token) → source.Cursor
  | .terminal => PUnit.unit
  | .step _ _ => .inl PUnit.unit

def cursorSource : (source : TokenSource Token) → source.Cursor → TokenSource Token
  | .terminal, _ => .terminal
  | .step p next, .inl _ => .step p next
  | .step _ next, .inr ⟨t, c⟩ => (next t).cursorSource c

def cursorHistory : (source : TokenSource Token) → source.Cursor → List Token
  | .terminal, _ => []
  | .step _ _, .inl _ => []
  | .step _ next, .inr ⟨t, c⟩ => t :: (next t).cursorHistory c

def LegalCursor : (source : TokenSource Token) → source.Cursor → Prop
  | .terminal, _ => True
  | .step _ _, .inl _ => True
  | .step p next, .inr ⟨t, c⟩ => 0 < p.prob t ∧ (next t).LegalCursor c

noncomputable def cursorLaw (word : Token → List Byte) :
    (source : TokenSource Token) → source.Cursor → FiniteLaw source.decisionTree.Leaves
  | .terminal, _ => FiniteLaw.point PUnit.unit
  | .step p next, .inl _ => (step p next).sourceLaw word
  | .step p next, .inr ⟨t, c⟩ => ((next t).cursorLaw word c).mix
      (fun l => FiniteLaw.point ⟨NativeGrouping.label word p t, ⟨t, l⟩⟩)

theorem cursorLaw_root (word : Token → List Byte) (source : TokenSource Token) :
    source.cursorLaw word source.rootCursor = source.sourceLaw word := by
  cases source <;> rfl

theorem rootCursor_legal (source : TokenSource Token) : source.LegalCursor source.rootCursor := by
  cases source <;> trivial

theorem cursorHistory_injective (source : TokenSource Token) :
    Function.Injective source.cursorHistory := by
  induction source with
  | terminal => exact fun _ _ _ => Subsingleton.elim _ _
  | step p next ih =>
    intro c d h
    cases c with
    | inl c =>
      cases d with
      | inl d => exact congrArg Sum.inl (Subsingleton.elim _ _)
      | inr d => cases h
    | inr c =>
      cases d with
      | inl d => cases h
      | inr d =>
        rcases c with ⟨t, c⟩
        rcases d with ⟨u, d⟩
        have he := List.cons.inj h
        obtain rfl := he.1
        have hd := ih t he.2
        subst d
        rfl

theorem cursorLaw_support (word : Token → List Byte) (source : TokenSource Token)
    (c : source.Cursor) (hc : source.LegalCursor c) (l : source.decisionTree.Leaves)
    (hl : 0 < (source.cursorLaw word c).prob l) : 0 < (source.sourceLaw word).prob l := by
  classical
  induction source with
  | terminal => exact hl
  | step p next ih =>
    cases c with
    | inl _ => exact hl
    | inr c =>
      rcases c with ⟨t, c⟩
      obtain ⟨m, hm, heq⟩ := (FiniteLaw.mix_positive_iff _ _ l).mp hl
      have he : l = ⟨NativeGrouping.label word p t, ⟨t, m⟩⟩ := by
        by_contra hn
        simp [FiniteLaw.point, hn] at heq
      subst l
      exact (sourceLaw_step_positive word p next _ t m).mpr
        ⟨rfl, hc.1, ih t c hc.2 m hm⟩

theorem cursorLaw_prefix (word : Token → List Byte) (source : TokenSource Token)
    (c : source.Cursor) (l : source.decisionTree.Leaves)
    (hl : 0 < (source.cursorLaw word c).prob l) : source.cursorHistory c <+: source.tokenPath l := by
  classical
  induction source with
  | terminal => simp [cursorHistory, tokenPath]
  | step p next ih =>
    cases c with
    | inl _ => simp [cursorHistory]
    | inr c =>
      rcases c with ⟨t, c⟩
      obtain ⟨m, hm, heq⟩ := (FiniteLaw.mix_positive_iff _ _ l).mp hl
      have he : l = ⟨NativeGrouping.label word p t, ⟨t, m⟩⟩ := by
        by_contra hn
        simp [FiniteLaw.point, hn] at heq
      subst l
      exact List.cons_prefix_cons.mpr ⟨rfl, ih t c m hm⟩

theorem terminal_cursor_resolved (word : Token → List Byte) (source : TokenSource Token)
    (c : source.Cursor) (ht : source.cursorSource c = terminal) :
    ∃ l, source.cursorLaw word c = FiniteLaw.point l ∧ source.cursorHistory c = source.tokenPath l := by
  induction source with
  | terminal => exact ⟨PUnit.unit, rfl, rfl⟩
  | step p next ih =>
    cases c with
    | inl _ => cases ht
    | inr c =>
      rcases c with ⟨t, c⟩
      obtain ⟨l, hl, hh⟩ := ih t c ht
      refine ⟨⟨NativeGrouping.label word p t, ⟨t, l⟩⟩, ?_, ?_⟩
      · simp only [cursorLaw, hl, FiniteLaw.point_mix]
      · simp only [cursorHistory, tokenPath, hh]

/-- An active source node expands using exactly its own token law. -/
theorem cursorLaw_step (word : Token → List Byte) (p : FiniteLaw Token)
    (next : Token → TokenSource Token) :
    (step p next).sourceLaw word = p.mix (fun t =>
      (step p next).cursorLaw word (.inr ⟨t, (next t).rootCursor⟩)) := by
  apply FiniteLaw.ext_expect
  intro f
  change (DecisionTree.groupedLaw (fun t => (next t).decisionTree) p
    (NativeGrouping.label word p) (fun t => (next t).sourceLaw word)).expect f = _
  rw [DecisionTree.groupedLaw_expect, FiniteLaw.expect_mix]
  congr 1
  funext t
  simp only [cursorLaw, cursorLaw_root, FiniteLaw.expect_mix, FiniteLaw.expect_point]

end TokenSource

end StegaTextProofs
