module

public import StegaTextProofs.FrontierSeparation

/-!
Structural invariants of LAS frontiers. Fresh expansion replaces the active
history by its labelled token children; retained histories do not expand.
Group selection and exact collapse take subsets of this expanded antichain.
-/

@[expose] public section

namespace StegaTextProofs

variable {Token Byte Retained Exact Strict : Type*}

def expandedHistory (base : List Token) (retained : Retained → List Token) :
    Token ⊕ Retained → List Token :=
  Sum.elim (fun token => base ++ [token]) retained

theorem history_antichain_expand (base : List Token) (retained : Retained → List Token)
    (hanti : HistoryAntichain (Sum.elim (fun _ : Unit => base) retained)) :
    HistoryAntichain (expandedHistory base retained) := by
  intro c d hp
  cases c with
  | inl t =>
    cases d with
    | inl u =>
      have ht : [t] <+: [u] := (List.prefix_append_right_inj base).mp hp
      have heq : [t] = [u] := ht.eq_of_length_le (by simp)
      simpa using heq
    | inr r =>
      have hb : base <+: retained r := (List.prefix_append base [t]).trans hp
      have heq := hanti (.inl ()) (.inr r) hb
      cases heq
  | inr r =>
    cases d with
    | inl t =>
      have hb := List.prefix_append base [t]
      rcases List.prefix_or_prefix_of_prefix hp hb with hrb | hbr
      · have heq := hanti (.inr r) (.inl ()) hrb
        cases heq
      · have heq := hanti (.inl ()) (.inr r) hbr
        cases heq
    | inr s => exact congrArg Sum.inr (Sum.inr.inj (hanti (.inr r) (.inr s) hp))

/-- Group restriction preserves the labelled-history antichain. -/
theorem history_antichain_restrict {E₁ S₁ E₂ S₂ : Type*}
    (history : E₁ ⊕ S₁ → List Token) (hanti : HistoryAntichain history)
    (select : E₂ ⊕ S₂ → E₁ ⊕ S₁) (hinj : Function.Injective select) :
    HistoryAntichain (fun c => history (select c)) := by
  intro c d hp
  exact hinj (hanti (select c) (select d) hp)

/-- After exact collapse, the chosen exact history is active and every strict
history remains retained, with no new ancestry relationships. -/
theorem history_antichain_collapse (history : Exact ⊕ Strict → List Token)
    (hanti : HistoryAntichain history) (e : Exact) :
    HistoryAntichain (Sum.elim (fun _ : Unit => history (.inl e))
      (fun r => history (.inr r))) := by
  intro c d hp
  rcases c with ⟨⟩ | r <;> rcases d with ⟨⟩ | s
  · rfl
  · have heq := hanti (.inl e) (.inr s) hp
    cases heq
  · have heq := hanti (.inr r) (.inl e) hp
    cases heq
  · exact congrArg Sum.inr (Sum.inr.inj (hanti (.inr r) (.inr s) hp))

/-- Consuming one common group key makes each exact history a visible prefix
of every surviving candidate, including the strict candidates. -/
theorem visible_boundary_of_group (word : Token → List Byte)
    (history : Exact ⊕ Strict → List Token) (residual : Exact ⊕ Strict → List Byte)
    (emitted key : List Byte)
    (hrealize : ∀ c, PrefixGrouping.render word (history c) = emitted ++ residual c)
    (hexact : ∀ e, residual (.inl e) = key)
    (hgroup : ∀ c, key <+: residual c) :
    ∀ e c, PrefixGrouping.render word (history (.inl e)) <+:
      PrefixGrouping.render word (history c) := by
  intro e c
  rw [hrealize, hrealize, hexact]
  exact (List.prefix_append_right_inj emitted).mpr (hgroup c)

end StegaTextProofs
