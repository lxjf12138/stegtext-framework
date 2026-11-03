module

public import StegaTextProofs.LASProtocolTree

/-!
Every reachable residual is a suffix of one token word. Distinct group keys
therefore lie in a fixed finite dictionary, independently of source depth.
This supplies a uniform entropy bound needed in integrable stopping arguments.
-/

@[expose] public section

namespace StegaTextProofs

open scoped BigOperators
attribute [local instance] Classical.propDecidable

namespace FiniteLaw

theorem entropyNat_le_card {α : Type*} [Fintype α] (p : FiniteLaw α) :
    p.entropyNat ≤ Fintype.card α := by
  calc
    _ ≤ ∑ _a : α, (1 : ℝ) := Finset.sum_le_sum (fun a _ =>
      (Real.negMulLog_le_one_sub_self (p.nonneg a)).trans (sub_le_self _ (p.nonneg a)))
    _ = _ := by simp

end FiniteLaw

namespace ProtocolTree

universe u v
variable {Byte : Type v}

def PayloadEntropyBound (C : ℝ) : ProtocolTree.{u} Byte → Prop
  | .leaf => True
  | @ProtocolTree.payload _ _C _ p _ next => p.entropyNat ≤ C ∧ ∀ c, PayloadEntropyBound C (next c)
  | @ProtocolTree.sync _ _C _ _ _ next => ∀ c, PayloadEntropyBound C (next c)

end ProtocolTree

namespace TokenSource

universe u v
open PrefixGrouping
variable {Token : Type u} {Byte : Type v} [Fintype Token]

noncomputable def suffixAlphabet (word : Token → List Byte) : Finset (List Byte) := by
  classical
  exact Finset.univ.biUnion (fun t => (word t).tails.toFinset)

theorem mem_suffixAlphabet (word : Token → List Byte) (r : List Byte)
    (h : ∃ t, r <:+ word t) : r ∈ suffixAlphabet word := by
  classical
  obtain ⟨t, ht⟩ := h
  exact Finset.mem_biUnion.mpr ⟨t, Finset.mem_univ t, List.mem_toFinset.mpr ((List.mem_tails r (word t)).mpr ht)⟩

namespace LASState

variable {word : Token → List Byte} {eos : Byte} {source : TokenSource Token}

def ResidualSuffix (s : LASState word eos source) : Prop :=
  ∀ c, 0 < s.prior.prob c → c ≠ s.active →
    ∃ t, (render word (source.cursorHistory c)).drop s.emitted.length <:+ word t

theorem initial_residualSuffix (word : Token → List Byte) (eos : Byte) (source : TokenSource Token) :
    (initial word eos source).ResidualSuffix := by
  intro c hc hn
  change c ≠ source.rootCursor at hn
  simp [initial, FiniteLaw.point, hn] at hc

theorem pool_residual_suffix (s : LASState word eos source) (ha : source.cursorSource s.active ≠ terminal)
    (hS : s.ResidualSuffix) (c : (source.expandActive s.prior s.active).Positive) :
    ∃ t, s.pool.residual c <:+ word t := by
  rcases (source.expandActive_positive s.prior s.active c.val).mp c.property with
    ⟨_, hc⟩ | ⟨hc, hn⟩
  · obtain ⟨t, ht⟩ := source.advance_history s.active c.val ha hc
    refine ⟨t, ?_⟩
    rw [s.fresh_residual c t ht]
  · exact hS c.val hc hn

theorem next_residualSuffix (s : LASState word eos source)
    (ha : source.cursorSource s.active ≠ terminal) (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (hS : s.ResidualSuffix) (g : s.pool.Group) (e : s.pool.Exact g) :
    (s.next ha hs hne hiso g e).ResidualSuffix := by
  intro c hc hn
  change 0 < (s.pool.nextLaw Subtype.val g e).prob c at hc
  change c ≠ e.val.val.val at hn
  rcases s.pool.nextLaw_support Subtype.val g e c hc with he | ⟨r, he⟩
  · exact False.elim (hn he)
  · subst c
    obtain ⟨t, ht⟩ := s.pool_residual_suffix ha hS r.val.val
    refine ⟨t, ?_⟩
    change (render word (source.cursorHistory r.val.val.val)).drop
      (s.emitted ++ s.pool.residual g.val).length <:+ word t
    rw [s.next_retained_render ha g e r, List.drop_left]
    exact (List.drop_suffix _ _).trans ht

theorem group_card_le_suffixAlphabet (s : LASState word eos source)
    (ha : source.cursorSource s.active ≠ terminal) (hS : s.ResidualSuffix) :
    Fintype.card s.pool.Group ≤ (suffixAlphabet word).card := by
  classical
  let f : s.pool.Group → {r // r ∈ suffixAlphabet word} := fun g =>
    ⟨s.pool.residual g.val, mem_suffixAlphabet word _ (s.pool_residual_suffix ha hS g.val)⟩
  have hf : Function.Injective f := by
    intro g h he
    have hk : s.pool.residual g.val = s.pool.residual h.val := congrArg Subtype.val he
    exact s.pool.keys_prefix_free g h (hk ▸ List.prefix_refl _)
  simpa using Fintype.card_le_of_injective f hf

theorem group_entropy_le_suffixAlphabet (s : LASState word eos source)
    (ha : source.cursorSource s.active ≠ terminal) (hS : s.ResidualSuffix) :
    s.pool.groups.entropyNat ≤ (suffixAlphabet word).card := by
  exact s.pool.groups.entropyNat_le_card.trans
    (Nat.cast_le.mpr (s.group_card_le_suffixAlphabet ha hS))

theorem protocol_payload_entropy_bound (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (n : ℕ) (s : LASState word eos source) (hS : s.ResidualSuffix) :
    (protocol hs hne hiso n s).PayloadEntropyBound (suffixAlphabet word).card := by
  induction n generalizing s with
  | zero => trivial
  | succ n ih =>
    by_cases ht : source.cursorSource s.active = terminal
    · simp only [protocol, dite_eq_left ht, ProtocolTree.PayloadEntropyBound]
    · simp only [protocol, dite_eq_right ht, ProtocolTree.PayloadEntropyBound]
      exact ⟨s.group_entropy_le_suffixAlphabet ht hS,
        fun g e => ih (s.next ht hs hne hiso g e) (s.next_residualSuffix ht hs hne hiso hS g e)⟩

theorem initial_protocol_payload_entropy_bound (word : Token → List Byte) (eos : Byte)
    (source : TokenSource Token) (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t)) (n : ℕ) :
    (protocol hs hne hiso n (initial word eos source)).PayloadEntropyBound (suffixAlphabet word).card :=
  protocol_payload_entropy_bound hs hne hiso n _ (initial_residualSuffix word eos source)

end LASState
end TokenSource
end StegaTextProofs
