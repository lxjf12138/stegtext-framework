module

public import StegaTextProofs.FiniteLaw
public import StegaTextProofs.PrefixGrouping

/-!
Native groups of a finite token law are the minimal supported visible
prefixes. A token realizing each key supplies a finite group label; only
positive-probability tokens participate in constructing the support.
-/

@[expose] public section

namespace StegaTextProofs.NativeGrouping

variable {Token Byte : Type*} [Fintype Token]

noncomputable def support (word : Token → List Byte) (p : FiniteLaw Token) :
    Finset (List Byte) := by
  classical
  exact (Finset.univ.filter (fun t => 0 < p.prob t)).image word

theorem mem_support (word : Token → List Byte) (p : FiniteLaw Token) (w : List Byte) :
    w ∈ support word p ↔ ∃ t, 0 < p.prob t ∧ word t = w := by
  classical
  simp [support, and_comm]

theorem token_mem_support (word : Token → List Byte) (p : FiniteLaw Token)
    (t : Token) (ht : 0 < p.prob t) : word t ∈ support word p :=
  (mem_support word p (word t)).mpr ⟨t, ht, rfl⟩

noncomputable def representative (word : Token → List Byte) (p : FiniteLaw Token)
    (w : List Byte) (hw : w ∈ support word p) : Token :=
  Classical.choose ((mem_support word p w).mp hw)

theorem representative_spec (word : Token → List Byte) (p : FiniteLaw Token)
    (w : List Byte) (hw : w ∈ support word p) :
    0 < p.prob (representative word p w hw) ∧ word (representative word p w hw) = w :=
  Classical.choose_spec ((mem_support word p w).mp hw)

noncomputable def label (word : Token → List Byte) (p : FiniteLaw Token) (t : Token) :
    Token := by
  classical
  exact if ht : 0 < p.prob t then
    representative word p
      (PrefixGrouping.key (support word p) (word t) (token_mem_support word p t ht))
      (PrefixGrouping.key_spec (support word p) (word t) (token_mem_support word p t ht)).1.1
    else t

theorem label_word (word : Token → List Byte) (p : FiniteLaw Token)
    (t : Token) (ht : 0 < p.prob t) :
    word (label word p t) =
      PrefixGrouping.key (support word p) (word t) (token_mem_support word p t ht) := by
  classical
  simp only [label, dite_eq_left ht]
  exact (representative_spec word p _ _).2

theorem label_positive (word : Token → List Byte) (p : FiniteLaw Token)
    (t : Token) (ht : 0 < p.prob t) : 0 < p.prob (label word p t) := by
  classical
  simp only [label, dite_eq_left ht]
  exact (representative_spec word p _ _).1

/-- The representative labels induce exactly the native visible-key partition. -/
theorem same_label_iff_same_key (word : Token → List Byte) (p : FiniteLaw Token)
    (t u : Token) (ht : 0 < p.prob t) (hu : 0 < p.prob u) :
    label word p t = label word p u ↔
      PrefixGrouping.key (support word p) (word t) (token_mem_support word p t ht) =
        PrefixGrouping.key (support word p) (word u) (token_mem_support word p u hu) := by
  constructor
  · intro h
    have hw := congrArg word h
    simpa only [label_word word p t ht, label_word word p u hu] using hw
  · intro h
    classical
    simp only [label, dite_eq_left ht, dite_eq_left hu]
    congr 1

end StegaTextProofs.NativeGrouping
