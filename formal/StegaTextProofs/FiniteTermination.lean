module

public import StegaTextProofs.LASSelectiveComparison

/-!
Finite termination from a decreasing natural rank. Only positive-probability
transitions matter. A normalized complete-path posterior and visible prefix
compatibility give an explicit rank: maximum output length minus emitted
length. Strict visible progress then eliminates the terminal correction.
-/

@[expose] public section

namespace StegaTextProofs

open scoped BigOperators

namespace FiniteLaw

variable {α : Type*} [Fintype α]

theorem exists_positive (p : FiniteLaw α) : ∃ a, 0 < p.prob a := by
  by_contra h
  push Not at h
  have hz : ∑ a, p.prob a = 0 := Finset.sum_eq_zero (fun a _ =>
    le_antisymm (h a) (p.nonneg a))
  linarith [p.sum_one]

theorem expect_zero_of_support (p : FiniteLaw α) (f : α → ℝ)
    (hf : ∀ a, 0 < p.prob a → f a = 0) : p.expect f = 0 := by
  apply Finset.sum_eq_zero
  intro a _
  by_cases ha : 0 < p.prob a
  · rw [hf a ha, mul_zero]
  · rw [le_antisymm (le_of_not_gt ha) (p.nonneg a), zero_mul]

end FiniteLaw

namespace FiniteHorizon

variable {State : Type*} [Fintype State]

theorem after_zero_of_terminal (kernel : Kernel State) (terminal : State → Prop)
    (f : State → ℝ)
    (hclosed : ∀ s t, terminal s → 0 < (kernel s).prob t → terminal t)
    (hzero : ∀ s, terminal s → f s = 0) (n : ℕ) (s : State) (hs : terminal s) :
    after kernel n f s = 0 := by
  induction n generalizing s with
  | zero => exact hzero s hs
  | succ n ih =>
    exact (kernel s).expect_zero_of_support _ (fun t ht => ih t (hclosed s t hs ht))

theorem rank_zero_terminal (kernel : Kernel State) (terminal : State → Prop)
    (rank : State → ℕ)
    (hdecrease : ∀ s t, ¬ terminal s → 0 < (kernel s).prob t → rank t < rank s)
    (s : State) (hs : rank s = 0) : terminal s := by
  by_contra hn
  obtain ⟨t, ht⟩ := (kernel s).exists_positive
  have hd := hdecrease s t hn ht
  omega

/-- The zero terminal expectation is a conclusion, not a hypothesis. -/
theorem after_zero_of_rank (kernel : Kernel State) (terminal : State → Prop)
    (rank : State → ℕ) (f : State → ℝ)
    (hdecrease : ∀ s t, ¬ terminal s → 0 < (kernel s).prob t → rank t < rank s)
    (hclosed : ∀ s t, terminal s → 0 < (kernel s).prob t → terminal t)
    (hzero : ∀ s, terminal s → f s = 0) (n : ℕ) (s : State) (hs : rank s ≤ n) :
    after kernel n f s = 0 := by
  induction n generalizing s with
  | zero =>
    exact hzero s (rank_zero_terminal kernel terminal rank hdecrease s (by omega))
  | succ n ih =>
    by_cases hterm : terminal s
    · exact after_zero_of_terminal kernel terminal f hclosed hzero (n + 1) s hterm
    · apply (kernel s).expect_zero_of_support
      intro t ht
      exact ih t (by have := hdecrease s t hterm ht; omega)

end FiniteHorizon

namespace VisibleTermination

variable {State Leaf Byte : Type*} [Fintype Leaf]

noncomputable def maxLength (output : Leaf → List Byte) : ℕ :=
  Finset.univ.sup (fun l => (output l).length)

theorem length_le_maxLength (output : Leaf → List Byte) (l : Leaf) :
    (output l).length ≤ maxLength output :=
  Finset.le_sup (f := fun l => (output l).length) (Finset.mem_univ l)

/-- A normalized posterior supplies at least one compatible complete path. -/
theorem emitted_length_le (belief : State → FiniteLaw Leaf)
    (output : Leaf → List Byte) (emitted : State → List Byte)
    (hprefix : ∀ s l, 0 < (belief s).prob l → emitted s <+: output l) (s : State) :
    (emitted s).length ≤ maxLength output := by
  obtain ⟨l, hl⟩ := (belief s).exists_positive
  exact (hprefix s l hl).length_le.trans (length_le_maxLength output l)

theorem rank_decreases (belief : State → FiniteLaw Leaf)
    (output : Leaf → List Byte) (emitted : State → List Byte)
    (hprefix : ∀ s l, 0 < (belief s).prob l → emitted s <+: output l)
    (s t : State) (hprogress : (emitted s).length < (emitted t).length) :
    maxLength output - (emitted t).length < maxLength output - (emitted s).length := by
  have ht := emitted_length_le belief output emitted hprefix t
  omega

theorem length_progress_of_append (emitted : State → List Byte) (s t : State)
    (key : List Byte) (hne : key ≠ []) (happend : emitted t = emitted s ++ key) :
    (emitted s).length < (emitted t).length := by
  rw [happend, List.length_append]
  have : 0 < key.length := List.length_pos_iff.mpr hne
  omega

/-- An explicit bound from finite complete outputs and nonempty visible progress. -/
theorem after_zero_of_visible_progress [Fintype State] (kernel : FiniteHorizon.Kernel State)
    (terminal : State → Prop) (belief : State → FiniteLaw Leaf)
    (output : Leaf → List Byte) (emitted : State → List Byte) (f : State → ℝ)
    (hprefix : ∀ s l, 0 < (belief s).prob l → emitted s <+: output l)
    (hprogress : ∀ s t, ¬ terminal s → 0 < (kernel s).prob t →
      (emitted s).length < (emitted t).length)
    (hclosed : ∀ s t, terminal s → 0 < (kernel s).prob t → terminal t)
    (hzero : ∀ s, terminal s → f s = 0) (s : State) :
    FiniteHorizon.after kernel (maxLength output - (emitted s).length) f s = 0 := by
  exact FiniteHorizon.after_zero_of_rank kernel terminal
    (fun s => maxLength output - (emitted s).length) f
    (fun s t hs ht => rank_decreases belief output emitted hprefix s t
      (hprogress s t hs ht)) hclosed hzero _ s le_rfl

end VisibleTermination

end StegaTextProofs
