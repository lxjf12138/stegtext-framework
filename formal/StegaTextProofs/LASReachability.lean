module

public import StegaTextProofs.LASSelectiveComparison

/-!
Positive transitions of the LAS kernel are exactly positive group and
surrogate choices leading to that state. This transfers deterministic
invariants of the protocol update to the finite Markov kernel.
-/

@[expose] public section

namespace StegaTextProofs

namespace FiniteLaw

variable {α β : Type*} [Fintype α] [Fintype β]

theorem mix_positive_iff (p : FiniteLaw α) (q : α → FiniteLaw β) (b : β) :
    0 < (p.mix q).prob b ↔ ∃ a, 0 < p.prob a ∧ 0 < (q a).prob b := by
  classical
  constructor
  · intro h
    by_contra hn
    have hz : (p.mix q).prob b = 0 := by
      apply Finset.sum_eq_zero
      intro a _
      by_cases ha : 0 < p.prob a
      · have hb : (q a).prob b = 0 := le_antisymm
          (le_of_not_gt (fun hb => hn ⟨a, ha, hb⟩)) ((q a).nonneg b)
        rw [hb, mul_zero]
      · rw [le_antisymm (le_of_not_gt ha) (p.nonneg a), zero_mul]
    linarith
  · rintro ⟨a, ha, hb⟩
    exact (mul_pos ha hb).trans_le (Finset.single_le_sum
      (fun i _ => mul_nonneg (p.nonneg i) ((q i).nonneg b)) (Finset.mem_univ a))

end FiniteLaw

namespace LASDecisionModel

variable {Branch State : Type*} [Fintype Branch] [Fintype State]
variable {Group : State → Type*} [∀ s, Fintype (Group s)]
variable {Exact Strict : (s : State) → Group s → Type*}
  [∀ s k, Fintype (Exact s k)] [∀ s k, Fintype (Strict s k)]
variable {tree : DecisionTree Branch}

theorem kernel_positive_iff (model : LASDecisionModel tree State Group Exact Strict)
    (s t : State) :
    0 < (model.kernel s).prob t ↔ ∃ k e,
      0 < (model.groups s).prob k ∧ 0 < (model.selected s k).surrogate.prob e ∧
        model.next s k e = t := by
  classical
  simp only [kernel, FiniteLaw.mix_positive_iff]
  constructor
  · rintro ⟨k, hk, e, he, ht⟩
    refine ⟨k, e, hk, he, ?_⟩
    by_contra hn
    simp [FiniteLaw.point, Ne.symm hn] at ht
  · rintro ⟨k, e, hk, he, rfl⟩
    exact ⟨k, hk, e, he, by simp [FiniteLaw.point]⟩

theorem transition_property (model : LASDecisionModel tree State Group Exact Strict)
    (P : State → State → Prop)
    (hnext : ∀ s k e, 0 < (model.groups s).prob k →
      0 < (model.selected s k).surrogate.prob e → P s (model.next s k e))
    (s t : State) (ht : 0 < (model.kernel s).prob t) : P s t := by
  obtain ⟨k, e, hk, he, rfl⟩ := (model.kernel_positive_iff s t).mp ht
  exact hnext s k e hk he

end LASDecisionModel

end StegaTextProofs
