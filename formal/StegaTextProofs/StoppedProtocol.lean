module

public import StegaTextProofs.FiniteTermination

/-!
Stopping means zero reward and a point-mass self-loop at terminal states.
The stopped recursion equals the padded finite-horizon budget. A decreasing
rank proves stabilization of the budget without terminal reward assumptions
on the original, unpadded kernel.
-/

@[expose] public section

namespace StegaTextProofs.FiniteHorizon

attribute [local instance] Classical.propDecidable

variable {State : Type*} [Fintype State]

noncomputable def stoppedKernel (kernel : Kernel State) (terminal : State → Prop) :
    Kernel State := fun s => if terminal s then FiniteLaw.point s else kernel s

noncomputable def stoppedReward (reward : State → ℝ) (terminal : State → Prop) :
    State → ℝ := fun s => if terminal s then 0 else reward s

noncomputable def stoppedBudget (kernel : Kernel State) (reward : State → ℝ)
    (terminal : State → Prop) : ℕ → State → ℝ
  | 0, _ => 0
  | n + 1, s => if terminal s then 0 else
      reward s + (kernel s).expect (fun t => stoppedBudget kernel reward terminal n t)

theorem stoppedBudget_terminal (kernel : Kernel State) (reward : State → ℝ)
    (terminal : State → Prop) (n : ℕ) (s : State) (hs : terminal s) :
    stoppedBudget kernel reward terminal n s = 0 := by
  cases n <;> simp [stoppedBudget, hs]

theorem stoppedBudget_eq_budget (kernel : Kernel State) (reward : State → ℝ)
    (terminal : State → Prop) (n : ℕ) (s : State) :
    stoppedBudget kernel reward terminal n s =
      budget (stoppedKernel kernel terminal) (stoppedReward reward terminal) n s := by
  classical
  induction n generalizing s with
  | zero => rfl
  | succ n ih =>
    by_cases hs : terminal s
    · simp [stoppedBudget, budget, stoppedKernel, stoppedReward, hs, ← ih,
        stoppedBudget_terminal kernel reward terminal n s hs]
    · simp [stoppedBudget, budget, stoppedKernel, stoppedReward, hs, ih]

theorem stoppedBudget_smul (kernel : Kernel State) (reward : State → ℝ)
    (terminal : State → Prop) (c : ℝ) (n : ℕ) (s : State) :
    stoppedBudget kernel (fun t => c * reward t) terminal n s =
      c * stoppedBudget kernel reward terminal n s := by
  induction n generalizing s with
  | zero => simp [stoppedBudget]
  | succ n ih =>
    by_cases hs : terminal s
    · simp [stoppedBudget, hs]
    · simp only [stoppedBudget, hs, ↓reduceIte, ih, FiniteLaw.expect_mul]
      ring

theorem stoppedKernel_positive (kernel : Kernel State) (terminal : State → Prop)
    (s t : State) :
    0 < (stoppedKernel kernel terminal s).prob t ↔
      (terminal s ∧ t = s) ∨ (¬ terminal s ∧ 0 < (kernel s).prob t) := by
  classical
  by_cases hs : terminal s
  · by_cases ht : t = s <;> simp [stoppedKernel, hs, FiniteLaw.point, ht]
  · simp [stoppedKernel, hs]

/-- Once the rank bound is reached, the expected stopped reward is constant. -/
theorem stoppedBudget_stable (kernel : Kernel State) (reward : State → ℝ)
    (terminal : State → Prop) (rank : State → ℕ)
    (hdecrease : ∀ s t, ¬ terminal s → 0 < (kernel s).prob t → rank t < rank s)
    (n m : ℕ) (s : State) (hs : rank s ≤ n) :
    stoppedBudget kernel reward terminal (n + m) s =
      stoppedBudget kernel reward terminal n s := by
  classical
  induction n generalizing s with
  | zero =>
    have ht := rank_zero_terminal kernel terminal rank hdecrease s (by omega)
    simp [stoppedBudget_terminal kernel reward terminal _ s ht]
  | succ n ih =>
    by_cases ht : terminal s
    · simp only [stoppedBudget_terminal kernel reward terminal _ s ht]
    · rw [show n + 1 + m = (n + m) + 1 by omega]
      simp only [stoppedBudget, ht, ↓reduceIte]
      congr 1
      apply sub_eq_zero.mp
      rw [← FiniteLaw.expect_sub]
      apply (kernel s).expect_zero_of_support
      intro t hprob
      rw [ih t (by have := hdecrease s t ht hprob; omega), sub_self]

/-- The comparison accounts only for rewards earned before stopping. -/
theorem potential_le_stoppedBudget (kernel : Kernel State) (reward potential : State → ℝ)
    (terminal : State → Prop) (rank : State → ℕ)
    (hdecrease : ∀ s t, ¬ terminal s → 0 < (kernel s).prob t → rank t < rank s)
    (hzero : ∀ s, terminal s → potential s = 0)
    (hstep : ∀ s, ¬ terminal s → potential s ≤ reward s + (kernel s).expect potential)
    (n : ℕ) (s : State) (hs : rank s ≤ n) :
    potential s ≤ stoppedBudget kernel reward terminal n s := by
  classical
  induction n generalizing s with
  | zero =>
    rw [hzero s (rank_zero_terminal kernel terminal rank hdecrease s (by omega))]
    exact le_rfl
  | succ n ih =>
    by_cases ht : terminal s
    · rw [hzero s ht, stoppedBudget_terminal kernel reward terminal _ s ht]
    · rw [stoppedBudget, ite_eq_right ht]
      apply (hstep s ht).trans
      apply add_le_add le_rfl
      apply Finset.sum_le_sum
      intro t _
      by_cases hp : 0 < (kernel s).prob t
      · exact mul_le_mul_of_nonneg_left (ih t (by
          have := hdecrease s t ht hp; omega)) (kernel s |>.nonneg t)
      · have hz : (kernel s).prob t = 0 :=
          le_antisymm (le_of_not_gt hp) ((kernel s).nonneg t)
        simp only [hz, zero_mul, le_refl]

end StegaTextProofs.FiniteHorizon
