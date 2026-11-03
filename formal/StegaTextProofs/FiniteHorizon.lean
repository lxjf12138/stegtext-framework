module

public import StegaTextProofs.FiniteLaw

/-!
Finite-horizon expectation and potential accounting. The telescoping theorem
has no sign assumption. Later corollaries explicitly require a baseline drift
identity and a zero terminal potential. Instantiating these with the complete
LAS state machine is a separate obligation, not an axiom of this module.
-/

@[expose] public section

namespace StegaTextProofs.FiniteHorizon

variable {State : Type*} [Fintype State]

abbrev Kernel (State : Type*) [Fintype State] := State → FiniteLaw State

def after (kernel : Kernel State) : ℕ → (State → ℝ) → State → ℝ
  | 0, f, s => f s
  | n + 1, f, s => (kernel s).expect (fun t => after kernel n f t)

def budget (kernel : Kernel State) (reward : State → ℝ) : ℕ → State → ℝ
  | 0, _ => 0
  | n + 1, s => reward s + (kernel s).expect (fun t => budget kernel reward n t)

theorem budget_sub (kernel : Kernel State) (f g : State → ℝ) (n : ℕ) (s : State) :
    budget kernel (fun t => f t - g t) n s = budget kernel f n s - budget kernel g n s := by
  induction n generalizing s with
  | zero => simp [budget]
  | succ n ih =>
    simp only [budget, ih, FiniteLaw.expect_sub]
    ring

theorem budget_mono (kernel : Kernel State) {f g : State → ℝ}
    (hfg : ∀ s, f s ≤ g s) (n : ℕ) (s : State) :
    budget kernel f n s ≤ budget kernel g n s := by
  induction n generalizing s with
  | zero => exact le_rfl
  | succ n ih =>
    exact add_le_add (hfg s) ((kernel s).expect_mono ih)

@[simp] theorem budget_zero (kernel : Kernel State) (n : ℕ) (s : State) :
    budget kernel (fun _ => 0) n s = 0 := by
  induction n generalizing s with
  | zero => rfl
  | succ n ih => simp only [budget, ih, FiniteLaw.expect_const, add_zero]

/-- Finite telescoping, retaining the terminal correction rather than silently
assuming it vanishes. -/
theorem potential_telescoping (kernel : Kernel State) (potential : State → ℝ)
    (n : ℕ) (s : State) :
    budget kernel (fun t => potential t - (kernel t).expect potential) n s =
      potential s - after kernel n potential s := by
  induction n generalizing s with
  | zero => simp [budget, after]
  | succ n ih =>
    simp only [budget, ih, FiniteLaw.expect_sub, after]
    ring

/-- The baseline is accounted for by the potential drift, with no assumption
that the resulting difference is nonnegative. -/
theorem cross_protocol_finite_telescope (kernel : Kernel State)
    (reward baseline potential : State → ℝ)
    (hdrift : ∀ s, potential s = baseline s + (kernel s).expect potential)
    (n : ℕ) (s : State) :
    budget kernel reward n s - potential s + after kernel n potential s =
      budget kernel (fun t => reward t - baseline t) n s := by
  have hb : baseline = fun t => potential t - (kernel t).expect potential := by
    funext t
    linarith [hdrift t]
  rw [budget_sub, hb, potential_telescoping]
  ring

theorem finite_compensation_identity (kernel : Kernel State)
    (reward baseline potential credit loss : State → ℝ)
    (hdrift : ∀ s, potential s = baseline s + (kernel s).expect potential)
    (hlocal : ∀ s, reward s - baseline s = credit s - loss s)
    (n : ℕ) (s : State) (hterminal : after kernel n potential s = 0) :
    budget kernel reward n s - potential s =
      budget kernel (fun t => credit t - loss t) n s := by
  have h := cross_protocol_finite_telescope kernel reward baseline potential hdrift n s
  simpa only [hterminal, add_zero, hlocal] using h

/-- A local upper bound on losses yields a global lower bound. This theorem
does not assert that the lower bound is positive. -/
theorem finite_compensation_lower_bound (kernel : Kernel State)
    (reward baseline potential credit loss bound : State → ℝ)
    (hdrift : ∀ s, potential s = baseline s + (kernel s).expect potential)
    (hlocal : ∀ s, reward s - baseline s = credit s - loss s)
    (hbound : ∀ s, loss s ≤ bound s)
    (n : ℕ) (s : State) (hterminal : after kernel n potential s = 0) :
    budget kernel (fun t => credit t - bound t) n s ≤
      budget kernel reward n s - potential s := by
  rw [finite_compensation_identity kernel reward baseline potential credit loss
    hdrift hlocal n s hterminal]
  exact budget_mono kernel (fun t => sub_le_sub_left (hbound t) _) n s

end StegaTextProofs.FiniteHorizon
