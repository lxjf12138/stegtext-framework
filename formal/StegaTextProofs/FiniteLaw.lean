module

public import Mathlib

/-!
Finite probability laws with real weights. Normalization and nonnegativity
are proved fields, not additional axioms. Expectations are finite sums.
-/

@[expose] public section

namespace StegaTextProofs

open scoped BigOperators

structure FiniteLaw (α : Type*) [Fintype α] where
  prob : α → ℝ
  nonneg : ∀ a, 0 ≤ prob a
  sum_one : ∑ a, prob a = 1

namespace FiniteLaw

variable {α β : Type*} [Fintype α] [Fintype β]

def expect (p : FiniteLaw α) (f : α → ℝ) : ℝ :=
  ∑ a, p.prob a * f a

@[simp] theorem expect_const (p : FiniteLaw α) (c : ℝ) :
    p.expect (fun _ => c) = c := by
  simp only [expect, ← Finset.sum_mul, p.sum_one, one_mul]

theorem expect_add (p : FiniteLaw α) (f g : α → ℝ) :
    p.expect (fun a => f a + g a) = p.expect f + p.expect g := by
  simp only [expect, mul_add, Finset.sum_add_distrib]

theorem expect_sub (p : FiniteLaw α) (f g : α → ℝ) :
    p.expect (fun a => f a - g a) = p.expect f - p.expect g := by
  simp only [expect, mul_sub, Finset.sum_sub_distrib]

theorem expect_mul (p : FiniteLaw α) (c : ℝ) (f : α → ℝ) :
    p.expect (fun a => c * f a) = c * p.expect f := by
  simp only [expect, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro a _
  ring

theorem expect_mono (p : FiniteLaw α) {f g : α → ℝ} (h : ∀ a, f a ≤ g a) :
    p.expect f ≤ p.expect g := by
  exact Finset.sum_le_sum fun a _ => mul_le_mul_of_nonneg_left (h a) (p.nonneg a)

noncomputable def point (a : α) : FiniteLaw α := by
  classical
  exact {
    prob := fun b => if b = a then 1 else 0
    nonneg := fun b => by split_ifs <;> norm_num
    sum_one := by simp
  }

@[simp] theorem expect_point (a : α) (f : α → ℝ) :
    (point a).expect f = f a := by
  classical
  simp [expect, point]

noncomputable def mix (p : FiniteLaw α) (q : α → FiniteLaw β) : FiniteLaw β where
  prob b := ∑ a, p.prob a * (q a).prob b
  nonneg b := Finset.sum_nonneg fun a _ => mul_nonneg (p.nonneg a) ((q a).nonneg b)
  sum_one := by
    rw [Finset.sum_comm]
    simp only [← Finset.mul_sum, sum_one, mul_one]

theorem expect_mix (p : FiniteLaw α) (q : α → FiniteLaw β) (f : β → ℝ) :
    (p.mix q).expect f = p.expect (fun a => (q a).expect f) := by
  simp only [expect, mix, Finset.sum_mul, Finset.mul_sum, mul_assoc]
  exact Finset.sum_comm

/-- Sampling a full candidate and replacing the law by its point mass is unbiased. -/
theorem belief_fullsync_unbiased (p : FiniteLaw α) (f : α → ℝ) :
    p.expect (fun a => (point a).expect f) = p.expect f := by
  simp only [expect_point]

end FiniteLaw

end StegaTextProofs
