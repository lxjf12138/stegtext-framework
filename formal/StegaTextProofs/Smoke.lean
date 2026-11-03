module

public import Mathlib

/-!
Environment checks only. These elementary theorems do not claim correctness
of any StegaText algorithm or implementation.
-/

public section

namespace StegaTextProofs

theorem add_zero_nat (n : ℕ) : n + 0 = n := by
  simp

theorem square_nonnegative (x : ℝ) : 0 ≤ x ^ 2 := by
  exact sq_nonneg x

theorem probability_split (p : ℝ) : p + (1 - p) = 1 := by
  ring

theorem order_transitive (a b c : ℝ) (hab : a ≤ b) (hbc : b ≤ c) : a ≤ c := by
  linarith

end StegaTextProofs

#print axioms StegaTextProofs.add_zero_nat
#print axioms StegaTextProofs.square_nonnegative
#print axioms StegaTextProofs.probability_split
#print axioms StegaTextProofs.order_transitive
