module

public import StegaTextProofs.SelectiveEntropy

/-!
A token draw can be represented by its deterministic group label followed
by the token. Inconsistent group/token pairs have zero mass. This module
proves normalization, preservation of expectations, and the payload entropy
recursion for this expansion, including arbitrary child continuation laws.
-/

@[expose] public section

namespace StegaTextProofs

open scoped BigOperators

namespace DecisionTree

variable {Branch : Type*} [Fintype Branch]

theorem selective_smul (selected : Bool) (tree : DecisionTree Branch)
    (c : ℝ) (w : tree.Leaves → ℝ) :
    tree.selective selected (fun l => c * w l) = c * tree.selective selected w := by
  induction tree with
  | leaf => simp [selective]
  | node tag children ih =>
    simp only [selective, ← Finset.mul_sum, weightedEntropy_smul, ih]
    split_ifs <;> ring

theorem selective_zero (selected : Bool) (tree : DecisionTree Branch) :
    tree.selective selected (fun _ => 0) = 0 := by
  simpa using tree.selective_smul selected 0 (fun _ => 0)

@[reducible] def grouped (children : Branch → DecisionTree Branch) : DecisionTree Branch :=
  .node true (fun _ => .node false children)

noncomputable def groupedLaw (children : Branch → DecisionTree Branch)
    (p : FiniteLaw Branch) (group : Branch → Branch)
    (q : (t : Branch) → FiniteLaw (children t).Leaves) :
    FiniteLaw (grouped children).Leaves := by
  classical
  exact {
    prob := fun l => (if group l.2.1 = l.1 then p.prob l.2.1 else 0) *
      (q l.2.1).prob l.2.2
    nonneg := fun l => mul_nonneg (by split_ifs; exact p.nonneg _; exact le_rfl)
      ((q l.2.1).nonneg l.2.2)
    sum_one := by
      change (∑ l : (_ : Branch) × (t : Branch) × (children t).Leaves,
        (if group l.2.1 = l.1 then p.prob l.2.1 else 0) * (q l.2.1).prob l.2.2) = 1
      simp only [Fintype.sum_sigma]
      calc
        _ = ∑ g, ∑ t, if group t = g then p.prob t else 0 := by
          apply Finset.sum_congr rfl
          intro g _
          apply Finset.sum_congr rfl
          intro t _
          rw [← Finset.mul_sum, (q t).sum_one, mul_one]
        _ = 1 := by rw [Finset.sum_comm]; simpa using p.sum_one
  }

theorem groupedLaw_expect (children : Branch → DecisionTree Branch)
    (p : FiniteLaw Branch) (group : Branch → Branch)
    (q : (t : Branch) → FiniteLaw (children t).Leaves)
    (f : (grouped children).Leaves → ℝ) :
    (groupedLaw children p group q).expect f =
      p.expect (fun t => (q t).expect (fun l => f ⟨group t, ⟨t, l⟩⟩)) := by
  classical
  simp only [FiniteLaw.expect, groupedLaw, Fintype.sum_sigma]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro t _
  rw [Finset.sum_comm]
  simp [ite_mul, mul_assoc, Finset.mul_sum]

theorem groupedLaw_selective (children : Branch → DecisionTree Branch)
    (p : FiniteLaw Branch) (group : Branch → Branch)
    (q : (t : Branch) → FiniteLaw (children t).Leaves) :
    (grouped children).selective true (groupedLaw children p group q).prob =
      (p.mix (fun t => FiniteLaw.point (group t))).entropyNat +
        p.expect (fun t => (children t).selective true (q t).prob) := by
  classical
  simp only [grouped, selective, Bool.false_eq_true, ↓reduceIte, zero_add,
    groupedLaw, Fintype.sum_sigma, ← Finset.mul_sum,
    selective_smul]
  congr 1
  · rw [← weightedEntropy_law]
    congr 1
    funext g
    simp only [FiniteLaw.mix, FiniteLaw.point]
    apply Finset.sum_congr rfl
    intro t _
    split_ifs <;> simp_all [ (q t).sum_one]
  · rw [Finset.sum_comm]
    simp [FiniteLaw.expect, ite_mul]

end DecisionTree

end StegaTextProofs
