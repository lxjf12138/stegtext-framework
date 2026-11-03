module

public import StegaTextProofs.LASSelectiveComparison
public import StegaTextProofs.FrontierSeparation

/-!
The A/AB, then BA/BC example. The root synchronization choice distinguishes
A from AB. On the A branch one payload choice distinguishes BA from BC.
The source law is AB:1/2, ABA:1/4, ABC:1/4. A zero-weight dummy leaf pads
the AB branch to the same binary shape and contributes no entropy.
-/

@[expose] public section

namespace StegaTextProofs.SelectiveExamples

open scoped BigOperators

def mergerTree : DecisionTree Bool :=
  .node false (fun _ => .node true (fun _ => .leaf))

noncomputable def mergerWeights : mergerTree.Leaves → ℝ := fun leaf =>
  if leaf.1 then 1 / 4 else if leaf.2.1 then 0 else 1 / 2

theorem merger_weights_normalized : ∑ l, mergerWeights l = 1 := by
  change (∑ l : (_ : Bool) × (_ : Bool) × PUnit,
    if l.1 then (1 : ℝ) / 4 else if l.2.1 then 0 else 1 / 2) = 1
  norm_num [Fintype.sum_sigma]

/-- The missing half-bit remains in the source-tree potential even after both
fresh token branches have become retained completed-token histories. -/
theorem merger_pending_entropy :
    mergerTree.selective true mergerWeights = Real.log 2 / 2 := by
  simp [mergerTree, DecisionTree.selective, weightedEntropy, mergerWeights,
    Real.negMulLog]
  rw [show (4 : ℝ) = 2 ^ 2 by norm_num, Real.log_pow]
  norm_num
  simp only [one_div, Real.log_inv]
  ring

theorem merger_total_entropy : weightedEntropy mergerWeights = 3 * Real.log 2 / 2 := by
  rw [← mergerTree.selective_add_complement mergerWeights, merger_pending_entropy]
  simp [mergerTree, DecisionTree.selective, weightedEntropy, mergerWeights,
    Real.negMulLog, Fintype.sum_sigma]
  norm_num
  simp only [one_div, Real.log_inv]
  ring

/-- A single selected group has zero entropy, while the original active
source supplied half a bit. This is the negative old one-step increment. -/
theorem merger_old_increment_negative :
    0 - Real.log 2 / 2 < 0 := by
  have hlog : 0 < Real.log 2 := Real.log_pos (by norm_num)
  linarith

/-- The total visible-law entropy exceeds selective native entropy by one
bit. Identifying these quantities with protocol budgets is a separate step. -/
theorem merger_entropy_gap :
    (weightedEntropy mergerWeights - mergerTree.selective true mergerWeights) /
      Real.log 2 = 1 := by
  rw [merger_total_entropy, merger_pending_entropy]
  have hlog : Real.log 2 ≠ 0 := ne_of_gt (Real.log_pos (by norm_num))
  field_simp
  ring

end StegaTextProofs.SelectiveExamples
