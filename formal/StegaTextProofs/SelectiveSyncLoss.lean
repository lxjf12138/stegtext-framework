module

public import StegaTextProofs.CollapsedFrontier
public import StegaTextProofs.SelectiveEntropy

/-! The selective-potential loss of FullSync relative to exact collapse. -/

@[expose] public section

namespace StegaTextProofs

open scoped BigOperators

namespace LASGroup

variable {Exact Strict Branch : Type*}
  [Fintype Exact] [Fintype Strict] [Fintype Branch]

theorem nextFrontier_expect (g : LASGroup Exact Strict) (e : Exact)
    (f : Exact ⊕ Strict → ℝ) :
    g.nextFrontier.expect (Sum.elim (fun _ => f (.inl e)) (fun r => f (.inr r))) =
      (g.collapse e).expect f := by
  rw [g.collapse_expect]
  simp [FiniteLaw.expect, nextFrontier, Fintype.sum_sum_type]

noncomputable def selectiveSyncLoss (g : LASGroup Exact Strict) (tree : DecisionTree Branch)
    (channel : Exact ⊕ Strict → FiniteLaw tree.Leaves) : ℝ :=
  g.surrogate.expect (fun e => tree.selective true ((g.collapse e).mix channel).prob) -
    g.prior.expect (fun c => tree.selective true (channel c).prob)

theorem selectiveSyncLoss_nonneg (g : LASGroup Exact Strict) (tree : DecisionTree Branch)
    (channel : Exact ⊕ Strict → FiniteLaw tree.Leaves) :
    0 ≤ g.selectiveSyncLoss tree channel := by
  have h := g.surrogate.expect_mono (fun e =>
    tree.selective_expect_le true (g.collapse e) (fun c => (channel c).prob)
      (fun c l => (channel c).nonneg l))
  change g.surrogate.expect (fun e => (g.collapse e).expect
    (fun c => tree.selective true (channel c).prob)) ≤
      g.surrogate.expect (fun e => tree.selective true ((g.collapse e).mix channel).prob) at h
  rw [g.las_collapse_unbiased] at h
  exact sub_nonneg.mpr h

theorem selectiveSyncLoss_le_entropy (g : LASGroup Exact Strict) (tree : DecisionTree Branch)
    (channel : Exact ⊕ Strict → FiniteLaw tree.Leaves) :
    g.selectiveSyncLoss tree channel ≤ g.nextFrontier.entropyNat := by
  have hpoint : ∀ e, tree.selective true ((g.collapse e).mix channel).prob ≤
      g.nextFrontier.entropyNat + (g.collapse e).expect
        (fun c => tree.selective true (channel c).prob) := by
    intro e
    have h := tree.selective_mix_le g.nextFrontier
      (Sum.elim (fun _ => channel (.inl e)) (fun r => channel (.inr r)))
    rw [g.nextFrontier_channel] at h
    have he : g.nextFrontier.expect (fun c => tree.selective true
        ((Sum.elim (fun _ => channel (.inl e)) (fun r => channel (.inr r))) c).prob) =
          (g.collapse e).expect (fun c => tree.selective true (channel c).prob) := by
      convert g.nextFrontier_expect e (fun c => tree.selective true (channel c).prob) using 1
      congr 1
      funext c
      cases c <;> rfl
    rw [he] at h
    exact h
  have h := g.surrogate.expect_mono hpoint
  rw [FiniteLaw.expect_add, FiniteLaw.expect_const, g.las_collapse_unbiased] at h
  exact sub_le_iff_le_add.mpr (by simpa only [add_comm] using h)

theorem selectiveSyncLoss_eq_gap (g : LASGroup Exact Strict) (tree : DecisionTree Branch)
    (channel : Exact ⊕ Strict → FiniteLaw tree.Leaves)
    (hsep : tree.Separated (fun c => (channel c).prob)) :
    g.selectiveSyncLoss tree channel = tree.selective true (g.prior.mix channel).prob -
      g.prior.expect (fun c => tree.selective true (channel c).prob) := by
  unfold selectiveSyncLoss
  have h := tree.selective_collapse_invariant g (fun c => (channel c).prob) hsep
  change g.surrogate.expect (fun e => tree.selective true ((g.collapse e).mix channel).prob) =
    tree.selective true (g.prior.mix channel).prob at h
  rw [h]

end LASGroup

end StegaTextProofs
