module

public import StegaTextProofs.WeightedEntropy

/-!
The three structural reasons why an exact-surrogate update preserves a
native decision-node entropy. Rows can have arbitrary total masses: a
candidate may reach a source node with probability less than one.
-/

@[expose] public section

namespace StegaTextProofs

open scoped BigOperators

variable {Candidate Exact Strict Branch : Type*}
  [Fintype Candidate] [Fintype Exact] [Fintype Strict]

def nodeWeights (p : FiniteLaw Candidate) (row : Candidate → Branch → ℝ) : Branch → ℝ :=
  fun b => p.expect (fun c => row c b)

theorem nodeWeights_single (p : FiniteLaw Candidate) (row : Candidate → Branch → ℝ)
    (c₀ : Candidate) (h : ∀ c, c ≠ c₀ → ∀ b, row c b = 0) :
    nodeWeights p row = fun b => p.prob c₀ * row c₀ b := by
  classical
  funext b
  apply Finset.sum_eq_single c₀
  · intro c _ hc
    simp only [h c hc b, mul_zero]
  · simp

/-- These are structural alternatives on candidate-to-node contributions,
not a presumed entropy identity or a presumed sign of the protocol gap. -/
def NodeSeparation (row : Exact ⊕ Strict → Branch → ℝ) : Prop :=
  (∀ e b, row (.inl e) b = 0) ∨
  (∃ e, ∀ c, c ≠ .inl e → ∀ b, row c b = 0) ∨
  (∃ b₀, ∀ c b, b ≠ b₀ → row c b = 0)

namespace LASGroup

theorem nodeWeights_no_exact (g : LASGroup Exact Strict)
    (row : Exact ⊕ Strict → Branch → ℝ) (h : ∀ e b, row (.inl e) b = 0)
    (e : Exact) : nodeWeights (g.collapse e) row = nodeWeights g.prior row := by
  funext b
  unfold nodeWeights
  rw [g.collapse_expect]
  simp [FiniteLaw.expect, Fintype.sum_sum_type, h]

/-- Each exact draw preserves the row if no exact candidate can reach it;
one exact contributor is handled by homogeneity; one branch has zero entropy. -/
theorem node_entropy_collapse_invariant [Fintype Branch] (g : LASGroup Exact Strict)
    (row : Exact ⊕ Strict → Branch → ℝ) (h : NodeSeparation row) :
    g.surrogate.expect (fun e => weightedEntropy (nodeWeights (g.collapse e) row)) =
      weightedEntropy (nodeWeights g.prior row) := by
  rcases h with hzero | ⟨e₀, hsingle⟩ | ⟨b₀, hbranch⟩
  · simp only [g.nodeWeights_no_exact row hzero, FiniteLaw.expect_const]
  · simp only [nodeWeights_single _ row (.inl e₀) hsingle, weightedEntropy_smul,
      FiniteLaw.expect, ← mul_assoc, ← Finset.sum_mul, g.collapse_mean_prob]
  · have hz : ∀ p : FiniteLaw (Exact ⊕ Strict), weightedEntropy (nodeWeights p row) = 0 := by
      intro p
      apply weightedEntropy_single b₀
      intro b hb
      simp only [nodeWeights, hbranch _ b hb, FiniteLaw.expect_const]
    simp only [hz, FiniteLaw.expect_const]

end LASGroup

end StegaTextProofs
