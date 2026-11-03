module

public import StegaTextProofs.LASNodeInvariant

/-!
A finite decision tree marks each node as payload (true) or synchronization
(false). The selective entropy is the sum of unnormalised entropies at just
one class of nodes. No independence of the leaf law is assumed.
-/

@[expose] public section

namespace StegaTextProofs

open scoped BigOperators

namespace FiniteLaw

theorem expect_sum {α ι : Type*} [Fintype α] [Fintype ι]
    (p : FiniteLaw α) (f : ι → α → ℝ) :
    p.expect (fun a => ∑ i, f i a) = ∑ i, p.expect (f i) := by
  simp only [expect, Finset.mul_sum]
  exact Finset.sum_comm

end FiniteLaw

universe u

inductive DecisionTree (Branch : Type u) where
  | leaf
  | node (payload : Bool) (children : Branch → DecisionTree Branch)

namespace DecisionTree

variable {Branch : Type u} {Index : Type*} [Fintype Branch] [Fintype Index]

@[reducible] def Leaves : DecisionTree Branch → Type u
  | .leaf => PUnit
  | .node _ children => (b : Branch) × Leaves (children b)

noncomputable instance leavesFintype (tree : DecisionTree Branch) : Fintype tree.Leaves :=
  match tree with
  | .leaf => inferInstanceAs (Fintype PUnit.{u + 1})
  | .node _ children =>
    letI := fun b => leavesFintype (children b)
    inferInstanceAs (Fintype ((b : Branch) × Leaves (children b)))

noncomputable def selective (selected : Bool) :
    (tree : DecisionTree Branch) → (tree.Leaves → ℝ) → ℝ
  | .leaf, _ => 0
  | .node tag children, w =>
    (if tag = selected then weightedEntropy (fun b => ∑ l, w ⟨b, l⟩) else 0) +
      ∑ b, selective selected (children b) (fun l => w ⟨b, l⟩)

/-- The two classes of nodes account for the entire leaf entropy. -/
theorem selective_add_complement (tree : DecisionTree Branch) (w : tree.Leaves → ℝ) :
    tree.selective true w + tree.selective false w = weightedEntropy w := by
  induction tree with
  | leaf =>
    have hz := weightedEntropy_single (α := (leaf : DecisionTree Branch).Leaves)
      PUnit.unit w (fun a ha => by cases a; exact False.elim (ha rfl))
    simpa only [selective, add_zero] using hz.symm
  | node tag children ih =>
    calc
      _ = weightedEntropy (fun b => ∑ l, w ⟨b, l⟩) +
          ∑ b, ((children b).selective true (fun l => w ⟨b, l⟩) +
            (children b).selective false (fun l => w ⟨b, l⟩)) := by
        cases tag <;> simp [selective, Finset.sum_add_distrib] <;> ring
      _ = weightedEntropy (fun b => ∑ l, w ⟨b, l⟩) +
          ∑ b, weightedEntropy (fun l => w ⟨b, l⟩) := by simp only [ih]
      _ = weightedEntropy w := (weightedEntropy_sigma_chain w).symm

theorem selective_nonneg (selected : Bool) (tree : DecisionTree Branch)
    (w : tree.Leaves → ℝ) (hw : ∀ l, 0 ≤ w l) : 0 ≤ tree.selective selected w := by
  induction tree with
  | leaf => exact le_rfl
  | node tag children ih =>
    apply add_nonneg
    · split_ifs
      · exact weightedEntropy_nonneg _ fun b => Finset.sum_nonneg fun l _ => hw ⟨b, l⟩
      · exact le_rfl
    · exact Finset.sum_nonneg fun b _ => ih b _ (fun l => hw ⟨b, l⟩)

/-- Concavity holds for subprobability vectors, not only normalized laws. -/
theorem selective_expect_le (selected : Bool) (tree : DecisionTree Branch)
    (p : FiniteLaw Index) (w : Index → tree.Leaves → ℝ) (hw : ∀ i l, 0 ≤ w i l) :
    p.expect (fun i => tree.selective selected (w i)) ≤
      tree.selective selected (fun l => p.expect (fun i => w i l)) := by
  induction tree with
  | leaf => simp [selective]
  | node tag children ih =>
    simp only [selective, FiniteLaw.expect_add, FiniteLaw.expect_sum]
    apply add_le_add
    · by_cases htag : tag = selected
      · simp only [htag, ↓reduceIte]
        simpa only [FiniteLaw.expect_sum] using weightedEntropy_expect_le p
          (fun i b => ∑ l, w i ⟨b, l⟩)
          (fun i b => Finset.sum_nonneg fun l _ => hw i ⟨b, l⟩)
      · simp [htag]
    · exact Finset.sum_le_sum fun b _ => ih b _ (fun i l => hw i ⟨b, l⟩)

/-- A mixture observation can decrease selective entropy by at most the
entropy of its label. The complementary nodes absorb the remaining decrease. -/
theorem selective_mix_le (tree : DecisionTree Branch) (p : FiniteLaw Index)
    (q : Index → FiniteLaw tree.Leaves) :
    tree.selective true (p.mix q).prob ≤ p.entropyNat +
      p.expect (fun i => tree.selective true (q i).prob) := by
  have hc := tree.selective_expect_le false p (fun i => (q i).prob)
    (fun i l => (q i).nonneg l)
  have htotal := tree.selective_add_complement (p.mix q).prob
  have heach : p.expect (fun i => (q i).entropyNat) =
      p.expect (fun i => tree.selective true (q i).prob) +
      p.expect (fun i => tree.selective false (q i).prob) := by
    rw [← p.expect_add]
    congr 1
    funext i
    rw [tree.selective_add_complement, weightedEntropy_law]
  rw [weightedEntropy_law] at htotal
  change _ ≤ tree.selective false (p.mix q).prob at hc
  have hb := p.entropyNat_mix_le q
  linarith

/-- A fully resolved terminal path has no remaining selective entropy. -/
theorem selective_point (tree : DecisionTree Branch) (l : tree.Leaves) :
    tree.selective true (FiniteLaw.point l).prob = 0 := by
  have ht := tree.selective_add_complement (FiniteLaw.point l).prob
  have hz : weightedEntropy (FiniteLaw.point l).prob = 0 := by
    apply weightedEntropy_single l
    intro k hk
    simp [FiniteLaw.point, hk]
  have hp := tree.selective_nonneg true _ (FiniteLaw.point l).nonneg
  have hc := tree.selective_nonneg false _ (FiniteLaw.point l).nonneg
  linarith

/-- At every payload node, candidate contributions satisfy one of the three
frontier separation cases. Synchronization nodes impose no such condition. -/
def Separated {Exact Strict : Type*} : (tree : DecisionTree Branch) →
    (Exact ⊕ Strict → tree.Leaves → ℝ) → Prop
  | .leaf, _ => True
  | .node tag children, channel =>
      (tag = true → NodeSeparation (fun c b => ∑ l, channel c ⟨b, l⟩)) ∧
      ∀ b, Separated (children b) (fun c l => channel c ⟨b, l⟩)

/-- The selective source-tree entropy is conserved in expectation by exact
collapse when the frontier separation property holds. -/
theorem selective_collapse_invariant {Exact Strict : Type*}
    [Fintype Exact] [Fintype Strict] (tree : DecisionTree Branch)
    (g : LASGroup Exact Strict) (channel : Exact ⊕ Strict → tree.Leaves → ℝ)
    (hsep : tree.Separated channel) :
    g.surrogate.expect (fun e => tree.selective true (nodeWeights (g.collapse e) channel)) =
      tree.selective true (nodeWeights g.prior channel) := by
  induction tree with
  | leaf => simp [selective]
  | node tag children ih =>
    simp only [selective, FiniteLaw.expect_add, FiniteLaw.expect_sum]
    congr 1
    · by_cases htag : tag = true
      · simp only [htag, ↓reduceIte, nodeWeights, ← FiniteLaw.expect_sum]
        exact g.node_entropy_collapse_invariant _ (hsep.1 htag)
      · simp [htag]
    · apply Finset.sum_congr rfl
      intro b _
      exact ih b _ (hsep.2 b)

end DecisionTree

end StegaTextProofs
