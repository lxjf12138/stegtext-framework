module

public import StegaTextProofs.NativeGrouping
public import StegaTextProofs.GroupedTree

/-!
A finite token source has a separate law at every labelled history.
Its native tree inserts a payload group node and a synchronization token
node at every draw. The compiled leaf law preserves every expectation on
complete token paths, and its selective entropy is the recursive native
group-entropy budget. No equality of continuation laws is required.
-/

@[expose] public section

namespace StegaTextProofs

inductive TokenSource (Token : Type*) [Fintype Token] where
  | terminal
  | step (law : FiniteLaw Token) (next : Token → TokenSource Token)

namespace TokenSource

variable {Token Byte : Type*} [Fintype Token]

@[reducible] def decisionTree : TokenSource Token → DecisionTree Token
  | .terminal => .leaf
  | .step _ next => .grouped (fun t => (next t).decisionTree)

noncomputable def sourceLaw (word : Token → List Byte) :
    (source : TokenSource Token) → FiniteLaw source.decisionTree.Leaves
  | .terminal => FiniteLaw.point PUnit.unit
  | .step p next => DecisionTree.groupedLaw (fun t => (next t).decisionTree)
      p (NativeGrouping.label word p) (fun t => (next t).sourceLaw word)

def tokenPath : (source : TokenSource Token) → source.decisionTree.Leaves → List Token
  | .terminal, _ => []
  | .step _ next, l => l.2.1 :: (next l.2.1).tokenPath l.2.2

/-- Direct recursive expectation under the original token draws. -/
noncomputable def pathExpect : TokenSource Token → (List Token → ℝ) → ℝ
  | .terminal, f => f []
  | .step p next, f => p.expect (fun t => (next t).pathExpect (fun tail => f (t :: tail)))

/-- Native SyncPool's ideal recursive group-entropy budget, in nats. -/
noncomputable def nativeValueNat (word : Token → List Byte) : TokenSource Token → ℝ
  | .terminal => 0
  | .step p next =>
      (p.mix (fun t => FiniteLaw.point (NativeGrouping.label word p t))).entropyNat +
        p.expect (fun t => (next t).nativeValueNat word)

noncomputable def nativeValue (word : Token → List Byte) (source : TokenSource Token) : ℝ :=
  source.nativeValueNat word / Real.log 2

/-- Compilation preserves the law of complete labelled token paths. -/
theorem sourceLaw_pathExpect (word : Token → List Byte) (source : TokenSource Token)
    (f : List Token → ℝ) :
    (source.sourceLaw word).expect (fun l => f (source.tokenPath l)) = source.pathExpect f := by
  induction source generalizing f with
  | terminal => simp [sourceLaw, tokenPath, pathExpect]
  | step p next ih =>
    change (DecisionTree.groupedLaw (fun t => (next t).decisionTree) p
      (NativeGrouping.label word p) (fun t => (next t).sourceLaw word)).expect
      (fun l => f (l.2.1 :: (next l.2.1).tokenPath l.2.2)) = _
    rw [DecisionTree.groupedLaw_expect]
    exact congrArg p.expect (funext fun t => ih t (fun tail => f (t :: tail)))

/-- The root selective entropy identity is derived from the source recursion. -/
theorem sourceLaw_selective (word : Token → List Byte) (source : TokenSource Token) :
    source.decisionTree.selective true (source.sourceLaw word).prob =
      source.nativeValueNat word := by
  induction source with
  | terminal => rfl
  | step p next ih =>
    change (DecisionTree.grouped (fun t => (next t).decisionTree)).selective true
      (DecisionTree.groupedLaw _ p (NativeGrouping.label word p)
        (fun t => (next t).sourceLaw word)).prob = _
    rw [DecisionTree.groupedLaw_selective]
    simp only [nativeValueNat, ih]

theorem sourceLaw_selective_bits (word : Token → List Byte) (source : TokenSource Token) :
    source.decisionTree.selective true (source.sourceLaw word).prob / Real.log 2 =
      source.nativeValue word := by
  rw [sourceLaw_selective, nativeValue]

theorem nativeValue_step (word : Token → List Byte) (p : FiniteLaw Token)
    (next : Token → TokenSource Token) :
    (step p next).nativeValue word =
      (p.mix (fun t => FiniteLaw.point (NativeGrouping.label word p t))).entropy +
        p.expect (fun t => (next t).nativeValue word) := by
  simp only [nativeValue, nativeValueNat, FiniteLaw.entropy, div_eq_mul_inv,
    mul_comm _ (Real.log 2)⁻¹, FiniteLaw.expect_mul]
  ring

/-- Inconsistent group labels and zero-probability tokens have no leaf mass. -/
theorem sourceLaw_step_positive (word : Token → List Byte) (p : FiniteLaw Token)
    (next : Token → TokenSource Token) (g t : Token) (l : (next t).decisionTree.Leaves) :
    0 < ((step p next).sourceLaw word).prob ⟨g, ⟨t, l⟩⟩ ↔
      NativeGrouping.label word p t = g ∧ 0 < p.prob t ∧
        0 < ((next t).sourceLaw word).prob l := by
  classical
  change 0 < (if NativeGrouping.label word p t = g then p.prob t else 0) *
    ((next t).sourceLaw word).prob l ↔ _
  by_cases hg : NativeGrouping.label word p t = g
  · simp only [hg, ↓reduceIte, true_and]
    constructor
    · intro h
      rcases mul_pos_iff.mp h with h | h
      · exact h
      · linarith [p.nonneg t]
    · exact fun h => mul_pos h.1 h.2
  · simp [hg]

end TokenSource

end StegaTextProofs
