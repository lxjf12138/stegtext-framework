module

public import StegaTextProofs.NativeFrontierModel

/-!
The LAS kernel preserves the mean complete-path belief at every horizon.
At a resolved terminal horizon this gives the original source law for the
actual labelled token path, in addition to the entropy-budget comparison.
-/

@[expose] public section

namespace StegaTextProofs

namespace LASDecisionModel

variable {Branch State : Type*} [Fintype Branch] [Fintype State]
variable {Group : State → Type*} [∀ s, Fintype (Group s)]
variable {Exact Strict : (s : State) → Group s → Type*}
  [∀ s k, Fintype (Exact s k)] [∀ s k, Fintype (Strict s k)]
variable {tree : DecisionTree Branch}

theorem expected_belief_expect (model : LASDecisionModel tree State Group Exact Strict)
    (f : tree.Leaves → ℝ) (s : State) :
    (model.kernel s).expect (fun t => (model.belief t).expect f) = (model.belief s).expect f := by
  simp only [kernel, FiniteLaw.expect_mix, FiniteLaw.expect_point, model.posterior]
  rw [← model.disintegrate s, FiniteLaw.expect_mix]
  congr 1
  funext k
  simp only [FiniteLaw.expect_mix]
  exact (model.selected s k).las_collapse_unbiased
    (fun c => (model.continuation s k c).expect f)

theorem after_belief_expect (model : LASDecisionModel tree State Group Exact Strict)
    (f : tree.Leaves → ℝ) (n : ℕ) (s : State) :
    FiniteHorizon.after model.kernel n (fun t => (model.belief t).expect f) s =
      (model.belief s).expect f := by
  induction n generalizing s with
  | zero => rfl
  | succ n ih =>
    simp only [FiniteHorizon.after, ih, model.expected_belief_expect]

end LASDecisionModel

namespace FiniteHorizon

theorem after_sub {State : Type*} [Fintype State] (kernel : Kernel State)
    (f g : State → ℝ) (n : ℕ) (s : State) :
    after kernel n (fun t => f t - g t) s = after kernel n f s - after kernel n g s := by
  induction n generalizing s with
  | zero => rfl
  | succ n ih => simp only [after, ih, FiniteLaw.expect_sub]

end FiniteHorizon

namespace TokenSource

variable {Token Byte State : Type*} [Fintype Token] [Fintype State]
variable {Group : State → Type*} [∀ s, Fintype (Group s)]
variable {Exact Strict : (s : State) → Group s → Type*}
  [∀ s k, Fintype (Exact s k)] [∀ s k, Fintype (Strict s k)]

theorem las_path_law_preserved (word : Token → List Byte) (source : TokenSource Token)
    (model : LASDecisionModel source.decisionTree State Group Exact Strict)
    (initial : State) (hroot : model.belief initial = source.sourceLaw word)
    (f : List Token → ℝ) (n : ℕ) :
    FiniteHorizon.after model.kernel n
      (fun s => (model.belief s).expect (fun l => f (source.tokenPath l))) initial =
        source.pathExpect f := by
  rw [model.after_belief_expect, hroot, source.sourceLaw_pathExpect]

/-- At a resolved horizon, readout token paths themselves have the source law. -/
theorem terminal_readout_law (word : Token → List Byte) (source : TokenSource Token)
    (model : LASDecisionModel source.decisionTree State Group Exact Strict)
    (initial : State) (hroot : model.belief initial = source.sourceLaw word)
    (terminal : State → Prop) (rank : State → ℕ) (readout : State → List Token)
    (hdecrease : ∀ s t, ¬ terminal s → 0 < (model.kernel s).prob t → rank t < rank s)
    (hclosed : ∀ s t, terminal s → 0 < (model.kernel s).prob t → terminal t)
    (hresolved : ∀ s, terminal s → ∃ l,
      model.belief s = FiniteLaw.point l ∧ readout s = source.tokenPath l)
    (f : List Token → ℝ) :
    FiniteHorizon.after model.kernel (rank initial) (fun s => f (readout s)) initial =
      source.pathExpect f := by
  have hzero : ∀ s, terminal s →
      (model.belief s).expect (fun l => f (source.tokenPath l)) - f (readout s) = 0 := by
    intro s hs
    obtain ⟨l, hl, hr⟩ := hresolved s hs
    simp [hl, hr]
  have h := FiniteHorizon.after_zero_of_rank model.kernel terminal rank _
    hdecrease hclosed hzero (rank initial) initial le_rfl
  rw [FiniteHorizon.after_sub, source.las_path_law_preserved word model initial hroot] at h
  linarith

end TokenSource

end StegaTextProofs
