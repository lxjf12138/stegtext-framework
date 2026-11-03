module

public import StegaTextProofs.SelectiveEntropy
public import StegaTextProofs.FiniteHorizon

/-!
A finite LAS transition model with an explicit complete-path belief at each
state. Its kernel is constructed from group selection and exact-surrogate
selection. The comparison inequality is derived from those transitions and
the structural node separation property; no drift or sign is assumed.

An arbitrary token-source implementation still has to be embedded into this
model, including its frontier invariants, stopping and output semantics.
-/

@[expose] public section

namespace StegaTextProofs

variable {Branch State : Type*} [Fintype Branch] [Fintype State]
variable {Group : State → Type*} [∀ s, Fintype (Group s)]
variable {Exact Strict : (s : State) → Group s → Type*}
  [∀ s k, Fintype (Exact s k)] [∀ s k, Fintype (Strict s k)]

namespace FiniteHorizon

theorem budget_smul (kernel : Kernel State) (reward : State → ℝ)
    (c : ℝ) (n : ℕ) (s : State) :
    budget kernel (fun t => c * reward t) n s = c * budget kernel reward n s := by
  induction n generalizing s with
  | zero => simp [budget]
  | succ n ih =>
    simp only [budget, ih, FiniteLaw.expect_mul]
    ring

end FiniteHorizon

structure LASDecisionModel (tree : DecisionTree Branch)
    (State : Type*) [Fintype State]
    (Group : State → Type*) [∀ s, Fintype (Group s)]
    (Exact Strict : (s : State) → Group s → Type*)
    [∀ s k, Fintype (Exact s k)] [∀ s k, Fintype (Strict s k)] where
  belief : State → FiniteLaw tree.Leaves
  groups : (s : State) → FiniteLaw (Group s)
  selected : (s : State) → (k : Group s) → LASGroup (Exact s k) (Strict s k)
  continuation : (s : State) → (k : Group s) → Exact s k ⊕ Strict s k → FiniteLaw tree.Leaves
  next : (s : State) → (k : Group s) → Exact s k → State
  disintegrate : ∀ s, (groups s).mix (fun k =>
    (selected s k).prior.mix (continuation s k)) = belief s
  posterior : ∀ s k e, belief (next s k e) =
    ((selected s k).collapse e).mix (continuation s k)
  separated : ∀ s k, tree.Separated (fun c => (continuation s k c).prob)

namespace LASDecisionModel

variable {tree : DecisionTree Branch}

noncomputable def kernel (model : LASDecisionModel tree State Group Exact Strict) :
    FiniteHorizon.Kernel State := fun s =>
  (model.groups s).mix (fun k => (model.selected s k).surrogate.mix
    (fun e => FiniteLaw.point (model.next s k e)))

noncomputable def potential (model : LASDecisionModel tree State Group Exact Strict)
    (s : State) : ℝ := tree.selective true (model.belief s).prob

noncomputable def rewardNat (model : LASDecisionModel tree State Group Exact Strict)
    (s : State) : ℝ := (model.groups s).entropyNat

/-- Exact collapse conserves the entire selective entropy potential, including
group choices that were sampled earlier and are still pending in the frontier. -/
theorem expected_potential_after_collapse
    (model : LASDecisionModel tree State Group Exact Strict) (s : State) :
    (model.kernel s).expect model.potential = (model.groups s).expect
      (fun k => tree.selective true ((model.selected s k).prior.mix
        (model.continuation s k)).prob) := by
  simp only [kernel, FiniteLaw.expect_mix, FiniteLaw.expect_point]
  congr 1
  funext k
  simp only [potential, model.posterior]
  exact tree.selective_collapse_invariant (model.selected s k)
    (fun c => (model.continuation s k c).prob) (model.separated s k)

/-- The local comparison is a conclusion of the concrete two-stage kernel. -/
theorem selective_step_inequality (model : LASDecisionModel tree State Group Exact Strict)
    (s : State) :
    model.potential s ≤ model.rewardNat s + (model.kernel s).expect model.potential := by
  rw [model.expected_potential_after_collapse]
  have h := tree.selective_mix_le (model.groups s)
    (fun k => (model.selected s k).prior.mix (model.continuation s k))
  rw [model.disintegrate] at h
  exact h

/-- The terminal correction is retained for every finite horizon. -/
theorem finite_selective_comparison (model : LASDecisionModel tree State Group Exact Strict)
    (n : ℕ) (s : State) :
    model.potential s ≤ FiniteHorizon.budget model.kernel model.rewardNat n s +
      FiniteHorizon.after model.kernel n model.potential s := by
  induction n generalizing s with
  | zero => simp [FiniteHorizon.budget, FiniteHorizon.after]
  | succ n ih =>
    calc
      model.potential s ≤ model.rewardNat s + (model.kernel s).expect model.potential :=
        model.selective_step_inequality s
      _ ≤ model.rewardNat s + (model.kernel s).expect (fun t =>
          FiniteHorizon.budget model.kernel model.rewardNat n t +
            FiniteHorizon.after model.kernel n model.potential t) :=
        add_le_add le_rfl ((model.kernel s).expect_mono ih)
      _ = _ := by
        simp only [FiniteHorizon.budget, FiniteHorizon.after, FiniteLaw.expect_add]
        ring

theorem terminating_selective_comparison
    (model : LASDecisionModel tree State Group Exact Strict) (n : ℕ) (s : State)
    (hterminal : FiniteHorizon.after model.kernel n model.potential s = 0) :
    tree.selective true (model.belief s).prob ≤
      FiniteHorizon.budget model.kernel model.rewardNat n s := by
  simpa only [hterminal, add_zero, potential] using model.finite_selective_comparison n s

theorem potential_zero_of_resolved (model : LASDecisionModel tree State Group Exact Strict)
    (s : State) (l : tree.Leaves) (hresolved : model.belief s = FiniteLaw.point l) :
    model.potential s = 0 := by
  simp only [potential, hresolved, tree.selective_point]

/-- The same finite comparison in bits, matching the native/LAS group-entropy
units. The terminal condition is explicit. -/
theorem terminating_selective_comparison_bits
    (model : LASDecisionModel tree State Group Exact Strict) (n : ℕ) (s : State)
    (hterminal : FiniteHorizon.after model.kernel n model.potential s = 0) :
    tree.selective true (model.belief s).prob / Real.log 2 ≤
      FiniteHorizon.budget model.kernel (fun t => (model.groups t).entropy) n s := by
  have hlog : 0 < Real.log 2 := Real.log_pos (by norm_num)
  have hreward : model.rewardNat = fun t => Real.log 2 * (model.groups t).entropy := by
    funext t
    dsimp [rewardNat, FiniteLaw.entropy]
    exact (mul_div_cancel₀ (model.groups t).entropyNat (ne_of_gt hlog)).symm
  have h := model.terminating_selective_comparison n s hterminal
  rw [hreward, FiniteHorizon.budget_smul] at h
  apply (div_le_iff₀ hlog).2
  simpa only [mul_comm] using h

end LASDecisionModel

end StegaTextProofs
