module

public import StegaTextProofs.AdaptiveLASRun
public import StegaTextProofs.SelectiveSyncLoss

/-! Accumulated native selective-potential losses along the adaptive run. -/

@[expose] public section

namespace StegaTextProofs.TokenSource

attribute [local instance] Classical.propDecidable
open PrefixGrouping
variable {Token Byte : Type*} [Fintype Token]

namespace LASState

variable {word : Token → List Byte} {eos : Byte} {source : TokenSource Token}

noncomputable def groupLoss (s : LASState word eos source) (g : s.pool.Group) : ℝ :=
  (s.pool.selected g).selectiveSyncLoss source.decisionTree
    (fun c => source.cursorLaw word (s.pool.candidate g c).val)

noncomputable def groupUncertaintyNat (s : LASState word eos source) (g : s.pool.Group) : ℝ :=
  (s.pool.selected g).nextFrontier.entropyNat

theorem groupLoss_nonneg (s : LASState word eos source) (g : s.pool.Group) :
    0 ≤ s.groupLoss g :=
  (s.pool.selected g).selectiveSyncLoss_nonneg _ _

theorem groupLoss_le_uncertainty (s : LASState word eos source) (g : s.pool.Group) :
    s.groupLoss g ≤ s.groupUncertaintyNat g :=
  (s.pool.selected g).selectiveSyncLoss_le_entropy _ _

theorem groupLoss_eq_native_gap (s : LASState word eos source)
    (ha : source.cursorSource s.active ≠ terminal) (g : s.pool.Group) :
    s.groupLoss g = source.decisionTree.selective true
      ((s.pool.selected g).prior.mix
        (fun c => source.cursorLaw word (s.pool.candidate g c).val)).prob -
      (s.pool.selected g).prior.expect (fun c => source.decisionTree.selective true
        (source.cursorLaw word (s.pool.candidate g c).val).prob) := by
  have he := s.pool.expected_selective_after_collapse word source
    (fun c => source.cursorHistory c.val) s.emitted (fun c => source.cursorLaw word c.val)
    (fun c d hp => Subtype.ext (source.expandActive_antichain s.prior s.active ha s.antichain
      c.val d.val c.property d.property hp)) (s.pool_render ha)
    (fun c l hl => source.cursorLaw_support word c.val
      (source.expandActive_legal s.prior s.active s.legal _ c.property) l
      (lt_of_le_of_ne (source.cursorLaw word c.val |>.nonneg l) (Ne.symm hl)))
    (fun c l hl => source.cursorLaw_prefix word c.val l
      (lt_of_le_of_ne (source.cursorLaw word c.val |>.nonneg l) (Ne.symm hl))) g
  unfold groupLoss LASGroup.selectiveSyncLoss
  change _ - _ = _ - _
  exact congrArg (fun x => x - (s.pool.selected g).prior.expect
    (fun c => source.decisionTree.selective true
      (source.cursorLaw word (s.pool.candidate g c).val).prob)) he

variable (hs : source.EOSSource word eos) (hne : ∀ t, word t ≠ [])
  (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))

theorem groupLoss_eq_transitions (s : LASState word eos source)
    (ha : source.cursorSource s.active ≠ terminal) (g : s.pool.Group) :
    s.groupLoss g = (s.pool.selected g).surrogate.expect
      (fun e => (s.next ha hs hne hiso g e).potential) -
      (s.pool.selected g).prior.expect (fun c => (s.fullSync g c).potential) := by
  simp only [groupLoss, LASGroup.selectiveSyncLoss, potential,
    next_posterior, PrefixPool.posterior, fullSync_posterior]

variable (gate : AdaptiveGate word eos source)

noncomputable def stepLoss (s : LASState word eos source) : ℝ :=
  s.pool.groups.expect (fun g => if gate s g then 0 else s.groupLoss g)

noncomputable def stepUncertainty (s : LASState word eos source) : ℝ :=
  s.pool.groups.expect (fun g => if gate s g then 0 else s.groupUncertaintyNat g)

theorem stepLoss_nonneg (s : LASState word eos source) : 0 ≤ stepLoss gate s := by
  have h := s.pool.groups.expect_mono (fun g => show (0 : ℝ) ≤
      (if gate s g then 0 else s.groupLoss g) by split <;> simp_all [groupLoss_nonneg])
  simpa only [stepLoss, FiniteLaw.expect_const] using h

theorem stepLoss_le_uncertainty (s : LASState word eos source) :
    stepLoss gate s ≤ stepUncertainty gate s := by
  apply FiniteLaw.expect_mono
  intro g
  split
  · exact le_rfl
  · exact s.groupLoss_le_uncertainty g

theorem potential_adaptive_step (s : LASState word eos source)
    (ha : source.cursorSource s.active ≠ terminal) :
    s.potential ≤ s.pool.groups.entropyNat +
      adaptiveStep hs hne hiso gate s ha potential + stepLoss gate s := by
  have he : adaptiveStep hs hne hiso gate s ha potential + stepLoss gate s =
      s.pool.groups.expect (fun g => (s.pool.selected g).surrogate.expect
        (fun e => (s.next ha hs hne hiso g e).potential)) := by
    unfold adaptiveStep stepLoss
    rw [← FiniteLaw.expect_add]
    congr 1
    funext g
    unfold groupExpect
    split
    · simp only [add_zero]
    · rw [groupLoss_eq_transitions hs hne hiso s ha g]
      ring
  rw [add_assoc, he]
  exact s.potential_step ha hs hne hiso

/-- Actual expected cumulative state reward, stopped at the source terminal. -/
noncomputable def adaptiveSum (reward : LASState word eos source → ℝ) :
    ℕ → LASState word eos source → ℝ
  | 0, _ => 0
  | n + 1, s => if ha : source.cursorSource s.active = terminal then 0
      else reward s + adaptiveStep hs hne hiso gate s ha (fun t => adaptiveSum reward n t)

theorem adaptiveSum_terminal (reward : LASState word eos source → ℝ)
    (n : ℕ) (s : LASState word eos source) (ht : source.cursorSource s.active = terminal) :
    adaptiveSum hs hne hiso gate reward n s = 0 := by
  cases n <;> simp [adaptiveSum, ht]

theorem adaptiveSum_add (f k : LASState word eos source → ℝ)
    (n : ℕ) (s : LASState word eos source) :
    adaptiveSum hs hne hiso gate (fun t => f t + k t) n s =
      adaptiveSum hs hne hiso gate f n s + adaptiveSum hs hne hiso gate k n s := by
  induction n generalizing s with
  | zero => simp [adaptiveSum]
  | succ n ih =>
    by_cases ht : source.cursorSource s.active = terminal
    · simp only [adaptiveSum, dite_eq_left ht, add_zero]
    · simp only [adaptiveSum, dite_eq_right ht, ih, adaptiveStep_add]
      ring

theorem adaptiveSum_mono (f k : LASState word eos source → ℝ) (h : ∀ t, f t ≤ k t)
    (n : ℕ) (s : LASState word eos source) :
    adaptiveSum hs hne hiso gate f n s ≤ adaptiveSum hs hne hiso gate k n s := by
  induction n generalizing s with
  | zero => exact le_rfl
  | succ n ih =>
    by_cases ht : source.cursorSource s.active = terminal
    · simp only [adaptiveSum, dite_eq_left ht, le_refl]
    · simp only [adaptiveSum, dite_eq_right ht]
      exact add_le_add (h s) (adaptiveStep_mono_rank hs hne hiso gate s ht _ _ (fun t _ => ih t))

theorem adaptiveSum_mul (a : ℝ) (f : LASState word eos source → ℝ)
    (n : ℕ) (s : LASState word eos source) :
    adaptiveSum hs hne hiso gate (fun t => a * f t) n s =
      a * adaptiveSum hs hne hiso gate f n s := by
  induction n generalizing s with
  | zero => simp [adaptiveSum]
  | succ n ih =>
    by_cases ht : source.cursorSource s.active = terminal
    · simp only [adaptiveSum, dite_eq_left ht, mul_zero]
    · simp only [adaptiveSum, dite_eq_right ht, ih, adaptiveStep_mul]
      ring

theorem adaptiveSum_stable (reward : LASState word eos source → ℝ)
    (n m : ℕ) (s : LASState word eos source) (hn : s.rank ≤ n) :
    adaptiveSum hs hne hiso gate reward (n + m) s = adaptiveSum hs hne hiso gate reward n s := by
  induction n generalizing s with
  | zero =>
    have ht : source.cursorSource s.active = terminal := by
      by_contra ha
      have := s.rank_positive ha hne
      omega
    rw [adaptiveSum_terminal hs hne hiso gate reward (0 + m) s ht]
    rfl
  | succ n ih =>
    by_cases ht : source.cursorSource s.active = terminal
    · rw [adaptiveSum_terminal hs hne hiso gate reward (n + 1 + m) s ht,
        adaptiveSum_terminal hs hne hiso gate reward (n + 1) s ht]
    · rw [Nat.succ_add]
      simp only [adaptiveSum, dite_eq_right ht]
      congr 1
      apply adaptiveStep_congr_rank
      intro t hlt
      apply ih
      omega

noncomputable def adaptiveBudgetNat (n : ℕ) (s : LASState word eos source) : ℝ :=
  adaptiveSum hs hne hiso gate (fun t => t.pool.groups.entropyNat) n s

noncomputable def adaptiveLossNat (n : ℕ) (s : LASState word eos source) : ℝ :=
  adaptiveSum hs hne hiso gate (stepLoss gate) n s

noncomputable def adaptiveUncertaintyNat (n : ℕ) (s : LASState word eos source) : ℝ :=
  adaptiveSum hs hne hiso gate (stepUncertainty gate) n s

theorem potential_le_adaptive_sum (n : ℕ) (s : LASState word eos source) (hn : s.rank ≤ n) :
    s.potential ≤ adaptiveSum hs hne hiso gate
      (fun t => t.pool.groups.entropyNat + stepLoss gate t) n s := by
  induction n generalizing s with
  | zero =>
    have ht : source.cursorSource s.active = terminal := by
      by_contra ha
      have := s.rank_positive ha hne
      omega
    rw [s.potential_terminal ht]
    exact le_rfl
  | succ n ih =>
    by_cases ht : source.cursorSource s.active = terminal
    · rw [s.potential_terminal ht, adaptiveSum_terminal hs hne hiso gate _ (n + 1) s ht]
    · rw [adaptiveSum, dite_eq_right ht]
      have hnext := adaptiveStep_mono_rank hs hne hiso gate s ht potential
        (fun t => adaptiveSum hs hne hiso gate
          (fun u => u.pool.groups.entropyNat + stepLoss gate u) n t)
        (fun t hlt => ih t (by omega))
      have h := potential_adaptive_step hs hne hiso gate s ht
      linarith

theorem potential_le_adaptive_budget_loss (n : ℕ) (s : LASState word eos source)
    (hn : s.rank ≤ n) :
    s.potential ≤ adaptiveBudgetNat hs hne hiso gate n s +
      adaptiveLossNat hs hne hiso gate n s := by
  have h := potential_le_adaptive_sum hs hne hiso gate n s hn
  rw [adaptiveSum_add] at h
  exact h

theorem adaptiveLoss_le_uncertainty (n : ℕ) (s : LASState word eos source) :
    adaptiveLossNat hs hne hiso gate n s ≤ adaptiveUncertaintyNat hs hne hiso gate n s :=
  adaptiveSum_mono hs hne hiso gate _ _ (stepLoss_le_uncertainty gate) n s

end LASState

noncomputable def adaptiveValue (word : Token → List Byte) (eos : Byte)
    (source : TokenSource Token) (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (gate : LASState.AdaptiveGate word eos source) : ℝ :=
  LASState.adaptiveBudgetNat hs hne hiso gate (source.maxRenderedLength word)
    (LASState.initial word eos source) / Real.log 2

theorem native_le_adaptive_add_loss (word : Token → List Byte) (eos : Byte)
    (source : TokenSource Token) (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (gate : LASState.AdaptiveGate word eos source) :
    source.nativeValue word ≤ source.adaptiveValue word eos hs hne hiso gate +
      LASState.adaptiveLossNat hs hne hiso gate (source.maxRenderedLength word)
        (LASState.initial word eos source) / Real.log 2 := by
  have h := LASState.potential_le_adaptive_budget_loss hs hne hiso gate
    (source.maxRenderedLength word) (LASState.initial word eos source) (Nat.sub_le _ _)
  rw [LASState.potential_initial] at h
  have hd := div_le_div_of_nonneg_right h (le_of_lt (Real.log_pos (by norm_num : (1 : ℝ) < 2)))
  simpa only [nativeValue, adaptiveValue, add_div] using hd

theorem native_le_adaptive_add_uncertainty (word : Token → List Byte) (eos : Byte)
    (source : TokenSource Token) (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (gate : LASState.AdaptiveGate word eos source) :
    source.nativeValue word ≤ source.adaptiveValue word eos hs hne hiso gate +
      LASState.adaptiveUncertaintyNat hs hne hiso gate (source.maxRenderedLength word)
        (LASState.initial word eos source) / Real.log 2 := by
  apply (source.native_le_adaptive_add_loss word eos hs hne hiso gate).trans
  apply add_le_add le_rfl
  exact div_le_div_of_nonneg_right
    (LASState.adaptiveLoss_le_uncertainty hs hne hiso gate _ _)
    (le_of_lt (Real.log_pos (by norm_num : (1 : ℝ) < 2)))

end StegaTextProofs.TokenSource
