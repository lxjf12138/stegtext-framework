module

public import StegaTextProofs.VisibleLASPotential

/-! The cumulative absolute capacity gap of the concrete finite stopped LAS run. -/

@[expose] public section

namespace StegaTextProofs.TokenSource

attribute [local instance] Classical.propDecidable
variable {Token Byte : Type*} [Fintype Token]

namespace LASState

variable {word : Token → List Byte} {eos : Byte} {source : TokenSource Token}

/-- Expected sum of a state cost, paid only at nonterminal LAS states. -/
noncomputable def accumulated (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (cost : LASState word eos source → ℝ) : ℕ → LASState word eos source → ℝ
  | 0, _ => 0
  | n + 1, s => if ha : source.cursorSource s.active = terminal then 0 else
      cost s + s.pool.groups.expect (fun g => (s.pool.selected g).surrogate.expect
        (fun e => accumulated hs hne hiso cost n (s.next ha hs hne hiso g e)))

theorem accumulated_terminal (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (cost : LASState word eos source → ℝ) (n : ℕ) (s : LASState word eos source)
    (ht : source.cursorSource s.active = terminal) : accumulated hs hne hiso cost n s = 0 := by
  cases n <;> simp [accumulated, ht]

theorem accumulated_mono (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (f g : LASState word eos source → ℝ) (hfg : ∀ s, f s ≤ g s)
    (n : ℕ) (s : LASState word eos source) :
    accumulated hs hne hiso f n s ≤ accumulated hs hne hiso g n s := by
  induction n generalizing s with
  | zero => exact le_rfl
  | succ n ih =>
    by_cases ht : source.cursorSource s.active = terminal
    · simp only [accumulated, dite_eq_left ht, le_refl]
    · simp only [accumulated, dite_eq_right ht]
      exact add_le_add (hfg s) (s.pool.groups.expect_mono (fun g =>
        (s.pool.selected g).surrogate.expect_mono (fun e => ih (s.next ht hs hne hiso g e))))

theorem accumulated_zero (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (n : ℕ) (s : LASState word eos source) : accumulated hs hne hiso (fun _ => 0) n s = 0 := by
  induction n generalizing s with
  | zero => rfl
  | succ n ih =>
    by_cases ht : source.cursorSource s.active = terminal <;>
      simp [accumulated, ht, ih]

theorem accumulated_gap_nonneg (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (n : ℕ) (s : LASState word eos source) :
    0 ≤ accumulated hs hne hiso localGapNat n s := by
  simpa only [accumulated_zero] using accumulated_mono hs hne hiso (fun _ => 0)
    localGapNat localGapNat_nonneg n s

/-- Exact telescope, with no assumed drift or terminal certificate. -/
theorem visible_gap_telescope (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (n : ℕ) (s : LASState word eos source) (hn : s.rank ≤ n) :
    s.visiblePotentialNat = budgetNat hs hne hiso n s +
      accumulated hs hne hiso localGapNat n s := by
  induction n generalizing s with
  | zero =>
    have ht : source.cursorSource s.active = terminal := by
      by_contra ha
      have := s.rank_positive ha hne
      omega
    simpa only [budgetNat, accumulated, add_zero] using s.visiblePotential_terminal ht
  | succ n ih =>
    by_cases ht : source.cursorSource s.active = terminal
    · rw [budgetNat_terminal hs hne hiso _ _ ht, accumulated_terminal hs hne hiso _ _ _ ht,
        s.visiblePotential_terminal ht, add_zero]
    · have hi : ∀ g e, (s.next ht hs hne hiso g e).visiblePotentialNat =
          budgetNat hs hne hiso n (s.next ht hs hne hiso g e) +
          accumulated hs hne hiso localGapNat n (s.next ht hs hne hiso g e) := by
        intro g e
        apply ih
        have := s.rank_next_lt ht hs hne hiso g e
        omega
      rw [s.visiblePotential_step ht hs hne hiso]
      simp only [hi, FiniteLaw.expect_add, budgetNat, accumulated, dite_eq_right ht]
      ring

theorem visible_gap_bounds (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (n : ℕ) (s : LASState word eos source) (hn : s.rank ≤ n) :
    0 ≤ s.visiblePotentialNat - budgetNat hs hne hiso n s ∧
    s.visiblePotentialNat - budgetNat hs hne hiso n s ≤
      accumulated hs hne hiso localBoundNat n s := by
  rw [visible_gap_telescope hs hne hiso n s hn, add_sub_cancel_left]
  exact ⟨accumulated_gap_nonneg hs hne hiso n s,
    accumulated_mono hs hne hiso localGapNat localBoundNat localGapNat_le_bound n s⟩

end LASState

noncomputable def lasGap (word : Token → List Byte) (eos : Byte) (source : TokenSource Token)
    (hs : source.EOSSource word eos) (hne : ∀ t, word t ≠ [])
    (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t)) : ℝ :=
  LASState.accumulated hs hne hiso LASState.localGapNat
    (source.maxRenderedLength word) (LASState.initial word eos source) / Real.log 2

noncomputable def lasGapBound (word : Token → List Byte) (eos : Byte) (source : TokenSource Token)
    (hs : source.EOSSource word eos) (hne : ∀ t, word t ≠ [])
    (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t)) : ℝ :=
  LASState.accumulated hs hne hiso LASState.localBoundNat
    (source.maxRenderedLength word) (LASState.initial word eos source) / Real.log 2

noncomputable def lasPredictiveGapBound (word : Token → List Byte) (eos : Byte) (source : TokenSource Token)
    (hs : source.EOSSource word eos) (hne : ∀ t, word t ≠ [])
    (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t)) : ℝ :=
  LASState.accumulated hs hne hiso LASState.localPredictiveBoundNat
    (source.maxRenderedLength word) (LASState.initial word eos source) / Real.log 2

theorem las_absolute_gap_eq (word : Token → List Byte) (eos : Byte) (source : TokenSource Token)
    (hs : source.EOSSource word eos) (hne : ∀ t, word t ≠ [])
    (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t)) :
    source.visibleEntropy word - source.lasValue word eos hs hne hiso =
      source.lasGap word eos hs hne hiso := by
  have h := LASState.visible_gap_telescope hs hne hiso (source.maxRenderedLength word)
    (LASState.initial word eos source) (Nat.sub_le _ _)
  have hp : (LASState.initial word eos source).visiblePotentialNat =
      ((source.sourceLaw word).map (source.visibleReadout word)).entropyNat := by
    unfold LASState.visiblePotentialNat LASState.visibleLaw LASState.initial
    rw [FiniteLaw.point_mix, cursorLaw_root]
  rw [hp] at h
  unfold visibleEntropy lasValue lasGap
  rw [h, add_div]
  ring

/-- Absolute, visible-entropy upper gap bound, in bits. -/
theorem las_absolute_gap_bounds (word : Token → List Byte) (eos : Byte) (source : TokenSource Token)
    (hs : source.EOSSource word eos) (hne : ∀ t, word t ≠ [])
    (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t)) :
    0 ≤ source.visibleEntropy word - source.lasValue word eos hs hne hiso ∧
    source.visibleEntropy word - source.lasValue word eos hs hne hiso ≤
      source.lasGapBound word eos hs hne hiso := by
  rw [source.las_absolute_gap_eq word eos hs hne hiso]
  exact ⟨div_nonneg (LASState.accumulated_gap_nonneg hs hne hiso _ _) (le_of_lt (Real.log_pos (by norm_num))),
    div_le_div_of_nonneg_right (LASState.accumulated_mono hs hne hiso _ _
      LASState.localGapNat_le_bound _ _) (le_of_lt (Real.log_pos (by norm_num)))⟩

/-- The sharper cumulative full-future predictive bound sits between gap and identity bound. -/
theorem las_predictive_gap_bounds (word : Token → List Byte) (eos : Byte) (source : TokenSource Token)
    (hs : source.EOSSource word eos) (hne : ∀ t, word t ≠ [])
    (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t)) :
    source.visibleEntropy word - source.lasValue word eos hs hne hiso ≤
      source.lasPredictiveGapBound word eos hs hne hiso ∧
    source.lasPredictiveGapBound word eos hs hne hiso ≤ source.lasGapBound word eos hs hne hiso := by
  rw [source.las_absolute_gap_eq word eos hs hne hiso]
  exact ⟨div_le_div_of_nonneg_right (LASState.accumulated_mono hs hne hiso _ _
      LASState.localGapNat_le_predictive _ _) (le_of_lt (Real.log_pos (by norm_num))),
    div_le_div_of_nonneg_right (LASState.accumulated_mono hs hne hiso _ _
      LASState.localPredictiveBoundNat_le_bound _ _) (le_of_lt (Real.log_pos (by norm_num)))⟩

/-- A certified bound relative to the budget implies a near-upper-limit ratio. -/
theorem las_near_visible_upper (word : Token → List Byte) (eos : Byte) (source : TokenSource Token)
    (hs : source.EOSSource word eos) (hne : ∀ t, word t ≠ [])
    (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t)) (η : ℝ)
    (hη : 0 ≤ η) (hv : 0 < source.lasValue word eos hs hne hiso)
    (hb : source.lasGapBound word eos hs hne hiso ≤ η * source.lasValue word eos hs hne hiso) :
    1 / (1 + η) ≤ source.lasValue word eos hs hne hiso / source.visibleEntropy word := by
  obtain ⟨hlo, hhi⟩ := source.las_absolute_gap_bounds word eos hs hne hiso
  have hy : 0 < source.visibleEntropy word := by linarith
  have he : 0 < 1 + η := by linarith
  apply (div_le_div_iff₀ he hy).mpr
  nlinarith

end StegaTextProofs.TokenSource
