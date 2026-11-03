module

public import StegaTextProofs.AdaptiveLASProtocol

/-! The implemented entropy threshold, measured in bits, and its loss-count bound. -/

@[expose] public section

namespace StegaTextProofs.TokenSource

attribute [local instance] Classical.propDecidable
variable {Token Byte : Type*} [Fintype Token]

namespace LASState

variable {word : Token → List Byte} {eos : Byte} {source : TokenSource Token}

noncomputable def groupUncertainty (s : LASState word eos source) (g : s.pool.Group) : ℝ :=
  s.groupUncertaintyNat g / Real.log 2

theorem groupUncertaintyNat_nonneg (s : LASState word eos source) (g : s.pool.Group) :
    0 ≤ s.groupUncertaintyNat g :=
  (s.groupLoss_nonneg g).trans (s.groupLoss_le_uncertainty g)

theorem groupUncertainty_nonneg (s : LASState word eos source) (g : s.pool.Group) :
    0 ≤ s.groupUncertainty g :=
  div_nonneg (s.groupUncertaintyNat_nonneg g)
    (le_of_lt (Real.log_pos (by norm_num : (1 : ℝ) < 2)))

noncomputable def thresholdGate (tau : ℝ) : AdaptiveGate word eos source :=
  fun s g => decide (tau ≤ s.groupUncertainty g)

theorem thresholdGate_fullSync_iff (tau : ℝ) (s : LASState word eos source) (g : s.pool.Group) :
    thresholdGate tau s g = false ↔ s.groupUncertainty g < tau := by
  simp only [thresholdGate, decide_eq_false_iff_not, not_le]

theorem thresholdGate_zero (s : LASState word eos source) (g : s.pool.Group) :
    thresholdGate 0 s g = true := by
  simp only [thresholdGate, decide_eq_true_eq]
  exact s.groupUncertainty_nonneg g

/-- Expected indicator of a FullSync choice with positive U at this state. -/
noncomputable def stepPositiveSyncCount (gate : AdaptiveGate word eos source)
    (s : LASState word eos source) : ℝ :=
  s.pool.groups.expect (fun g => if gate s g then 0
    else if 0 < s.groupUncertaintyNat g then 1 else 0)

theorem stepUncertainty_threshold_le (tau : ℝ) (s : LASState word eos source) :
    stepUncertainty (thresholdGate tau) s ≤
      (tau * Real.log 2) * stepPositiveSyncCount (thresholdGate tau) s := by
  unfold stepUncertainty stepPositiveSyncCount
  rw [← FiniteLaw.expect_mul]
  apply FiniteLaw.expect_mono
  intro g
  by_cases hg : thresholdGate tau s g = true
  · simp only [hg, ↓reduceIte, mul_zero, le_refl]
  · have hf : thresholdGate tau s g = false := Bool.eq_false_iff.mpr hg
    have hlt := (thresholdGate_fullSync_iff tau s g).mp hf
    have hlog : 0 < Real.log 2 := Real.log_pos (by norm_num)
    have hbound : s.groupUncertaintyNat g < tau * Real.log 2 :=
      (div_lt_iff₀ hlog).mp hlt
    simp only [hf, Bool.false_eq_true, ↓reduceIte]
    by_cases hu : 0 < s.groupUncertaintyNat g
    · simpa only [hu, ↓reduceIte, mul_one] using le_of_lt hbound
    · have hz : s.groupUncertaintyNat g = 0 :=
        le_antisymm (le_of_not_gt hu) (s.groupUncertaintyNat_nonneg g)
      simp [hz]

variable (hs : source.EOSSource word eos) (hne : ∀ t, word t ≠ [])
  (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))

theorem adaptiveUncertainty_threshold_le (tau : ℝ) (n : ℕ) (s : LASState word eos source) :
    adaptiveUncertaintyNat hs hne hiso (thresholdGate tau) n s ≤ (tau * Real.log 2) *
      adaptiveSum hs hne hiso (thresholdGate tau)
        (stepPositiveSyncCount (thresholdGate tau)) n s := by
  have h := adaptiveSum_mono hs hne hiso (thresholdGate tau)
    (stepUncertainty (thresholdGate tau))
    (fun t => (tau * Real.log 2) * stepPositiveSyncCount (thresholdGate tau) t)
    (stepUncertainty_threshold_le tau) n s
  rw [adaptiveSum_mul] at h
  exact h

end LASState

/-- C_SP <= C_A + tau E[M_tau], with M_tau counting positive-U FullSync rounds. -/
theorem native_le_threshold_adaptive_add_count (word : Token → List Byte) (eos : Byte)
    (source : TokenSource Token) (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t)) (tau : ℝ) :
    source.nativeValue word ≤
      source.adaptiveValue word eos hs hne hiso (LASState.thresholdGate tau) + tau *
        LASState.adaptiveSum hs hne hiso (LASState.thresholdGate tau)
          (LASState.stepPositiveSyncCount (LASState.thresholdGate tau))
          (source.maxRenderedLength word) (LASState.initial word eos source) := by
  apply (source.native_le_adaptive_add_uncertainty word eos hs hne hiso
    (LASState.thresholdGate tau)).trans
  apply add_le_add le_rfl
  have hlog : 0 < Real.log 2 := Real.log_pos (by norm_num)
  have h := div_le_div_of_nonneg_right
    (LASState.adaptiveUncertainty_threshold_le hs hne hiso tau
      (source.maxRenderedLength word) (LASState.initial word eos source)) (le_of_lt hlog)
  convert h using 1
  field_simp

end StegaTextProofs.TokenSource
