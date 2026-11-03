module

public import StegaTextProofs.MeasureConditionalEntropy

/-! Integrable tails for stopping times with no deterministic upper bound. -/

@[expose] public section

namespace StegaTextProofs.StoppingTails

open MeasureTheory Filter
open scoped Topology

variable {Ω : Type*}

def tail (T : Ω → ℕ) (B : Ω → ℝ) (n : ℕ) (ω : Ω) : ℝ :=
  if n < T ω then B ω else 0

theorem tail_norm_le (T : Ω → ℕ) (B : Ω → ℝ) (n : ℕ) (ω : Ω) :
    ‖tail T B n ω‖ ≤ ‖B ω‖ := by
  unfold tail
  split_ifs <;> simp

theorem tail_nonneg (T : Ω → ℕ) (B : Ω → ℝ) (hB : ∀ ω, 0 ≤ B ω) (n : ℕ) (ω : Ω) :
    0 ≤ tail T B n ω := by
  unfold tail
  split_ifs
  · exact hB ω
  · exact le_rfl

variable [mΩ : MeasurableSpace Ω]

theorem tail_integrable (μ : Measure Ω) (T : Ω → ℕ) (hT : Measurable T)
    (B : Ω → ℝ) (hB : Integrable B μ) (n : ℕ) : Integrable (tail T B n) μ := by
  have he : tail T B n = {ω | n < T ω}.indicator B := by
    funext ω
    by_cases h : n < T ω <;> simp [tail, Set.indicator, h]
  rw [he]
  exact hB.indicator (measurableSet_lt measurable_const hT)

omit mΩ in
theorem tail_tendsto_zero (T : Ω → ℕ) (B : Ω → ℝ) (ω : Ω) :
    Tendsto (fun n => tail T B n ω) atTop (𝓝 0) := by
  apply tendsto_const_nhds.congr'
  filter_upwards [eventually_ge_atTop (T ω)] with n hn
  simp [tail, not_lt.mpr hn]

theorem integral_tail_tendsto_zero (μ : Measure Ω) (T : Ω → ℕ) (hT : Measurable T)
    (B : Ω → ℝ) (hB : Integrable B μ) :
    Tendsto (fun n => ∫ ω, tail T B n ω ∂μ) atTop (𝓝 0) := by
  have h := tendsto_integral_of_dominated_convergence (μ := μ) (fun ω => ‖B ω‖)
    (fun n => (tail_integrable μ T hT B hB n).aestronglyMeasurable) hB.norm
    (fun n => Filter.Eventually.of_forall (tail_norm_le T B n))
    (Filter.Eventually.of_forall (tail_tendsto_zero T B))
  simpa only [integral_zero] using h

def remaining (T : Ω → ℕ) (n : ℕ) (ω : Ω) : ℝ := (T ω - n : ℕ)

omit mΩ in
theorem remaining_nonneg (T : Ω → ℕ) (n : ℕ) (ω : Ω) : 0 ≤ remaining T n ω :=
  Nat.cast_nonneg _

omit mΩ in
theorem remaining_le_time (T : Ω → ℕ) (n : ℕ) (ω : Ω) :
    remaining T n ω ≤ (T ω : ℝ) := by
  exact Nat.cast_le.mpr (Nat.sub_le (T ω) n)

theorem remaining_measurable (T : Ω → ℕ) (hT : Measurable T) (n : ℕ) :
    Measurable (remaining T n) :=
  (measurable_of_countable (fun t : ℕ => ((t - n : ℕ) : ℝ))).comp hT

theorem remaining_integrable (μ : Measure Ω) (T : Ω → ℕ) (hT : Measurable T)
    (hmean : Integrable (fun ω => (T ω : ℝ)) μ) (n : ℕ) :
    Integrable (remaining T n) μ := by
  apply hmean.mono_nonneg (remaining_measurable T hT n).aestronglyMeasurable
    (Filter.Eventually.of_forall (remaining_nonneg T n))
  exact Filter.Eventually.of_forall (remaining_le_time T n)

theorem integral_remaining_tendsto_zero (μ : Measure Ω) (T : Ω → ℕ) (hT : Measurable T)
    (hmean : Integrable (fun ω => (T ω : ℝ)) μ) :
    Tendsto (fun n => ∫ ω, remaining T n ω ∂μ) atTop (𝓝 0) := by
  have h := tendsto_integral_of_dominated_convergence (μ := μ) (f := fun _ => (0 : ℝ))
    (fun ω => (T ω : ℝ))
    (fun n => (remaining_measurable T hT n).aestronglyMeasurable) hmean
    (fun n => Filter.Eventually.of_forall (fun ω => by
      rw [Real.norm_of_nonneg (remaining_nonneg T n ω)]
      exact remaining_le_time T n ω))
    (Filter.Eventually.of_forall (fun ω => by
      apply tendsto_const_nhds.congr'
      filter_upwards [eventually_ge_atTop (T ω)] with n hn
      simp [remaining, Nat.sub_eq_zero_of_le hn]))
  simpa only [integral_zero] using h

variable (μ : Measure Ω) [IsProbabilityMeasure μ]

omit [IsProbabilityMeasure μ] in
/-- Conditional tail domination proves integrability; it is not an extra hypothesis. -/
theorem integrable_of_conditional_tail_bound (T : Ω → ℕ) (B Φ : Ω → ℝ) (n : ℕ)
    (m : MeasurableSpace Ω) (hΦ : AEStronglyMeasurable[mΩ] Φ μ)
    (h0 : ∀ᵐ ω ∂μ, 0 ≤ Φ ω)
    (hbound : Φ ≤ᵐ[μ] μ[tail T B n | m]) : Integrable Φ μ :=
  integrable_condExp.mono_nonneg hΦ h0 hbound

theorem integral_le_tail_of_conditional_bound (T : Ω → ℕ) (B Φ : Ω → ℝ) (n : ℕ)
    (m : MeasurableSpace Ω) (hm : m ≤ mΩ) (hΦ : AEStronglyMeasurable[mΩ] Φ μ)
    (h0 : ∀ᵐ ω ∂μ, 0 ≤ Φ ω)
    (hbound : Φ ≤ᵐ[μ] μ[tail T B n | m]) :
    (∫ ω, Φ ω ∂μ) ≤ ∫ ω, tail T B n ω ∂μ := by
  let : MeasurableSpace Ω := mΩ
  calc
    _ ≤ ∫ ω, μ[tail T B n | m] ω ∂μ :=
      integral_mono_ae (integrable_of_conditional_tail_bound μ T B Φ n m hΦ h0 hbound)
        integrable_condExp hbound
    _ = _ := integral_condExp (μ := μ) hm

theorem integral_potential_tendsto_zero (T : Ω → ℕ) (hT : Measurable T)
    (B : Ω → ℝ) (hB : Integrable B μ) (Φ : ℕ → Ω → ℝ)
    (m : ℕ → MeasurableSpace Ω) (hm : ∀ n, m n ≤ mΩ)
    (hΦ : ∀ n, AEStronglyMeasurable (Φ n) μ) (h0 : ∀ n, ∀ᵐ ω ∂μ, 0 ≤ Φ n ω)
    (hbound : ∀ n, Φ n ≤ᵐ[μ] μ[tail T B n | m n]) :
    Tendsto (fun n => ∫ ω, Φ n ω ∂μ) atTop (𝓝 0) := by
  apply squeeze_zero (fun n => integral_nonneg_of_ae (h0 n))
    (fun n => integral_le_tail_of_conditional_bound μ T B (Φ n) n (m n) (hm n)
      (hΦ n) (h0 n) (hbound n))
  exact integral_tail_tendsto_zero μ T hT B hB

end StegaTextProofs.StoppingTails
