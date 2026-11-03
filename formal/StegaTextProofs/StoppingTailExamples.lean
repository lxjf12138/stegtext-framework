module

public import StegaTextProofs.UnboundedStoppedComparison
public import StegaTextProofs.MeasureTapeExamples
public import Mathlib.Analysis.SpecificLimits.Basic

/-! Eventual zero on every sample does not imply convergence of expectations. -/

@[expose] public section

namespace StegaTextProofs.StoppingTailExamples

open MeasureTheory Set Filter
open scoped Topology
open MeasureTapeExamples
attribute [local instance] Classical.propDecidable

noncomputable def spike (n : ℕ) : ℝ → ℝ :=
  (Ioc 0 ((1 / 2 : ℝ) ^ n)).indicator (fun _ => (2 : ℝ) ^ n)

theorem spike_nonneg (n : ℕ) (x : ℝ) : 0 ≤ spike n x := by
  exact Set.indicator_nonneg (fun _ _ => pow_nonneg (by norm_num) n) x

theorem spike_measurable (n : ℕ) : Measurable (spike n) :=
  measurable_const.indicator measurableSet_Ioc

theorem spike_integrable (n : ℕ) : Integrable (spike n) uniform01 :=
  (integrable_const ((2 : ℝ) ^ n)).indicator measurableSet_Ioc

theorem integral_spike (n : ℕ) : (∫ x, spike n x ∂uniform01) = 1 := by
  have ha : (1 / 2 : ℝ) ^ n ≤ 1 := pow_le_one₀ (by norm_num) (by norm_num)
  have hsub : Ioc 0 ((1 / 2 : ℝ) ^ n) ⊆ Ioc 0 (1 : ℝ) :=
    fun _ hx => ⟨hx.1, hx.2.trans ha⟩
  rw [spike, integral_indicator_const _ measurableSet_Ioc, uniform01,
    measureReal_restrict_apply measurableSet_Ioc, inter_eq_left.mpr hsub,
    Real.volume_real_Ioc_of_le (pow_nonneg (by norm_num) n), sub_zero, smul_eq_mul, ← mul_pow]
  norm_num

theorem spike_eventually_zero (x : ℝ) : ∀ᶠ n in atTop, spike n x = 0 := by
  by_cases hx : 0 < x
  · have ht : Tendsto (fun n : ℕ => (1 / 2 : ℝ) ^ n) atTop (𝓝 0) :=
      tendsto_pow_atTop_nhds_zero_of_lt_one (by norm_num) (by norm_num)
    filter_upwards [ht.eventually (gt_mem_nhds hx)] with n hn
    exact Set.indicator_of_notMem (fun h => (not_le.mpr hn) h.2) _
  · exact Filter.Eventually.of_forall (fun _ =>
      Set.indicator_of_notMem (fun h => hx h.1) _)

theorem spike_tendsto_zero (x : ℝ) : Tendsto (fun n => spike n x) atTop (𝓝 0) := by
  apply tendsto_const_nhds.congr'
  filter_upwards [spike_eventually_zero x] with n hn
  exact hn.symm

theorem integral_spike_not_tendsto_zero :
    ¬ Tendsto (fun n => ∫ x, spike n x ∂uniform01) atTop (𝓝 0) := by
  simp_rw [integral_spike]
  intro h
  have he : (1 : ℝ) = 0 := tendsto_nhds_unique tendsto_const_nhds h
  norm_num at he

/-- The missing domination is substantive, even though each sample eventually vanishes. -/
theorem no_integrable_spike_envelope :
    ¬ ∃ B : ℝ → ℝ, Integrable B uniform01 ∧
      ∀ n, ∀ᵐ x ∂uniform01, ‖spike n x‖ ≤ B x := by
  rintro ⟨B, hB, hb⟩
  have h := tendsto_integral_of_dominated_convergence (μ := uniform01)
    (f := fun _ => (0 : ℝ)) B
    (fun n => (spike_measurable n).aestronglyMeasurable) hB hb
    (Filter.Eventually.of_forall spike_tendsto_zero)
  simp only [integral_zero] at h
  exact integral_spike_not_tendsto_zero h

end StegaTextProofs.StoppingTailExamples
