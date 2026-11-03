module

public import StegaTextProofs.IntegrableStoppingTails

/-! Pathwise stopped sums, integrability, and quantitative truncation errors. -/

@[expose] public section

namespace StegaTextProofs.StoppedSum

open MeasureTheory Filter
open scoped BigOperators Topology
variable {Ω : Type*}

def partialSum (T : Ω → ℕ) (c : ℕ → Ω → ℝ) (n : ℕ) (ω : Ω) : ℝ :=
  ∑ k ∈ Finset.range (min n (T ω)), c k ω

def total (T : Ω → ℕ) (c : ℕ → Ω → ℝ) (ω : Ω) : ℝ :=
  ∑ k ∈ Finset.range (T ω), c k ω

theorem partial_zero (T : Ω → ℕ) (c : ℕ → Ω → ℝ) : partialSum T c 0 = 0 := by
  funext ω
  simp [partialSum]

theorem partial_succ (T : Ω → ℕ) (c : ℕ → Ω → ℝ) (n : ℕ) (ω : Ω) :
    partialSum T c (n + 1) ω = partialSum T c n ω + StoppingTails.tail T (c n) n ω := by
  by_cases h : n < T ω
  · simp [partialSum, StoppingTails.tail, h, Nat.min_eq_left (Nat.le_of_lt h),
      Finset.sum_range_succ]
  · have hn : T ω ≤ n := Nat.le_of_not_gt h
    simp [partialSum, StoppingTails.tail, h, Nat.min_eq_right hn,
      Nat.min_eq_right (hn.trans (Nat.le_succ n))]

theorem partial_eq_total (T : Ω → ℕ) (c : ℕ → Ω → ℝ) (n : ℕ) (ω : Ω)
    (h : T ω ≤ n) : partialSum T c n ω = total T c ω := by
  simp only [partialSum, total, min_eq_right h]

theorem partial_nonneg (T : Ω → ℕ) (c : ℕ → Ω → ℝ)
    (hc : ∀ k ω, 0 ≤ c k ω) (n : ℕ) (ω : Ω) : 0 ≤ partialSum T c n ω :=
  Finset.sum_nonneg (fun k _ => hc k ω)

theorem total_nonneg (T : Ω → ℕ) (c : ℕ → Ω → ℝ)
    (hc : ∀ k ω, 0 ≤ c k ω) (ω : Ω) : 0 ≤ total T c ω :=
  Finset.sum_nonneg (fun k _ => hc k ω)

theorem partial_le_total (T : Ω → ℕ) (c : ℕ → Ω → ℝ)
    (hc : ∀ k ω, 0 ≤ c k ω) (n : ℕ) (ω : Ω) : partialSum T c n ω ≤ total T c ω := by
  exact Finset.sum_le_sum_of_subset_of_nonneg (Finset.range_mono (min_le_right _ _))
    (fun k _ _ => hc k ω)

theorem total_le_time (T : Ω → ℕ) (c : ℕ → Ω → ℝ) (C : ℝ)
    (hc : ∀ k ω, c k ω ≤ C) (ω : Ω) : total T c ω ≤ C * (T ω : ℝ) := by
  calc
    _ ≤ ∑ _k ∈ Finset.range (T ω), C := Finset.sum_le_sum (fun k _ => hc k ω)
    _ = _ := by simp [mul_comm]

theorem truncation_error_le (T : Ω → ℕ) (c : ℕ → Ω → ℝ) (C : ℝ)
    (hc : ∀ k ω, c k ω ≤ C) (n : ℕ) (ω : Ω) :
    total T c ω - partialSum T c n ω ≤ C * StoppingTails.remaining T n ω := by
  by_cases hn : n ≤ T ω
  · have he := Finset.sum_range_add (fun k => c k ω) n (T ω - n)
    rw [Nat.add_sub_of_le hn] at he
    have hb : (∑ k ∈ Finset.range (T ω - n), c (n + k) ω) ≤ C * (T ω - n : ℕ) := by
      calc
        _ ≤ ∑ _k ∈ Finset.range (T ω - n), C := Finset.sum_le_sum (fun k _ => hc (n + k) ω)
        _ = _ := by simp [mul_comm]
    simp only [total, partialSum, min_eq_left hn, StoppingTails.remaining]
    linarith
  · have ht : T ω ≤ n := le_of_not_ge hn
    rw [partial_eq_total T c n ω ht]
    simp [StoppingTails.remaining, Nat.sub_eq_zero_of_le ht]

theorem partial_tendsto_total (T : Ω → ℕ) (c : ℕ → Ω → ℝ) (ω : Ω) :
    Tendsto (fun n => partialSum T c n ω) atTop (𝓝 (total T c ω)) := by
  apply tendsto_const_nhds.congr'
  filter_upwards [eventually_ge_atTop (T ω)] with n hn
  exact (partial_eq_total T c n ω hn).symm

variable [MeasurableSpace Ω]

theorem total_measurable (T : Ω → ℕ) (hT : Measurable T) (c : ℕ → Ω → ℝ)
    (hc : ∀ k, Measurable (c k)) : Measurable (total T c) := by
  have hm : Measurable (fun z : Ω × ℕ => ∑ k ∈ Finset.range z.2, c k z.1) :=
    measurable_from_prod_countable_left (fun n => (Finset.range n).measurable_sum (fun k _ => hc k))
  exact hm.comp (measurable_id.prodMk hT)

theorem partial_measurable (T : Ω → ℕ) (hT : Measurable T) (c : ℕ → Ω → ℝ)
    (hc : ∀ k, Measurable (c k)) (n : ℕ) : Measurable (partialSum T c n) :=
  total_measurable (fun ω => min n (T ω)) (measurable_const.min hT) c hc

theorem total_integrable (μ : Measure Ω) (T : Ω → ℕ) (hT : Measurable T)
    (hmean : Integrable (fun ω => (T ω : ℝ)) μ) (c : ℕ → Ω → ℝ)
    (hc : ∀ k, Measurable (c k)) (h0 : ∀ k ω, 0 ≤ c k ω)
    (C : ℝ) (hbound : ∀ k ω, c k ω ≤ C) : Integrable (total T c) μ := by
  exact (hmean.const_mul C).mono_nonneg (total_measurable T hT c hc).aestronglyMeasurable
    (Filter.Eventually.of_forall (total_nonneg T c h0))
    (Filter.Eventually.of_forall (total_le_time T c C hbound))

theorem partial_integrable (μ : Measure Ω) (T : Ω → ℕ) (hT : Measurable T)
    (hmean : Integrable (fun ω => (T ω : ℝ)) μ) (c : ℕ → Ω → ℝ)
    (hc : ∀ k, Measurable (c k)) (h0 : ∀ k ω, 0 ≤ c k ω)
    (C : ℝ) (hbound : ∀ k ω, c k ω ≤ C) (n : ℕ) : Integrable (partialSum T c n) μ := by
  exact (total_integrable μ T hT hmean c hc h0 C hbound).mono_nonneg
    (partial_measurable T hT c hc n).aestronglyMeasurable
    (Filter.Eventually.of_forall (partial_nonneg T c h0 n))
    (Filter.Eventually.of_forall (partial_le_total T c h0 n))

theorem integral_partial_tendsto_total (μ : Measure Ω) (T : Ω → ℕ) (hT : Measurable T)
    (hmean : Integrable (fun ω => (T ω : ℝ)) μ) (c : ℕ → Ω → ℝ)
    (hc : ∀ k, Measurable (c k)) (h0 : ∀ k ω, 0 ≤ c k ω)
    (C : ℝ) (hbound : ∀ k ω, c k ω ≤ C) :
    Tendsto (fun n => ∫ ω, partialSum T c n ω ∂μ) atTop (𝓝 (∫ ω, total T c ω ∂μ)) := by
  apply tendsto_integral_of_dominated_convergence (fun ω => C * (T ω : ℝ))
    (fun n => (partial_measurable T hT c hc n).aestronglyMeasurable) (hmean.const_mul C)
  · intro n
    apply Filter.Eventually.of_forall
    intro ω
    rw [Real.norm_of_nonneg (partial_nonneg T c h0 n ω)]
    exact (partial_le_total T c h0 n ω).trans (total_le_time T c C hbound ω)
  · exact Filter.Eventually.of_forall (partial_tendsto_total T c)

theorem integral_truncation_error (μ : Measure Ω) (T : Ω → ℕ) (hT : Measurable T)
    (hmean : Integrable (fun ω => (T ω : ℝ)) μ) (c : ℕ → Ω → ℝ)
    (hc : ∀ k, Measurable (c k)) (h0 : ∀ k ω, 0 ≤ c k ω)
    (C : ℝ) (hbound : ∀ k ω, c k ω ≤ C) (n : ℕ) :
    0 ≤ (∫ ω, total T c ω ∂μ) - (∫ ω, partialSum T c n ω ∂μ) ∧
      (∫ ω, total T c ω ∂μ) - (∫ ω, partialSum T c n ω ∂μ) ≤
        C * ∫ ω, StoppingTails.remaining T n ω ∂μ := by
  have htotal := total_integrable μ T hT hmean c hc h0 C hbound
  have hpartial := partial_integrable μ T hT hmean c hc h0 C hbound n
  rw [← integral_sub htotal hpartial]
  constructor
  · exact integral_nonneg (fun ω => sub_nonneg.mpr (partial_le_total T c h0 n ω))
  · rw [← integral_const_mul]
    exact integral_mono (htotal.sub hpartial)
      ((StoppingTails.remaining_integrable μ T hT hmean n).const_mul C)
      (truncation_error_le T c C hbound n)

end StegaTextProofs.StoppedSum
