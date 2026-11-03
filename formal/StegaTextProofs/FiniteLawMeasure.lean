module

public import StegaTextProofs.FiniteEntropyChannel

/-! Finite laws as probability measures, and measurable finite observations. -/

@[expose] public section

namespace StegaTextProofs.FiniteLaw

open MeasureTheory
open scoped BigOperators ENNReal

variable {α : Type*} [Fintype α]

noncomputable def toPMF (p : FiniteLaw α) : PMF α :=
  PMF.ofFintype (fun a => ENNReal.ofReal (p.prob a)) (by
    rw [← ENNReal.ofReal_sum_of_nonneg (fun a _ => p.nonneg a), p.sum_one]
    simp)

noncomputable def toMeasure [MeasurableSpace α] (p : FiniteLaw α) : Measure α :=
  p.toPMF.toMeasure

instance toMeasure_probability [MeasurableSpace α] (p : FiniteLaw α) :
    IsProbabilityMeasure p.toMeasure := inferInstanceAs (IsProbabilityMeasure p.toPMF.toMeasure)

theorem integral_toMeasure [MeasurableSpace α] [MeasurableSingletonClass α]
    (p : FiniteLaw α) (f : α → ℝ) :
    ∫ a, f a ∂p.toMeasure = p.expect f := by
  rw [toMeasure, PMF.integral_eq_sum]
  simp only [toPMF, PMF.ofFintype_apply, ENNReal.toReal_ofReal (p.nonneg _), smul_eq_mul, expect]

noncomputable def ofMeasure [MeasurableSpace α] [MeasurableSingletonClass α]
    (μ : Measure α) [IsProbabilityMeasure μ] : FiniteLaw α where
  prob a := μ.real {a}
  nonneg a := ENNReal.toReal_nonneg
  sum_one := by simp

theorem ofMeasure_expect [MeasurableSpace α] [MeasurableSingletonClass α]
    (μ : Measure α) [IsProbabilityMeasure μ] (f : α → ℝ) :
    (ofMeasure μ).expect f = ∫ a, f a ∂μ := by
  rw [integral_fintype Integrable.of_finite]
  rfl

theorem ofMeasure_toMeasure [MeasurableSpace α] [MeasurableSingletonClass α]
    (p : FiniteLaw α) : ofMeasure p.toMeasure = p := by
  apply ext_prob
  intro a
  change (p.toPMF.toMeasure {a}).toReal = p.prob a
  rw [PMF.toMeasure_apply_singleton _ _ (measurableSet_singleton a)]
  exact ENNReal.toReal_ofReal (p.nonneg a)

variable {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
variable [MeasurableSpace α] [MeasurableSingletonClass α]

theorem integral_finite_read (read : Ω → α) (hr : Measurable read) (f : α → ℝ) :
    ∫ ω, f (read ω) ∂μ = (ofMeasure (μ.map read)).expect f := by
  rw [ofMeasure_expect]
  exact (integral_map hr.aemeasurable (.of_discrete)).symm

theorem integrable_finite_read (read : Ω → α) (hr : Measurable read) (f : α → ℝ) :
    Integrable (fun ω => f (read ω)) μ := by
  exact (Integrable.of_finite (μ := μ.map read)).comp_aemeasurable hr.aemeasurable

end StegaTextProofs.FiniteLaw
