module

public import StegaTextProofs.UnboundedStoppedSum

/-!
Conditional one-step drift yields a finite telescoping inequality. Integrable
tail domination removes the terminal correction at unbounded stopping.
The concrete infinite-source protocol must separately verify these interfaces.
-/

@[expose] public section

namespace StegaTextProofs.StoppedSum

open MeasureTheory Filter
open scoped Topology
variable {Ω : Type*} [mΩ : MeasurableSpace Ω]
variable (μ : Measure Ω) [IsProbabilityMeasure μ]

omit [IsProbabilityMeasure μ] in
theorem stopped_step_integrable (T : Ω → ℕ) (hT : Measurable T)
    (hmean : Integrable (fun ω => (T ω : ℝ)) μ) (c : ℕ → Ω → ℝ)
    (hc : ∀ k, Measurable (c k)) (h0 : ∀ k ω, 0 ≤ c k ω)
    (C : ℝ) (hbound : ∀ k ω, c k ω ≤ C) (n : ℕ) :
    Integrable (StoppingTails.tail T (c n) n) μ := by
  have he : StoppingTails.tail T (c n) n = fun ω => partialSum T c (n + 1) ω - partialSum T c n ω := by
    funext ω
    rw [partial_succ]
    ring
  rw [he]
  exact (partial_integrable μ T hT hmean c hc h0 C hbound (n + 1)).sub
    (partial_integrable μ T hT hmean c hc h0 C hbound n)

theorem finite_conditional_comparison
    (T : Ω → ℕ) (c Φ : ℕ → Ω → ℝ) (m : ℕ → MeasurableSpace Ω)
    (hm : ∀ n, m n ≤ mΩ)
    (hpart : ∀ n, Integrable (partialSum T c n) μ)
    (hstepint : ∀ n, Integrable (StoppingTails.tail T (c n) n) μ)
    (hΦ : ∀ n, Integrable (Φ n) μ)
    (hstep : ∀ n, Φ n ≤ᵐ[μ] fun ω => StoppingTails.tail T (c n) n ω + μ[Φ (n + 1) | m n] ω)
    (n : ℕ) :
    (∫ ω, Φ 0 ω ∂μ) ≤ (∫ ω, partialSum T c n ω ∂μ) + ∫ ω, Φ n ω ∂μ := by
  induction n with
  | zero => simp [partial_zero]
  | succ n ih =>
    have hd : (∫ ω, Φ n ω ∂μ) ≤
        (∫ ω, StoppingTails.tail T (c n) n ω ∂μ) + ∫ ω, Φ (n + 1) ω ∂μ := by
      calc
        _ ≤ ∫ ω, StoppingTails.tail T (c n) n ω + μ[Φ (n + 1) | m n] ω ∂μ :=
          integral_mono_ae (hΦ n) ((hstepint n).add integrable_condExp) (hstep n)
        _ = _ := by rw [integral_add (hstepint n) integrable_condExp, integral_condExp (hm n)]
    have he : (∫ ω, partialSum T c (n + 1) ω ∂μ) =
        (∫ ω, partialSum T c n ω ∂μ) + ∫ ω, StoppingTails.tail T (c n) n ω ∂μ := by
      simp_rw [partial_succ]
      exact integral_add (hpart n) (hstepint n)
    linarith

/-- No deterministic horizon, final-comparison assumption, or zero terminal expectation is used. -/
theorem unbounded_stopped_comparison
    (T : Ω → ℕ) (hT : Measurable T) (hmean : Integrable (fun ω => (T ω : ℝ)) μ)
    (c : ℕ → Ω → ℝ) (hc : ∀ k, Measurable (c k)) (h0 : ∀ k ω, 0 ≤ c k ω)
    (C : ℝ) (hbound : ∀ k ω, c k ω ≤ C)
    (Φ : ℕ → Ω → ℝ) (hΦ : ∀ n, AEStronglyMeasurable (Φ n) μ)
    (hΦ0 : ∀ n, ∀ᵐ ω ∂μ, 0 ≤ Φ n ω)
    (m : ℕ → MeasurableSpace Ω) (hm : ∀ n, m n ≤ mΩ)
    (B : Ω → ℝ) (hB : Integrable B μ)
    (htail : ∀ n, Φ n ≤ᵐ[μ] μ[StoppingTails.tail T B n | m n])
    (hstep : ∀ n, Φ n ≤ᵐ[μ] fun ω => StoppingTails.tail T (c n) n ω + μ[Φ (n + 1) | m n] ω) :
    (∫ ω, Φ 0 ω ∂μ) ≤ ∫ ω, total T c ω ∂μ := by
  have hiΦ : ∀ n, Integrable (Φ n) μ := fun n =>
    StoppingTails.integrable_of_conditional_tail_bound μ T B (Φ n) n (m n) (hΦ n) (hΦ0 n) (htail n)
  have hfinite := finite_conditional_comparison μ T c Φ m hm
    (partial_integrable μ T hT hmean c hc h0 C hbound)
    (stopped_step_integrable μ T hT hmean c hc h0 C hbound) hiΦ hstep
  have hsum := integral_partial_tendsto_total μ T hT hmean c hc h0 C hbound
  have hzero := StoppingTails.integral_potential_tendsto_zero μ T hT B hB Φ m hm hΦ hΦ0 htail
  have hlimit := hsum.add hzero
  simp only [add_zero] at hlimit
  exact ge_of_tendsto hlimit (Filter.Eventually.of_forall hfinite)

theorem finite_conditional_error_bound
    (T : Ω → ℕ) (c Φ : ℕ → Ω → ℝ) (m : ℕ → MeasurableSpace Ω)
    (hm : ∀ n, m n ≤ mΩ)
    (hpart : ∀ n, Integrable (partialSum T c n) μ)
    (hstepint : ∀ n, Integrable (StoppingTails.tail T (c n) n) μ)
    (hΦ : ∀ n, AEStronglyMeasurable (Φ n) μ) (hΦ0 : ∀ n, ∀ᵐ ω ∂μ, 0 ≤ Φ n ω)
    (B : Ω → ℝ) (htail : ∀ n, Φ n ≤ᵐ[μ] μ[StoppingTails.tail T B n | m n])
    (hstep : ∀ n, Φ n ≤ᵐ[μ] fun ω => StoppingTails.tail T (c n) n ω + μ[Φ (n + 1) | m n] ω)
    (n : ℕ) :
    (∫ ω, Φ 0 ω ∂μ) ≤ (∫ ω, partialSum T c n ω ∂μ) + ∫ ω, StoppingTails.tail T B n ω ∂μ := by
  have hiΦ : ∀ k, Integrable (Φ k) μ := fun k =>
    StoppingTails.integrable_of_conditional_tail_bound μ T B (Φ k) k (m k) (hΦ k) (hΦ0 k) (htail k)
  exact (finite_conditional_comparison μ T c Φ m hm hpart hstepint hiΦ hstep n).trans
    (add_le_add le_rfl (StoppingTails.integral_le_tail_of_conditional_bound μ T B (Φ n) n
      (m n) (hm n) (hΦ n) (hΦ0 n) (htail n)))

end StegaTextProofs.StoppedSum
