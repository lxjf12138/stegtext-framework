module

public import StegaTextProofs.FiniteLaw

/-!
A selected LAS group is a finite labelled disjoint union of exact and strict
candidates. The exact mass is positive. A surrogate is sampled only from the
exact part; its posterior weight is the whole exact mass, while every strict
candidate keeps its own weight. These are local distributional statements.
-/

@[expose] public section

namespace StegaTextProofs

open scoped BigOperators

structure LASGroup (Exact Strict : Type*) [Fintype Exact] [Fintype Strict] where
  prior : FiniteLaw (Exact ⊕ Strict)
  exact_pos : 0 < ∑ e, prior.prob (.inl e)

namespace LASGroup

variable {Exact Strict : Type*} [Fintype Exact] [Fintype Strict]

def exactMass (g : LASGroup Exact Strict) : ℝ :=
  ∑ e, g.prior.prob (.inl e)

theorem exactMass_pos (g : LASGroup Exact Strict) : 0 < g.exactMass :=
  g.exact_pos

theorem mass_partition (g : LASGroup Exact Strict) :
    g.exactMass + ∑ r, g.prior.prob (.inr r) = 1 := by
  simpa only [exactMass, Fintype.sum_sum_type] using g.prior.sum_one

noncomputable def surrogate (g : LASGroup Exact Strict) : FiniteLaw Exact where
  prob e := g.prior.prob (.inl e) / g.exactMass
  nonneg e := div_nonneg (g.prior.nonneg _) (le_of_lt g.exactMass_pos)
  sum_one := by
    rw [← Finset.sum_div]
    exact div_self (ne_of_gt g.exactMass_pos)

theorem surrogate_pmf_normalized (g : LASGroup Exact Strict) :
    ∑ e, g.surrogate.prob e = 1 :=
  g.surrogate.sum_one

noncomputable def collapse (g : LASGroup Exact Strict) (e : Exact) :
    FiniteLaw (Exact ⊕ Strict) := by
  classical
  exact {
    prob := Sum.elim (fun c => if c = e then g.exactMass else 0)
      (fun r => g.prior.prob (.inr r))
    nonneg := by
      intro c
      cases c with
      | inl c =>
        dsimp
        split_ifs
        · exact le_of_lt g.exactMass_pos
        · exact le_rfl
      | inr r => exact g.prior.nonneg _
    sum_one := by
      simp only [Fintype.sum_sum_type, Sum.elim_inl, Sum.elim_inr]
      simpa using g.mass_partition
  }

theorem las_collapse_mass_one (g : LASGroup Exact Strict) (e : Exact) :
    ∑ c, (g.collapse e).prob c = 1 :=
  (g.collapse e).sum_one

theorem collapse_expect (g : LASGroup Exact Strict) (e : Exact)
    (f : Exact ⊕ Strict → ℝ) :
    (g.collapse e).expect f =
      g.exactMass * f (.inl e) + ∑ r, g.prior.prob (.inr r) * f (.inr r) := by
  classical
  simp [FiniteLaw.expect, collapse, Fintype.sum_sum_type]

theorem surrogate_mul_exactMass (g : LASGroup Exact Strict) (e : Exact) :
    g.surrogate.prob e * g.exactMass = g.prior.prob (.inl e) := by
  exact div_mul_cancel₀ _ (ne_of_gt g.exactMass_pos)

/-- LAS preserves the prior expectation of every function on the full candidate labels. -/
theorem las_collapse_unbiased (g : LASGroup Exact Strict) (f : Exact ⊕ Strict → ℝ) :
    g.surrogate.expect (fun e => (g.collapse e).expect f) = g.prior.expect f := by
  simp only [collapse_expect, FiniteLaw.expect_add, FiniteLaw.expect_const]
  simp only [FiniteLaw.expect, Fintype.sum_sum_type]
  congr 1
  apply Finset.sum_congr rfl
  intro e _
  rw [← mul_assoc, g.surrogate_mul_exactMass]

theorem collapse_mean_prob (g : LASGroup Exact Strict) (c : Exact ⊕ Strict) :
    ∑ e, g.surrogate.prob e * (g.collapse e).prob c = g.prior.prob c := by
  classical
  cases c with
  | inl c =>
    simpa [collapse, mul_ite, eq_comm] using g.surrogate_mul_exactMass c
  | inr r =>
    simp only [collapse, Sum.elim_inr, ← Finset.sum_mul, g.surrogate.sum_one, one_mul]

/-- Joint weights of the complete latent candidate C and the exact surrogate S. -/
noncomputable def coupling (g : LASGroup Exact Strict) :
    FiniteLaw ((Exact ⊕ Strict) × Exact) where
  prob ce := g.surrogate.prob ce.2 * (g.collapse ce.2).prob ce.1
  nonneg ce := mul_nonneg (g.surrogate.nonneg _) ((g.collapse ce.2).nonneg _)
  sum_one := by
    rw [Fintype.sum_prod_type, Finset.sum_comm]
    simp only [← Finset.mul_sum, las_collapse_mass_one, mul_one]
    exact g.surrogate.sum_one

theorem surrogate_coupling_candidate_marginal (g : LASGroup Exact Strict)
    (c : Exact ⊕ Strict) :
    ∑ e, g.coupling.prob (c, e) = g.prior.prob c :=
  g.collapse_mean_prob c

theorem surrogate_coupling_surrogate_marginal (g : LASGroup Exact Strict) (e : Exact) :
    ∑ c, g.coupling.prob (c, e) = g.surrogate.prob e := by
  simp only [coupling, ← Finset.mul_sum, las_collapse_mass_one, mul_one]

/-- An exact latent candidate forces its own surrogate identity. -/
theorem coupling_exact [DecidableEq Exact] (g : LASGroup Exact Strict) (c e : Exact) :
    g.coupling.prob (.inl c, e) = if c = e then g.prior.prob (.inl c) else 0 := by
  classical
  by_cases h : c = e
  · subst c
    simpa [coupling, collapse] using g.surrogate_mul_exactMass e
  · simp [coupling, collapse, h]

/-- For a strict latent candidate, the surrogate is drawn from the same exact law. -/
theorem coupling_strict (g : LASGroup Exact Strict) (r : Strict) (e : Exact) :
    g.coupling.prob (.inr r, e) = g.prior.prob (.inr r) * g.surrogate.prob e := by
  simp only [coupling, collapse, Sum.elim_inr, mul_comm]

/-- Bayes' formula is asserted only for a positive-probability surrogate. -/
theorem surrogate_conditional_law (g : LASGroup Exact Strict) (e : Exact)
    (he : 0 < g.prior.prob (.inl e)) (c : Exact ⊕ Strict) :
    g.coupling.prob (c, e) / (∑ c', g.coupling.prob (c', e)) =
      (g.collapse e).prob c := by
  rw [g.surrogate_coupling_surrogate_marginal]
  have hp : g.surrogate.prob e ≠ 0 := ne_of_gt (div_pos he g.exactMass_pos)
  change g.surrogate.prob e * (g.collapse e).prob c / g.surrogate.prob e = _
  exact mul_div_cancel_left₀ _ hp

end LASGroup

end StegaTextProofs
