module

public import StegaTextProofs.LASLocal

/-!
Shannon entropy for finite laws, with natural-log and bit versions. The local
LAS inequality uses one fixed candidate-to-output channel. It does not compare
the different future state processes of LAS and native SyncPool.
-/

@[expose] public section

namespace StegaTextProofs

open scoped BigOperators

namespace FiniteLaw

variable {α β : Type*} [Fintype α] [Fintype β]

noncomputable def entropyNat (p : FiniteLaw α) : ℝ :=
  ∑ a, Real.negMulLog (p.prob a)

noncomputable def entropy (p : FiniteLaw α) : ℝ :=
  p.entropyNat / Real.log 2

theorem entropyNat_mixture_concave (p : FiniteLaw α) (q : α → FiniteLaw β) :
    p.expect (fun a => (q a).entropyNat) ≤ (p.mix q).entropyNat := by
  calc
    p.expect (fun a => (q a).entropyNat) =
        ∑ b, ∑ a, p.prob a * Real.negMulLog ((q a).prob b) := by
      simp only [expect, entropyNat, Finset.mul_sum]
      exact Finset.sum_comm
    _ ≤ (p.mix q).entropyNat := by
      apply Finset.sum_le_sum
      intro b _
      have h := Real.concaveOn_negMulLog.le_map_sum
        (t := Finset.univ) (w := p.prob) (p := fun a => (q a).prob b)
        (fun a _ => p.nonneg a) p.sum_one (fun a _ => (q a).nonneg b)
      simpa only [smul_eq_mul, mix] using h

/-- Finite Jensen inequality for Shannon entropy measured in bits. -/
theorem entropy_mixture_concave (p : FiniteLaw α) (q : α → FiniteLaw β) :
    p.expect (fun a => (q a).entropy) ≤ (p.mix q).entropy := by
  have hlog : 0 < Real.log 2 := Real.log_pos (by norm_num)
  calc
    p.expect (fun a => (q a).entropy) =
        p.expect (fun a => (q a).entropyNat) / Real.log 2 := by
      simp only [expect, entropy, mul_div_assoc, Finset.sum_div]
    _ ≤ (p.mix q).entropy :=
      div_le_div_of_nonneg_right (entropyNat_mixture_concave p q) (le_of_lt hlog)

end FiniteLaw

namespace LASGroup

variable {Exact Strict Output : Type*}
  [Fintype Exact] [Fintype Strict] [Fintype Output]

/-- Averaging LAS's posterior future laws recovers the original mixture,
pointwise for every observable output. -/
theorem las_fixed_channel_law_preserved (g : LASGroup Exact Strict)
    (channel : Exact ⊕ Strict → FiniteLaw Output) (output : Output) :
    (g.surrogate.mix (fun e => (g.collapse e).mix channel)).prob output =
      (g.prior.mix channel).prob output := by
  exact g.las_collapse_unbiased (fun c => (channel c).prob output)

/-- For a fixed future channel, partial collapse retains at least as much
expected future entropy as synchronizing the full candidate identity. -/
theorem las_fixed_channel_entropy_ge_fullsync (g : LASGroup Exact Strict)
    (channel : Exact ⊕ Strict → FiniteLaw Output) :
    g.prior.expect (fun c => (channel c).entropy) ≤
      g.surrogate.expect (fun e => ((g.collapse e).mix channel).entropy) := by
  calc
    g.prior.expect (fun c => (channel c).entropy) =
        g.surrogate.expect
          (fun e => (g.collapse e).expect (fun c => (channel c).entropy)) :=
      (g.las_collapse_unbiased _).symm
    _ ≤ g.surrogate.expect (fun e => ((g.collapse e).mix channel).entropy) :=
      g.surrogate.expect_mono fun e =>
        FiniteLaw.entropy_mixture_concave (g.collapse e) channel

end LASGroup

end StegaTextProofs
