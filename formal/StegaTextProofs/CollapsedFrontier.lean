module

public import StegaTextProofs.PrefixLASStep

/-!
After a surrogate is chosen, the next runtime frontier has one active atom
of the entire exact mass and every strict atom at its unchanged conditional
weight. Mixing the corresponding continuation laws is precisely the
complete-path posterior of exact collapse.
-/

@[expose] public section

namespace StegaTextProofs

open scoped BigOperators

namespace LASGroup

variable {Exact Strict Output : Type*} [Fintype Exact] [Fintype Strict] [Fintype Output]

noncomputable def nextFrontier (g : LASGroup Exact Strict) : FiniteLaw (Unit ⊕ Strict) where
  prob := Sum.elim (fun _ => g.exactMass) (fun r => g.prior.prob (.inr r))
  nonneg c := by
    cases c with
    | inl _ => exact le_of_lt g.exactMass_pos
    | inr r => exact g.prior.nonneg _
  sum_one := by simpa only [Fintype.sum_sum_type, Sum.elim_inl, Sum.elim_inr,
    Fintype.sum_unique] using g.mass_partition

theorem nextFrontier_active (g : LASGroup Exact Strict) :
    g.nextFrontier.prob (.inl ()) = g.exactMass := rfl

theorem nextFrontier_retained (g : LASGroup Exact Strict) (r : Strict) :
    g.nextFrontier.prob (.inr r) = g.prior.prob (.inr r) := rfl

theorem nextFrontier_channel (g : LASGroup Exact Strict) (e : Exact)
    (channel : Exact ⊕ Strict → FiniteLaw Output) :
    g.nextFrontier.mix (Sum.elim (fun _ => channel (.inl e)) (fun r => channel (.inr r))) =
      (g.collapse e).mix channel := by
  apply FiniteLaw.ext_prob
  intro y
  change g.nextFrontier.expect (fun c =>
    (Sum.elim (fun _ => channel (.inl e)) (fun r => channel (.inr r)) c).prob y) =
      (g.collapse e).expect (fun c => (channel c).prob y)
  rw [g.collapse_expect]
  simp [FiniteLaw.expect, nextFrontier, Fintype.sum_sum_type]

end LASGroup

namespace PrefixPool

attribute [local instance] Classical.propDecidable
variable {Candidate Byte : Type*} [Fintype Candidate]

theorem nextFrontier_strict_probability (pool : PrefixPool Candidate Byte)
    (g : pool.Group) (r : pool.Strict g) :
    (pool.selected g).nextFrontier.prob (.inr r) =
      pool.prior.prob r.val.val / pool.groups.prob g := rfl

theorem nextFrontier_active_probability (pool : PrefixPool Candidate Byte) (g : pool.Group) :
    (pool.selected g).nextFrontier.prob (.inl ()) =
      (∑ e : pool.Exact g, pool.prior.prob e.val.val) / pool.groups.prob g := by
  change (∑ e : pool.Exact g, pool.prior.prob e.val.val / pool.groups.prob g) = _
  exact (Finset.sum_div _ _ _).symm

end PrefixPool

end StegaTextProofs
