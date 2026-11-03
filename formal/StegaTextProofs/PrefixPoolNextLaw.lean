module

public import StegaTextProofs.CollapsedFrontier
public import StegaTextProofs.PrefixPoolNext
public import StegaTextProofs.LASReachability
public import StegaTextProofs.FrontierExpansionLaw

/-! Push the concrete collapsed frontier to fixed labels, retaining all strict atoms. -/

@[expose] public section

namespace StegaTextProofs.PrefixPool

attribute [local instance] Classical.propDecidable
variable {Candidate Byte Label Output : Type*}
  [Fintype Candidate] [Fintype Label] [Fintype Output]

noncomputable def nextLaw (pool : PrefixPool Candidate Byte) (label : Candidate → Label)
    (g : pool.Group) (e : pool.Exact g) : FiniteLaw Label :=
  (pool.selected g).nextFrontier.mix
    (Sum.elim (fun _ => FiniteLaw.point (label e.val.val))
      (fun r => FiniteLaw.point (label r.val.val)))

theorem nextLaw_support (pool : PrefixPool Candidate Byte) (label : Candidate → Label)
    (g : pool.Group) (e : pool.Exact g) (d : Label)
    (hd : 0 < (pool.nextLaw label g e).prob d) :
    d = label e.val.val ∨ ∃ r : pool.Strict g, d = label r.val.val := by
  obtain ⟨c, _, hc⟩ := (FiniteLaw.mix_positive_iff _ _ d).mp hd
  cases c with
  | inl _ =>
    left
    by_contra hn
    simp [FiniteLaw.point, hn] at hc
  | inr r =>
    refine Or.inr ⟨r, ?_⟩
    by_contra hn
    simp [FiniteLaw.point, hn] at hc

theorem nextLaw_active_positive (pool : PrefixPool Candidate Byte) (label : Candidate → Label)
    (g : pool.Group) (e : pool.Exact g) :
    0 < (pool.nextLaw label g e).prob (label e.val.val) := by
  apply (FiniteLaw.mix_positive_iff _ _ _).mpr
  exact ⟨.inl (), (pool.selected g).exactMass_pos, by simp [FiniteLaw.point]⟩

theorem nextLaw_eq_collapse (pool : PrefixPool Candidate Byte) (label : Candidate → Label)
    (g : pool.Group) (e : pool.Exact g) :
    pool.nextLaw label g e = ((pool.selected g).collapse e).mix
      (fun c => FiniteLaw.point (label (pool.candidate g c))) :=
  (pool.selected g).nextFrontier_channel e
    (fun c => FiniteLaw.point (label (pool.candidate g c)))

theorem nextLaw_channel (pool : PrefixPool Candidate Byte) (label : Candidate → Label)
    (channel : Label → FiniteLaw Output) (g : pool.Group) (e : pool.Exact g) :
    (pool.nextLaw label g e).mix channel = ((pool.selected g).collapse e).mix
      (fun c => channel (label (pool.candidate g c))) := by
  rw [pool.nextLaw_eq_collapse, FiniteLaw.mix_assoc]
  simp only [FiniteLaw.point_mix]

theorem nextLaw_eos (pool : PrefixPool Candidate Byte) (label : Candidate → Label)
    (eos : Byte) (hisolated : ∀ c, EOSIsolation.IsolatedWord eos (pool.residual c))
    (g : pool.Group) (hg : pool.residual g.val = [eos]) (e : pool.Exact g) :
    pool.nextLaw label g e = FiniteLaw.point (label e.val.val) := by
  rw [pool.nextLaw_eq_collapse, pool.eos_collapse_point eos hisolated g hg,
    FiniteLaw.point_mix]
  rfl

end StegaTextProofs.PrefixPool
