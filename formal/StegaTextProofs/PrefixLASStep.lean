module

public import StegaTextProofs.FinitePartition
public import StegaTextProofs.NativeGrouping

/-!
An actual LAS selected-group construction from a strictly supported finite
pool of residual words. Groups are the attained native prefix labels; exact
and strict types partition each fibre. Their law is constructed from pool
weights, with positive exact mass proved from minimal-key realization.
-/

@[expose] public section

namespace StegaTextProofs

open scoped BigOperators
attribute [local instance] Classical.propDecidable

structure PrefixPool (Candidate Byte : Type*) [Fintype Candidate] where
  prior : FiniteLaw Candidate
  positive : ∀ c, 0 < prior.prob c
  residual : Candidate → List Byte

namespace PrefixPool

variable {Candidate Byte Output : Type*} [Fintype Candidate] [Fintype Output]

abbrev Group (pool : PrefixPool Candidate Byte) :=
  {g : Candidate // ∃ c, NativeGrouping.label pool.residual pool.prior c = g}

noncomputable def key (pool : PrefixPool Candidate Byte) (c : Candidate) : pool.Group :=
  ⟨NativeGrouping.label pool.residual pool.prior c, ⟨c, rfl⟩⟩

noncomputable instance groupFintype (pool : PrefixPool Candidate Byte) : Fintype pool.Group :=
  inferInstanceAs (Fintype {g : Candidate //
    ∃ c, NativeGrouping.label pool.residual pool.prior c = g})

abbrev Fiber (pool : PrefixPool Candidate Byte) (g : pool.Group) :=
  FinitePartition.Fiber pool.key g

theorem group_mass_positive (pool : PrefixPool Candidate Byte) (g : pool.Group) :
    0 < FinitePartition.mass pool.prior pool.key g := by
  obtain ⟨c, hc⟩ := g.property
  exact FinitePartition.mass_positive_of_member pool.prior pool.key g c
    (Subtype.ext hc) (pool.positive c)

noncomputable def groups (pool : PrefixPool Candidate Byte) : FiniteLaw pool.Group :=
  FinitePartition.groups pool.prior pool.key

noncomputable def conditional (pool : PrefixPool Candidate Byte) (g : pool.Group) :
    FiniteLaw (pool.Fiber g) :=
  FinitePartition.conditional pool.prior pool.key g (pool.group_mass_positive g)

theorem key_word (pool : PrefixPool Candidate Byte) (c : Candidate) :
    pool.residual (pool.key c).val = PrefixGrouping.key
      (NativeGrouping.support pool.residual pool.prior) (pool.residual c)
      (NativeGrouping.token_mem_support pool.residual pool.prior c (pool.positive c)) :=
  NativeGrouping.label_word pool.residual pool.prior c (pool.positive c)

theorem group_key_prefix (pool : PrefixPool Candidate Byte) (g : pool.Group)
    (c : pool.Fiber g) : pool.residual g.val <+: pool.residual c.val := by
  have hg : pool.residual (pool.key c.val).val = pool.residual g.val :=
    congrArg (fun k : pool.Group => pool.residual k.val) c.property
  rw [← hg, pool.key_word]
  exact (PrefixGrouping.key_spec _ _ _).2

theorem key_idempotent (pool : PrefixPool Candidate Byte) (c : Candidate) :
    pool.key (pool.key c).val = pool.key c := by
  apply Subtype.ext
  apply (NativeGrouping.same_label_iff_same_key pool.residual pool.prior
    (pool.key c).val c (pool.positive _) (pool.positive c)).mpr
  have hk := PrefixGrouping.key_spec (NativeGrouping.support pool.residual pool.prior)
    (pool.residual c) (NativeGrouping.token_mem_support _ _ _ (pool.positive c))
  have hk' := PrefixGrouping.key_spec (NativeGrouping.support pool.residual pool.prior)
    (pool.residual (pool.key c).val)
    (NativeGrouping.token_mem_support _ _ _ (pool.positive (pool.key c).val))
  apply hk.1.2 _ hk'.1.1
  simpa only [pool.key_word] using hk'.2

theorem group_fixed (pool : PrefixPool Candidate Byte) (g : pool.Group) : pool.key g.val = g := by
  obtain ⟨c, hc⟩ := g.property
  have hg : pool.key c = g := Subtype.ext hc
  rw [← hg, pool.key_idempotent]

abbrev Exact (pool : PrefixPool Candidate Byte) (g : pool.Group) :=
  {c : pool.Fiber g // pool.residual c.val = pool.residual g.val}

abbrev Strict (pool : PrefixPool Candidate Byte) (g : pool.Group) :=
  {c : pool.Fiber g // pool.residual c.val ≠ pool.residual g.val}

noncomputable def splitEquiv (pool : PrefixPool Candidate Byte) (g : pool.Group) :
    (pool.Exact g ⊕ pool.Strict g) ≃ pool.Fiber g :=
  Equiv.sumCompl (fun c : pool.Fiber g => pool.residual c.val = pool.residual g.val)

noncomputable def candidate (pool : PrefixPool Candidate Byte) (g : pool.Group)
    (c : pool.Exact g ⊕ pool.Strict g) : Candidate := (pool.splitEquiv g c).val

noncomputable def selected (pool : PrefixPool Candidate Byte) (g : pool.Group) :
    LASGroup (pool.Exact g) (pool.Strict g) where
  prior := (pool.conditional g).reindex (pool.splitEquiv g)
  exact_pos := by
    let witness : pool.Exact g := ⟨⟨g.val, pool.group_fixed g⟩, rfl⟩
    have hp : 0 < ((pool.conditional g).reindex (pool.splitEquiv g)).prob (.inl witness) :=
      div_pos (pool.positive g.val) (pool.group_mass_positive g)
    exact hp.trans_le (Finset.single_le_sum
      (fun e _ => ((pool.conditional g).reindex (pool.splitEquiv g)).nonneg (.inl e))
      (Finset.mem_univ witness))

theorem selected_probability (pool : PrefixPool Candidate Byte) (g : pool.Group)
    (c : pool.Exact g ⊕ pool.Strict g) :
    (pool.selected g).prior.prob c =
      pool.prior.prob (pool.candidate g c) / (pool.groups).prob g := rfl

theorem disintegrate_channel (pool : PrefixPool Candidate Byte)
    (channel : Candidate → FiniteLaw Output) :
    pool.groups.mix (fun g => (pool.selected g).prior.mix (fun c =>
      channel (pool.candidate g c))) = pool.prior.mix channel := by
  change pool.groups.mix (fun g => ((pool.conditional g).reindex (pool.splitEquiv g)).mix
    (fun c => channel ((pool.splitEquiv g c).val))) = _
  calc
    _ = pool.groups.mix (fun g => (pool.conditional g).mix (fun c => channel c.val)) := by
      congr 1
      funext g
      exact (pool.conditional g).reindex_mix (pool.splitEquiv g) (fun c => channel c.val)
    _ = _ := FinitePartition.disintegrate_channel pool.prior pool.key pool.group_mass_positive channel

theorem key_nonempty (pool : PrefixPool Candidate Byte)
    (hne : ∀ c, pool.residual c ≠ []) (g : pool.Group) : pool.residual g.val ≠ [] := hne g.val

end PrefixPool

end StegaTextProofs
