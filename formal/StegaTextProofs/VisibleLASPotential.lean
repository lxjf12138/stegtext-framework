module

public import StegaTextProofs.LASAbsoluteLocal
public import StegaTextProofs.FiniteConditionalComparison

/-! The visible-output entropy potential of the actual constructed LAS state. -/

@[expose] public section

namespace StegaTextProofs

attribute [local instance] Classical.propDecidable
open PrefixGrouping
variable {Token Byte : Type*} [Fintype Token]

namespace TokenSource

abbrev VisibleText (word : Token → List Byte) (source : TokenSource Token) :=
  Set.range (fun l : source.decisionTree.Leaves => render word (source.tokenPath l))

noncomputable instance visibleTextFintype (word : Token → List Byte) (source : TokenSource Token) :
    Fintype (source.VisibleText word) := Fintype.ofFinite _

def visibleReadout (word : Token → List Byte) (source : TokenSource Token)
    (l : source.decisionTree.Leaves) : source.VisibleText word :=
  ⟨render word (source.tokenPath l), ⟨l, rfl⟩⟩

noncomputable def visibleEntropy (word : Token → List Byte) (source : TokenSource Token) : ℝ :=
  ((source.sourceLaw word).map (source.visibleReadout word)).entropyNat / Real.log 2

namespace LASState

variable {word : Token → List Byte} {eos : Byte} {source : TokenSource Token}

noncomputable def visibleLaw (s : LASState word eos source) : FiniteLaw (source.VisibleText word) :=
  (s.prior.mix (source.cursorLaw word)).map (source.visibleReadout word)

noncomputable def visiblePotentialNat (s : LASState word eos source) : ℝ := s.visibleLaw.entropyNat

noncomputable def candidateVisibleLaw (s : LASState word eos source)
    (c : (source.expandActive s.prior s.active).Positive) : FiniteLaw (source.VisibleText word) :=
  (source.cursorLaw word c.val).map (source.visibleReadout word)

noncomputable def groupVisibleChannel (s : LASState word eos source) (g : s.pool.Group) :=
  fun c => s.candidateVisibleLaw (s.pool.candidate g c)

noncomputable def localGapNat (s : LASState word eos source) : ℝ :=
  s.pool.groups.expect (fun g => (s.pool.selected g).absoluteGapNat (s.groupVisibleChannel g))

noncomputable def localBoundNat (s : LASState word eos source) : ℝ :=
  s.pool.groups.expect (fun g => (s.pool.selected g).exactMass * (s.pool.selected g).surrogate.entropyNat)

noncomputable def localPredictiveBoundNat (s : LASState word eos source) : ℝ :=
  s.pool.groups.expect (fun g => (s.pool.selected g).exactMass *
    (s.pool.selected g).surrogate.informationNat (fun e => s.groupVisibleChannel g (.inl e)))

theorem pool_visibleLaw (s : LASState word eos source) :
    s.pool.prior.mix s.candidateVisibleLaw = s.visibleLaw := by
  have hp : s.pool.prior.mix (fun c => source.cursorLaw word c.val) =
      s.prior.mix (source.cursorLaw word) :=
    ((source.expandActive s.prior s.active).supportLaw_mix (source.cursorLaw word)).trans
      (source.expandActive_continuation word s.prior s.active)
  rw [visibleLaw, ← hp, FiniteLaw.mix_map]
  rfl

theorem next_visibleLaw (s : LASState word eos source)
    (ha : source.cursorSource s.active ≠ terminal) (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (g : s.pool.Group) (e : s.pool.Exact g) :
    (s.next ha hs hne hiso g e).visibleLaw =
      ((s.pool.selected g).collapse e).mix (s.groupVisibleChannel g) := by
  rw [visibleLaw, s.next_posterior ha hs hne hiso]
  exact FiniteLaw.mix_map _ _ _

theorem group_visible_prefix (s : LASState word eos source)
    (ha : source.cursorSource s.active ≠ terminal) (g : s.pool.Group)
    (y : source.VisibleText word)
    (hy : 0 < ((s.pool.selected g).prior.mix (s.groupVisibleChannel g)).prob y) :
    s.emitted ++ s.pool.residual g.val <+: y.val := by
  obtain ⟨c, _, hc⟩ := (FiniteLaw.mix_positive_iff _ _ y).mp hy
  obtain ⟨l, hl, he⟩ := (FiniteLaw.map_positive_iff _ _ y).mp hc
  obtain ⟨tail, ht⟩ := source.cursorLaw_prefix word (s.pool.candidate g c).val l hl
  have hr : render word (source.tokenPath l) =
      s.emitted ++ s.pool.residual (s.pool.candidate g c) ++ render word tail := by
    rw [← ht]
    simp only [render, List.flatMap_append]
    change render word (source.cursorHistory (s.pool.candidate g c).val) ++ render word tail = _
    rw [s.pool_render ha]
    rfl
  have hk := s.pool.group_key_prefix g (s.pool.splitEquiv g c)
  obtain ⟨rest, hk⟩ := hk
  refine ⟨rest ++ render word tail, ?_⟩
  have he' := congrArg Subtype.val he
  change render word (source.tokenPath l) = y.val at he'
  rw [← he', hr]
  change _ = s.emitted ++ s.pool.residual (s.pool.splitEquiv g c).val ++ render word tail
  rw [← hk]
  simp only [List.append_assoc]

theorem group_visible_disjoint (s : LASState word eos source)
    (ha : source.cursorSource s.active ≠ terminal) (g h : s.pool.Group)
    (y : source.VisibleText word)
    (hg : 0 < ((s.pool.selected g).prior.mix (s.groupVisibleChannel g)).prob y)
    (hh : 0 < ((s.pool.selected h).prior.mix (s.groupVisibleChannel h)).prob y) : g = h := by
  have hp := List.prefix_or_prefix_of_prefix (s.group_visible_prefix ha g y hg)
    (s.group_visible_prefix ha h y hh)
  rcases hp with hp | hp
  · exact s.pool.keys_prefix_free g h ((List.prefix_append_right_inj s.emitted).mp hp)
  · exact (s.pool.keys_prefix_free h g ((List.prefix_append_right_inj s.emitted).mp hp)).symm

theorem visiblePotential_terminal (s : LASState word eos source)
    (ht : source.cursorSource s.active = terminal) : s.visiblePotentialNat = 0 := by
  obtain ⟨l, hl, _⟩ := source.terminal_cursor_resolved word s.active ht
  unfold visiblePotentialNat visibleLaw
  rw [s.terminal_point ht, FiniteLaw.point_mix, hl]
  simp only [FiniteLaw.map, FiniteLaw.point_mix, FiniteLaw.entropyNat_point]

/-- Exact one-step decomposition; the drift is proved from prefix separation. -/
theorem visiblePotential_step (s : LASState word eos source)
    (ha : source.cursorSource s.active ≠ terminal) (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t)) :
    s.visiblePotentialNat = s.pool.groups.entropyNat +
      s.pool.groups.expect (fun g => (s.pool.selected g).surrogate.expect
        (fun e => (s.next ha hs hne hiso g e).visiblePotentialNat)) + s.localGapNat := by
  have hm : s.pool.groups.mix (fun g => (s.pool.selected g).prior.mix
      (s.groupVisibleChannel g)) = s.visibleLaw :=
    (s.pool.disintegrate_channel s.candidateVisibleLaw).trans s.pool_visibleLaw
  have hh := s.pool.groups.entropyNat_mix_eq_of_disjoint
    (fun g => (s.pool.selected g).prior.mix (s.groupVisibleChannel g)) (s.group_visible_disjoint ha)
  rw [hm] at hh
  simp only [visiblePotentialNat, s.next_visibleLaw ha hs hne hiso,
    localGapNat, LASGroup.absoluteGapNat, FiniteLaw.expect_sub]
  linarith

theorem localGapNat_nonneg (s : LASState word eos source) : 0 ≤ s.localGapNat := by
  simpa only [FiniteLaw.expect_const, localGapNat] using s.pool.groups.expect_mono
    (fun g => (s.pool.selected g).absoluteGapNat_nonneg (s.groupVisibleChannel g))

theorem localGapNat_le_bound (s : LASState word eos source) : s.localGapNat ≤ s.localBoundNat :=
  s.pool.groups.expect_mono (fun g => (s.pool.selected g).absoluteGapNat_le_exact_entropy
    (s.groupVisibleChannel g))

theorem localGapNat_le_predictive (s : LASState word eos source) :
    s.localGapNat ≤ s.localPredictiveBoundNat :=
  s.pool.groups.expect_mono (fun g => (s.pool.selected g).absoluteGapNat_le_exact_information
    (s.groupVisibleChannel g))

theorem localPredictiveBoundNat_le_bound (s : LASState word eos source) :
    s.localPredictiveBoundNat ≤ s.localBoundNat :=
  s.pool.groups.expect_mono (fun g => mul_le_mul_of_nonneg_left
    (FiniteLaw.informationNat_le_entropy _ _) (le_of_lt (s.pool.selected g).exactMass_pos))

end LASState
end TokenSource
end StegaTextProofs
