module

public import StegaTextProofs.ConcreteLASOutput

/-! Full-candidate synchronization on the same concrete LAS state space. -/

@[expose] public section

namespace StegaTextProofs.TokenSource

attribute [local instance] Classical.propDecidable
open PrefixGrouping
variable {Token Byte : Type*} [Fintype Token]

namespace LASState

variable {word : Token → List Byte} {eos : Byte} {source : TokenSource Token}

/-- A legal singleton frontier. Its continuation can still be random. -/
noncomputable def resolved (c : source.Cursor) (hc : source.LegalCursor c) :
    LASState word eos source where
  prior := FiniteLaw.point c
  active := c
  emitted := render word (source.cursorHistory c)
  active_pos := by simp [FiniteLaw.point]
  legal := by
    intro d hd
    by_cases he : d = c
    · exact he ▸ hc
    · simp [FiniteLaw.point, he] at hd
  antichain := by
    intro d e hd he _
    have hd' : d = c := by
      by_contra hn
      simp [FiniteLaw.point, hn] at hd
    have he' : e = c := by
      by_contra hn
      simp [FiniteLaw.point, hn] at he
    exact hd'.trans he'.symm
  active_render := rfl
  visible := by
    intro d hd
    by_cases he : d = c
    · simp [he]
    · simp [FiniteLaw.point, he] at hd
  strict_length := by intro d hd hn; simp [FiniteLaw.point, hn] at hd
  isolated := by intro d hd hn; simp [FiniteLaw.point, hn] at hd
  retained_nonterminal := by intro d hd hn; simp [FiniteLaw.point, hn] at hd
  terminal_point := fun _ => rfl

noncomputable def fullSync (s : LASState word eos source) (g : s.pool.Group)
    (c : s.pool.Exact g ⊕ s.pool.Strict g) : LASState word eos source :=
  resolved (s.pool.candidate g c).val
    (source.expandActive_legal s.prior s.active s.legal _ (s.pool.candidate g c).property)

theorem fullSync_emitted (s : LASState word eos source)
    (ha : source.cursorSource s.active ≠ terminal) (g : s.pool.Group)
    (c : s.pool.Exact g ⊕ s.pool.Strict g) :
    (s.fullSync g c).emitted = s.emitted ++ s.pool.residual (s.pool.candidate g c) :=
  s.pool_render ha (s.pool.candidate g c)

theorem fullSync_posterior (s : LASState word eos source) (g : s.pool.Group)
    (c : s.pool.Exact g ⊕ s.pool.Strict g) :
    (s.fullSync g c).prior.mix (source.cursorLaw word) =
      source.cursorLaw word (s.pool.candidate g c).val :=
  FiniteLaw.point_mix _ _

theorem rank_fullSync_lt (s : LASState word eos source)
    (ha : source.cursorSource s.active ≠ terminal) (hne : ∀ t, word t ≠ [])
    (g : s.pool.Group) (c : s.pool.Exact g ⊕ s.pool.Strict g) :
    (s.fullSync g c).rank < s.rank := by
  have hb := (s.fullSync g c).emitted_length_le_max
  have hn := List.length_pos_iff.mpr (s.pool_nonempty ha hne (s.pool.candidate g c))
  rw [s.fullSync_emitted ha, List.length_append] at hb
  unfold rank
  rw [s.fullSync_emitted ha, List.length_append]
  omega

/-- FullSync consumes precisely the group key followed by a replayable suffix. -/
theorem fullSync_residual_split (s : LASState word eos source) (g : s.pool.Group)
    (c : s.pool.Exact g ⊕ s.pool.Strict g) :
    s.pool.residual (s.pool.candidate g c) = s.pool.residual g.val ++
      (s.pool.residual (s.pool.candidate g c)).drop (s.pool.residual g.val).length := by
  have hp := s.pool.group_key_prefix g (s.pool.splitEquiv g c)
  exact (List.prefix_iff_eq_append.mp hp).symm

end LASState

end StegaTextProofs.TokenSource
