module

public import StegaTextProofs.ConcreteLASState
public import StegaTextProofs.FiniteTermination

/-!
The finite stopped budget of the constructed LAS protocol. State weights may
be real: no finite-type assumption is imposed on the state space. The maximum
rendered cursor length gives a strictly decreasing rank on every update.
-/

@[expose] public section

namespace StegaTextProofs.TokenSource

attribute [local instance] Classical.propDecidable
open PrefixGrouping
variable {Token Byte : Type*} [Fintype Token]

noncomputable def maxRenderedLength (word : Token → List Byte) (source : TokenSource Token) : ℕ :=
  Finset.univ.sup (fun c : source.Cursor => (render word (source.cursorHistory c)).length)

theorem cursor_length_le_max (word : Token → List Byte) (source : TokenSource Token)
    (c : source.Cursor) : (render word (source.cursorHistory c)).length ≤
      source.maxRenderedLength word :=
  Finset.le_sup (f := fun c => (render word (source.cursorHistory c)).length) (Finset.mem_univ c)

namespace LASState

variable {word : Token → List Byte} {eos : Byte} {source : TokenSource Token}

noncomputable def rank (s : LASState word eos source) : ℕ :=
  source.maxRenderedLength word - s.emitted.length

theorem emitted_length_le_max (s : LASState word eos source) :
    s.emitted.length ≤ source.maxRenderedLength word := by
  rw [← s.active_render]
  exact source.cursor_length_le_max word s.active

theorem rank_positive (s : LASState word eos source)
    (ha : source.cursorSource s.active ≠ terminal) (hne : ∀ t, word t ≠ []) : 0 < s.rank := by
  obtain ⟨c, hc⟩ := (source.expandActive s.prior s.active).exists_positive
  let d : (source.expandActive s.prior s.active).Positive := ⟨c, hc⟩
  have hr := s.pool_render ha d
  have hn := List.length_pos_iff.mpr (s.pool_nonempty ha hne d)
  have hb := source.cursor_length_le_max word c
  change (render word (source.cursorHistory d.val)).length ≤ _ at hb
  rw [hr, List.length_append] at hb
  unfold rank
  omega

theorem rank_next_lt (s : LASState word eos source)
    (ha : source.cursorSource s.active ≠ terminal) (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (g : s.pool.Group) (e : s.pool.Exact g) :
    (s.next ha hs hne hiso g e).rank < s.rank := by
  have hb := (s.next ha hs hne hiso g e).emitted_length_le_max
  have hn := List.length_pos_iff.mpr (s.pool_nonempty ha hne g.val)
  change (s.emitted ++ s.pool.residual g.val).length ≤ _ at hb
  simp only [List.length_append] at hb
  change source.maxRenderedLength word - (s.emitted ++ s.pool.residual g.val).length <
    source.maxRenderedLength word - s.emitted.length
  simp only [List.length_append]
  omega

noncomputable def potential (s : LASState word eos source) : ℝ :=
  source.decisionTree.selective true (s.prior.mix (source.cursorLaw word)).prob

theorem potential_terminal (s : LASState word eos source)
    (ht : source.cursorSource s.active = terminal) : s.potential = 0 := by
  obtain ⟨l, hl, _⟩ := source.terminal_cursor_resolved word s.active ht
  unfold potential
  rw [s.terminal_point ht, FiniteLaw.point_mix, hl]
  exact source.decisionTree.selective_point l

theorem potential_initial (word : Token → List Byte) (eos : Byte) (source : TokenSource Token) :
    (initial word eos source).potential = source.nativeValueNat word := by
  change source.decisionTree.selective true
    ((FiniteLaw.point source.rootCursor).mix (source.cursorLaw word)).prob = _
  rw [FiniteLaw.point_mix, cursorLaw_root, sourceLaw_selective]

theorem potential_step (s : LASState word eos source)
    (ha : source.cursorSource s.active ≠ terminal) (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t)) :
    s.potential ≤ s.pool.groups.entropyNat + s.pool.groups.expect
      (fun g => (s.pool.selected g).surrogate.expect
        (fun e => (s.next ha hs hne hiso g e).potential)) := by
  have h := source.activeLAS_selective_step word s.prior s.active s.emitted
    ha s.legal s.antichain s.active_render s.visible
  simp only [potential, s.next_posterior ha hs hne hiso]
  unfold pool
  exact h

noncomputable def budgetNat (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t)) :
    ℕ → LASState word eos source → ℝ
  | 0, _ => 0
  | n + 1, s => if ha : source.cursorSource s.active = terminal then 0
      else s.pool.groups.entropyNat + s.pool.groups.expect
        (fun g => (s.pool.selected g).surrogate.expect
          (fun e => budgetNat hs hne hiso n (s.next ha hs hne hiso g e)))

theorem budgetNat_terminal (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (n : ℕ) (s : LASState word eos source) (ht : source.cursorSource s.active = terminal) :
    budgetNat hs hne hiso n s = 0 := by
  cases n <;> simp [budgetNat, ht]

theorem potential_le_budgetNat (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (n : ℕ) (s : LASState word eos source) (hn : s.rank ≤ n) :
    s.potential ≤ budgetNat hs hne hiso n s := by
  induction n generalizing s with
  | zero =>
    have ht : source.cursorSource s.active = terminal := by
      by_contra ha
      have := s.rank_positive ha hne
      omega
    rw [s.potential_terminal ht, budgetNat_terminal hs hne hiso 0 s ht]
  | succ n ih =>
    by_cases ht : source.cursorSource s.active = terminal
    · rw [s.potential_terminal ht, budgetNat_terminal hs hne hiso (n + 1) s ht]
    · rw [budgetNat, dite_eq_right ht]
      apply (s.potential_step ht hs hne hiso).trans
      apply add_le_add le_rfl
      apply FiniteLaw.expect_mono
      intro g
      apply FiniteLaw.expect_mono
      intro e
      apply ih
      have := s.rank_next_lt ht hs hne hiso g e
      omega

theorem budgetNat_stable (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (n m : ℕ) (s : LASState word eos source) (hn : s.rank ≤ n) :
    budgetNat hs hne hiso (n + m) s = budgetNat hs hne hiso n s := by
  induction n generalizing s with
  | zero =>
    have ht : source.cursorSource s.active = terminal := by
      by_contra ha
      have := s.rank_positive ha hne
      omega
    rw [budgetNat_terminal hs hne hiso (0 + m) s ht, budgetNat_terminal hs hne hiso 0 s ht]
  | succ n ih =>
    by_cases ht : source.cursorSource s.active = terminal
    · rw [budgetNat_terminal hs hne hiso (n + 1 + m) s ht,
        budgetNat_terminal hs hne hiso (n + 1) s ht]
    · rw [Nat.succ_add]
      simp only [budgetNat, dite_eq_right ht]
      congr 1
      congr 1
      funext g
      congr 1
      funext e
      apply ih
      have := s.rank_next_lt ht hs hne hiso g e
      omega

end LASState

/-- The full ideal LAS budget, in bits, of the constructed finite EOS protocol. -/
noncomputable def lasValue (word : Token → List Byte) (eos : Byte) (source : TokenSource Token)
    (hs : source.EOSSource word eos) (hne : ∀ t, word t ≠ [])
    (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t)) : ℝ :=
  LASState.budgetNat hs hne hiso (source.maxRenderedLength word)
    (LASState.initial word eos source) / Real.log 2

/-- Native SyncPool's ideal budget is bounded by the concrete stopped LAS budget. -/
theorem native_le_concrete_las (word : Token → List Byte) (eos : Byte) (source : TokenSource Token)
    (hs : source.EOSSource word eos) (hne : ∀ t, word t ≠ [])
    (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t)) :
    source.nativeValue word ≤ source.lasValue word eos hs hne hiso := by
  have hr : (LASState.initial word eos source).rank ≤ source.maxRenderedLength word :=
    Nat.sub_le _ _
  have h := LASState.potential_le_budgetNat hs hne hiso (source.maxRenderedLength word)
    (LASState.initial word eos source) hr
  rw [LASState.potential_initial] at h
  exact div_le_div_of_nonneg_right h (le_of_lt (Real.log_pos (by norm_num)))

end StegaTextProofs.TokenSource
