module

public import StegaTextProofs.ConcreteLASBudget

/-!
Expectations for the same concrete stopped LAS execution. Its final token
history and emitted text have exactly the source laws, without an abstract
finite-state model or assumed terminal certificates.
-/

@[expose] public section

namespace StegaTextProofs.TokenSource

attribute [local instance] Classical.propDecidable
open PrefixGrouping
variable {Token Byte : Type*} [Fintype Token]

namespace LASState

variable {word : Token → List Byte} {eos : Byte} {source : TokenSource Token}

theorem next_belief_expect (s : LASState word eos source)
    (ha : source.cursorSource s.active ≠ terminal) (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (f : source.decisionTree.Leaves → ℝ) :
    s.pool.groups.expect (fun g => (s.pool.selected g).surrogate.expect
      (fun e => ((s.next ha hs hne hiso g e).prior.mix (source.cursorLaw word)).expect f)) =
        (s.prior.mix (source.cursorLaw word)).expect f := by
  have hpool : s.pool.prior.mix (fun c => source.cursorLaw word c.val) =
      s.prior.mix (source.cursorLaw word) :=
    ((source.expandActive s.prior s.active).supportLaw_mix (source.cursorLaw word)).trans
      (source.expandActive_continuation word s.prior s.active)
  calc
    _ = (s.pool.groups.mix (fun g => (s.pool.selected g).prior.mix
        (fun c => source.cursorLaw word (s.pool.candidate g c).val))).expect f := by
      simp only [s.next_posterior ha hs hne hiso, PrefixPool.posterior,
        FiniteLaw.expect_mix, LASGroup.las_collapse_unbiased]
    _ = _ := by rw [s.pool.disintegrate_channel (fun c => source.cursorLaw word c.val), hpool]

noncomputable def runExpect (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (f : LASState word eos source → ℝ) : ℕ → LASState word eos source → ℝ
  | 0, s => f s
  | n + 1, s => if ha : source.cursorSource s.active = terminal then f s
      else s.pool.groups.expect (fun g => (s.pool.selected g).surrogate.expect
        (fun e => runExpect hs hne hiso f n (s.next ha hs hne hiso g e)))

theorem runExpect_terminal (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (f : LASState word eos source → ℝ) (n : ℕ) (s : LASState word eos source)
    (ht : source.cursorSource s.active = terminal) : runExpect hs hne hiso f n s = f s := by
  cases n <;> simp [runExpect, ht]

theorem runExpect_sub (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (f g : LASState word eos source → ℝ) (n : ℕ) (s : LASState word eos source) :
    runExpect hs hne hiso (fun t => f t - g t) n s =
      runExpect hs hne hiso f n s - runExpect hs hne hiso g n s := by
  induction n generalizing s with
  | zero => rfl
  | succ n ih =>
    by_cases ht : source.cursorSource s.active = terminal
    · simp only [runExpect, dite_eq_left ht]
    · simp only [runExpect, dite_eq_right ht, ih, FiniteLaw.expect_sub]

theorem runExpect_zero_of_terminal (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (f : LASState word eos source → ℝ)
    (hf : ∀ s, source.cursorSource s.active = terminal → f s = 0)
    (n : ℕ) (s : LASState word eos source) (hn : s.rank ≤ n) :
    runExpect hs hne hiso f n s = 0 := by
  induction n generalizing s with
  | zero =>
    have ht : source.cursorSource s.active = terminal := by
      by_contra ha
      have := s.rank_positive ha hne
      omega
    exact hf s ht
  | succ n ih =>
    by_cases ht : source.cursorSource s.active = terminal
    · rw [runExpect_terminal hs hne hiso f (n + 1) s ht, hf s ht]
    · rw [runExpect, dite_eq_right ht]
      have hh : ∀ g e, runExpect hs hne hiso f n (s.next ht hs hne hiso g e) = 0 := by
        intro g e
        apply ih
        have := s.rank_next_lt ht hs hne hiso g e
        omega
      simp only [hh, FiniteLaw.expect_const]

/-- The actual run is terminal with probability one by the rank bound. -/
theorem runExpect_nonterminal_zero (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (n : ℕ) (s : LASState word eos source) (hn : s.rank ≤ n) :
    runExpect hs hne hiso (fun t => if source.cursorSource t.active = terminal then 0 else 1)
      n s = 0 := by
  apply runExpect_zero_of_terminal hs hne hiso _ _ n s hn
  intro t ht
  simp only [ite_eq_left ht]

theorem runExpect_belief (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (f : source.decisionTree.Leaves → ℝ) (n : ℕ) (s : LASState word eos source) :
    runExpect hs hne hiso (fun t => (t.prior.mix (source.cursorLaw word)).expect f) n s =
      (s.prior.mix (source.cursorLaw word)).expect f := by
  induction n generalizing s with
  | zero => rfl
  | succ n ih =>
    by_cases ht : source.cursorSource s.active = terminal
    · rw [runExpect_terminal hs hne hiso _ (n + 1) s ht]
    · simp only [runExpect, dite_eq_right ht, ih]
      exact s.next_belief_expect ht hs hne hiso f

theorem runExpect_tokenPath (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (f : List Token → ℝ) (n : ℕ) (s : LASState word eos source) (hn : s.rank ≤ n) :
    runExpect hs hne hiso (fun t => f (source.cursorHistory t.active)) n s =
      (s.prior.mix (source.cursorLaw word)).expect (fun l => f (source.tokenPath l)) := by
  have hz := runExpect_zero_of_terminal hs hne hiso
    (fun t => (t.prior.mix (source.cursorLaw word)).expect (fun l => f (source.tokenPath l)) -
      f (source.cursorHistory t.active)) (by
        intro t ht
        obtain ⟨l, hl, hh⟩ := source.terminal_cursor_resolved word t.active ht
        rw [t.terminal_point ht, FiniteLaw.point_mix, hl, FiniteLaw.expect_point, hh, sub_self]) n s hn
  rw [runExpect_sub, runExpect_belief] at hz
  linarith

theorem runExpect_visible (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (f : List Byte → ℝ) (n : ℕ) (s : LASState word eos source) (hn : s.rank ≤ n) :
    runExpect hs hne hiso (fun t => f t.emitted) n s =
      (s.prior.mix (source.cursorLaw word)).expect (fun l => f (render word (source.tokenPath l))) := by
  have h := runExpect_tokenPath hs hne hiso (fun h => f (render word h)) n s hn
  simpa only [LASState.active_render] using h

end LASState

/-- Equality for every token-path observable, for the actual stopped execution. -/
theorem concrete_las_tokenPath_law (word : Token → List Byte) (eos : Byte) (source : TokenSource Token)
    (hs : source.EOSSource word eos) (hne : ∀ t, word t ≠ [])
    (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t)) (f : List Token → ℝ) :
    LASState.runExpect hs hne hiso (fun s => f (source.cursorHistory s.active))
      (source.maxRenderedLength word) (LASState.initial word eos source) = source.pathExpect f := by
  rw [LASState.runExpect_tokenPath hs hne hiso f _ _ (Nat.sub_le _ _)]
  change ((FiniteLaw.point source.rootCursor).mix (source.cursorLaw word)).expect _ = _
  rw [FiniteLaw.point_mix, cursorLaw_root, sourceLaw_pathExpect]

/-- Equality for every visible-text observable, including indicators of individual outputs. -/
theorem concrete_las_visible_law (word : Token → List Byte) (eos : Byte) (source : TokenSource Token)
    (hs : source.EOSSource word eos) (hne : ∀ t, word t ≠ [])
    (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t)) (f : List Byte → ℝ) :
    LASState.runExpect hs hne hiso (fun s => f s.emitted)
      (source.maxRenderedLength word) (LASState.initial word eos source) =
        source.pathExpect (fun h => f (render word h)) := by
  have h := source.concrete_las_tokenPath_law word eos hs hne hiso (fun h => f (render word h))
  simpa only [LASState.active_render] using h

end StegaTextProofs.TokenSource
