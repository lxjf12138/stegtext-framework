module

public import StegaTextProofs.FullSyncState

/-! The actual finite adaptive run, with state/group-dependent gates. -/

@[expose] public section

namespace StegaTextProofs.TokenSource

attribute [local instance] Classical.propDecidable
open PrefixGrouping
variable {Token Byte : Type*} [Fintype Token]

namespace LASState

variable {word : Token → List Byte} {eos : Byte} {source : TokenSource Token}

/-- True selects LAS; false selects full-candidate synchronization. -/
abbrev AdaptiveGate (word : Token → List Byte) (eos : Byte) (source : TokenSource Token) :=
  (s : LASState word eos source) → s.pool.Group → Bool

variable (hs : source.EOSSource word eos) (hne : ∀ t, word t ≠ [])
  (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))

noncomputable def groupExpect (s : LASState word eos source)
    (ha : source.cursorSource s.active ≠ terminal) (useLAS : Bool) (g : s.pool.Group)
    (f : LASState word eos source → ℝ) : ℝ :=
  if useLAS then (s.pool.selected g).surrogate.expect
    (fun e => f (s.next ha hs hne hiso g e))
  else (s.pool.selected g).prior.expect (fun c => f (s.fullSync g c))

theorem groupExpect_const (s : LASState word eos source)
    (ha : source.cursorSource s.active ≠ terminal) (b : Bool) (g : s.pool.Group) (a : ℝ) :
    groupExpect hs hne hiso s ha b g (fun _ => a) = a := by
  cases b <;> simp [groupExpect]

theorem groupExpect_add (s : LASState word eos source)
    (ha : source.cursorSource s.active ≠ terminal) (b : Bool) (g : s.pool.Group)
    (f k : LASState word eos source → ℝ) :
    groupExpect hs hne hiso s ha b g (fun t => f t + k t) =
      groupExpect hs hne hiso s ha b g f + groupExpect hs hne hiso s ha b g k := by
  cases b <;> simp only [groupExpect, Bool.false_eq_true, ↓reduceIte,
    FiniteLaw.expect_add]

theorem groupExpect_sub (s : LASState word eos source)
    (ha : source.cursorSource s.active ≠ terminal) (b : Bool) (g : s.pool.Group)
    (f k : LASState word eos source → ℝ) :
    groupExpect hs hne hiso s ha b g (fun t => f t - k t) =
      groupExpect hs hne hiso s ha b g f - groupExpect hs hne hiso s ha b g k := by
  cases b <;> simp only [groupExpect, Bool.false_eq_true, ↓reduceIte,
    FiniteLaw.expect_sub]

theorem groupExpect_mul (s : LASState word eos source)
    (ha : source.cursorSource s.active ≠ terminal) (b : Bool) (g : s.pool.Group)
    (a : ℝ) (f : LASState word eos source → ℝ) :
    groupExpect hs hne hiso s ha b g (fun t => a * f t) =
      a * groupExpect hs hne hiso s ha b g f := by
  cases b <;> simp only [groupExpect, Bool.false_eq_true, ↓reduceIte, FiniteLaw.expect_mul]

theorem groupExpect_belief (s : LASState word eos source)
    (ha : source.cursorSource s.active ≠ terminal) (b : Bool) (g : s.pool.Group)
    (f : source.decisionTree.Leaves → ℝ) :
    groupExpect hs hne hiso s ha b g
      (fun t => (t.prior.mix (source.cursorLaw word)).expect f) =
        ((s.pool.selected g).prior.mix
          (fun c => source.cursorLaw word (s.pool.candidate g c).val)).expect f := by
  cases b <;> simp only [groupExpect, Bool.false_eq_true, ↓reduceIte,
    fullSync_posterior, next_posterior, PrefixPool.posterior,
    FiniteLaw.expect_mix, LASGroup.las_collapse_unbiased]

variable (gate : AdaptiveGate word eos source)

noncomputable def adaptiveStep (s : LASState word eos source)
    (ha : source.cursorSource s.active ≠ terminal) (f : LASState word eos source → ℝ) : ℝ :=
  s.pool.groups.expect (fun g => groupExpect hs hne hiso s ha (gate s g) g f)

theorem adaptiveStep_const (s : LASState word eos source)
    (ha : source.cursorSource s.active ≠ terminal) (a : ℝ) :
    adaptiveStep hs hne hiso gate s ha (fun _ => a) = a := by
  simp only [adaptiveStep, groupExpect_const, FiniteLaw.expect_const]

theorem adaptiveStep_add (s : LASState word eos source)
    (ha : source.cursorSource s.active ≠ terminal) (f k : LASState word eos source → ℝ) :
    adaptiveStep hs hne hiso gate s ha (fun t => f t + k t) =
      adaptiveStep hs hne hiso gate s ha f + adaptiveStep hs hne hiso gate s ha k := by
  simp only [adaptiveStep, groupExpect_add, FiniteLaw.expect_add]

theorem adaptiveStep_sub (s : LASState word eos source)
    (ha : source.cursorSource s.active ≠ terminal) (f k : LASState word eos source → ℝ) :
    adaptiveStep hs hne hiso gate s ha (fun t => f t - k t) =
      adaptiveStep hs hne hiso gate s ha f - adaptiveStep hs hne hiso gate s ha k := by
  simp only [adaptiveStep, groupExpect_sub, FiniteLaw.expect_sub]

theorem adaptiveStep_mul (s : LASState word eos source)
    (ha : source.cursorSource s.active ≠ terminal) (a : ℝ) (f : LASState word eos source → ℝ) :
    adaptiveStep hs hne hiso gate s ha (fun t => a * f t) =
      a * adaptiveStep hs hne hiso gate s ha f := by
  simp only [adaptiveStep, groupExpect_mul, FiniteLaw.expect_mul]

theorem adaptiveStep_congr_rank (s : LASState word eos source)
    (ha : source.cursorSource s.active ≠ terminal) (f k : LASState word eos source → ℝ)
    (h : ∀ t, t.rank < s.rank → f t = k t) :
    adaptiveStep hs hne hiso gate s ha f = adaptiveStep hs hne hiso gate s ha k := by
  unfold adaptiveStep groupExpect
  congr 1
  funext g
  split
  · congr 1
    funext e
    exact h _ (s.rank_next_lt ha hs hne hiso g e)
  · congr 1
    funext c
    exact h _ (s.rank_fullSync_lt ha hne g c)

theorem adaptiveStep_mono_rank (s : LASState word eos source)
    (ha : source.cursorSource s.active ≠ terminal) (f k : LASState word eos source → ℝ)
    (h : ∀ t, t.rank < s.rank → f t ≤ k t) :
    adaptiveStep hs hne hiso gate s ha f ≤ adaptiveStep hs hne hiso gate s ha k := by
  apply FiniteLaw.expect_mono
  intro g
  unfold groupExpect
  split
  · exact FiniteLaw.expect_mono _ (fun e => h _ (s.rank_next_lt ha hs hne hiso g e))
  · exact FiniteLaw.expect_mono _ (fun c => h _ (s.rank_fullSync_lt ha hne g c))

theorem adaptiveStep_belief (s : LASState word eos source)
    (ha : source.cursorSource s.active ≠ terminal) (f : source.decisionTree.Leaves → ℝ) :
    adaptiveStep hs hne hiso gate s ha
      (fun t => (t.prior.mix (source.cursorLaw word)).expect f) =
        (s.prior.mix (source.cursorLaw word)).expect f := by
  simp only [adaptiveStep, groupExpect_belief]
  rw [← FiniteLaw.expect_mix,
    s.pool.disintegrate_channel (fun c => source.cursorLaw word c.val)]
  have hpool : s.pool.prior.mix (fun c => source.cursorLaw word c.val) =
      s.prior.mix (source.cursorLaw word) :=
    ((source.expandActive s.prior s.active).supportLaw_mix (source.cursorLaw word)).trans
      (source.expandActive_continuation word s.prior s.active)
  rw [hpool]

noncomputable def adaptiveRun (f : LASState word eos source → ℝ) :
    ℕ → LASState word eos source → ℝ
  | 0, s => f s
  | n + 1, s => if ha : source.cursorSource s.active = terminal then f s
      else adaptiveStep hs hne hiso gate s ha (fun t => adaptiveRun f n t)

theorem adaptiveRun_terminal (f : LASState word eos source → ℝ)
    (n : ℕ) (s : LASState word eos source) (ht : source.cursorSource s.active = terminal) :
    adaptiveRun hs hne hiso gate f n s = f s := by
  cases n <;> simp [adaptiveRun, ht]

theorem adaptiveRun_sub (f k : LASState word eos source → ℝ)
    (n : ℕ) (s : LASState word eos source) :
    adaptiveRun hs hne hiso gate (fun t => f t - k t) n s =
      adaptiveRun hs hne hiso gate f n s - adaptiveRun hs hne hiso gate k n s := by
  induction n generalizing s with
  | zero => rfl
  | succ n ih =>
    by_cases ht : source.cursorSource s.active = terminal
    · simp only [adaptiveRun, dite_eq_left ht]
    · simp only [adaptiveRun, dite_eq_right ht, ih, adaptiveStep_sub]

theorem adaptiveRun_zero_of_terminal (f : LASState word eos source → ℝ)
    (hf : ∀ s, source.cursorSource s.active = terminal → f s = 0)
    (n : ℕ) (s : LASState word eos source) (hn : s.rank ≤ n) :
    adaptiveRun hs hne hiso gate f n s = 0 := by
  induction n generalizing s with
  | zero =>
    have ht : source.cursorSource s.active = terminal := by
      by_contra ha
      have := s.rank_positive ha hne
      omega
    exact hf s ht
  | succ n ih =>
    by_cases ht : source.cursorSource s.active = terminal
    · rw [adaptiveRun_terminal hs hne hiso gate f (n + 1) s ht, hf s ht]
    · rw [adaptiveRun, dite_eq_right ht]
      calc
        _ = adaptiveStep hs hne hiso gate s ht (fun _ => 0) := by
          apply adaptiveStep_congr_rank
          intro t hlt
          apply ih
          omega
        _ = 0 := adaptiveStep_const hs hne hiso gate s ht 0

theorem adaptiveRun_nonterminal_zero (n : ℕ) (s : LASState word eos source)
    (hn : s.rank ≤ n) :
    adaptiveRun hs hne hiso gate
      (fun t => if source.cursorSource t.active = terminal then 0 else 1) n s = 0 := by
  apply adaptiveRun_zero_of_terminal hs hne hiso gate _ _ n s hn
  intro t ht
  simp only [ite_eq_left ht]

theorem adaptiveRun_stable (f : LASState word eos source → ℝ)
    (n m : ℕ) (s : LASState word eos source) (hn : s.rank ≤ n) :
    adaptiveRun hs hne hiso gate f (n + m) s = adaptiveRun hs hne hiso gate f n s := by
  induction n generalizing s with
  | zero =>
    have ht : source.cursorSource s.active = terminal := by
      by_contra ha
      have := s.rank_positive ha hne
      omega
    rw [adaptiveRun_terminal hs hne hiso gate f (0 + m) s ht]
    rfl
  | succ n ih =>
    by_cases ht : source.cursorSource s.active = terminal
    · rw [adaptiveRun_terminal hs hne hiso gate f (n + 1 + m) s ht,
        adaptiveRun_terminal hs hne hiso gate f (n + 1) s ht]
    · rw [Nat.succ_add]
      simp only [adaptiveRun, dite_eq_right ht]
      apply adaptiveStep_congr_rank
      intro t hlt
      apply ih
      omega

theorem adaptiveRun_belief (f : source.decisionTree.Leaves → ℝ)
    (n : ℕ) (s : LASState word eos source) :
    adaptiveRun hs hne hiso gate
      (fun t => (t.prior.mix (source.cursorLaw word)).expect f) n s =
        (s.prior.mix (source.cursorLaw word)).expect f := by
  induction n generalizing s with
  | zero => rfl
  | succ n ih =>
    by_cases ht : source.cursorSource s.active = terminal
    · rw [adaptiveRun_terminal hs hne hiso gate _ (n + 1) s ht]
    · simp only [adaptiveRun, dite_eq_right ht, ih]
      exact adaptiveStep_belief hs hne hiso gate s ht f

theorem adaptiveRun_tokenPath (f : List Token → ℝ) (n : ℕ)
    (s : LASState word eos source) (hn : s.rank ≤ n) :
    adaptiveRun hs hne hiso gate (fun t => f (source.cursorHistory t.active)) n s =
      (s.prior.mix (source.cursorLaw word)).expect (fun l => f (source.tokenPath l)) := by
  have hz := adaptiveRun_zero_of_terminal hs hne hiso gate
    (fun t => (t.prior.mix (source.cursorLaw word)).expect (fun l => f (source.tokenPath l)) -
      f (source.cursorHistory t.active)) (by
        intro t ht
        obtain ⟨l, hl, hh⟩ := source.terminal_cursor_resolved word t.active ht
        rw [t.terminal_point ht, FiniteLaw.point_mix, hl, FiniteLaw.expect_point, hh, sub_self]) n s hn
  rw [adaptiveRun_sub, adaptiveRun_belief] at hz
  linarith

theorem adaptiveRun_visible (f : List Byte → ℝ) (n : ℕ)
    (s : LASState word eos source) (hn : s.rank ≤ n) :
    adaptiveRun hs hne hiso gate (fun t => f t.emitted) n s =
      (s.prior.mix (source.cursorLaw word)).expect
        (fun l => f (render word (source.tokenPath l))) := by
  have h := adaptiveRun_tokenPath hs hne hiso gate (fun h => f (render word h)) n s hn
  simpa only [LASState.active_render] using h

end LASState

theorem concrete_adaptive_tokenPath_law (word : Token → List Byte) (eos : Byte)
    (source : TokenSource Token) (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (gate : LASState.AdaptiveGate word eos source) (f : List Token → ℝ) :
    LASState.adaptiveRun hs hne hiso gate (fun s => f (source.cursorHistory s.active))
      (source.maxRenderedLength word) (LASState.initial word eos source) = source.pathExpect f := by
  rw [LASState.adaptiveRun_tokenPath hs hne hiso gate f _ _ (Nat.sub_le _ _)]
  change ((FiniteLaw.point source.rootCursor).mix (source.cursorLaw word)).expect _ = _
  rw [FiniteLaw.point_mix, cursorLaw_root, sourceLaw_pathExpect]

theorem concrete_adaptive_visible_law (word : Token → List Byte) (eos : Byte)
    (source : TokenSource Token) (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (gate : LASState.AdaptiveGate word eos source) (f : List Byte → ℝ) :
    LASState.adaptiveRun hs hne hiso gate (fun s => f s.emitted)
      (source.maxRenderedLength word) (LASState.initial word eos source) =
        source.pathExpect (fun h => f (render word h)) := by
  have h := source.concrete_adaptive_tokenPath_law word eos hs hne hiso gate
    (fun h => f (render word h))
  simpa only [LASState.active_render] using h

end StegaTextProofs.TokenSource
