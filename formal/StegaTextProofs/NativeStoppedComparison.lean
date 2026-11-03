module

public import StegaTextProofs.NativeFrontierModel
public import StegaTextProofs.StoppedProtocol

/-!
Native source entropy is bounded by the LAS budget earned strictly before
stopping. No entropy from artificial terminal steps is counted, and no
closure condition on the original kernel at terminal states is required.
-/

@[expose] public section

namespace StegaTextProofs.TokenSource

variable {Token Byte State : Type*} [Fintype Token] [Fintype State]
variable {Group : State → Type*} [∀ s, Fintype (Group s)]
variable {Exact Strict : (s : State) → Group s → Type*}
  [∀ s k, Fintype (Exact s k)] [∀ s k, Fintype (Strict s k)]

theorem native_le_stopped_las (word : Token → List Byte) (source : TokenSource Token)
    (model : LASDecisionModel source.decisionTree State Group Exact Strict)
    (initial : State) (hroot : model.belief initial = source.sourceLaw word)
    (terminal : State → Prop) (rank : State → ℕ)
    (hdecrease : ∀ s t, ¬ terminal s → 0 < (model.kernel s).prob t → rank t < rank s)
    (hresolved : ∀ s, terminal s → ∃ l, model.belief s = FiniteLaw.point l) :
    source.nativeValue word ≤ FiniteHorizon.stoppedBudget model.kernel
      (fun s => (model.groups s).entropy) terminal (rank initial) initial := by
  have hzero : ∀ s, terminal s → model.potential s = 0 := by
    intro s hs
    obtain ⟨l, hl⟩ := hresolved s hs
    exact model.potential_zero_of_resolved s l hl
  have h := FiniteHorizon.potential_le_stoppedBudget model.kernel model.rewardNat model.potential
    terminal rank hdecrease hzero (fun s _ => model.selective_step_inequality s)
    (rank initial) initial le_rfl
  have hlog : 0 < Real.log 2 := Real.log_pos (by norm_num)
  have hreward : model.rewardNat = fun s => Real.log 2 * (model.groups s).entropy := by
    funext s
    dsimp [LASDecisionModel.rewardNat, FiniteLaw.entropy]
    exact (mul_div_cancel₀ _ (ne_of_gt hlog)).symm
  rw [LASDecisionModel.potential, hroot, source.sourceLaw_selective word,
    hreward, FiniteHorizon.stoppedBudget_smul] at h
  apply (div_le_iff₀ hlog).mpr
  simpa only [mul_comm] using h

theorem native_le_stopped_las_visible (word : Token → List Byte) (source : TokenSource Token)
    (model : LASDecisionModel source.decisionTree State Group Exact Strict)
    (initial : State) (hroot : model.belief initial = source.sourceLaw word)
    (terminal : State → Prop) (emitted : State → List Byte)
    (hprefix : ∀ s l, 0 < (model.belief s).prob l →
      emitted s <+: PrefixGrouping.render word (source.tokenPath l))
    (hprogress : ∀ s t, ¬ terminal s → 0 < (model.kernel s).prob t →
      ∃ key : List Byte, key ≠ [] ∧ emitted t = emitted s ++ key)
    (hresolved : ∀ s, terminal s → ∃ l, model.belief s = FiniteLaw.point l) :
    source.nativeValue word ≤ FiniteHorizon.stoppedBudget model.kernel
      (fun s => (model.groups s).entropy) terminal
        (VisibleTermination.maxLength (fun l =>
          PrefixGrouping.render word (source.tokenPath l)) - (emitted initial).length) initial := by
  apply source.native_le_stopped_las word model initial hroot terminal
    (fun s => VisibleTermination.maxLength (fun l =>
      PrefixGrouping.render word (source.tokenPath l)) - (emitted s).length) _ hresolved
  intro s t hs ht
  obtain ⟨key, hne, heq⟩ := hprogress s t hs ht
  exact VisibleTermination.rank_decreases model.belief _ emitted hprefix s t
    (VisibleTermination.length_progress_of_append emitted s t key hne heq)

end StegaTextProofs.TokenSource
