module

public import StegaTextProofs.ProtocolTapeRealization

/-! Exact examples distinguish full independent coins from executed synchronization values. -/

@[expose] public section

namespace StegaTextProofs.SyncTapeExamples

noncomputable def fair : FiniteLaw Bool where
  prob _ := 1 / 2
  nonneg _ := by norm_num
  sum_one := by norm_num

theorem fair_entropyNat : fair.entropyNat = Real.log 2 := by
  simp [fair, FiniteLaw.entropyNat, Real.negMulLog]

/-- The complete tape contains separate fair sync choices for both root branches. -/
noncomputable def unusedBranchTape : ProtocolTree Bool :=
  .payload fair (fun b => [b]) (fun _ => .sync fair (fun b => [b]) (fun _ => .leaf))

theorem unusedBranchTape_decodable : unusedBranchTape.Decodable := by
  refine ⟨?_, fun _ _ => trivial⟩
  intro a b h
  simpa using h

theorem unusedBranchTape_conditionalEntropy : unusedBranchTape.observedConditionalEntropy = 1 := by
  rw [unusedBranchTape.observedConditionalEntropy_eq_budget unusedBranchTape_decodable]
  simp only [unusedBranchTape, ProtocolTree.budgetNat, FiniteLaw.expect_const, add_zero, fair_entropyNat]
  exact div_self (ne_of_gt (Real.log_pos (by norm_num)))

/-- The synchronized value equals the payload branch, although the required coins are deterministic. -/
noncomputable def stateDependentSync : ProtocolTree Bool :=
  .payload fair (fun b => [b]) (fun b => .sync (FiniteLaw.point b) (fun _ => []) (fun _ => .leaf))

theorem stateDependentSync_decodable : stateDependentSync.Decodable := by
  refine ⟨?_, fun _ _ => trivial⟩
  intro a b h
  simpa using h

theorem stateDependentSync_fullTape_entropy : stateDependentSync.observedConditionalEntropy = 1 := by
  rw [stateDependentSync.observedConditionalEntropy_eq_budget stateDependentSync_decodable]
  simp only [stateDependentSync, ProtocolTree.budgetNat, FiniteLaw.expect_const, add_zero, fair_entropyNat]
  exact div_self (ne_of_gt (Real.log_pos (by norm_num)))

/-- Replacing the independent tape by the executed sync value would incorrectly remove one bit. -/
theorem executedSyncValue_conditionalEntropy :
    fair.conditionalEntropyNat (fun b => FiniteLaw.point b) / Real.log 2 = 0 := by
  simp only [FiniteLaw.conditionalEntropyNat_eq_expect, FiniteLaw.entropyNat_point,
    FiniteLaw.expect_const, zero_div]

end StegaTextProofs.SyncTapeExamples
