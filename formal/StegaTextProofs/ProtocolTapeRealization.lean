module

public import StegaTextProofs.IndependentProtocolTapes

/-!
The entropy result depends on independence from payload and correct executed
trace law, not on a particular representation of unused synchronization coins.
Arbitrary finite raw tapes may correlate counterfactual branch choices.
-/

@[expose] public section

namespace StegaTextProofs

namespace FiniteLaw

variable {α : Type*} [Fintype α]

theorem entropyNat_eq_expect_neglog (p : FiniteLaw α) :
    p.entropyNat = p.expect (fun a => -Real.log (p.prob a)) := by
  simp only [entropyNat, expect, Real.negMulLog, mul_neg, neg_mul]

end FiniteLaw

namespace ProtocolTree

universe u v
attribute [local instance] Classical.propDecidable
variable {Byte : Type v}

noncomputable def payloadCost : (tree : ProtocolTree.{u} Byte) → tree.Leaves → ℝ
  | .leaf, _ => 0
  | @ProtocolTree.payload _ _C _ p _ next, ⟨c, l⟩ => -Real.log (p.prob c) + payloadCost (next c) l
  | @ProtocolTree.sync _ _C _ _ _ next, ⟨c, l⟩ => payloadCost (next c) l

theorem given_entropy_eq_cost (tree : ProtocolTree.{u} Byte) (r : tree.Tape) :
    (tree.givenLaw r).entropyNat = (tree.givenLaw r).expect tree.payloadCost := by
  induction tree with
  | leaf => simp only [givenLaw, FiniteLaw.entropyNat_point, payloadCost, FiniteLaw.expect_const]
  | payload p emit next ih =>
    simp only [givenLaw, FiniteLaw.joint_entropyNat, FiniteLaw.joint_expect, payloadCost,
      FiniteLaw.expect_add, FiniteLaw.expect_const, ih]
    rw [p.entropyNat_eq_expect_neglog]
  | sync p emit next ih =>
    simp only [givenLaw, FiniteLaw.joint_entropyNat, FiniteLaw.entropyNat_point, zero_add,
      FiniteLaw.expect_point, FiniteLaw.joint_expect, payloadCost, ih]

theorem pathCost_eq_budget (tree : ProtocolTree.{u} Byte) :
    tree.pathLaw.expect tree.payloadCost = tree.budgetNat := by
  induction tree with
  | leaf => simp only [pathLaw, payloadCost, FiniteLaw.expect_const, budgetNat]
  | payload p emit next ih =>
    simp only [pathLaw, FiniteLaw.joint_expect, payloadCost, FiniteLaw.expect_add,
      FiniteLaw.expect_const, ih, budgetNat, FiniteLaw.entropyNat_eq_expect_neglog]
  | sync p emit next ih => simp only [pathLaw, FiniteLaw.joint_expect, payloadCost, ih, budgetNat]

variable {Raw : Type*} [Fintype Raw]

noncomputable def rawObservedLaw (tree : ProtocolTree.{u} Byte) (p : FiniteLaw Raw)
    (read : Raw → tree.Tape) : FiniteLaw ((_ : Raw) × tree.Visible) :=
  (p.joint (fun _ => tree.payloadLaw)).map
    (fun rz => ⟨rz.1, tree.readout (tree.evaluate (read rz.1) rz.2)⟩)

theorem rawObservedLaw_eq_joint (tree : ProtocolTree.{u} Byte) (p : FiniteLaw Raw)
    (read : Raw → tree.Tape) :
    tree.rawObservedLaw p read = p.joint (fun r => tree.visibleGivenLaw (read r)) := by
  have h : ∀ f : ((_ : Raw) × tree.Visible) → ℝ,
      (tree.rawObservedLaw p read).expect f =
        (p.joint (fun r => tree.visibleGivenLaw (read r))).expect f := by
    intro f
    simp only [rawObservedLaw, FiniteLaw.map_expect, FiniteLaw.joint_expect]
    congr 1
    funext r
    rw [tree.evaluate_expect (read r) (fun l => f ⟨r, tree.readout l⟩)]
    exact ((tree.givenLaw (read r)).map_expect tree.readout (fun y => f ⟨r, y⟩)).symm
  apply FiniteLaw.ext_prob
  intro x
  simpa [FiniteLaw.expect] using h (fun y => if y = x then 1 else 0)

noncomputable def rawConditionalEntropy (tree : ProtocolTree.{u} Byte) (p : FiniteLaw Raw)
    (read : Raw → tree.Tape) : ℝ :=
  ((tree.rawObservedLaw p read).entropyNat - p.entropyNat) / Real.log 2

/-- Correct trace generation and independent payload suffice; the final inequality is not assumed. -/
theorem rawConditionalEntropy_eq_budget (tree : ProtocolTree.{u} Byte) (hd : tree.Decodable)
    (p : FiniteLaw Raw) (read : Raw → tree.Tape)
    (hcorrect : p.mix (fun r => tree.givenLaw (read r)) = tree.pathLaw) :
    tree.rawConditionalEntropy p read = tree.budgetNat / Real.log 2 := by
  rw [rawConditionalEntropy, tree.rawObservedLaw_eq_joint]
  change p.conditionalEntropyNat (fun r => tree.visibleGivenLaw (read r)) / Real.log 2 = _
  rw [FiniteLaw.conditionalEntropyNat_eq_expect]
  simp only [tree.visibleGiven_entropy hd, tree.given_entropy_eq_cost]
  rw [← p.expect_mix (fun r => tree.givenLaw (read r)) tree.payloadCost, hcorrect, tree.pathCost_eq_budget]

end ProtocolTree

end StegaTextProofs
