module

public import StegaTextProofs.ProtocolTranscript

/-! Independent complete payload and synchronization tables generate the conditional law. -/

@[expose] public section

namespace StegaTextProofs.ProtocolTree

universe u v
attribute [local instance] Classical.propDecidable
variable {Byte : Type v}

@[reducible] def PayloadTape : ProtocolTree.{u} Byte → Type u
  | .leaf => PUnit
  | @ProtocolTree.payload _ C _ _ _ next => (_ : C) × (∀ c, PayloadTape (next c))
  | @ProtocolTree.sync _ _C _ _ _ next => ∀ c, PayloadTape (next c)

noncomputable instance payloadTapeFintype (tree : ProtocolTree.{u} Byte) : Fintype tree.PayloadTape :=
  match tree with
  | .leaf => inferInstanceAs (Fintype PUnit.{u + 1})
  | @ProtocolTree.payload _ C _ _ _ next =>
    letI := fun c => payloadTapeFintype (next c)
    inferInstanceAs (Fintype ((_ : C) × (∀ c, PayloadTape (next c))))
  | @ProtocolTree.sync _ _C _ _ _ next =>
    letI := fun c => payloadTapeFintype (next c)
    inferInstanceAs (Fintype (∀ c, PayloadTape (next c)))

noncomputable def payloadLaw : (tree : ProtocolTree.{u} Byte) → FiniteLaw tree.PayloadTape
  | .leaf => FiniteLaw.point PUnit.unit
  | @ProtocolTree.payload _ _C _ p _ next =>
      p.joint (fun _ => FiniteLaw.pi (fun c => payloadLaw (next c)))
  | @ProtocolTree.sync _ _C _ _ _ next => FiniteLaw.pi (fun c => payloadLaw (next c))

def evaluate : (tree : ProtocolTree.{u} Byte) → tree.Tape → tree.PayloadTape → tree.Leaves
  | .leaf, _, _ => PUnit.unit
  | @ProtocolTree.payload _ _C _ _ _ next, r, z =>
      ⟨z.1, evaluate (next z.1) (r z.1) (z.2 z.1)⟩
  | @ProtocolTree.sync _ _C _ _ _ next, r, z =>
      ⟨r.1, evaluate (next r.1) (r.2 r.1) (z r.1)⟩

/-- This is a product distribution: the complete two tapes are independent. -/
noncomputable def independentTapes (tree : ProtocolTree.{u} Byte) :
    FiniteLaw ((_ : tree.Tape) × tree.PayloadTape) :=
  tree.tapeLaw.joint (fun _ => tree.payloadLaw)

theorem independentTapes_prob (tree : ProtocolTree.{u} Byte) (r : tree.Tape) (z : tree.PayloadTape) :
    tree.independentTapes.prob ⟨r, z⟩ = tree.tapeLaw.prob r * tree.payloadLaw.prob z := rfl

theorem evaluate_expect (tree : ProtocolTree.{u} Byte) (r : tree.Tape) (f : tree.Leaves → ℝ) :
    tree.payloadLaw.expect (fun z => f (tree.evaluate r z)) = (tree.givenLaw r).expect f := by
  induction tree with
  | leaf => simp [payloadLaw, evaluate, givenLaw]
  | payload p emit next ih =>
    simp only [payloadLaw, evaluate, givenLaw, FiniteLaw.joint_expect]
    congr 1
    funext c
    rw [FiniteLaw.pi_expect_coordinate (fun c => (next c).payloadLaw) c
      (fun z => f ⟨c, (next c).evaluate (r c) z⟩)]
    exact ih c (r c) (fun l => f ⟨c, l⟩)
  | sync p emit next ih =>
    simp only [payloadLaw, evaluate, givenLaw, FiniteLaw.joint_expect, FiniteLaw.expect_point]
    rw [FiniteLaw.pi_expect_coordinate (fun c => (next c).payloadLaw) r.1
      (fun z => f ⟨r.1, (next r.1).evaluate (r.2 r.1) z⟩)]
    exact ih r.1 (r.2 r.1) (fun l => f ⟨r.1, l⟩)

theorem independentTapes_expect (tree : ProtocolTree.{u} Byte)
    (f : tree.Tape → tree.Leaves → ℝ) :
    tree.independentTapes.expect (fun rz => f rz.1 (tree.evaluate rz.1 rz.2)) =
      tree.tapeLaw.expect (fun r => (tree.givenLaw r).expect (f r)) := by
  simp only [independentTapes, FiniteLaw.joint_expect, tree.evaluate_expect]

/-- The actual joint law of the full sync tape and final visible text. -/
noncomputable def observedLaw (tree : ProtocolTree.{u} Byte) :
    FiniteLaw ((_ : tree.Tape) × tree.Visible) :=
  tree.independentTapes.map (fun rz => ⟨rz.1, tree.readout (tree.evaluate rz.1 rz.2)⟩)

theorem observedLaw_eq_joint (tree : ProtocolTree.{u} Byte) :
    tree.observedLaw = tree.tapeLaw.joint tree.visibleGivenLaw := by
  have h : ∀ f : ((_ : tree.Tape) × tree.Visible) → ℝ,
      tree.observedLaw.expect f = (tree.tapeLaw.joint tree.visibleGivenLaw).expect f := by
    intro f
    rw [observedLaw, FiniteLaw.map_expect,
      tree.independentTapes_expect (fun r l => f ⟨r, tree.readout l⟩), FiniteLaw.joint_expect]
    simp only [visibleGivenLaw, FiniteLaw.map_expect]
  apply FiniteLaw.ext_prob
  intro x
  simpa [FiniteLaw.expect] using h (fun y => if y = x then 1 else 0)

theorem observedLaw_tape_expect (tree : ProtocolTree.{u} Byte) (f : tree.Tape → ℝ) :
    tree.observedLaw.expect (fun ry => f ry.1) = tree.tapeLaw.expect f := by
  rw [tree.observedLaw_eq_joint, FiniteLaw.joint_expect]
  simp only [FiniteLaw.expect_const]

noncomputable def observedConditionalEntropy (tree : ProtocolTree.{u} Byte) : ℝ :=
  (tree.observedLaw.entropyNat - tree.tapeLaw.entropyNat) / Real.log 2

theorem observedConditionalEntropy_eq_budget (tree : ProtocolTree.{u} Byte) (hd : tree.Decodable) :
    tree.observedConditionalEntropy = tree.budgetNat / Real.log 2 := by
  rw [observedConditionalEntropy, tree.observedLaw_eq_joint]
  exact tree.visibleConditionalEntropy_eq_budget hd

noncomputable def pathLaw : (tree : ProtocolTree.{u} Byte) → FiniteLaw tree.Leaves
  | .leaf => FiniteLaw.point PUnit.unit
  | @ProtocolTree.payload _ _C _ p _ next => p.joint (fun c => pathLaw (next c))
  | @ProtocolTree.sync _ _C _ p _ next => p.joint (fun c => pathLaw (next c))

noncomputable def outputExpect (tree : ProtocolTree.{u} Byte) (f : List Byte → ℝ) : ℝ :=
  tree.pathLaw.expect (fun l => f (tree.output l))

theorem outputExpect_leaf (f : List Byte → ℝ) :
    (ProtocolTree.leaf : ProtocolTree.{u} Byte).outputExpect f = f [] := by
  exact FiniteLaw.expect_const (FiniteLaw.point PUnit.unit) (f [])

theorem outputExpect_payload {C : Type u} [Fintype C] (p : FiniteLaw C)
    (emit : C → List Byte) (next : C → ProtocolTree.{u} Byte) (f : List Byte → ℝ) :
    (ProtocolTree.payload p emit next).outputExpect f =
      p.expect (fun c => (next c).outputExpect (fun w => f (emit c ++ w))) :=
  FiniteLaw.joint_expect p (fun c => (next c).pathLaw) _

theorem outputExpect_sync {C : Type u} [Fintype C] (p : FiniteLaw C)
    (emit : C → List Byte) (next : C → ProtocolTree.{u} Byte) (f : List Byte → ℝ) :
    (ProtocolTree.sync p emit next).outputExpect f =
      p.expect (fun c => (next c).outputExpect (fun w => f (emit c ++ w))) :=
  FiniteLaw.joint_expect p (fun c => (next c).pathLaw) _

theorem averaged_given_expect (tree : ProtocolTree.{u} Byte) (f : tree.Leaves → ℝ) :
    tree.tapeLaw.expect (fun r => (tree.givenLaw r).expect f) = tree.pathLaw.expect f := by
  induction tree with
  | leaf => simp [tapeLaw, givenLaw, pathLaw]
  | payload p emit next ih =>
    simp only [tapeLaw, givenLaw, pathLaw, FiniteLaw.joint_expect]
    rw [FiniteLaw.expect_comm]
    congr 1
    funext c
    rw [FiniteLaw.pi_expect_coordinate (fun c => (next c).tapeLaw) c
      (fun r => ((next c).givenLaw r).expect (fun l => f ⟨c, l⟩))]
    exact ih c (fun l => f ⟨c, l⟩)
  | sync p emit next ih =>
    simp only [tapeLaw, givenLaw, pathLaw, FiniteLaw.joint_expect, FiniteLaw.expect_point]
    congr 1
    funext c
    rw [FiniteLaw.pi_expect_coordinate (fun c => (next c).tapeLaw) c
      (fun r => ((next c).givenLaw r).expect (fun l => f ⟨c, l⟩))]
    exact ih c (fun l => f ⟨c, l⟩)

end StegaTextProofs.ProtocolTree
