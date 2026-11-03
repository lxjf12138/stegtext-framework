module

public import StegaTextProofs.IndependentFiniteTable

/-!
A finite protocol with payload choices and independent synchronization choices.
The full synchronization tape includes every unvisited subtree. Given this
tape, only payload choices remain random. Emitted words may also accompany
synchronization choices, as required by native SyncPool token completion.
-/

@[expose] public section

namespace StegaTextProofs

universe u v

attribute [local instance] Classical.propDecidable

inductive ProtocolTree (Byte : Type v) : Type (max (u + 1) v)
  | leaf
  | payload {Choice : Type u} [Fintype Choice] (law : FiniteLaw Choice)
      (emit : Choice → List Byte) (next : Choice → ProtocolTree Byte)
  | sync {Choice : Type u} [Fintype Choice] (law : FiniteLaw Choice)
      (emit : Choice → List Byte) (next : Choice → ProtocolTree Byte)

namespace ProtocolTree

variable {Byte : Type v}

@[reducible] def Leaves : ProtocolTree.{u} Byte → Type u
  | .leaf => PUnit
  | @ProtocolTree.payload _ _C _ _ _ next => (c : _C) × Leaves (next c)
  | @ProtocolTree.sync _ _C _ _ _ next => (c : _C) × Leaves (next c)

noncomputable instance leavesFintype (tree : ProtocolTree.{u} Byte) : Fintype tree.Leaves :=
  match tree with
  | .leaf => inferInstanceAs (Fintype PUnit.{u + 1})
  | @ProtocolTree.payload _ _C _ _ _ next =>
    letI := fun c => leavesFintype (next c)
    inferInstanceAs (Fintype ((c : _C) × Leaves (next c)))
  | @ProtocolTree.sync _ _C _ _ _ next =>
    letI := fun c => leavesFintype (next c)
    inferInstanceAs (Fintype ((c : _C) × Leaves (next c)))

@[reducible] def Tape : ProtocolTree.{u} Byte → Type u
  | .leaf => PUnit
  | @ProtocolTree.payload _ _C _ _ _ next => ∀ c, Tape (next c)
  | @ProtocolTree.sync _ _C _ _ _ next => (_ : _C) × (∀ c, Tape (next c))

noncomputable instance tapeFintype (tree : ProtocolTree.{u} Byte) : Fintype tree.Tape :=
  match tree with
  | .leaf => inferInstanceAs (Fintype PUnit.{u + 1})
  | @ProtocolTree.payload _ _C _ _ _ next =>
    letI := fun c => tapeFintype (next c)
    inferInstanceAs (Fintype (∀ c, Tape (next c)))
  | @ProtocolTree.sync _ _C _ _ _ next =>
    letI := fun c => tapeFintype (next c)
    inferInstanceAs (Fintype ((_ : _C) × (∀ c, Tape (next c))))

noncomputable def tapeLaw : (tree : ProtocolTree.{u} Byte) → FiniteLaw tree.Tape
  | .leaf => FiniteLaw.point PUnit.unit
  | @ProtocolTree.payload _ _C _ _ _ next => FiniteLaw.pi (fun c => tapeLaw (next c))
  | @ProtocolTree.sync _ _C _ p _ next => p.joint (fun _ => FiniteLaw.pi (fun c => tapeLaw (next c)))

noncomputable def givenLaw : (tree : ProtocolTree.{u} Byte) → tree.Tape → FiniteLaw tree.Leaves
  | .leaf, _ => FiniteLaw.point PUnit.unit
  | @ProtocolTree.payload _ _C _ p _ next, r => p.joint (fun c => givenLaw (next c) (r c))
  | @ProtocolTree.sync _ _C _ _ _ next, r => (FiniteLaw.point r.1).joint (fun c => givenLaw (next c) (r.2 c))

noncomputable def budgetNat : ProtocolTree.{u} Byte → ℝ
  | .leaf => 0
  | @ProtocolTree.payload _ _C _ p _ next => p.entropyNat + p.expect (fun c => budgetNat (next c))
  | @ProtocolTree.sync _ _C _ p _ next => p.expect (fun c => budgetNat (next c))

def output : (tree : ProtocolTree.{u} Byte) → tree.Leaves → List Byte
  | .leaf, _ => []
  | @ProtocolTree.payload _ _C _ _ emit next, ⟨c, l⟩ => emit c ++ output (next c) l
  | @ProtocolTree.sync _ _C _ _ emit next, ⟨c, l⟩ => emit c ++ output (next c) l

def Decodable : ProtocolTree.{u} Byte → Prop
  | .leaf => True
  | @ProtocolTree.payload _ _C _ _ emit next =>
      (∀ a b, emit a <+: emit b → a = b) ∧ ∀ c, Decodable (next c)
  | @ProtocolTree.sync _ _C _ _ _ next => ∀ c, Decodable (next c)

/-- Averaging over the full tape leaves exactly the payload entropy budget. -/
theorem expected_given_entropy (tree : ProtocolTree.{u} Byte) :
    tree.tapeLaw.expect (fun r => (tree.givenLaw r).entropyNat) = tree.budgetNat := by
  induction tree with
  | leaf => simp [tapeLaw, givenLaw, budgetNat, FiniteLaw.entropyNat_point]
  | payload p emit next ih =>
    simp only [tapeLaw, givenLaw, budgetNat, FiniteLaw.joint_entropyNat,
      FiniteLaw.expect_add, FiniteLaw.expect_const]
    congr 1
    rw [FiniteLaw.expect_comm]
    congr 1
    funext c
    rw [FiniteLaw.pi_expect_coordinate (fun c => (next c).tapeLaw) c
      (fun r => ((next c).givenLaw r).entropyNat)]
    exact ih c
  | sync p emit next ih =>
    simp only [tapeLaw, givenLaw, budgetNat, FiniteLaw.joint_expect,
      FiniteLaw.joint_entropyNat, FiniteLaw.entropyNat_point, zero_add,
      FiniteLaw.expect_point]
    congr 1
    funext c
    rw [FiniteLaw.pi_expect_coordinate (fun c => (next c).tapeLaw) c
      (fun r => ((next c).givenLaw r).entropyNat)]
    exact ih c

end ProtocolTree

end StegaTextProofs
