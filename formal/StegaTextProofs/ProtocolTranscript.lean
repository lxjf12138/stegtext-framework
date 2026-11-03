module

public import StegaTextProofs.FiniteProtocolTree

/-! Prefix-free payload words recover the transcript when the full sync tape is fixed. -/

@[expose] public section

namespace StegaTextProofs.ProtocolTree

universe u v
variable {Byte : Type v}

theorem output_injective_given (tree : ProtocolTree.{u} Byte) (hd : tree.Decodable)
    (r : tree.Tape) (a b : tree.Leaves)
    (ha : 0 < (tree.givenLaw r).prob a) (hb : 0 < (tree.givenLaw r).prob b)
    (he : tree.output a = tree.output b) : a = b := by
  classical
  induction tree with
  | leaf => exact Subsingleton.elim _ _
  | payload p emit next ih =>
    rcases a with ⟨c, a⟩
    rcases b with ⟨d, b⟩
    change emit c ++ (next c).output a = emit d ++ (next d).output b at he
    have hcd : c = d := by
      have hp : emit c <+: emit d ++ (next d).output b :=
        he ▸ List.prefix_append (emit c) ((next c).output a)
      rcases List.prefix_or_prefix_of_prefix hp (List.prefix_append (emit d) ((next d).output b)) with h | h
      · exact hd.1 c d h
      · exact (hd.1 d c h).symm
    subst d
    have hac : 0 < ((next c).givenLaw (r c)).prob a :=
      pos_of_mul_pos_right ha (p.nonneg c)
    have hbc : 0 < ((next c).givenLaw (r c)).prob b :=
      pos_of_mul_pos_right hb (p.nonneg c)
    have hab := ih c (hd.2 c) (r c) a b hac hbc (List.append_cancel_left he)
    exact congrArg (Sigma.mk c) hab
  | sync p emit next ih =>
    rcases a with ⟨c, a⟩
    rcases b with ⟨d, b⟩
    have hc : c = r.1 := by
      by_contra hn
      simp [givenLaw, FiniteLaw.joint, FiniteLaw.point, hn] at ha
    have hd' : d = r.1 := by
      by_contra hn
      simp [givenLaw, FiniteLaw.joint, FiniteLaw.point, hn] at hb
    have hcd : c = d := hc.trans hd'.symm
    cases hcd
    have hac : 0 < ((next c).givenLaw (r.2 c)).prob a :=
      pos_of_mul_pos_right ha ((FiniteLaw.point r.1).nonneg c)
    have hbc : 0 < ((next c).givenLaw (r.2 c)).prob b :=
      pos_of_mul_pos_right hb ((FiniteLaw.point r.1).nonneg c)
    have hab := ih c (hd c) (r.2 c) a b hac hbc (List.append_cancel_left he)
    exact congrArg (Sigma.mk c) hab

abbrev Visible (tree : ProtocolTree.{u} Byte) := Set.range tree.output

noncomputable instance visibleFintype (tree : ProtocolTree.{u} Byte) : Fintype tree.Visible :=
  Fintype.ofFinite _

def readout (tree : ProtocolTree.{u} Byte) (l : tree.Leaves) : tree.Visible :=
  ⟨tree.output l, ⟨l, rfl⟩⟩

noncomputable def visibleGivenLaw (tree : ProtocolTree.{u} Byte) (r : tree.Tape) :
    FiniteLaw tree.Visible := (tree.givenLaw r).map tree.readout

/-- Actual finite conditional entropy H(R,Y)-H(R), measured in bits. -/
noncomputable def visibleConditionalEntropy (tree : ProtocolTree.{u} Byte) : ℝ :=
  tree.tapeLaw.conditionalEntropyNat tree.visibleGivenLaw / Real.log 2

theorem visibleGiven_entropy (tree : ProtocolTree.{u} Byte) (hd : tree.Decodable) (r : tree.Tape) :
    (tree.visibleGivenLaw r).entropyNat = (tree.givenLaw r).entropyNat := by
  apply FiniteLaw.entropyNat_map_of_injective_support
  intro a b ha hb he
  exact tree.output_injective_given hd r a b ha hb (congrArg Subtype.val he)

/-- Decodability turns the payload budget into conditional entropy of the final text. -/
theorem visibleConditionalEntropy_eq_budget (tree : ProtocolTree.{u} Byte) (hd : tree.Decodable) :
    tree.visibleConditionalEntropy = tree.budgetNat / Real.log 2 := by
  simp only [visibleConditionalEntropy, FiniteLaw.conditionalEntropyNat_eq_expect,
    tree.visibleGiven_entropy hd, tree.expected_given_entropy]

end StegaTextProofs.ProtocolTree
