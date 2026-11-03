module

public import StegaTextProofs.TerminalCollapse

/-!
An EOS token consists of one reserved symbol; ordinary token realizations
do not contain that symbol. Any residual suffix that starts with EOS must
therefore equal the EOS word. An EOS group consequently has zero strict
mass, and exact collapse resolves its candidate.
-/

@[expose] public section

namespace StegaTextProofs.EOSIsolation

variable {Byte : Type*}

def IsolatedWord (eos : Byte) (word : List Byte) : Prop :=
  word = [eos] ∨ eos ∉ word

theorem eos_prefix_suffix_eq (eos : Byte) (word residual : List Byte)
    (hword : IsolatedWord eos word) (hsuffix : residual <:+ word)
    (hprefix : [eos] <+: residual) : residual = [eos] := by
  rcases hword with hw | hw
  · exact (hprefix.eq_of_length_le (hw ▸ hsuffix.length_le)).symm
  · exact False.elim (hw (hsuffix.sublist.subset (hprefix.sublist.subset (by simp))))

theorem suffix_isolated (eos : Byte) (word residual : List Byte)
    (hword : IsolatedWord eos word) (hsuffix : residual <:+ word) :
    IsolatedWord eos residual := by
  rcases hword with hw | hw
  · by_cases he : eos ∈ residual
    · left
      have hle : residual.length ≤ 1 := by simpa [hw] using hsuffix.length_le
      cases residual with
      | nil => simp at he
      | cons b tail =>
        have htail : tail = [] := List.eq_nil_of_length_eq_zero (by simp_all)
        subst tail
        simp only [List.mem_singleton] at he
        subst b
        rfl
    · exact Or.inr he
  · exact Or.inr (fun he => hw (hsuffix.sublist.subset he))

variable {Exact Strict : Type*} [Fintype Exact] [Fintype Strict]

theorem strict_mass_zero (eos : Byte) (g : LASGroup Exact Strict)
    (word residual : Strict → List Byte)
    (hword : ∀ r, 0 < g.prior.prob (.inr r) → IsolatedWord eos (word r))
    (hsuffix : ∀ r, 0 < g.prior.prob (.inr r) → residual r <:+ word r)
    (hprefix : ∀ r, 0 < g.prior.prob (.inr r) → [eos] <+: residual r)
    (hstrict : ∀ r, residual r ≠ [eos]) :
    ∀ r, g.prior.prob (.inr r) = 0 := by
  intro r
  apply le_antisymm _ (g.prior.nonneg _)
  apply le_of_not_gt
  intro hr
  exact hstrict r (eos_prefix_suffix_eq eos (word r) (residual r)
    (hword r hr) (hsuffix r hr) (hprefix r hr))

theorem collapse_point (eos : Byte) (g : LASGroup Exact Strict)
    (word residual : Strict → List Byte)
    (hword : ∀ r, 0 < g.prior.prob (.inr r) → IsolatedWord eos (word r))
    (hsuffix : ∀ r, 0 < g.prior.prob (.inr r) → residual r <:+ word r)
    (hprefix : ∀ r, 0 < g.prior.prob (.inr r) → [eos] <+: residual r)
    (hstrict : ∀ r, residual r ≠ [eos]) (e : Exact) :
    g.collapse e = FiniteLaw.point (.inl e) :=
  g.collapse_point_of_no_strict (strict_mass_zero eos g word residual hword hsuffix hprefix hstrict) e

end StegaTextProofs.EOSIsolation
