module

public import Mathlib

/-!
Prefix grouping on a finite support of words. Minimal keys are actual support
elements, not arbitrary common string prefixes. Duplicate candidate identities
may share a word; this module only establishes the combinatorics of group keys.
-/

@[expose] public section

namespace StegaTextProofs.PrefixGrouping

variable {Byte : Type*}

def MinimalKey (support : Finset (List Byte)) (key : List Byte) : Prop :=
  key ∈ support ∧ ∀ other ∈ support, other <+: key → other = key

theorem minimal_keys_prefix_free {support : Finset (List Byte)} {a b : List Byte}
    (ha : MinimalKey support a) (hb : MinimalKey support b) (hab : a <+: b) : a = b :=
  hb.2 a ha.1 hab

theorem minimal_key_unique {support : Finset (List Byte)} {a b word : List Byte}
    (ha : MinimalKey support a) (hb : MinimalKey support b)
    (haw : a <+: word) (hbw : b <+: word) : a = b := by
  rcases List.prefix_or_prefix_of_prefix haw hbw with hab | hba
  · exact minimal_keys_prefix_free ha hb hab
  · exact (minimal_keys_prefix_free hb ha hba).symm

/-- Every supported word has a support-minimal prefix, including a nonempty
exact subset when words are realized by labelled positive-mass candidates. -/
theorem exists_minimal_key (support : Finset (List Byte)) (word : List Byte)
    (hw : word ∈ support) : ∃ key, MinimalKey support key ∧ key <+: word := by
  classical
  let prefixes := support.filter (fun key => key <+: word)
  have hn : prefixes.Nonempty := ⟨word, by simp [prefixes, hw]⟩
  obtain ⟨key, hk, hmin⟩ := prefixes.exists_min_image List.length hn
  have hk' : key ∈ support ∧ key <+: word := Finset.mem_filter.mp hk
  refine ⟨key, ⟨hk'.1, ?_⟩, hk'.2⟩
  intro other ho hop
  apply hop.eq_of_length_le
  exact hmin other (Finset.mem_filter.mpr ⟨ho, hop.trans hk'.2⟩)

noncomputable def key (support : Finset (List Byte)) (word : List Byte)
    (hw : word ∈ support) : List Byte :=
  Classical.choose (exists_minimal_key support word hw)

theorem key_spec (support : Finset (List Byte)) (word : List Byte)
    (hw : word ∈ support) : MinimalKey support (key support word hw) ∧
      key support word hw <+: word :=
  Classical.choose_spec (exists_minimal_key support word hw)

/-- The key in the expanded support is a function of the original key.
This is the deterministic coarsening needed for fresh-group entropy accounting. -/
theorem fresh_group_coarsening (fresh combined : Finset (List Byte))
    (hsub : fresh ⊆ combined) (word : List Byte) (hw : word ∈ fresh) :
    key combined word (hsub hw) =
      key combined (key fresh word hw) (hsub (key_spec fresh word hw).1.1) := by
  have hf := key_spec fresh word hw
  have hc := key_spec combined word (hsub hw)
  have hb := key_spec combined (key fresh word hw) (hsub hf.1.1)
  exact minimal_key_unique hc.1 hb.1 hc.2 (hb.2.trans hf.2)

/-- Two candidates in one native fresh group stay together after adding
retained residuals to the support. -/
theorem same_fresh_group_stays_together (fresh combined : Finset (List Byte))
    (hsub : fresh ⊆ combined) (u v : List Byte) (hu : u ∈ fresh) (hv : v ∈ fresh)
    (hgroup : key fresh u hu = key fresh v hv) :
    key combined u (hsub hu) = key combined v (hsub hv) := by
  rw [fresh_group_coarsening fresh combined hsub u hu,
    fresh_group_coarsening fresh combined hsub v hv]
  congr 1

end StegaTextProofs.PrefixGrouping
