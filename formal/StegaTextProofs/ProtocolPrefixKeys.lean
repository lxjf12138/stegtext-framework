module

public import StegaTextProofs.PrefixLASStep
public import StegaTextProofs.FiniteConditionalLaw

/-! The actual native and LAS group labels have prefix-free emitted keys. -/

@[expose] public section

namespace StegaTextProofs

attribute [local instance] Classical.propDecidable
variable {Token Byte : Type*} [Fintype Token]

namespace NativeGrouping

theorem positive_labels_prefix_free (word : Token → List Byte) (p : FiniteLaw Token)
    (g h : Token) (hg : 0 < (p.map (label word p)).prob g)
    (hh : 0 < (p.map (label word p)).prob h) (hp : word g <+: word h) : g = h := by
  obtain ⟨t, ht, rfl⟩ := (p.map_positive_iff (label word p) g).mp hg
  obtain ⟨u, hu, rfl⟩ := (p.map_positive_iff (label word p) h).mp hh
  apply (same_label_iff_same_key word p t u ht hu).mpr
  apply PrefixGrouping.minimal_keys_prefix_free
    (PrefixGrouping.key_spec _ _ (token_mem_support word p t ht)).1
    (PrefixGrouping.key_spec _ _ (token_mem_support word p u hu)).1
  simpa only [label_word word p t ht, label_word word p u hu] using hp

theorem conditional_key_prefix (word : Token → List Byte) (p : FiniteLaw Token)
    (g : (p.map (label word p)).Positive) (t : Token)
    (ht : 0 < (p.fiberLaw (label word p) g.val g.property).prob t) :
    word g.val <+: word t := by
  obtain ⟨he, hp⟩ := p.fiberLaw_positive (label word p) g.val g.property t ht
  rw [← he, label_word word p t hp]
  exact (PrefixGrouping.key_spec _ _ (token_mem_support word p t hp)).2

end NativeGrouping

namespace PrefixPool

theorem keys_prefix_free (pool : PrefixPool Token Byte) (g h : pool.Group)
    (hp : pool.residual g.val <+: pool.residual h.val) : g = h := by
  have hg : 0 < (pool.prior.map (NativeGrouping.label pool.residual pool.prior)).prob g.val :=
    (pool.prior.map_positive_iff _ _).mpr
      ⟨g.val, pool.positive g.val, congrArg Subtype.val (pool.group_fixed g)⟩
  have hh : 0 < (pool.prior.map (NativeGrouping.label pool.residual pool.prior)).prob h.val :=
    (pool.prior.map_positive_iff _ _).mpr
      ⟨h.val, pool.positive h.val, congrArg Subtype.val (pool.group_fixed h)⟩
  exact Subtype.ext (NativeGrouping.positive_labels_prefix_free pool.residual pool.prior
    g.val h.val hg hh hp)

end PrefixPool

end StegaTextProofs
