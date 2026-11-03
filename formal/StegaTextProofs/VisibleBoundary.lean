module

public import StegaTextProofs.PrefixGrouping

/-!
A completed token fixes its native group on every compatible longer visible
history. The keys belong to the original native support. In particular, they
are not recomputed after conditioning on an LAS state.
-/

@[expose] public section

namespace StegaTextProofs.PrefixGrouping

variable {Byte Token : Type*}

/-- A completed native token and any compatible continuation have the same
native group, even when the other token crosses the completed boundary. -/
theorem native_group_fixed_by_visible_prefix (support : Finset (List Byte))
    (u v tailU tailV : List Byte) (hu : u ∈ support) (hv : v ∈ support)
    (hvisible : u ++ tailU <+: v ++ tailV) :
    key support u hu = key support v hv := by
  have hku := key_spec support u hu
  have hkv := key_spec support v hv
  exact minimal_key_unique hku.1 hkv.1
    ((hku.2.trans (List.prefix_append u tailU)).trans hvisible)
    (hkv.2.trans (List.prefix_append v tailV))

def render (word : Token → List Byte) (history : List Token) : List Byte :=
  history.flatMap word

/-- The same lemma at an arbitrary labelled source history. Equal visible
histories are not identified as token histories. -/
theorem native_group_fixed_at_history (word : Token → List Byte)
    (support : Finset (List Byte)) (history : List Token)
    (t u : Token) (tailT tailU : List Token)
    (ht : word t ∈ support) (hu : word u ∈ support)
    (hvisible : render word (history ++ t :: tailT) <+:
      render word (history ++ u :: tailU)) :
    key support (word t) ht = key support (word u) hu := by
  have hp : word t ++ render word tailT <+: word u ++ render word tailU := by
    simpa only [render, List.flatMap_append, List.flatMap_cons,
      List.prefix_append_right_inj] using hvisible
  exact native_group_fixed_by_visible_prefix support _ _ _ _ ht hu hp

end StegaTextProofs.PrefixGrouping
