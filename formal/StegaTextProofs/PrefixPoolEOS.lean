module

public import StegaTextProofs.PrefixLASStep
public import StegaTextProofs.EOSIsolation

/-!
EOS isolation applied to the constructed prefix groups. No strict-mass-zero
premise is needed: every residual is EOS or excludes its reserved symbol.
-/

@[expose] public section

namespace StegaTextProofs.PrefixPool

attribute [local instance] Classical.propDecidable
variable {Candidate Byte : Type*} [Fintype Candidate]

theorem eos_no_strict (pool : PrefixPool Candidate Byte) (eos : Byte)
    (hisolated : ∀ c, EOSIsolation.IsolatedWord eos (pool.residual c))
    (g : pool.Group) (hg : pool.residual g.val = [eos]) (r : pool.Strict g) : False := by
  have hp : [eos] <+: pool.residual r.val.val := hg ▸ pool.group_key_prefix g r.val
  have heq := EOSIsolation.eos_prefix_suffix_eq eos (pool.residual r.val.val)
    (pool.residual r.val.val) (hisolated _) (by simp) hp
  exact r.property (heq.trans hg.symm)

theorem eos_collapse_point (pool : PrefixPool Candidate Byte) (eos : Byte)
    (hisolated : ∀ c, EOSIsolation.IsolatedWord eos (pool.residual c))
    (g : pool.Group) (hg : pool.residual g.val = [eos]) (e : pool.Exact g) :
    (pool.selected g).collapse e = FiniteLaw.point (.inl e) := by
  apply (pool.selected g).collapse_point_of_no_strict
  intro r
  exact False.elim (pool.eos_no_strict eos hisolated g hg r)

omit [Fintype Candidate] in
theorem eos_residual_from_suffix (eos : Byte) (word residual : Candidate → List Byte)
    (hword : ∀ c, EOSIsolation.IsolatedWord eos (word c))
    (hsuffix : ∀ c, residual c <:+ word c) :
    ∀ c, EOSIsolation.IsolatedWord eos (residual c) :=
  fun c => EOSIsolation.suffix_isolated eos (word c) (residual c) (hword c) (hsuffix c)

end StegaTextProofs.PrefixPool
