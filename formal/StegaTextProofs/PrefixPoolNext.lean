module

public import StegaTextProofs.ConstructedLASStep
public import StegaTextProofs.FrontierTransitions
public import StegaTextProofs.PrefixPoolEOS

/-!
The selected exact history becomes active. Each retained history keeps its
identity, while its residual drops precisely the emitted key. The next
frontier preserves antichains, rendering equations and EOS isolation.
-/

@[expose] public section

namespace StegaTextProofs.PrefixPool

attribute [local instance] Classical.propDecidable
open PrefixGrouping
variable {Candidate Token Byte : Type*} [Fintype Candidate]

noncomputable def nextHistory (pool : PrefixPool Candidate Byte) (history : Candidate → List Token)
    (g : pool.Group) (e : pool.Exact g) : Unit ⊕ pool.Strict g → List Token :=
  Sum.elim (fun _ => history e.val.val) (fun r => history r.val.val)

noncomputable def nextResidual (pool : PrefixPool Candidate Byte) (g : pool.Group)
    (r : pool.Strict g) : List Byte := (pool.residual r.val.val).drop (pool.residual g.val).length

theorem nextHistory_antichain (pool : PrefixPool Candidate Byte) (history : Candidate → List Token)
    (hanti : ∀ c d, history c <+: history d → c = d) (g : pool.Group) (e : pool.Exact g) :
    HistoryAntichain (pool.nextHistory history g e) :=
  history_antichain_collapse (fun c => history (pool.candidate g c))
    (pool.selected_antichain g history hanti) e

theorem key_append_nextResidual (pool : PrefixPool Candidate Byte) (g : pool.Group)
    (r : pool.Strict g) : pool.residual g.val ++ pool.nextResidual g r = pool.residual r.val.val := by
  obtain ⟨tail, ht⟩ := pool.group_key_prefix g r.val
  simp [nextResidual, ← ht]

theorem nextResidual_nonempty (pool : PrefixPool Candidate Byte) (g : pool.Group)
    (r : pool.Strict g) : pool.nextResidual g r ≠ [] := by
  intro hz
  have h := pool.key_append_nextResidual g r
  rw [hz, List.append_nil] at h
  exact r.property h.symm

theorem next_active_render (pool : PrefixPool Candidate Byte) (word : Token → List Byte)
    (history : Candidate → List Token) (emitted : List Byte)
    (hrender : ∀ c, render word (history c) = emitted ++ pool.residual c)
    (g : pool.Group) (e : pool.Exact g) :
    render word (pool.nextHistory history g e (.inl ())) = emitted ++ pool.residual g.val := by
  change render word (history e.val.val) = _
  rw [hrender, e.property]

theorem next_retained_render (pool : PrefixPool Candidate Byte) (word : Token → List Byte)
    (history : Candidate → List Token) (emitted : List Byte)
    (hrender : ∀ c, render word (history c) = emitted ++ pool.residual c)
    (g : pool.Group) (e : pool.Exact g) (r : pool.Strict g) :
    render word (pool.nextHistory history g e (.inr r)) =
      (emitted ++ pool.residual g.val) ++ pool.nextResidual g r := by
  change render word (history r.val.val) = _
  rw [hrender, List.append_assoc, pool.key_append_nextResidual]

theorem nextResidual_isolated (pool : PrefixPool Candidate Byte) (eos : Byte)
    (hisolated : ∀ c, EOSIsolation.IsolatedWord eos (pool.residual c))
    (g : pool.Group) (r : pool.Strict g) : EOSIsolation.IsolatedWord eos (pool.nextResidual g r) :=
  EOSIsolation.suffix_isolated eos _ _ (hisolated r.val.val) (List.drop_suffix _ _)

end StegaTextProofs.PrefixPool
