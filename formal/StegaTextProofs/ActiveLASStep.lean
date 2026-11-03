module

public import StegaTextProofs.CursorActiveExpansion
public import StegaTextProofs.CursorLASStep
public import StegaTextProofs.PrefixPoolNext

/-!
A complete concrete one-step entropy comparison at an active source frontier:
advance only its active cursor, prune zero mass, group residual prefixes,
select an exact surrogate and form the posterior. Source continuation laws
and all probability decompositions are constructed. Only ordinary state
invariants (legal histories, antichain, visible boundary) remain as premises.
-/

@[expose] public section

namespace StegaTextProofs.TokenSource

attribute [local instance] Classical.propDecidable
open PrefixGrouping
variable {Token Byte : Type*} [Fintype Token]

noncomputable def activePool (word : Token → List Byte) (source : TokenSource Token)
    (prior : FiniteLaw source.Cursor) (active : source.Cursor) (emitted : List Byte) :
    PrefixPool (source.expandActive prior active).Positive Byte :=
  PrefixPool.ofLaw (source.expandActive prior active)
    (fun c => (render word (source.cursorHistory c)).drop emitted.length)

theorem expandActive_visible (word : Token → List Byte) (source : TokenSource Token)
    (prior : FiniteLaw source.Cursor) (active : source.Cursor) (emitted : List Byte)
    (hactive : source.cursorSource active ≠ terminal)
    (hbase : render word (source.cursorHistory active) = emitted)
    (hvisible : ∀ c, 0 < prior.prob c → emitted <+: render word (source.cursorHistory c))
    (d : source.Cursor) (hd : 0 < (source.expandActive prior active).prob d) :
    emitted <+: render word (source.cursorHistory d) := by
  rcases (source.expandActive_positive prior active d).mp hd with ⟨_, hd⟩ | ⟨hd, _⟩
  · obtain ⟨t, ht⟩ := source.advance_history active d hactive hd
    rw [ht, render, List.flatMap_append]
    change emitted <+: render word (source.cursorHistory active) ++ _
    rw [hbase]
    exact List.prefix_append _ _
  · exact hvisible d hd

theorem activePool_render (word : Token → List Byte) (source : TokenSource Token)
    (prior : FiniteLaw source.Cursor) (active : source.Cursor) (emitted : List Byte)
    (hactive : source.cursorSource active ≠ terminal)
    (hbase : render word (source.cursorHistory active) = emitted)
    (hvisible : ∀ c, 0 < prior.prob c → emitted <+: render word (source.cursorHistory c))
    (c : (source.expandActive prior active).Positive) :
    render word (source.cursorHistory c.val) =
      emitted ++ (source.activePool word prior active emitted).residual c := by
  obtain ⟨tail, ht⟩ := source.expandActive_visible word prior active emitted
    hactive hbase hvisible c.val c.property
  change render word (source.cursorHistory c.val) =
    emitted ++ (render word (source.cursorHistory c.val)).drop emitted.length
  simp [← ht]

theorem activePool_nonempty (word : Token → List Byte) (source : TokenSource Token)
    (prior : FiniteLaw source.Cursor) (active : source.Cursor) (emitted : List Byte)
    (hactive : source.cursorSource active ≠ terminal)
    (hbase : render word (source.cursorHistory active) = emitted)
    (hword : ∀ t, word t ≠ [])
    (hstrict : ∀ c, 0 < prior.prob c → c ≠ active →
      emitted.length < (render word (source.cursorHistory c)).length)
    (c : (source.expandActive prior active).Positive) :
    (source.activePool word prior active emitted).residual c ≠ [] := by
  have hlength : emitted.length < (render word (source.cursorHistory c.val)).length := by
    rcases (source.expandActive_positive prior active c.val).mp c.property with ⟨_, hc⟩ | ⟨hc, hca⟩
    · obtain ⟨t, ht⟩ := source.advance_history active c.val hactive hc
      have he : render word (source.cursorHistory c.val) = emitted ++ word t := by
        simp only [ht, render, List.flatMap_append, List.flatMap_cons, List.flatMap_nil,
          List.append_nil]
        exact congrArg (fun xs => xs ++ word t) hbase
      rw [he, List.length_append]
      have : 0 < (word t).length := List.length_pos_iff.mpr (hword t)
      omega
    · exact hstrict c.val hc hca
  change (render word (source.cursorHistory c.val)).drop emitted.length ≠ []
  intro hz
  have := congrArg List.length hz
  simp only [List.length_drop, List.length_nil] at this
  omega

/-- The LAS drift bound for the actual active-only source expansion and prefix groups. -/
theorem activeLAS_selective_step (word : Token → List Byte) (source : TokenSource Token)
    (prior : FiniteLaw source.Cursor) (active : source.Cursor) (emitted : List Byte)
    (hactive : source.cursorSource active ≠ terminal)
    (hlegal : ∀ c, 0 < prior.prob c → source.LegalCursor c)
    (hanti : ∀ c d, 0 < prior.prob c → 0 < prior.prob d →
      source.cursorHistory c <+: source.cursorHistory d → c = d)
    (hbase : render word (source.cursorHistory active) = emitted)
    (hvisible : ∀ c, 0 < prior.prob c → emitted <+: render word (source.cursorHistory c)) :
    let pool := source.activePool word prior active emitted
    source.decisionTree.selective true (prior.mix (source.cursorLaw word)).prob ≤
      pool.groups.entropyNat + pool.groups.expect (fun g => (pool.selected g).surrogate.expect
        (fun e => source.decisionTree.selective true
          (pool.posterior source (fun c => source.cursorLaw word c.val) g e).prob)) := by
  let pool := source.activePool word prior active emitted
  have h := pool.cursor_selective_step word source (fun c => c.val) emitted
    (fun c => source.expandActive_legal prior active hlegal c.val c.property)
    (fun c d hp => Subtype.ext (source.expandActive_antichain prior active hactive hanti
      c.val d.val c.property d.property hp))
    (source.activePool_render word prior active emitted hactive hbase hvisible)
  have hbelief : pool.prior.mix (fun c => source.cursorLaw word c.val) =
      prior.mix (source.cursorLaw word) :=
    ((source.expandActive prior active).supportLaw_mix (source.cursorLaw word)).trans
      (source.expandActive_continuation word prior active)
  rw [hbelief] at h
  exact h

end StegaTextProofs.TokenSource
