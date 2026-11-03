module

public import StegaTextProofs.SourceCursorAdvance
public import StegaTextProofs.ConstructedLASStep

/-!
The constructed LAS step instantiated with actual continuation laws of
source cursors. Source support and future-history conditions are derived
from cursor legality, rather than required as channel assumptions.
-/

@[expose] public section

namespace StegaTextProofs.PrefixPool

attribute [local instance] Classical.propDecidable
variable {Candidate Token Byte : Type*} [Fintype Candidate] [Fintype Token]

theorem cursor_selective_step (pool : PrefixPool Candidate Byte)
    (word : Token → List Byte) (source : TokenSource Token)
    (position : Candidate → source.Cursor) (emitted : List Byte)
    (hlegal : ∀ c, source.LegalCursor (position c))
    (hanti : ∀ c d, source.cursorHistory (position c) <+: source.cursorHistory (position d) → c = d)
    (hrender : ∀ c, PrefixGrouping.render word (source.cursorHistory (position c)) =
      emitted ++ pool.residual c) :
    source.decisionTree.selective true
      (pool.prior.mix (fun c => source.cursorLaw word (position c))).prob ≤
        pool.groups.entropyNat + pool.groups.expect (fun g => (pool.selected g).surrogate.expect
          (fun e => source.decisionTree.selective true
            (pool.posterior source (fun c => source.cursorLaw word (position c)) g e).prob)) := by
  apply pool.selective_step_inequality word source (fun c => source.cursorHistory (position c))
    emitted (fun c => source.cursorLaw word (position c)) hanti hrender
  · intro c l hl
    exact source.cursorLaw_support word (position c) (hlegal c) l
      (lt_of_le_of_ne (source.cursorLaw word (position c) |>.nonneg l) (Ne.symm hl))
  · intro c l hl
    exact source.cursorLaw_prefix word (position c) l
      (lt_of_le_of_ne (source.cursorLaw word (position c) |>.nonneg l) (Ne.symm hl))

end StegaTextProofs.PrefixPool
