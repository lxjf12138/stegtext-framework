module

public import StegaTextProofs.IndependentProtocolTapes
public import StegaTextProofs.ProtocolPrefixKeys
public import StegaTextProofs.ConcreteLASOutput

/-! Compile the constructed LAS execution to an explicit two-tape protocol tree. -/

@[expose] public section

namespace StegaTextProofs.TokenSource.LASState

universe u v
attribute [local instance] Classical.propDecidable
variable {Token : Type u} {Byte : Type v} [Fintype Token]
variable {word : Token → List Byte} {eos : Byte} {source : TokenSource Token}

noncomputable def protocol (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t)) :
    ℕ → LASState word eos source → ProtocolTree.{u} Byte
  | 0, _ => .leaf
  | n + 1, s => if ha : source.cursorSource s.active = terminal then .leaf else
      .payload s.pool.groups (fun g => s.pool.residual g.val) (fun g =>
        .sync (s.pool.selected g).surrogate (fun _ => [])
          (fun e => protocol hs hne hiso n (s.next ha hs hne hiso g e)))

theorem protocol_budget (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (n : ℕ) (s : LASState word eos source) :
    (protocol hs hne hiso n s).budgetNat = budgetNat hs hne hiso n s := by
  induction n generalizing s with
  | zero => rfl
  | succ n ih =>
    by_cases ht : source.cursorSource s.active = terminal
    · simp only [protocol, budgetNat, dite_eq_left ht, ProtocolTree.budgetNat]
    · simp only [protocol, budgetNat, dite_eq_right ht, ProtocolTree.budgetNat, ih]

theorem protocol_decodable (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (n : ℕ) (s : LASState word eos source) : (protocol hs hne hiso n s).Decodable := by
  induction n generalizing s with
  | zero => trivial
  | succ n ih =>
    by_cases ht : source.cursorSource s.active = terminal
    · simp only [protocol, dite_eq_left ht, ProtocolTree.Decodable]
    · simp only [protocol, dite_eq_right ht, ProtocolTree.Decodable]
      exact ⟨s.pool.keys_prefix_free, fun g e => ih (s.next ht hs hne hiso g e)⟩

theorem protocol_conditional_entropy (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (n : ℕ) (s : LASState word eos source) :
    (protocol hs hne hiso n s).visibleConditionalEntropy = budgetNat hs hne hiso n s / Real.log 2 := by
  rw [ProtocolTree.visibleConditionalEntropy_eq_budget _ (protocol_decodable hs hne hiso n s),
    protocol_budget]

theorem protocol_output_expect (hs : source.EOSSource word eos)
    (hne : ∀ t, word t ≠ []) (hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t))
    (f : List Byte → ℝ) (n : ℕ) (s : LASState word eos source) :
    (protocol hs hne hiso n s).outputExpect (fun w => f (s.emitted ++ w)) =
        runExpect hs hne hiso (fun t => f t.emitted) n s := by
  induction n generalizing s with
  | zero => simp only [protocol, ProtocolTree.outputExpect_leaf, List.append_nil, runExpect]
  | succ n ih =>
    by_cases ht : source.cursorSource s.active = terminal
    · simp only [protocol, dite_eq_left ht, ProtocolTree.outputExpect_leaf, List.append_nil, runExpect]
    · simp only [protocol, dite_eq_right ht, ProtocolTree.outputExpect_payload,
        ProtocolTree.outputExpect_sync, List.nil_append, runExpect]
      congr 1
      funext g
      congr 1
      funext e
      simpa only [next, List.append_assoc] using ih (s.next ht hs hne hiso g e)

end StegaTextProofs.TokenSource.LASState
