module

public import StegaTextProofs.SourceEOS
public import StegaTextProofs.PrefixPoolNextLaw
public import StegaTextProofs.ConcreteLASState
public import StegaTextProofs.ConcreteLASBudget
public import StegaTextProofs.ConcreteLASOutput

/-! Audit every fifth-batch theorem and the concrete recursive constructors. -/

#print axioms StegaTextProofs.TokenSource.cursorSource_root
#print axioms StegaTextProofs.TokenSource.advance_eos
#print axioms StegaTextProofs.PrefixPool.nextLaw_support
#print axioms StegaTextProofs.PrefixPool.nextLaw_active_positive
#print axioms StegaTextProofs.PrefixPool.nextLaw_eq_collapse
#print axioms StegaTextProofs.PrefixPool.nextLaw_channel
#print axioms StegaTextProofs.PrefixPool.nextLaw_eos
#print axioms StegaTextProofs.TokenSource.LASState.pool_render
#print axioms StegaTextProofs.TokenSource.LASState.pool_nonempty
#print axioms StegaTextProofs.TokenSource.LASState.fresh_residual
#print axioms StegaTextProofs.TokenSource.LASState.pool_isolated
#print axioms StegaTextProofs.TokenSource.LASState.pool_terminal
#print axioms StegaTextProofs.TokenSource.LASState.strict_nonterminal
#print axioms StegaTextProofs.TokenSource.LASState.next_retained_render
#print axioms StegaTextProofs.TokenSource.LASState.next_posterior
#print axioms StegaTextProofs.TokenSource.cursor_length_le_max
#print axioms StegaTextProofs.TokenSource.LASState.emitted_length_le_max
#print axioms StegaTextProofs.TokenSource.LASState.rank_positive
#print axioms StegaTextProofs.TokenSource.LASState.rank_next_lt
#print axioms StegaTextProofs.TokenSource.LASState.potential_terminal
#print axioms StegaTextProofs.TokenSource.LASState.potential_initial
#print axioms StegaTextProofs.TokenSource.LASState.potential_step
#print axioms StegaTextProofs.TokenSource.LASState.budgetNat_terminal
#print axioms StegaTextProofs.TokenSource.LASState.potential_le_budgetNat
#print axioms StegaTextProofs.TokenSource.LASState.budgetNat_stable
#print axioms StegaTextProofs.TokenSource.native_le_concrete_las
#print axioms StegaTextProofs.TokenSource.LASState.next_belief_expect
#print axioms StegaTextProofs.TokenSource.LASState.runExpect_terminal
#print axioms StegaTextProofs.TokenSource.LASState.runExpect_sub
#print axioms StegaTextProofs.TokenSource.LASState.runExpect_zero_of_terminal
#print axioms StegaTextProofs.TokenSource.LASState.runExpect_nonterminal_zero
#print axioms StegaTextProofs.TokenSource.LASState.runExpect_belief
#print axioms StegaTextProofs.TokenSource.LASState.runExpect_tokenPath
#print axioms StegaTextProofs.TokenSource.LASState.runExpect_visible
#print axioms StegaTextProofs.TokenSource.concrete_las_tokenPath_law
#print axioms StegaTextProofs.TokenSource.concrete_las_visible_law
#print axioms StegaTextProofs.TokenSource.LASState.initial
#print axioms StegaTextProofs.TokenSource.LASState.next
#print axioms StegaTextProofs.TokenSource.LASState.budgetNat
#print axioms StegaTextProofs.TokenSource.LASState.runExpect
#print axioms StegaTextProofs.TokenSource.lasValue
