module

public import StegaTextProofs.ParameterizedLAS

public section

/-! Axiom audit for the public, paper-facing parameterized LAS interface. -/

#print axioms StegaTextProofs.TokenSource.LASState.LASGate
#print axioms StegaTextProofs.TokenSource.LASState.thresholdLASGate
#print axioms StegaTextProofs.TokenSource.LASState.alwaysLASGate
#print axioms StegaTextProofs.TokenSource.LASState.thresholdLASGate_zero
#print axioms StegaTextProofs.TokenSource.LASState.thresholdLASGate_zero_eq_always
#print axioms StegaTextProofs.TokenSource.LASState.adaptiveBudgetNat_alwaysLAS_eq_budgetNat
#print axioms StegaTextProofs.TokenSource.parameterizedLASValue
#print axioms StegaTextProofs.TokenSource.parameterizedLASProtocol
#print axioms StegaTextProofs.TokenSource.parameterizedLASValue_zero_eq_lasValue
#print axioms StegaTextProofs.TokenSource.parameterizedLAS_source_law
#print axioms StegaTextProofs.TokenSource.parameterizedLAS_value_eq_budget
#print axioms StegaTextProofs.TokenSource.native_le_parameterizedLAS_add_uncertainty
