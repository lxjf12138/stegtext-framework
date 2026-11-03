# Lean 4 formalization

This directory machine-checks the ideal analysis of LAS(τ) in Section IV of the
paper for finite token sources. The library `StegaTextProofs` has 92 files and
about 9,300 lines.

## Build

Install [elan](https://github.com/leanprover/elan), then run:

```bash
cd formal
lake exe cache get   # prebuilt mathlib v4.35.0
lake build
```

The toolchain is pinned to Lean 4.35.0 (`lean-toolchain`) and mathlib v4.35.0
(`lakefile.toml`, `lake-manifest.json`). The build uses `autoImplicit = false`
and `warningAsError = true`. The library contains no `sorry`, `admit`, or
`axiom` declarations. The `AxiomAudit*.lean` modules run `#print axioms` on the
main results during the build; every audited declaration depends only on
`propext`, `Classical.choice`, and `Quot.sound`.

## Model

A `TokenSource` is a finite tree of token histories with a next-token law at
each history. The main theorems assume what Section IV-A states:

| Paper assumption | Lean hypothesis |
|---|---|
| Finite source; natural EOS, or the common token cap, ends every history | `hs : source.EOSSource word eos` (`SourceEOS.lean`) |
| Token realizations are nonempty | `hne : ∀ t, word t ≠ []` |
| The end marker occurs only at termination | `hiso : ∀ t, EOSIsolation.IsolatedWord eos (word t)` |

The coder is ideal: each round selects a group with its conditional group law
\(P_t^{\mathrm{ua}}\), and synchronization draws use randomness independent of
the payload. Probabilities are real numbers; budgets are computed in nats and
divided by `Real.log 2` to give bits.

The LAS state (`LASState`, `ConcreteLASState.lean`) keeps the active history,
the retained histories, and their masses. The retaining update and the
whole-group update (`fullSync`, `FullSyncState.lean`) are both constructed on
it. A gate (`AdaptiveGate`) chooses between them after group selection.
`thresholdGate τ` retains exactly when \(\tau \le U\), where `groupUncertainty`
is the uncertainty \(U\) of Eq. (8).

## Correspondence with the paper

| Paper | Lean declaration | File |
|---|---|---|
| Theorem 1 (complete-output distribution preservation), for every gate | `concrete_adaptive_visible_law` | `AdaptiveLASRun.lean` |
| Theorem 1 with explicit independent payload and synchronization tapes | `adaptive_independent_tapes_output` | `AdaptiveLASProtocol.lean` |
| Eq. (14): \(0 \le d(S,G) \le U\) | `groupLoss_nonneg`, `groupLoss_le_uncertainty`, `groupLoss_eq_native_gap` | `AdaptiveLASBudget.lean` |
| Theorem 2: \(V_{\mathrm{SP}} \le V_\tau + \tau\,\mathbb E_\tau[N_\tau]\), with \(N_\tau\) counting whole-group updates with \(U>0\) | `native_le_threshold_adaptive_add_count` | `AdaptiveLASThreshold.lean` |
| \(\tau = 0\) always retains, so \(V_{\mathrm{SP}} \le V_0\) | `thresholdGate_zero`; `native_le_concrete_las` | `AdaptiveLASThreshold.lean`; `ConcreteLASBudget.lean` |
| Group-key transcripts are recoverable from the visible output given the synchronization tape | `adaptiveProtocol_decodable`, `adaptiveProtocol_output_injective` | `AdaptiveLASProtocol.lean` |
| The ideal budgets equal conditional entropies \(H(Y \mid R)\) of the output given the synchronization tape | `adaptiveProtocol_conditional_entropy`, `finite_adaptive_conditionalEntropy_loss_bound`, `finite_las_conditionalEntropy_ge_native` | `AdaptiveLASProtocol.lean`; `FiniteConditionalComparison.lean` |

Termination, the antichain and visible-boundary invariants, and the rank that
decreases in every round are proved for the constructed state machine; the
theorems above take none of them as hypotheses.

## Scope

The formalization does not cover:

- concrete entropy coders, including the iMEC coder used in the experiments, or
  the probability and recovery contracts of Section IV-A that connect a coder
  to the ideal model; Proposition 1 is therefore checked only at the level of
  group-key transcripts;
- the computational-cost bound of Section IV-E;
- floating-point arithmetic, pseudorandom generators, and the Python
  implementation;
- sources without a finite token tree.
