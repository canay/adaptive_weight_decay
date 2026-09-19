# Analysis Amendment — Independence-Safe Claim Gate

Date/time: 2026-08-26 14:11 +03:00  
Tool: Codex  
Model, if known: GPT-5.6 Extra High (xhigh, user-attested; runtime exposed GPT-5 family)  
Operation ID: `F06-NOVELTY-ACCEPTABILITY-UPGRADE-20260826-01`

Status: `FROZEN_AFTER_COMPUTE_BEFORE_RESULT_INSPECTION`.  
Raw result identity seen before this amendment: row count 810, byte count
100387, SHA-256
`535DF250D5513A5D16C8816C9FAA1AD53A0B293C0055266EB935C5BA9E41D953`.
No method means, differences, rankings, p-values, or task outcomes were read
before this amendment.

## Reason

The frozen plan's Wilcoxon test over 90 task-seed cells treats five stochastic
seeds from the same task as independent evidence. That analysis is retained as
a declared sensitivity result but is too permissive for a superiority claim.
The correction is direction-neutral and strictly more conservative.

## Claim-gating analysis

1. Average each method over the five matched seeds within each of 18 task
   variants.
2. Run paired two-sided Wilcoxon tests over the 18 task-variant means for the
   predeclared 12 controller-versus-strong-baseline comparisons.
3. Apply Holm correction across all 12 tests.
4. Estimate paired median-difference uncertainty with a cluster bootstrap over
   the nine base datasets, retaining each base dataset's clean/noisy pair.
5. Report the original 90 task-seed analysis as sensitivity only.

Controller superiority requires both the existing practical threshold
`mean paired difference >= 0.005` and claim-gating Holm-adjusted `p < 0.05`.
Failure to pass preserves the benchmark/negative-result pivot.

