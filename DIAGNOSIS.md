# Acquisition diagnosis: loss direction, initialization and identifier copying

14 September 2026. Diagnostic phase complete on explanatory progress and a
functioning acquisition regime. Evidence and audit: Git `c183674`.

**The original learning failure is partly recoverable by changing the loss
direction.** With the same examples, teacher, adapter initialization and input
order, forward KL reaches 16/16 training calls at 128 updates; reverse KL reaches
4/16. From identical failed reverse-KL weights, a controlled switch to forward
KL repairs every routing decision and raises whole-call recall to 11/16, while
continued reverse KL remains at 4/16. The remaining rescue errors omit the fast
suffix. Independently, supplying correct routing does not repair new-identifier
errors; supplying the uppercase identifier allows both selected students to
complete all 48 development calls correctly.

This extends the [accepted findings](FINDINGS.md), which remain unchanged. It
does **not** establish a transfer advantage for forward KL. All new scores below
are diagnostic development observations. No new prospective transfer comparison
was selected or run, and no development score replaces the accepted 96-case test.

**What was changed and controlled.** The updated brief asked for an explanation
beyond finite gradients and failed acquisition. The [matrix plan](protocol/diagnosis-v2.md)
was committed at `966fd43` before execution; the [signal/component plan](protocol/diagnosis-v2-signal.md)
at `ec8c77d` preceded probability inspection, and the [rescue amendment](protocol/diagnosis-v2-rescue.md)
at `429b1e0` preceded rescue and privileged-prefix generation. The amendment also
added the acquired forward-128 checkpoint and retrospective probes of the six
original adapters. Diagnostic decisions were adaptive and are preserved in these
plans, rather than represented as one prospectively frozen confirmatory study.

Four new acquisition identifiers (`pelk`, `druvan`, `snebit`, `korvaz`) crossed
the two channels and priorities produce 16 checked calls. Four separate
development identifiers (`halmek`, `przuna`, `veskolit`, `dunfepazor`) produce
16 more queries. The task retains copper→kestrel, violet→marten, uppercase
identifier, and `-Q` iff fast. Applying these operations independently to unseen
identifiers is the intended composition assumption; finitely many examples do
not uniquely specify it.

The frozen base teacher sees all 16 checked calls and the previously selected
generic induction reminder. The student sees only schema and query at generation.
All soft-target training uses the same canonical checked-answer prefixes,
including EOS, and full-vocabulary teacher distributions. Thus the matrix tests
loss direction at matched prefix information; it does not test student-sampled
trajectories. Direct imitation uses the hard checked labels. Forward KL uses
`KL(teacher || student)` and reverse KL `KL(student || teacher)`, averaged over
answer positions, with detached teacher targets. Teacher pretraining knowledge
and the induction reminder are additional supervision beyond the hard labels.

The model is Qwen3-4B-Instruct-2507 bf16, revision
`cdbee75f17c01a7cc42f958dc650907174af0554`, with rank-8 q/v LoRA in the last eight
blocks, scale 2, no dropout, AdamW with zero weight decay, batch one. Seed 41 fixes
initialization and the 256-example order for every matrix arm. The teacher cache
is shared and the base remains frozen. Native MLX provides both teacher token
distributions and student gradients on the local M1 Ultra Mac Studio, 64 GiB.
MLX is 0.32.2 and MLX-LM is revision
`86b48c461feebf87c58788655b7e57b5574b9e6d`; the
[resource manifest](evidence/diagnosis-v2/resource.json) verifies model hashes.

This is an independently implemented MLX adaptation. The closest primary method
is OPCD, **2602.12275v1**, §3 and training appendices; the inspected author code
is Microsoft LMOps **4f2a9deb5f08e459fd44c2e4792344d78ca89fc3**. Its off-policy
baseline also changes loss orientation, motivating this separation of direction
from trajectory source. Its published top-256 approximation differs from our
full-vocabulary objective. Forward context distillation is an existing baseline,
not a newly invented method. PMD **2607.01480v1** has evolving memory/policy and is
not tested. Exact reading scopes and primary links are in the
[source record](sources/README.md).

**Acquisition succeeds under a different objective.** Every cell below is
whole-call correctness, **recall / development**, each out of 16. All scheduled
checkpoints were evaluated and saved; none was discarded. The untrained base
scores 0/16 on both splits.

| Objective | Learning rate | 32 updates | 128 updates | 256 updates |
|---|---:|---:|---:|---:|
| Hard-label imitation | 0.0005 | 7 / 3 | 16 / 12 | 16 / 12 |
| Reverse KL | 0.0005 | 2 / 3 | 4 / 3 | 4 / 3 |
| Reverse KL | 0.00005 | 0 / 0 | 0 / 0 | 0 / 0 |
| Forward KL | 0.0005 | 6 / 4 | 16 / 13 | 14 / 10 |
| Forward KL | 0.00005 | 6 / 2 | 13 / 10 | 14 / 11 |

At both rates the forward endpoint substantially exceeds the corresponding
reverse endpoint on acquisition. Reducing the reverse learning rate tenfold does
not establish acquisition in this budget. That comparison rules out this one
simple rate repair, not every possible reverse-KL schedule. Forward-128 establishes
a functioning checkpoint; the later decline means it is not evidence that longer
training monotonically improves acquisition. See the
[raw matrix](evidence/diagnosis-v2/responses.jsonl) and
[audited component counts](evidence/diagnosis-v2-audit/audit.json).

**The teacher is imperfect, but its mistakes do not explain the broad reverse
failure.** It generates 13/16 acquisition calls correctly. On the 13 correct
calls, generated answer/EOS tokenization exactly matches the canonical target.
It wrongly routes violet/slow `pelk` and `korvaz`, assigning the correct initial
token probabilities 0.119 and 0.037. Forward-128 nevertheless routes these calls
correctly; forward-256's only two recall errors reproduce these teacher mistakes.
For `pelk`, the student's correct-token probability falls from 0.804 to 0.242.
These observations are consistent with later fitting of erroneous teacher targets;
they do not isolate every effect of the additional updates.

The third greedy teacher error is violet/fast `korvaz`: canonical-prefix scoring
gives the correct first token probability 0.526 despite the wrong greedy output.
Full-sequence and prefix-only first-position log probabilities match exactly on
all 16 inputs. This excludes that tested scoring-alignment discrepancy, while
leaving the generation/scoring disagreement on this marginal case unresolved.
We do not count greedy teacher answers and scored distributions as interchangeable.
[Teacher audit](evidence/diagnosis-v2/teacher-audit.json).

On eight other cases where the teacher assigns over 0.9 probability to the
correct first routing token, the reverse-high endpoint predicts the wrong token
and gives the correct token negligible probability. With student distribution
`p` and teacher distribution `q`, the raw first-position logit derivatives are
`p_j * (log(p_j) - log(q_j) - KL(p || q))` for reverse KL and `p_j - q_j` for
forward KL. They are not full parameter gradients or whole-answer averaged losses.

For copper/slow `pelk`, the reverse-high endpoint has `p_correct = 2.65e-15`
while `q_correct = 0.999527`. Its correct-logit reverse derivative is approximately
`-1.34e-13`; the forward derivative is `-0.999527`. All eight teacher-clear wrong
decisions meet the prespecified weak-reverse/strong-forward thresholds. The base
had no cases meeting all these thresholds; reverse training developed this severe
suppression. CE-128 and forward-128 have no wrong first-token decisions on the
13 teacher-clear inputs. [Saved-distribution audit](evidence/diagnosis-v2-gradients/gradients.json).

The same retrospective probe on the **original** acquisition inputs finds this
pattern on 6 and 8 teacher-clear wrong decisions in the two checked-prefix reverse
adapters, and 8 each in the two on-policy adapters. Neither original imitation
adapter has a wrong first decision on the 14 teacher-clear inputs. This links
the diagnostic mechanism to both original failure variants without reusing their
test outputs for selection. It does not demonstrate that first-token suppression
is their sole cause, or that on-policy forward KL would succeed.
[Original-adapter signal audit](evidence/transfer-v1-gradients/gradients.json).

**A controlled intervention repairs the failed routing.** Both collapsed-start
arms begin from the identical reverse-high-256 checkpoint. Each resets AdamW at
0.0005 and receives the same next 128 input indices, canonical prefixes and fixed
teacher distributions. Only loss direction changes. The third arm starts from
CE-128 and tests whether reverse KL can preserve already acquired calls.

| Start → objective, 128 further updates | Recall /16 | Routing /16 | Identifier /16 | Suffix /16 | Development /16 |
|---|---:|---:|---:|---:|---:|
| Reverse-256 → reverse | 4 | 8 | 16 | 8 | 1 |
| Reverse-256 → forward | 11 | 16 | 16 | 11 | 6 |
| CE-128 → reverse | 16 | 16 | 16 | 16 | 8 |

Changing orientation repairs routing from the same failed parameters, giving
causal evidence beyond the observed gradient contrast. Whole-call rescue remains
partial: five fast cases omit `-Q`. The CE warm start shows reverse KL can preserve
training recall for 128 further updates even though base-start reverse KL acquired
only 4/16 over the same input order. Hard-label training created that policy and
its cost must be included. Development correctness falls from CE's 12/16 to 8/16;
this is not evidence of improved transfer or later retention under unrelated
learning. [Rescue responses](evidence/diagnosis-v2-rescue/responses.jsonl).

**Identifier production remains the main observed obstacle after acquisition.**
The component plan selects the earliest checkpoint reaching at least 15/16 recall
for each successful method: CE-128 and forward-high-128. Forward-low never reaches
the criterion. Twelve random lowercase identifiers (three each of lengths
4, 6, 8, 10), seed 2026091402, are crossed with all four conditions. Their identities
and exclusions were saved before any generation. They overlap neither acquisition,
earlier development, nor the accepted test identifiers. These are fresh
**development probes**, not a fresh confirmatory transfer test.

| Model | Free generation /48 | Correct route prefix supplied /48 | Correct route and uppercase identifier supplied /48 |
|---|---:|---:|---:|
| CE-128 | 28 | 28 | 48 |
| Forward-high-128 | 17 | 15 | 48 |
| Base teacher with 16 examples and reminder | 46 | — | — |

Both students freely route all 48 inputs correctly. Their identifier scores equal
their whole-call scores, 28 and 17, while suffix correctness is 45 and 41. Supplying
the correct `kestrel(text="` or `marten(text="` prefix does not restore identifier
production. The forward score changes by two despite an already correct route;
an externally tokenized prefix is itself an intervention, not necessarily the
same token path as free generation. Once the correct uppercase identifier is
also supplied, all suffix/closing completions succeed. The teacher copies all
48 identifiers and makes two suffix errors.

Thus wrong routing is insufficient to explain these copying failures. Correct
suffix use is available conditional on a correct preceding answer, but reliable
end-to-end identifier production is not established. These results do not prove
an abstract composition representation or identify the internal cause of copying
errors. The supplied prefixes contain oracle answer information and are not
deployable evidence-free student performance. The forward checkpoint's 13/16
versus CE's 12/16 on the earlier development words plainly did not justify a
transfer-advantage claim on random identifiers.
[Design](evidence/diagnosis-v2-components/design.json),
[raw interventions](evidence/diagnosis-v2-components/responses.jsonl).

**Costs and verification.** The matrix uses 1,280 optimizer updates and 512
student diagnostic generations, plus 16 teacher acquisition generations. Its
16 teacher distributions cost 7.25 seconds and 7,532 scored input tokens to cache;
each is reused across arms and epochs. Rescue uses 384 updates and 96 generations,
with a separately rebuilt 16-distribution cache costing 7.11 seconds. Component
work uses 336 generations, including 48 example-conditioned teacher calls. There
are 960 generation observations across these experiments, plus 18 saved-adapter
replays used only for verification. Probability audits add scoring calls, not
generation observations.

Measured run times are 494 seconds for the matrix, 149 for rescue, and 121 for
components; the peak recorded MLX allocation is 9.717 GB (decimal), below the
40 GB allocation ceilings. Probability-probe timers report another 28 seconds
each, excluding their initial model loading, and the reload audit reports
15 seconds. These are local timers, not total investigator time or energy.
No paid experimental model calls or new weight downloads were required. Teacher
pretraining, investigator effort, development labels/selection and hardware costs
are not priced. Caching differs from the accepted uncached run, so timing ratios
would not isolate objective efficiency. No useful-cost repayment claim follows.

The [audit](evidence/diagnosis-v2-audit/audit.json) verifies all 41 files listed in
the five input manifests, 944 response records, 16 teacher records, training
orders, initialization/reset controls, token decoding, prompt construction,
case separation and component counts. All 18 saved adapters reproduce the first
recall response's exact tokens on reload. Frozen-base hashes and reset logits
match. Three tests pass, including analytic forward/reverse values and gradients
against the actual diagnostic loss implementation. The accepted findings,
protocol and original evidence directories have no changes relative to `6960843`.
[Reproduction instructions](notes/reproduction.md).

The phase closes because it establishes a working soft-target acquisition
checkpoint, a controlled directional-loss effect on failed routing, and a
conditional separation of routing, copying and suffix completion. It does not
close on an unexplained negative result. Limits remain: one diagnostic training
seed and one small task, partial suffix rescue, teacher errors, imperfect copying,
and no test of later retention. A subsequent transfer claim would require a
separately fixed selection rule and new evaluation identifiers after any further
development. The present evidence warrants none of neural superiority, universal
reverse-KL failure, or a capacity limitation.
