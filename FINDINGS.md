# Procedure transfer: local findings

14 September 2026. Completed bounded investigation: two development pilots, six prospective adapter-training runs, 976 scored evaluation responses, and a separate artifact reload audit. Protocol and execution sources were frozen at **5f636fb**; results, adapters and audits are preserved at **ab87431**. This is a local method adaptation, not a reproduction of a published benchmark.

**Evidence-conditioned reverse-KL distillation did not improve procedural transfer under the tested recipe. On-policy students scored 0/96 and 2/96 on fresh inputs, versus direct imitation's 24/96 and 32/96. Retaining the checked examples with the selected generic induction reminder scored 89/96. Both distillation variants also failed to acquire most training calls.** The supported negative result concerns this acquisition recipe; it does not establish a general limitation of context distillation or neural procedural memory.

## What was compared

The new task routes an identifier to `kestrel` or `marten`, uppercases it, and optionally appends `-Q`. Four training identifiers, of lengths 3–6, cover **all four channel/priority combinations**, yielding 16 oracle-checked calls. Unlike the earlier study, no condition combination is withheld. Transfer uses 24 fresh identifiers, six each at lengths 4, 6, 8 and 10, crossed with the four conditions: 96 paired cases. Stable routing and transformation across identifiers remain inductive assumptions; finite examples do not uniquely determine arbitrary programs.

All treatments use the same bf16 **Qwen3-4B-Instruct-2507**, revision `cdbee75f17c01a7cc42f958dc650907174af0554`, with identical initial adapters and training-input orders within each seed. The two seeds, 17 and 29, vary initialization and order together, and rollout randomness for on-policy training; this study does not disentangle those factors. Students see only the schema and current input. The frozen base teacher additionally sees all 16 checked calls, including the current training answer, and a generic reminder to infer a consistent protocol accounting for the input fields and argument changes.

Three treatments each receive 128 updates:

- **Imitation:** answer/EOS cross-entropy on the checked calls.
- **Checked-prefix distillation:** full-vocabulary reverse KL against the evidence-conditioned teacher, evaluated on checked answer/EOS prefixes.
- **On-policy distillation:** the same reverse KL, evaluated on newly sampled student responses to the same training inputs. Sampling uses temperature 1, no top-k/top-p restriction, and at most 48 tokens. Teacher distributions and sampled token choices are detached from differentiation.

The checked-prefix control holds loss orientation, teacher, context and training inputs fixed when comparing response sources. Comparison to imitation changes the loss and teacher supervision together. No extra unique training inputs, teacher-generated answer dataset, verifier feedback, or supplied rule enters student training. The teacher nevertheless contributes pretrained capabilities and dense conditional probabilities beyond the hard labels. The generic reminder is a disclosed researcher-authored inductive cue; it does not state the routing map, uppercase operation or suffix.

The common LoRA starting point uses the last eight blocks' query/value projections, rank 8, scale 2, dropout 0, and AdamW at 0.0005 with no weight decay. Eight passes over the complete condition coverage yield 128 steps per run. Native checks supported execution of this economical starting point; they did not establish that its optimization settings were appropriate or optimal for reverse KL. There was no learning-rate or checkpoint search. All six final checkpoints were fixed before the evaluation identifiers were constructed. The [frozen protocol](protocol/transfer-v1.md) specifies the selection rule, analysis and resource ceiling.

## Outcomes

Paired entries below are seeds 17 / 29. Reference branches are single base-model evaluations, not additional training seeds.

| Method/reference | Training-call recall | Fresh-input transfer |
|---|---:|---:|
| No acquisition | Not measured | 0/96 |
| Retained examples + selected reminder | 14/16 | 89/96 |
| Supplied complete rule, privileged diagnostic | Not measured | 65/96 |
| Direct imitation | 16/16 / 16/16 | 24/96 / 32/96 |
| Reverse KL, checked prefixes | 3/16 / 4/16 | 3/96 / 1/96 |
| Reverse KL, on-policy prefixes | 0/16 / 3/16 | 0/96 / 2/96 |

On-policy minus imitation averages **−28.1 percentage points**, with per-seed differences of −25.0 and −31.25 points. The predeclared descriptive identifier-cluster bootstrap interval is **[−40.6, −16.7] points**. On-policy minus checked-prefix distillation averages −1.04 points, with interval [−2.60, 0.00] and opposite per-seed signs. Neither comparison satisfies the predeclared positive-advantage rule. Intervals resample the 24 identifiers while retaining all four conditions and both seeds; they condition on these two training runs and this task, not a population of models or procedures.

Imitation's perfect recall does not imply reliable transfer. On familiar lengths 4 and 6, each imitation seed gets 19/48 correct; on longer lengths 8 and 10, they get only 5/48 and 13/48. Retained examples get 45/48 and 44/48 respectively. Both distillation variants fail largely **before** the generalization question: their final acquisition recall is poor. The result therefore localizes a failure of this distillation acquisition recipe, rather than showing successful learning followed only by a transfer deficit. Covering all condition combinations was insufficient to make these recipes reliable on this new task; this comparison does not isolate the cause of the previous study's failure.

The [post-hoc error audit](evidence/transfer-v1-components/components.json) describes the failures without claiming their cause. Every distillation checkpoint emits only one leading tool name throughout the 96 transfer calls. The on-policy seed-17 checkpoint returns the same response across all four conditions for every identifier, and all its calls are malformed. Both imitation checkpoints preserve exact call syntax; identifier transformations remain the main difficulty. These are observed endpoint behaviors, not evidence of an irreversible attractor, permanent information loss or a proven optimizer mechanism.

The references also matter. The selected teacher's 89/96 shows that the evidence can support useful new-input behavior in this model. It is imperfect, including 14/16 acquisition recall. The complete-rule branch gets routing and suffix decisions right on all 96 cases, but mis-transforms 31 identifiers. A correct verbal rule is therefore not an automatic execution ceiling. The examples prompt was development-selected and the supplied-rule prompt was not optimized, so this is not a general ranking of examples and rules.

## Development and evidence accounting

The initial example prompt scored 9/16 on development, versus 0/16 without evidence and 16/16 with a supplied rule. One generic reminder was then tested under a recorded selection rule: adopt it only if it exceeds 9/16, with no further candidates. It scored 16/16 development and 14/16 acquisition recall. That selected prompt subsequently scored 89/96 on the untouched final identifiers. No student transfer scores selected the recipe. The original failures, candidate wording, selection rule and executed code are preserved in the [development note](notes/2026-09-14-development.md) and both pilot directories.

There are 16 unique training labels, 16 additional development labels used to assess/select the teacher prompt, and 96 final labels used only for scoring. All come from a researcher-defined deterministic contract; there is no claimed autonomous environment exploration. Repeated uses across branches are not new independent examples. On-policy training produces 256 additional sampled responses to existing inputs, not additional tasks or oracle answers. The teacher reminder and investigator/assistant effort are unmetered construction work. [Cross-study checks](evidence/cross-study-freshness.json) find no overlap between the 24 final identifiers and 38 identifiers extracted from the earlier study's response logs; no pretraining-contamination claim is made.

## Costs and verified resources

The actual machine is a **64 GiB Mac Studio M1 Ultra**. Docker Model Runner was inspected, but its available quantized model did not establish the gradient/distribution access needed for this comparison. Native MLX supplied full 151,936-token teacher distributions and student LoRA gradients. The downloaded checkpoint matches the prior study's model-file hashes. MLX-LM is pinned to `86b48c461feebf87c58788655b7e57b5574b9e6d`; exact packages, interpreter, model files and hardware are recorded in the [run resource manifest](evidence/transfer-v1/resource.json), [environment record](evidence/runtime-environment.json), and `uv.lock`.

| Treatment | Training seconds, seeds 17 / 29 | Teacher scoring seconds | Student rollout seconds |
|---|---:|---:|---:|
| Imitation | 23.6 / 23.7 | 0 / 0 | 0 / 0 |
| Checked-prefix reverse KL | 81.8 / 82.1 | 58.3 / 58.4 | 0 / 0 |
| On-policy reverse KL | 131.8 / 121.7 | 57.1 / 57.0 | 51.4 / 42.0 |

Per-treatment timers include sampling, scoring, updates and adapter saving; the main-run total also includes preparation and invariant-checking overhead.

Imitation and checked-prefix distillation each supervise 1,184 response positions per run; on-policy runs use 1,809 and 1,282. The four distillation runs additionally process 235,219 teacher input positions. Equal optimizer steps do not imply equal compute. Each adapter is 2,624,893 bytes.

For 96 fresh uses, retained examples require 43,352 prompt tokens and 59.5 generation seconds, with 89 successes. Imitation uses 6,872 prompt tokens and 28.3/28.9 seconds, with only 24/32 successes, before adding its training cost. Distillation adds greater acquisition cost and achieves much lower accuracy. **There is no useful-cost or repayment advantage at comparable performance.** No break-even projection is offered. Timings use one sequential workload with fresh generation caches and no persistent teacher-prefix cache; they are not energy, money, optimized-serving or repeated timing estimates.

The main run takes 887.24 seconds, including preparation, invariants and evaluation, with peak MLX allocation of 9.64 GB. This is not total host memory. Initial model download takes 741.7 seconds by the download progress timer and is separate from the main run; setup and development costs are preserved rather than treated as free resident resources. All experimental inference/training is local.

Independent analytic checks validate the reverse-KL value, student derivative and stopped teacher derivative. Native pilot checks verify response-position alignment at the first teacher-scored position, finite nonzero adapter gradients, unchanged base hashes and exact reset logits. The completed-run analyzer checks file hashes, scores, input frequencies, branch sizes, and evaluation after all checkpoints were fixed. The separate [reload audit](evidence/transfer-v1-reload-audit/audit.json) checks all 976 prompt/output token records and reproduces index-zero token streams for all nine branches after loading the saved adapters. Those nine repeated calls are verification, not additional test evidence.

## Interpretation and stop decision

The initial expectation is not supported here. A useful evidence-conditioned teacher did not become a useful context-free student through either tested reverse-KL route. Direct imitation acquired the checked calls but generalized poorly; retained evidence remained the strongest observed representation. Poor distillation recall, and failure of both prefix variants, mean these results do not specifically identify student-generated prefixes as the cause. Shared optimizer settings, the LoRA parameterization, loss geometry, teacher errors and token-prefix behavior remain possible contributors that this comparison does not separate.

This is one synthetic procedure, two coupled initialization/order seeds, one common training recipe and one developed teacher prompt. The full-vocabulary objective, small LoRA updates, fixed examples and unfiltered sampling are local adaptations of [OPCD, 2602.12275v1](https://arxiv.org/html/2602.12275v1), not its published training configuration. The [primary-method/code reading](sources/README.md) records exact versions and distinguishes the author implementation from this one. The broader [PMD, 2607.01480v1](https://arxiv.org/html/2607.01480v1) co-evolves memory and policy; that mechanism was not tested.

The bounded contribution is complete. No performance-driven retraining, later-learning, correction, or consolidation run follows these results. A future investigation of this route would first need to establish reliable acquisition under an independently developed optimization recipe, with a new prospective evaluation. This study establishes neither later retention nor a general inability to learn procedures.

[Reproduction instructions](notes/reproduction.md) · [Audited tables and costs](evidence/transfer-v1-analysis/README.md) · [Raw responses](evidence/transfer-v1/responses.jsonl) · [Training/cost ledger](evidence/transfer-v1/events.jsonl) · [Artifact checksums](evidence/transfer-v1/SHA256SUMS).
