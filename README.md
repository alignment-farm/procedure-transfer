# Procedure transfer

**Status: bounded investigation complete, 14 September 2026.**

**The tested evidence-conditioned distillation recipe did not improve transfer.**
On-policy students scored 0/96 and 2/96 on fresh inputs, versus direct imitation's
24/96 and 32/96. Retaining the checked examples with a development-selected
generic reminder scored 89/96. Both distillation variants also had poor training
recall. These are local results for one shared LoRA recipe, not a general
limitation of context distillation.

- [FINDINGS.md](FINDINGS.md): local publication, interpretation, scope and stop decision.
- [Frozen protocol](protocol/transfer-v1.md), committed as `5f636fb` before the prospective run.
- [Audited results](evidence/transfer-v1-analysis/README.md), [raw responses](evidence/transfer-v1/responses.jsonl),
  and [saved-adapter audit](evidence/transfer-v1-reload-audit/audit.json), preserved at `ab87431`.
- [Development history](notes/2026-09-14-development.md), [primary-source reading](sources/README.md),
  and [reproduction instructions](notes/reproduction.md).

The original starting question and expectations below are preserved for context.
The completed investigation stops at acquisition and immediate transfer; it does
not test later retention or correction.

This study asks whether a different acquisition method can turn checked
experience into behavior that transfers to new inputs. It follows the completed
[procedure-acquisition investigation](../procedure-acquisition-and-reuse/README.md)
and informs Construct-2's [S2 question](../../construct-2/studies/README.md#s2-can-a-learned-procedure-survive-later-learning-and-accept-a-scoped-correction).

## Question

**Does evidence-conditioned distillation produce more reliable procedural
transfer than direct imitation, under disclosed evidence and acquisition cost?**

The earlier fixed adapter recalled 12/12 training calls but scored 28/64 on its
fresh follow-up, against 51/64 with retained examples. A supplied correct rule
scored 64/64 but contained privileged information. The demonstrations did not
uniquely determine the intended composition rule. These results leave both the
acquisition method and the available evidence in question; they do not establish
that the model cannot learn the procedure.

## Initial expectation

Begin with the closest methods and a focused local comparison. The proposed
method lead is a teacher that consults the acquisition evidence while
supervising the learner's own attempted responses; the learner later acts
without that evidence. This is existing context-distillation work, not a new
mechanism claim. The investigator owns the task, precise objective, controls,
budget and implementation, and may revise the question when reading or results
warrant it.

A useful first outcome is a local findings note explaining the acquisition
method examined, what transferred, what did not, and what the evidence says
about further study. A bounded experiment is appropriate where feasible.
An explanation or a negative result can complete the contribution. There is no
requirement to obtain a neural advantage or proceed to later-learning,
correction or consolidation experiments.

## Starting comparison

Use a small independently checkable procedure, such as unfamiliar tool routing
or argument transformations. Make the composition assumptions and coverage of
conditions explicit. New inputs should require applying the procedure, with
fresh development and evaluation material beyond the previous study's cases.

Compare direct imitation of checked calls with the selected distillation
method, retaining no-update and explicit-evidence references. Use a common base
model and disclose teacher capabilities, prompts, training-input access and any
extra oracle feedback. A supplied complete rule can diagnose transfer under
known instructions, but is different from discovering that rule from experience.
Measure whether the evidence-conditioned teacher can use the evidence; teacher
success need not bound everything an updated student could learn.

A distillation gain could come from the learning objective, learner-generated
responses, additional training cases, or better supervision. Choose controls
that support the intended claim. If several change together, report a combined
method result. Record construction, teacher inference and student-training
costs; equal optimizer steps do not establish equal compute. Test retained
behavior after removing the teaching context, including acquisition recall
and new-input transfer separately. A cost advantage requires comparable useful
performance; repeated cheap errors do not establish repayment.

These are starting design considerations, not a frozen protocol. Preserve
exploration and unsuccessful attempts, and fix the evaluation and checkpoint
selection rule before a prospective test of a developed claim. Initialization
and data sampling may interact, as S3 found in a different mechanism; do not
assume its particular embedding intervention applies to a language-model adapter.

## Starting reading and reusable work

The root's [focused methods review](../../construct-2/sources/2026-09-13/README.md)
records exact reading scope and limits. Main leads:

- [On-Policy Context Distillation, 2602.12275v1](https://arxiv.org/html/2602.12275v1),
  §3 and training appendices: the closest acquisition-method lead.
- [Procedural Memory Distillation, 2607.01480v1](https://arxiv.org/html/2607.01480v1),
  §§3.1–3.2: a broader approach with evolving procedural memory and policy.
- [SEAL, 2506.10943v2](https://arxiv.org/html/2506.10943v2), §3, and
  [PERK, 2507.06415v3](https://arxiv.org/html/2507.06415v3), §3:
  alternative ways to learn an acquisition method.

Inspect the previous study's [protocol](../procedure-acquisition-and-reuse/protocol/acquisition-v1.md)
and [results/reproduction note](../procedure-acquisition-and-reuse/notes/2026-09-11-acquisition-results.md)
for the native MLX implementation. Reuse components with provenance, not its
evaluation cases or fixed training recipe by default. The [depth findings](../neural-memory-depth/FINDINGS.md)
motivate separating capacity from successful training; they do not explain the
earlier adapter's failure. This study is a focused local investigation with
substantial prior overlap, not a claim to reproduce either distillation paper.

## Resources and publication

[AGENTS.md](AGENTS.md#model-resources) lists current resource routes. Existing
native MLX adapter training is a feasibility lead; this distillation method
still needs verified gradient access and teacher token distributions. A chat
endpoint alone does not establish either capability. No model installation or
training run was performed during preparation.

Keep the initial work focused on existing local resources. No numerical budget
has been set by the root. The investigator sizes its bounded comparison after
checking feasibility; a large training campaign is outside this initial brief.

Publish the overview and findings here, with links to local methods, evidence
and reproduction instructions. An inspectable `FINDINGS.md` is sufficient;
a separate manuscript is optional. Preserve identifiable revisions in Git.
