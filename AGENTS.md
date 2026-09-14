# Procedure transfer

Read [README.md](README.md) first. It owns the question, initial expectation,
starting evidence and publication links. The root prepared this project for
an independent ancillary session; the investigator owns its methods, evidence
and local publication.

Investigate whether an evidence-conditioned acquisition method improves
procedural transfer beyond direct imitation. Context distillation is the
starting method lead. Revise the question when sources or findings warrant it,
and explain consequential changes. The parent [study map](../../construct-2/studies/README.md)
supplies theory and synthesis; it is not a mandatory experimental protocol.

## Research practice

- Read the closest primary methods and available implementations. Record exact
  paper versions and code revisions, and distinguish reported results, local
  adaptation and reproduction. Use the parent's [source guidance](../../construct-2/AGENTS.md#research-sources)
  for arXiv access, caching and rate limits.
- Distinguish the evidence available from the method used to learn from it.
  Identify added examples, teacher knowledge, oracle answers and validation
  feedback, and their acquisition costs. State composition assumptions.
- Separate training recall, transfer, later retention and useful-cost claims.
  Preserve exploratory changes and failures; use fresh evaluation material and
  a selection rule fixed before evaluating a developed claim.
- Keep notes proportional to the question. The study owns its implementation,
  protocol, resource sizing and repairs. A negative result alone is not a stop
  condition. When acquisition has not worked, pursue bounded diagnostic
  development to distinguish causes or establish a functioning learning regime.
  Calibration on separate development material is legitimate; finite gradients
  and exact resets establish mechanics, not acquisition. Publish findings and
  failed attempts with supporting evidence and identifiable Git revisions.

The user accepted the first publication on 14 September 2026 and asked for less
acceptance of unresolved learning failures. Follow the README's current
diagnostic direction. Preserve the accepted comparison and frozen protocol;
evaluate a subsequently developed transfer claim on fresh material. Close a
phase on explanatory progress, a demonstrated limitation or a concrete resource
constraint, without requiring a neural advantage or an indefinite search.

## Model resources

- Dedicated Mac Studio M1 (64 GB unified memory), serving over Tailscale through
  Docker Model Runner (preferred).
  Chat-completions endpoint:
  `https://mac-studio-7hr7.taile71f88.ts.net/engines/v1/chat/completions`
- Local open-weight models with `docker model`
- OpenAI models with `codex`
- SpaceXAI models with `agent`

Verify model revisions, teacher token-distribution access and student gradient
access before relying on a resource for distillation. The serving endpoint is
carried forward from the lab instructions; project preparation did not test it.
The completed acquisition study's native MLX route is a reusable lead, not
verification of this new objective or its resource needs.

## Dependency management

- Use `uv` for Python package and project management.
- Use Docker and Compose/Dockerfiles for supporting resources where needed.
