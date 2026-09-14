# Reproduction

Run from this repository on Apple Silicon with enough memory for the bf16 4B model and teacher/student forward passes (bounded at 40 GB of MLX allocation). Python packages are managed by uv. This study's uv.lock pins the full native environment; it differs from the sibling study's historical environment. Docker Model Runner was inspected but is not the distillation backend.

```
uv sync --python 3.14.7 --extra adaptation --frozen
uv run --no-sync python -c 'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen3-4B-Instruct-2507", revision="cdbee75f17c01a7cc42f958dc650907174af0554", local_dir="models/qwen3-4b-instruct", allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"])'
uv run --no-sync python -m unittest discover -s tests -v
uv run --no-sync python scripts/experiment.py --output evidence/NEW-RUN
uv run --no-sync python scripts/analyze.py evidence/NEW-RUN
uv run --no-sync python scripts/audit_saved.py evidence/NEW-RUN
uv run --no-sync python scripts/components.py evidence/NEW-RUN
```

The runner rejects an existing output directory, snapshots its sources and frozen protocol, and records Git provenance, model file hashes, raw responses, token IDs, loss/gradient/cost events, six final adapters and SHA256SUMS. The analyzer rejects incomplete runs and checks every recorded hash, score, branch size and training-input frequency before writing a separate analysis directory. Exact device timing and numeric behavior can depend on the runtime. Final checkpoints are never selected using test outputs.

The owned `sources/model-reference.json` preserves the original sibling model hash manifest with source-file hash and Git provenance. No sibling checkout is needed. The model reference is the Qwen revision above. No sibling training/evaluation data is loaded by the experiment. The local feasibility pilot is invoked by `uv run --no-sync python scripts/pilot.py` and uses a fixed development-only output path; preserve existing pilot evidence before any independently named rerun. Final reproduction should run the frozen comparison without rerunning development or changing the protocol.

Source study: `procedure-acquisition-and-reuse` revision 08d8ec1b0b757e8a537a14d708bd895e8829ac68. OPCD author implementation: microsoft/LMOps revision 4f2a9deb5f08e459fd44c2e4792344d78ca89fc3. MLX-LM revision and model revision are additionally recorded in uv.lock and each run's resource.json. See [sources/README.md](../sources/README.md) for exact paper reading and code scope.

The completed frozen execution is revision `5f636fb`; its evidence and supplemental audits are at `ab87431`. `audit_saved.py` independently checks all token records and replays only the index-zero transfer case per branch after checkpoint reload. These repeats are excluded from accuracy denominators. `components.py` is an explicitly post-hoc descriptive error parser; it does not change the predeclared outcome or select a checkpoint. The six original adapters are committed under `evidence/transfer-v1/`.

The subsequent diagnostic phase is published in [DIAGNOSIS.md](../DIAGNOSIS.md).
Its matrix source/plan is `966fd43`, signal plan `ec8c77d`, rescue/component source
and amendment `429b1e0`, and complete evidence/audit `c183674`. Run these sequentially
on the same GPU, using a new output name; every script rejects existing output:

```
uv run --no-sync python scripts/diagnosis_v2.py --output evidence/NEW-DIAGNOSIS
uv run --no-sync python scripts/diagnostic_gradients.py evidence/NEW-DIAGNOSIS
uv run --no-sync python scripts/diagnostic_rescue.py evidence/NEW-DIAGNOSIS
uv run --no-sync python scripts/diagnostic_components.py evidence/NEW-DIAGNOSIS
uv run --no-sync python scripts/audit_diagnosis.py evidence/NEW-DIAGNOSIS
```

The component generator reads the accepted `evidence/transfer-v1/evaluation.json`
only to exclude its identifiers. The audit also checks the already committed
`evidence/transfer-v1-gradients` retrospective bundle. To independently recompute
that probe, copy the original run directory to a newly named path and invoke
`diagnostic_gradients.py NEW-PATH --legacy`; the script will create a sibling
`NEW-PATH-gradients` directory. Preserve the committed original evidence.

Diagnostic scripts save source copies, raw records, checkpoints where applicable
and SHA256SUMS. The matrix contains the runtime/task snapshots and resource hashes
shared by the follow-ups. The audit checks 18 checkpoint replays, all generated
token records, construction and training controls. The mathematical checks run
through the unittest command above. This phase intentionally reports development
and intervention results; it does not generate a new confirmatory transfer test.
