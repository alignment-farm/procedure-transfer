# Reproduction

Run from this repository on Apple Silicon with enough memory for the bf16 4B model and teacher/student forward passes (bounded at 40 GB of MLX allocation). Python packages are managed by uv. This study's uv.lock pins the full native environment; it differs from the sibling study's historical environment. Docker Model Runner was inspected but is not the distillation backend.

```
uv sync --extra adaptation --frozen
uv run --no-sync python -c 'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen3-4B-Instruct-2507", revision="cdbee75f17c01a7cc42f958dc650907174af0554", local_dir="models/qwen3-4b-instruct", allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"])'
uv run --no-sync python -m unittest discover -s tests -v
uv run --no-sync python scripts/experiment.py --output evidence/NEW-RUN
uv run --no-sync python scripts/analyze.py evidence/NEW-RUN
```

The runner rejects an existing output directory, snapshots its sources and frozen protocol, and records Git provenance, model file hashes, raw responses, token IDs, loss/gradient/cost events, six final adapters and SHA256SUMS. The analyzer rejects incomplete runs and checks every recorded hash, score, branch size and training-input frequency before writing a separate analysis directory. Exact device timing and numeric behavior can depend on the runtime. Final checkpoints are never selected using test outputs.

The owned `sources/model-reference.json` preserves the original sibling model hash manifest with source-file hash and Git provenance. No sibling checkout is needed. The model reference is the Qwen revision above. No sibling training/evaluation data is loaded by the experiment. The local feasibility pilot is invoked by `uv run --no-sync python scripts/pilot.py` and uses a fixed development-only output path; preserve existing pilot evidence before any independently named rerun. Final reproduction should run the frozen comparison without rerunning development or changing the protocol.

Source study: `procedure-acquisition-and-reuse` revision 08d8ec1b0b757e8a537a14d708bd895e8829ac68. OPCD author implementation: microsoft/LMOps revision 4f2a9deb5f08e459fd44c2e4792344d78ca89fc3. MLX-LM revision and model revision are additionally recorded in uv.lock and each run's resource.json. See [sources/README.md](../sources/README.md) for exact paper reading and code scope.
