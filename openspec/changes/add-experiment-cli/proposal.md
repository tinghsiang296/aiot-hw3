# Change: Add experiment-runner CLI

## Why
Students and maintainers need a simple, reproducible way to run training experiments, log metrics, and compare results without manually editing notebooks. A CLI + config-driven runner reduces friction, improves reproducibility, and provides a single entrypoint for automated grading and CI.

## What Changes
- Add a new `scripts/run_experiment.py` CLI (thin wrapper) and `src/experiments/runner.py` implementation.
- Add a YAML-based config schema for experiments and example configs under `configs/`.
- Persist experiment outputs (metrics, model artifacts, logs) under `results/<timestamp>-<name>/`.
- Add small integration tests and docs explaining how to run experiments locally and in CI.

**BREAKING:** None expected. This is additive.

## Impact
- Affected specs/capabilities: `experiments` (new capability)
- Affected code: new scripts and modules only; existing modules should be used (no refactor required)
- Testing: new integration tests and CI job to test runner

## Acceptance Criteria
- `python scripts/run_experiment.py --config configs/example.yaml` runs end-to-end and writes `metrics.json` and `model.pkl` into `results/`.
- Runs are reproducible given same seed and config.
- CLI returns non-zero exit code on config validation errors.

## Migration
- None required; new capability is optional for existing workflows.
