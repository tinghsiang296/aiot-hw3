## 1. Implementation
- [ ] 1.1 Add `scripts/run_experiment.py` CLI entrypoint
- [ ] 1.2 Implement `src/experiments/runner.py` with config parsing and orchestrating pipeline
- [ ] 1.3 Add `configs/example.yaml` and config schema docs
- [ ] 1.4 Add result persistence (`results/` folder layout) and metrics JSON output
- [ ] 1.5 Add tests under `tests/test_runner.py` (unit + small integration using fixture data)

## 2. Docs
- [ ] 2.1 Update `README.md` with example run instructions
- [ ] 2.2 Add usage snippet to `docs/` if present

## 3. CI
- [ ] 3.1 Add a lightweight CI job that runs `scripts/run_experiment.py --config configs/example.yaml` on push to PR

## 4. Validation
- [ ] 4.1 Validate reproducibility with pinned random seed
- [ ] 4.2 Run `openspec validate add-experiment-cli --strict`
