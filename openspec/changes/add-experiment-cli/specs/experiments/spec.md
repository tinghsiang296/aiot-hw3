## ADDED Requirements

### Requirement: Config-driven Experiment Runner
The system SHALL provide a CLI-driven experiment runner that executes a training pipeline based on a YAML configuration and produces reproducible result artifacts.

#### Scenario: Run experiment with valid config
- **WHEN** the user runs `python scripts/run_experiment.py --config configs/example.yaml`
- **THEN** the runner executes data loading, preprocessing, model training, and evaluation
- **AND** writes `metrics.json`, `model.pkl`, and `run.log` into a new folder under `results/` named with timestamp and config label
- **AND** exits with status code `0`

#### Scenario: Reproducible runs with seed
- **WHEN** the user runs the same config twice with an explicit seed set in the config (e.g., `seed: 42`)
- **THEN** the numeric metrics in `metrics.json` shall be identical within floating-point tolerance

#### Scenario: Config validation failure
- **WHEN** the user provides an invalid or missing config key
- **THEN** the runner SHALL print a clear validation error to stderr
- **AND** exit with a non-zero status code

#### Scenario: Dry-run mode
- **WHEN** the user runs with `--dry-run`
- **THEN** the runner SHALL validate config and print planned steps without writing artifacts

### Requirement: Results format
The runner SHALL persist metrics as JSON using a small schema with fields: `timestamp`, `config`, `metrics` (MSE, MAE, R2), and `seed`.

#### Scenario: Metrics file present
- **WHEN** a run completes successfully
- **THEN** `metrics.json` exists and contains the expected keys: `timestamp`, `config.label`, `metrics` and `seed`

### Requirement: CLI arguments
The CLI SHALL accept at minimum:
- `--config <path>` (required)
- `--output <path>` (optional; defaults to `results/`)
- `--dry-run` (flag)
- `--seed <int>` (optional; overrides config seed)

#### Scenario: Missing config arg
- **WHEN** the user omits `--config`
- **THEN** the CLI SHALL print usage and exit with non-zero status
