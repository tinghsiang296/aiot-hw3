# Project Context
```markdown
# Project Context

## Purpose
This repository contains coursework and reference code for "Multiple Linear Regression" (AIoT / ML homework). The project provides reproducible data preprocessing, model training, evaluation, and simple experiment tracking utilities so students can run experiments, compare metrics, and reproduce results across environments.

## Tech Stack
- **Language:** Python 3.10+ (recommend 3.11)
- **Data & ML:** pandas, numpy, scikit-learn
- **Visualization:** matplotlib, seaborn
- **Experiment / CLI:** click or argparse (CLI), PyYAML for config
- **Dev / QA:** pytest, black, isort, flake8, mypy (optional)
- **Packaging / env:** `venv` or `virtualenv`, `requirements.txt` (or `pip-tools`/`poetry` if desired)

## Project Conventions

### Repo Layout
- `data/` - raw and derived small datasets (git-ignored large files)
- `notebooks/` - exploratory Jupyter notebooks
- `src/` - Python package / modules for data processing and models
- `scripts/` - convenience scripts and CLI entrypoints
- `tests/` - unit and integration tests
- `docs/` - usage docs and examples
- `requirements.txt` - pinned runtime deps

### Code Style
- Use **black** for formatting and **isort** for import ordering.
- Follow idiomatic Python and prefer explicit typing for public functions.
- Keep functions small and single-responsibility.

### Architecture Patterns
- Lightweight module structure: separate `data/` (I/O & transforms), `models/` (training and persistence), and `eval/` (metrics and plots).
- CLI-driven experiments that call into `src/` modules (logic in modules, CLI thin wrapper).

### Testing Strategy
- Unit tests for preprocessing, model training pipeline steps, and metric calculations using `pytest`.
- Deterministic tests: set random seeds in tests and CI to ensure reproducible output.
- Provide small fixture datasets under `tests/fixtures/` for fast test runs.

### Git Workflow
- Protect `main` branch; use short-lived feature branches: `feat/<name>`, `fix/<name>`, `chore/<name>`.
- Use PRs with descriptive titles and link to relevant OpenSpec change (when applicable).
- Prefer conventional commits but keep messages concise. Include `refs #<hw>` when referring to assignment numbers.

## Domain Context
- The project focuses on supervised regression (predicting continuous targets) using multiple linear regression.
- Typical tasks: feature engineering (scaling, categorical encoding), train/validation split, metrics (MSE, MAE, R^2), and simple model persistence.
- Datasets are small-to-moderate; workflows must be reproducible and easy to run on a laptop.

## Important Constraints
- Deterministic results: always expose and use a global random seed for experiments.
- Keep external, heavy dependencies minimal to make grading/running on student's machines feasible.
- Avoid GPU-specific code; rely on CPU-based scikit-learn.

## External Dependencies
- scikit-learn: core modeling
- pandas, numpy: data manipulation
- matplotlib/seaborn: plots
- PyYAML: config files
- CLI: click or built-in argparse

## How to run (developer quick-start)
1. Create virtualenv and install deps:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2. Run a quick experiment (example):
```bash
python scripts/run_experiment.py --config configs/example.yaml
```
3. Run tests:
```bash
pytest -q
```

## Notes for AI assistants
- Prefer non-breaking changes when possible; create an OpenSpec change for new features or breaking API changes (see `openspec/AGENTS.md`).
- When proposing features, include `proposal.md`, `tasks.md`, and spec deltas under `openspec/changes/<change-id>/`.

```
