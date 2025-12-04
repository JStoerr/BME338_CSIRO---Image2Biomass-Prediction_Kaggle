# Repository Guidelines

This Image2Biomass workspace pairs Kaggle imagery with notebook-first experimentation, so keep contributions reproducible, reviewable, and storage-aware.

## Project Structure & Module Organization
- `model_build.ipynb` is the canonical modeling notebook; once logic stabilizes, port reusable pieces into a `scripts/` helper or a new `src/` package so they can be imported elsewhere.
- Kaggle assets stay under `data/` (`train/`, `test/`, `*_csv` mirrors); write derived embeddings or tabular features to `data/derived/<artifact>.parquet` with brief provenance notes.
- The actively supported runtime lives in `wsl_venv/` (aligned with WSL paths); `environment.yml` exists for coworkers using Conda but is not part of the default workflow here.
- `scripts/mlp_head.py` trains the lightweight regression head on the `data/derived/dino_embeddings.parquet` artifact, reports validation R²/ RMSE alongside losses, and should remain CLI-driven so it can be versioned and reproduced outside notebooks.
- `scripts/parameter_tuning.py` wraps Ray Tune around the MLP head; it maximizes the validation R² emitted in `mlp_metrics.json` and exposes `--num-samples` plus `--time-budget-s` flags so you can decide how long sweeps should run.

## Build, Test, and Development Commands
- `source wsl_venv/bin/activate && python -m pip install -U pip` keeps the WSL virtualenv current; only regenerate `wsl_venv` when dependency drift becomes unmanageable.
- Coworkers who prefer Conda can run `conda env create -f environment.yml`, but aligned contributors should rely on `wsl_venv` so paths and compilers match WSL expectations.
- With the venv active, start notebooks via `jupyter lab`; commit checkpoints only after running top-to-bottom.
- Promote notebooks to scripts via parameterized CLIs, e.g., `python -m src.training.run_experiment --config configs/baseline.yaml`; add `--dry-run` to validate configs without GPU usage and finish each notebook refactor with `pytest tests -m "not slow"` for a quick regression signal.
- After generating embeddings, run `python scripts/mlp_head.py --epochs 200 --batch-size 64` (tweak flags as needed) to produce updated predictions, `mlp_head.pt`, `mlp_head_last.pt`, and metrics inside `data/derived/`; refresh these artifacts whenever notebook preprocessing changes.
- For longer sweeps, launch `python scripts/parameter_tuning.py --num-samples 60 [--time-budget-s 86400]` so Ray explores more of the search space while tracking metrics in `data/derived/tune_runs/`.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indents; auto-format using `black .` and `isort .` before pushing.
- Name notebooks `NN_description.ipynb`, Python helpers `verb_noun.py`, and config files `exp_<target>_<model>.yaml` for easy discovery.
- Prefer type hints, dataclasses for dataset schemas, and centralized logging (e.g., `logging` + `tqdm`) over ad-hoc prints.

## Testing Guidelines
- Every preprocessing or modeling utility should have a companion `tests/test_<module>.py`; mock tensors with `torch.rand(3, 224, 224)` instead of large fixtures.
- Target ≥80% line coverage on utility modules; expose notebook functions so tests can import them.
- Before merging, rerun `pytest` plus a short inference cell in `model_build.ipynb` to verify the active Conda environment.

## Commit & Pull Request Guidelines
- History favors short, present-tense subjects (e.g., “finished class specific UMAP”); stay under 60 characters and expand reasoning in the body when behavior changes.
- Reference linked issues, list regenerated artifacts (`data/derived/*`), and paste key metrics or screenshots so reviewers can validate outcomes.
- PR descriptions must capture dataset snapshots, commands executed, and remaining risks/next steps.

## Security & Data Handling
- Keep Kaggle API tokens, AWS credentials, and large raw dumps outside Git; rely on `.kaggle` configs or environment variables instead of hard-coded strings.
- When sharing intermediate data, strip PII and note the originating script plus command line in a README block within `data/derived/`.
