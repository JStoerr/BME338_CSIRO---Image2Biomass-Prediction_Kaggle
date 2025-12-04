# Image2Biomass – CSIRO Kaggle Experiments

This workspace contains the notebook-first pipeline we use to predict biomass targets from Kaggle-provided canopy imagery. Reusable logic is gradually promoted out of notebooks into `scripts/` helpers so collaborators can reproduce experiments without rerunning exploratory code.

## Repository Layout
- `data/`: Kaggle assets (`train/`, `test/`, CSVs); derived artifacts live under `data/derived/`.
- `scripts/model_build.ipynb`: canonical experimentation notebook for feature generation.
- `scripts/mlp_head.py`: CLI for training the lightweight regression head on DINO embeddings; writes predictions, checkpoints, and `mlp_metrics.json` (loss, RMSE, validation R²).
- `scripts/parameter_tuning.py`: Ray Tune sweep over the MLP head hyperparameters; maximizes the recorded validation R² and persists sweep summaries inside `data/derived/tune_runs/`.
- `wsl_venv/`: preferred virtual environment for day-to-day development on WSL.

## Environment & Tooling
1. `source wsl_venv/bin/activate`
2. `python -m pip install -U pip` plus any new dependencies from `requirements.txt`.
3. Notebook contributors can fall back to `conda env create -f environment.yml` if needed, but keep paths aligned with the WSL venv when sharing artifacts.

## Typical Workflow
1. Run `jupyter lab` (with the venv active) and execute `scripts/model_build.ipynb` top-to-bottom to refresh embeddings (`data/derived/dino_embeddings.parquet`).
2. Train the regression head:
   ```bash
   python scripts/mlp_head.py --epochs 200 --batch-size 64 --output-dir data/derived/$(date +%Y%m%d)
   ```
   The script reports validation loss/RMSE/R², saves predictions to `mlp_predictions.csv`, and exports checkpoints (`mlp_head.pt`, `mlp_head_last.pt`).
3. Hyperparameter tuning (optional, but encouraged after preprocessing changes):
   ```bash
   python scripts/parameter_tuning.py --num-samples 60 --time-budget-s 86400
   ```
   Ray Tune dispatches repeated `mlp_head.py` runs, targeting higher validation R². Results land in `data/derived/tune_runs/` with JSON summaries you can diff between sweeps.
4. Before sharing results, run the regression head once more with the chosen hyperparameters and attach the updated artifacts/metrics to your PR description.

## Testing & Quality
- Refactor notebook utilities into importable modules and cover them with `tests/test_<module>.py`; mock tensors with `torch.rand(3, 224, 224)`.
- Aim for ≥80 % coverage on utility modules and run `pytest tests -m "not slow"` before committing substantial changes.
- Format Python code with `black .` and `isort .` to keep diffs consistent.

## Data & Security Notes
- Keep Kaggle tokens and other credentials in environment variables or local config files outside version control.
- Document derived artifacts (generator script + invocation) in a short README inside `data/derived/<artifact>/` whenever you share them.
