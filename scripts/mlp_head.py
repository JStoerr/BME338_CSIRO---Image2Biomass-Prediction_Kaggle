"""Simple MLP regression head for DINO embeddings."""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def parse_hidden_dims(value: str) -> Tuple[int, ...]:
    parts = [int(chunk) for chunk in value.split(",") if chunk]
    if not parts:
        raise argparse.ArgumentTypeError("hidden dimensions must be non-empty")
    return tuple(parts)


@dataclass
class DatasetBundle:
    features: np.ndarray
    targets: np.ndarray
    image_ids: List[str]
    target_names: List[str]
    feature_cols: List[str]


def load_embeddings_and_targets(
    embeddings_path: Path,
    labels_csv: Path,
) -> DatasetBundle:
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Embeddings file {embeddings_path} does not exist. "
            "Run scripts/model_build.ipynb first."
        )
    if not labels_csv.exists():
        raise FileNotFoundError(f"Training labels {labels_csv} were not found.")

    emb_df = pd.read_parquet(embeddings_path)
    emb_df["image_id"] = emb_df["image_path"].apply(lambda p: Path(p).name)
    feature_cols = [col for col in emb_df.columns if col not in {"image_path", "image_id"}]

    labels = pd.read_csv(labels_csv)
    labels["image_id"] = labels["image_path"].apply(lambda p: Path(p).name)
    pivot = (
        labels.pivot_table(index="image_id", columns="target_name", values="target", aggfunc="first")
        .reset_index()
    )
    merged = emb_df.merge(pivot, on="image_id", how="inner")
    merged = merged.dropna()

    target_names = sorted(pivot.columns.drop("image_id"))
    features = merged[feature_cols].to_numpy(dtype=np.float32)
    targets = merged[target_names].to_numpy(dtype=np.float32)
    image_ids = merged["image_id"].tolist()

    return DatasetBundle(features, targets, image_ids, target_names, feature_cols)


class MLPHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        out_dim: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = in_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden
        layers.append(nn.Linear(prev_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_dataloader(
    features: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(features).float(),
        torch.from_numpy(targets).float(),
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        preds = model(batch_x)
        loss = loss_fn(preds, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        preds = model(batch_x)
        loss = loss_fn(preds, batch_y)
        total_loss += loss.item()
        total_batches += 1
    return total_loss / max(total_batches, 1)


@torch.no_grad()
def predict_all(
    model: nn.Module,
    features: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    tensor = torch.from_numpy(features).float().to(device)
    outputs = model(tensor)
    return outputs.cpu().numpy()


def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    bundle = load_embeddings_and_targets(args.embeddings, args.labels)
    (
        X_train,
        X_val,
        y_train,
        y_val,
        id_train,
        id_val,
    ) = train_test_split(
        bundle.features,
        bundle.targets,
        bundle.image_ids,
        test_size=args.val_split,
        random_state=args.seed,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    train_loader = make_dataloader(X_train_scaled, y_train, args.batch_size)
    val_loader = make_dataloader(X_val_scaled, y_val, args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = MLPHead(
        in_dim=X_train.shape[1],
        hidden_dims=args.hidden_dims,
        out_dim=y_train.shape[1],
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    last_val_loss = None
    last_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = evaluate(model, val_loader, loss_fn, device)
        last_val_loss = val_loss
        last_epoch = epoch
        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }
        if epoch % args.log_every == 0 or epoch == 1 or epoch == args.epochs:
            print(f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    if best_state is None or last_val_loss is None:
        raise RuntimeError("Training did not produce any model state.")
    last_state = {
        "model_state": model.state_dict(),
        "epoch": last_epoch,
        "val_loss": last_val_loss,
    }
    model.load_state_dict(best_state["model_state"])

    val_preds = predict_all(model, X_val_scaled, device)
    val_r2 = float(r2_score(y_val, val_preds, multioutput="variance_weighted"))

    scaled_full = scaler.transform(bundle.features)
    predictions = predict_all(model, scaled_full, device)

    derived_dir = args.output_dir
    derived_dir.mkdir(parents=True, exist_ok=True)

    predictions_df = pd.DataFrame(predictions, columns=bundle.target_names)
    predictions_df.insert(0, "image_id", bundle.image_ids)
    pred_path = derived_dir / "mlp_predictions.csv"
    predictions_df.to_csv(pred_path, index=False)
    print(f"Saved predictions to {pred_path}")

    checkpoint = {
        "config": vars(args),
        "model_state": best_state["model_state"],
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "feature_cols": bundle.feature_cols,
        "target_names": bundle.target_names,
        "best_val_loss": best_val,
        "best_val_r2": val_r2,
    }
    ckpt_path = derived_dir / "mlp_head.pt"
    torch.save(checkpoint, ckpt_path)
    print(f"Saved best model checkpoint to {ckpt_path}")

    last_checkpoint = {
        "config": vars(args),
        "model_state": last_state["model_state"],
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "feature_cols": bundle.feature_cols,
        "target_names": bundle.target_names,
        "last_val_loss": last_val_loss,
    }
    last_ckpt_path = derived_dir / "mlp_head_last.pt"
    torch.save(last_checkpoint, last_ckpt_path)
    print(f"Saved final-epoch checkpoint to {last_ckpt_path}")

    metrics = {
        "best_val_loss": best_val,
        "last_val_loss": last_val_loss,
        "val_rmse": float(np.sqrt(best_val)),
        "val_r2": val_r2,
        "num_train_samples": len(id_train),
        "num_val_samples": len(id_val),
        "targets": bundle.target_names,
    }
    metrics_path = derived_dir / "mlp_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Validation metrics stored in {metrics_path}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an MLP head on DINO embeddings.")
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("data/derived/dino_embeddings.parquet"),
        help="Path to the parquet file produced by model_build.ipynb.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/train.csv"),
        help="Training labels CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/derived"),
        help="Directory for checkpoints and predictions.",
    )
    parser.add_argument(
        "--hidden-dims",
        type=parse_hidden_dims,
        default=(256, 128),
        help="Comma separated hidden layer sizes, e.g. 256,128.",
    )
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate between hidden layers.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation set fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Log training progress every N epochs.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force training on CPU even if CUDA is available.",
    )
    return parser


if __name__ == "__main__":
    run(build_argparser().parse_args())
