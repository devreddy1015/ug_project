import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from .dataset import TrainingDataset, TrainingItem, collect_items
from .metrics import binary_metrics, multilabel_metrics
from .models import MLPClassifier


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Keep training runs reproducible across launches.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _save_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _random_split_indices(total_size: int, val_split: float, seed: int) -> Tuple[List[int], List[int]]:
    if total_size <= 1:
        return [0], [0]

    indices = list(range(total_size))
    rng = random.Random(seed)
    rng.shuffle(indices)

    val_size = max(1, int(round(total_size * val_split)))
    val_size = min(val_size, total_size - 1)
    return indices[val_size:], indices[:val_size]


def _binary_stratified_split_indices(
    items: Sequence[TrainingItem], val_split: float, seed: int
) -> Tuple[List[int], List[int]]:
    positives: List[int] = []
    negatives: List[int] = []

    for index, item in enumerate(items):
        target = float(item.label.item())
        if target >= 0.5:
            positives.append(index)
        else:
            negatives.append(index)

    if not positives or not negatives:
        return _random_split_indices(len(items), val_split, seed)

    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    def _split_bucket(bucket: List[int]) -> Tuple[List[int], List[int]]:
        if len(bucket) <= 1:
            return bucket[:], []
        val_size = int(round(len(bucket) * val_split))
        val_size = min(max(1, val_size), len(bucket) - 1)
        return bucket[val_size:], bucket[:val_size]

    pos_train, pos_val = _split_bucket(positives)
    neg_train, neg_val = _split_bucket(negatives)

    train_indices = [*pos_train, *neg_train]
    val_indices = [*pos_val, *neg_val]

    if not train_indices or not val_indices:
        return _random_split_indices(len(items), val_split, seed)

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def _resolve_split_indices(
    items: Sequence[TrainingItem], multi_label: bool, val_split: float, seed: int
) -> Tuple[List[int], List[int]]:
    if len(items) < 2:
        raise SystemExit("Need at least 2 labeled items for train/validation split.")

    if multi_label:
        return _random_split_indices(len(items), val_split, seed)
    return _binary_stratified_split_indices(items, val_split, seed)


def _stack_labels(items: Sequence[TrainingItem], indices: Sequence[int]) -> torch.Tensor:
    return torch.stack([items[index].label for index in indices], dim=0).float()


def _build_pos_weight(labels: torch.Tensor, multi_label: bool) -> torch.Tensor:
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)

    sample_count = labels.shape[0]
    if sample_count == 0:
        return torch.ones(labels.shape[1], dtype=torch.float32)

    if not multi_label:
        positives = float(labels.sum().item())
        negatives = float(sample_count - positives)
        if positives <= 0.0 or negatives <= 0.0:
            return torch.ones(1, dtype=torch.float32)
        return torch.tensor([negatives / positives], dtype=torch.float32)

    positive_counts = labels.sum(dim=0)
    negative_counts = float(sample_count) - positive_counts
    safe_positive_counts = positive_counts.clamp_min(1.0)
    weights = negative_counts / safe_positive_counts
    return torch.where(positive_counts > 0, weights, torch.ones_like(weights)).float()


def _build_balanced_sampler(
    labels: torch.Tensor, multi_label: bool
) -> Optional[WeightedRandomSampler]:
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)

    if not multi_label:
        flat = labels.view(-1)
        positive_count = int((flat >= 0.5).sum().item())
        negative_count = int((flat < 0.5).sum().item())
        if positive_count == 0 or negative_count == 0:
            return None
        weights = torch.where(
            flat >= 0.5,
            torch.tensor(1.0 / positive_count, dtype=torch.float32),
            torch.tensor(1.0 / negative_count, dtype=torch.float32),
        )
    else:
        class_frequencies = labels.sum(dim=0)
        inverse_class_frequencies = 1.0 / class_frequencies.clamp_min(1.0)
        no_label_mask = labels.sum(dim=1) < 0.5
        no_label_count = max(int(no_label_mask.sum().item()), 1)

        sample_weights: List[float] = []
        for row in labels:
            positive_mask = row >= 0.5
            if bool(positive_mask.any()):
                sample_weights.append(float(inverse_class_frequencies[positive_mask].mean().item()))
            else:
                sample_weights.append(1.0 / no_label_count)
        weights = torch.tensor(sample_weights, dtype=torch.float32)

    normalized = weights / weights.mean().clamp_min(1e-6)
    return WeightedRandomSampler(
        weights=normalized.double(),
        num_samples=int(normalized.shape[0]),
        replacement=True,
    )


def _evaluate_thresholds(
    labels: torch.Tensor,
    probs: torch.Tensor,
    multi_label: bool,
    base_threshold: float,
    disable_threshold_search: bool,
) -> Tuple[float, Dict[str, float], float]:
    base_threshold = max(0.01, min(0.99, float(base_threshold)))
    metric_fn = multilabel_metrics if multi_label else binary_metrics
    target_metric = "micro_f1" if multi_label else "f1"

    if disable_threshold_search:
        metrics = metric_fn(labels, probs, base_threshold)
        score = float(metrics.get(target_metric, 0.0))
        return base_threshold, metrics, score

    candidate_thresholds = {base_threshold}
    for step in range(15, 86, 2):
        candidate_thresholds.add(step / 100.0)

    best_threshold = base_threshold
    best_metrics = metric_fn(labels, probs, best_threshold)
    best_score = float(best_metrics.get(target_metric, 0.0))

    for threshold in sorted(candidate_thresholds):
        metrics = metric_fn(labels, probs, threshold)
        score = float(metrics.get(target_metric, 0.0))
        if score > best_score:
            best_threshold = threshold
            best_metrics = metrics
            best_score = score

    return best_threshold, best_metrics, best_score


def _label_distribution(
    labels: torch.Tensor,
    multi_label: bool,
    categories: Sequence[str],
) -> Dict[str, object]:
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)

    if not multi_label:
        positives = int((labels >= 0.5).sum().item())
        total = int(labels.shape[0])
        negatives = total - positives
        return {
            "total": total,
            "positives": positives,
            "negatives": negatives,
            "positive_rate": round(positives / total, 4) if total else 0.0,
        }

    per_class = labels.sum(dim=0).tolist()
    total = int(labels.shape[0])
    named_counts = {
        categories[index] if index < len(categories) else f"label_{index + 1}": int(count)
        for index, count in enumerate(per_class)
    }
    return {
        "total": total,
        "positive_counts": named_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SHIELD training runner")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="training_runs")
    parser.add_argument("--modality", type=str, default="video")
    parser.add_argument("--labels-file", type=str, default=None)
    parser.add_argument("--multi-label", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--frame-count", type=int, default=4)
    parser.add_argument("--no-frames", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim1", type=int, default=256)
    parser.add_argument("--hidden-dim2", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--disable-balanced-sampling", action="store_true")
    parser.add_argument("--disable-pos-weight", action="store_true")
    parser.add_argument("--disable-threshold-search", action="store_true")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(args.seed)
    labels_path = Path(args.labels_file).expanduser().resolve() if args.labels_file else None

    items, feature_spec, categories = collect_items(
        dataset_dir=dataset_dir,
        modality=args.modality,
        multi_label=args.multi_label,
        labels_path=labels_path,
        max_files=args.limit,
        frame_count=args.frame_count,
        use_frames=not args.no_frames,
    )

    if not items:
        raise SystemExit("No training items found. Check dataset path and labels.")

    dataset = TrainingDataset(items=items, feature_spec=feature_spec)

    train_indices, val_indices = _resolve_split_indices(
        items=items,
        multi_label=args.multi_label,
        val_split=float(args.val_split),
        seed=int(args.seed),
    )
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    train_labels = _stack_labels(items, train_indices)
    sampler = None
    if not args.disable_balanced_sampling:
        sampler = _build_balanced_sampler(train_labels, args.multi_label)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=torch.cuda.is_available(),
    )

    output_dim = len(categories) if args.multi_label else 1
    model = MLPClassifier(
        feature_spec.input_dim,
        output_dim,
        hidden_dims=(int(args.hidden_dim1), int(args.hidden_dim2)),
        dropout=float(args.dropout),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=max(0.0, float(args.weight_decay)),
    )

    pos_weight = None
    if not args.disable_pos_weight:
        candidate = _build_pos_weight(train_labels, args.multi_label)
        if not torch.isinf(candidate).any() and not torch.isnan(candidate).any():
            pos_weight = candidate.to(device)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=max(0.1, min(0.99, float(args.lr_factor))),
        patience=max(0, int(args.lr_patience)),
        min_lr=max(0.0, float(args.min_lr)),
    )

    history: List[Dict[str, object]] = []
    best_f1 = -1.0
    best_epoch = 0
    best_threshold = max(0.01, min(0.99, float(args.threshold)))
    best_path = output_dir / "model.pt"
    no_improve_epochs = 0
    early_stop_patience = max(0, int(args.early_stopping_patience))

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = loss_fn(logits, labels)
            loss.backward()

            clip_value = float(args.grad_clip)
            if clip_value > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)

            optimizer.step()
            epoch_loss += float(loss.item())

        model.eval()
        all_labels: List[torch.Tensor] = []
        all_probs: List[torch.Tensor] = []
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                logits = model(features)
                val_loss += float(loss_fn(logits, labels).item())
                probs = torch.sigmoid(logits)
                all_labels.append(labels.cpu())
                all_probs.append(probs.cpu())

        if not all_labels or not all_probs:
            continue

        labels_tensor = torch.cat(all_labels, dim=0)
        probs_tensor = torch.cat(all_probs, dim=0)

        threshold_used, metrics, score = _evaluate_thresholds(
            labels=labels_tensor,
            probs=probs_tensor,
            multi_label=args.multi_label,
            base_threshold=float(args.threshold),
            disable_threshold_search=bool(args.disable_threshold_search),
        )
        scheduler.step(float(score))

        train_loss = epoch_loss / max(len(train_loader), 1)
        avg_val_loss = val_loss / max(len(val_loader), 1)
        current_lr = float(optimizer.param_groups[0]["lr"])

        history.append(
            {
                "epoch": epoch,
                "loss": round(train_loss, 6),
                "val_loss": round(avg_val_loss, 6),
                "threshold": round(float(threshold_used), 4),
                "lr": round(current_lr, 8),
                **metrics,
            }
        )

        if score > best_f1:
            best_f1 = score
            best_epoch = epoch
            best_threshold = float(threshold_used)
            no_improve_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "input_dim": feature_spec.input_dim,
                    "output_dim": output_dim,
                    "hidden_dims": [int(args.hidden_dim1), int(args.hidden_dim2)],
                    "dropout": float(args.dropout),
                    "recommended_threshold": round(float(threshold_used), 4),
                    "best_metric": round(float(score), 6),
                    "best_metric_name": "micro_f1" if args.multi_label else "f1",
                },
                best_path,
            )
        else:
            no_improve_epochs += 1

        if early_stop_patience > 0 and no_improve_epochs >= early_stop_patience:
            break

    train_distribution = _label_distribution(train_labels, args.multi_label, categories)

    run_summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_dir),
        "modality": args.modality,
        "multi_label": args.multi_label,
        "categories": categories,
        "feature_spec": {
            "input_dim": feature_spec.input_dim,
            "frame_count": feature_spec.frame_count,
            "use_frames": feature_spec.use_frames,
        },
        "split": {
            "train_size": len(train_indices),
            "val_size": len(val_indices),
        },
        "architecture": {
            "hidden_dims": [int(args.hidden_dim1), int(args.hidden_dim2)],
            "dropout": float(args.dropout),
        },
        "training_config": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "grad_clip": float(args.grad_clip),
            "balanced_sampling": not bool(args.disable_balanced_sampling),
            "pos_weight": not bool(args.disable_pos_weight),
            "threshold_search": not bool(args.disable_threshold_search),
        },
        "label_distribution": train_distribution,
        "best_epoch": int(best_epoch),
        "recommended_threshold": round(float(best_threshold), 4),
        "best_f1": round(best_f1, 4),
    }

    _save_json(output_dir / "run_summary.json", run_summary)
    _save_json(output_dir / "metrics_history.json", {"history": history})

    if args.multi_label:
        _save_json(output_dir / "label_map.json", {"categories": categories})


if __name__ == "__main__":
    main()
