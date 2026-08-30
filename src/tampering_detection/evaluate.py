"""
evaluate.py

Loads a trained tampering-detection checkpoint and reports quantitative
performance metrics (accuracy, precision, recall, F1-score, and a confusion
matrix) on a held-out validation/test split. This complements train.py,
which only tracks accuracy during training -- this script is meant to be
run separately, after training, for a fuller evaluation.

Usage:
    python -m tampering_detection.evaluate
    python -m tampering_detection.evaluate --data_root data/casia_v2 --checkpoint checkpoints/best_model.pth
"""

import os
import argparse
import json

import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

from .dataset import TamperingDataset
from .model import build_model


def evaluate(data_root="data/casia_v2", checkpoint_path="checkpoints/best_model.pth",
             batch_size=32, val_split=0.2, seed=42, results_dir="results/metrics",
             figures_dir="results/figures"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'. Run training first: "
            f"python -m tampering_detection.train"
        )

    # --- Load the same validation split used during training, for a fair comparison ---
    dataset = TamperingDataset(data_root)
    val_len = int(val_split * len(dataset))
    train_len = len(dataset) - val_len
    generator = torch.Generator().manual_seed(seed)
    _, val_ds = random_split(dataset, [train_len, val_len], generator=generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # --- Load model ---
    model = build_model()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    # --- Compute metrics ---
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)
    report_text = classification_report(
        all_labels, all_preds, target_names=["Authentic", "Tampered"], zero_division=0
    )

    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nConfusion Matrix (rows=actual, cols=predicted):")
    print("                Authentic  Tampered")
    print(f"Actual Authentic   {cm[0][0]:>6}    {cm[0][1]:>6}")
    print(f"Actual Tampered    {cm[1][0]:>6}    {cm[1][1]:>6}")
    print("\nFull classification report:")
    print(report_text)

    # --- Save metrics to results/metrics/ so they aren't lost ---
    os.makedirs(results_dir, exist_ok=True)
    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm.tolist(),
        "checkpoint": checkpoint_path,
        "val_samples": len(val_ds),
    }
    out_path = os.path.join(results_dir, "evaluation_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {out_path}")

    # --- Save a confusion matrix plot to results/figures/ ---
    os.makedirs(figures_dir, exist_ok=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Authentic", "Tampered"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix \u2014 Tampering Detection")
    fig_path = os.path.join(figures_dir, "confusion_matrix.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix plot saved to: {fig_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the tampering detection model.")
    parser.add_argument("--data_root", default="data/casia_v2", help="Path to dataset root (with Au/ and Tp/ subfolders)")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth", help="Path to trained model checkpoint")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--results_dir", default="results/metrics", help="Where to save evaluation_metrics.json")
    parser.add_argument("--figures_dir", default="results/figures", help="Where to save confusion_matrix.png")
    args = parser.parse_args()

    evaluate(
        data_root=args.data_root,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
    )
