"""
SustainMine training pipeline for the multimodal v2 model.
Binary classification:
- 0 = low
- 1 = polluted
"""

import json
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, mean_absolute_error, precision_score, recall_score

from sustainmine_dataset_v2_binary import SustainMineDatasetV2
from sustainmine_model_v2_fixed_binary import SustainMineModel, compute_multi_task_loss


def _extract_labels_from_subset(subset) -> np.ndarray:
    labels = []
    for index in subset.indices:
        labels.append(int(subset.dataset.samples[index].label))
    return np.asarray(labels, dtype=np.int64)


def _compute_class_weights_from_subset(subset) -> torch.Tensor:
    labels = _extract_labels_from_subset(subset)
    counts = np.bincount(labels, minlength=2).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)


class SustainMineTrainer:
    def __init__(self, model, train_loader, val_loader, device=None, lr=1e-4, weight_decay=0.01, alpha=0.5, beta=0.5, class_weights=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.alpha = alpha
        self.beta = beta
        self.class_weights = class_weights.to(self.device) if class_weights is not None else None
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", patience=5, factor=0.5)
        self.history = {
            "train_loss": [],
            "train_class_loss": [],
            "train_forecast_loss": [],
            "val_loss": [],
            "val_class_loss": [],
            "val_forecast_loss": [],
            "val_accuracy": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
            "val_mae": [],
        }

    def train_epoch(self):
        self.model.train()
        epoch_losses = {"total": [], "classification": [], "forecast": []}

        for batch in tqdm(self.train_loader, desc="Training"):
            images = batch["image"].to(self.device)
            s5 = batch["s5"].to(self.device)
            sensors = batch["sensor"].to(self.device)
            labels = batch["label"].to(self.device)
            forecasts = batch["forecast"].to(self.device)

            outputs = self.model(images, s5, sensors)
            losses = compute_multi_task_loss(
                outputs,
                labels,
                forecasts,
                alpha=self.alpha,
                beta=self.beta,
                class_weights=self.class_weights,
            )

            self.optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            for key in epoch_losses:
                epoch_losses[key].append(losses[key].item())

        return {key: float(np.mean(values)) for key, values in epoch_losses.items()}

    def validate(self):
        self.model.eval()
        epoch_losses = {"total": [], "classification": [], "forecast": []}
        all_preds = []
        all_labels = []
        all_forecast_preds = []
        all_forecasts = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                images = batch["image"].to(self.device)
                s5 = batch["s5"].to(self.device)
                sensors = batch["sensor"].to(self.device)
                labels = batch["label"].to(self.device)
                forecasts = batch["forecast"].to(self.device)

                outputs = self.model(images, s5, sensors)
                losses = compute_multi_task_loss(
                    outputs,
                    labels,
                    forecasts,
                    alpha=self.alpha,
                    beta=self.beta,
                    class_weights=self.class_weights,
                )

                for key in epoch_losses:
                    epoch_losses[key].append(losses[key].item())

                preds = torch.argmax(outputs["classification"], dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_forecast_preds.append(outputs["forecast"].cpu().numpy())
                all_forecasts.append(forecasts.cpu().numpy())

        all_preds = np.asarray(all_preds, dtype=np.int64)
        all_labels = np.asarray(all_labels, dtype=np.int64)

        accuracy = float((all_preds == all_labels).mean()) if len(all_labels) else 0.0
        precision = float(precision_score(all_labels, all_preds, zero_division=0)) if len(all_labels) else 0.0
        recall = float(recall_score(all_labels, all_preds, zero_division=0)) if len(all_labels) else 0.0
        f1 = float(f1_score(all_labels, all_preds, zero_division=0)) if len(all_labels) else 0.0

        y_true = np.concatenate(all_forecasts, axis=0)
        y_pred = np.concatenate(all_forecast_preds, axis=0)
        mae = float(mean_absolute_error(y_true.reshape(-1), y_pred.reshape(-1)))

        results = {key: float(np.mean(values)) for key, values in epoch_losses.items()}
        results["accuracy"] = accuracy
        results["precision"] = precision
        results["recall"] = recall
        results["f1"] = f1
        results["mae"] = mae
        return results

    def train(self, num_epochs=20, save_dir="checkpoints"):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        best_val_loss = float("inf")

        print(f"\n{'=' * 60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Class weights: {self.class_weights.cpu().tolist() if self.class_weights is not None else 'None'}")
        print(f"{'=' * 60}\n")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 60)

            train_losses = self.train_epoch()
            val_results = self.validate()
            self.scheduler.step(val_results["total"])

            self.history["train_loss"].append(train_losses["total"])
            self.history["train_class_loss"].append(train_losses["classification"])
            self.history["train_forecast_loss"].append(train_losses["forecast"])
            self.history["val_loss"].append(val_results["total"])
            self.history["val_class_loss"].append(val_results["classification"])
            self.history["val_forecast_loss"].append(val_results["forecast"])
            self.history["val_accuracy"].append(val_results["accuracy"])
            self.history["val_precision"].append(val_results["precision"])
            self.history["val_recall"].append(val_results["recall"])
            self.history["val_f1"].append(val_results["f1"])
            self.history["val_mae"].append(val_results["mae"])

            print(
                f"Train Loss: {train_losses['total']:.4f} "
                f"(Class: {train_losses['classification']:.4f}, Forecast: {train_losses['forecast']:.4f})"
            )
            print(
                f"Val Loss: {val_results['total']:.4f} "
                f"(Class: {val_results['classification']:.4f}, Forecast: {val_results['forecast']:.4f})"
            )
            print(f"Val Accuracy: {val_results['accuracy']:.4f}")
            print(f"Val Precision: {val_results['precision']:.4f}")
            print(f"Val Recall: {val_results['recall']:.4f}")
            print(f"Val F1: {val_results['f1']:.4f}")
            print(f"Val MAE: {val_results['mae']:.4f}")

            if val_results["total"] < best_val_loss:
                best_val_loss = val_results["total"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": val_results["total"],
                        "val_accuracy": val_results["accuracy"],
                        "val_precision": val_results["precision"],
                        "val_recall": val_results["recall"],
                        "val_f1": val_results["f1"],
                        "val_mae": val_results["mae"],
                    },
                    save_dir / "best_model.pth",
                )
                print(f"✓ Saved best model (loss: {best_val_loss:.4f})")

        torch.save({"model_state_dict": self.model.state_dict(), "history": self.history}, save_dir / "final_model.pth")
        return self.history

    def plot_training_curves(self, save_path="training_curves.png"):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        axes[0, 0].plot(self.history["train_loss"], label="Train")
        axes[0, 0].plot(self.history["val_loss"], label="Validation")
        axes[0, 0].set_title("Total Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        axes[0, 1].plot(self.history["val_accuracy"], label="Accuracy")
        axes[0, 1].plot(self.history["val_f1"], label="F1")
        axes[0, 1].set_title("Validation Classification Metrics")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        axes[1, 0].plot(self.history["val_mae"])
        axes[1, 0].set_title("Forecast MAE")
        axes[1, 0].grid(True)

        axes[1, 1].plot(self.history["train_class_loss"], label="Train Class", linestyle="--")
        axes[1, 1].plot(self.history["val_class_loss"], label="Val Class", linestyle="--")
        axes[1, 1].plot(self.history["train_forecast_loss"], label="Train Forecast", linestyle="-.")
        axes[1, 1].plot(self.history["val_forecast_loss"], label="Val Forecast", linestyle="-.")
        axes[1, 1].set_title("Component Losses")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved training curves to {save_path}")


if __name__ == "__main__":
    config = {
        "sensor_data_path": "sensor_data_cleaned_with_binary_labels.csv",
        "sentinel2_path": "data/sentinel_2",
        "sentinel5_path": "data/sentinel_5",
        "img_size": 224,
        "forecast_horizon": 3,
        "label_column": "binary_label",
        "batch_size": 8,
        "num_epochs": 20,
        "train_split": 0.8,
        "learning_rate": 1e-4,
        "alpha": 0.5,
        "beta": 0.5,
        "weight_decay": 0.01,
        "model": {
            "img_size": 224,
            "patch_size": 16,
            "in_channels": 6,
            "s5_dim": 3,
            "sensor_dim": 8,
            "embed_dim": 384,
            "depth": 6,
            "num_heads": 6,
            "num_classes": 2,
            "num_forecast_steps": 3,
            "num_pollutants": 6,
            "dropout": 0.1,
        },
    }

    print("Preparing real dataset...")
    dataset = SustainMineDatasetV2(
        sensor_data_path=config["sensor_data_path"],
        sentinel2_path=config["sentinel2_path"],
        sentinel5_path=config["sentinel5_path"],
        img_size=config["img_size"],
        forecast_horizon=config["forecast_horizon"],
        label_column=config["label_column"],
    )

    print("Dataset summary:")
    for key, value in dataset.summary().items():
        print(f"  {key}: {value}")

    train_size = int(config["train_split"] * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    class_weights = _compute_class_weights_from_subset(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    print("Creating model...")
    model = SustainMineModel(**config["model"])

    trainer = SustainMineTrainer(
        model,
        train_loader,
        val_loader,
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        alpha=config["alpha"],
        beta=config["beta"],
        class_weights=class_weights,
    )

    history = trainer.train(num_epochs=config["num_epochs"])
    trainer.plot_training_curves("training_curves.png")

    with open("training_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": config,
                "dataset_summary": dataset.summary(),
                "class_weights": class_weights.tolist(),
                "final_metrics": {
                    "best_val_loss": min(history["val_loss"]) if history["val_loss"] else None,
                    "best_val_accuracy": max(history["val_accuracy"]) if history["val_accuracy"] else None,
                    "best_val_f1": max(history["val_f1"]) if history["val_f1"] else None,
                    "best_val_mae": min(history["val_mae"]) if history["val_mae"] else None,
                },
            },
            f,
            indent=2,
        )

    print("\n✓ Training completed successfully!")
    print("✓ Checkpoints saved to checkpoints/")
    print("✓ Training curves saved to training_curves.png")
