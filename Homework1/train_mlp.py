import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


DATASET_PATH = Path("dataset/Concrete_Data.xls")
OUTPUT_DIR = Path("outputs")
SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class StandardScaler:
    def __init__(self) -> None:
        self.mean = None
        self.std = None

    def fit(self, array: np.ndarray) -> None:
        self.mean = array.mean(axis=0, keepdims=True)
        self.std = array.std(axis=0, keepdims=True)
        self.std[self.std == 0] = 1.0

    def transform(self, array: np.ndarray) -> np.ndarray:
        return (array - self.mean) / self.std

    def fit_transform(self, array: np.ndarray) -> np.ndarray:
        self.fit(array)
        return self.transform(array)

    def inverse_transform(self, array: np.ndarray) -> np.ndarray:
        return array * self.std + self.mean


class MLPRegressor(torch.nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an MLP on the concrete strength dataset.")
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=120)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    return pd.read_excel(dataset_path)


def save_correlation_artifacts(train_df: pd.DataFrame, target_col: str, output_dir: Path) -> pd.Series:
    correlations = (
        train_df.corr(numeric_only=True)[target_col]
        .drop(target_col)
        .sort_values(key=lambda series: series.abs(), ascending=False)
    )
    correlation_df = correlations.rename("correlation_with_target").reset_index()
    correlation_df.columns = ["feature", "correlation_with_target"]
    correlation_df.to_csv(output_dir / "correlation_rank.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 5))
    plt.barh(correlation_df["feature"], correlation_df["correlation_with_target"], color="#2f6db2")
    plt.xlabel("Pearson correlation with target")
    plt.title("Feature Correlation Analysis")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_bar.png", dpi=200)
    plt.close()
    return correlations


def plot_loss_curve(history: dict[str, list[float]], output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss", linewidth=2)
    plt.plot(history["val_loss"], label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=200)
    plt.close()


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path) -> None:
    min_value = min(float(y_true.min()), float(y_pred.min()))
    max_value = max(float(y_true.max()), float(y_pred.max()))

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.75, color="#d67229", edgecolors="none")
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--", color="black")
    plt.xlabel("True Strength (MPa)")
    plt.ylabel("Predicted Strength (MPa)")
    plt.title("Prediction vs Ground Truth")
    plt.tight_layout()
    plt.savefig(output_dir / "prediction_scatter.png", dpi=200)
    plt.close()


def train_model(
    model: torch.nn.Module,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    val_features: torch.Tensor,
    val_targets: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.nn.Module, dict[str, list[float]], int]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.MSELoss()
    history = {"train_loss": [], "val_loss": []}

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    remaining_patience = args.patience
    train_indices = np.arange(len(train_features))

    for epoch in range(1, args.epochs + 1):
        model.train()
        shuffled_indices = np.random.permutation(train_indices)
        epoch_losses = []

        for start in range(0, len(shuffled_indices), args.batch_size):
            batch_indices = shuffled_indices[start : start + args.batch_size]
            batch_x = train_features[batch_indices]
            batch_y = train_targets[batch_indices]

            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = loss_fn(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_features), val_targets).item()

        train_loss = float(np.mean(epoch_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
            remaining_patience = args.patience
        else:
            remaining_patience -= 1
            if remaining_patience == 0:
                break

    model.load_state_dict(best_state)
    return model, history, best_epoch


def main() -> None:
    args = build_argparser().parse_args()
    set_seed(args.seed)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(DATASET_PATH)
    feature_cols = list(df.columns[:-1])
    target_col = df.columns[-1]

    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()

    correlations = save_correlation_artifacts(train_df, target_col, OUTPUT_DIR)

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df[target_col].to_numpy(dtype=np.float32).reshape(-1, 1)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_test = test_df[target_col].to_numpy(dtype=np.float32).reshape(-1, 1)

    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)

    # The target is strictly positive; log1p stabilizes training on this fixed split.
    y_train_log = np.log1p(y_train)
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train_log)

    val_count = int(len(X_train_scaled) * args.val_ratio)
    train_features = X_train_scaled[:-val_count]
    val_features = X_train_scaled[-val_count:]
    train_targets = y_train_scaled[:-val_count]
    val_targets = y_train_scaled[-val_count:]

    train_features_tensor = torch.tensor(train_features, dtype=torch.float32)
    val_features_tensor = torch.tensor(val_features, dtype=torch.float32)
    train_targets_tensor = torch.tensor(train_targets, dtype=torch.float32)
    val_targets_tensor = torch.tensor(val_targets, dtype=torch.float32)
    test_features_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    model = MLPRegressor(input_dim=len(feature_cols))
    model, history, best_epoch = train_model(
        model=model,
        train_features=train_features_tensor,
        train_targets=train_targets_tensor,
        val_features=val_features_tensor,
        val_targets=val_targets_tensor,
        args=args,
    )

    model.eval()
    with torch.no_grad():
        test_predictions_scaled = model(test_features_tensor).cpu().numpy()

    test_predictions_log = target_scaler.inverse_transform(test_predictions_scaled)
    test_predictions = np.expm1(test_predictions_log)

    mse = mean_squared_error(y_test.ravel(), test_predictions.ravel())
    r2 = r2_score(y_test.ravel(), test_predictions.ravel())

    baseline = LinearRegression()
    baseline.fit(X_train_scaled, y_train.ravel())
    baseline_predictions = baseline.predict(X_test_scaled)
    baseline_mse = mean_squared_error(y_test.ravel(), baseline_predictions)
    baseline_r2 = r2_score(y_test.ravel(), baseline_predictions)

    predictions_df = pd.DataFrame(
        {
            "true_strength_mpa": y_test.ravel(),
            "predicted_strength_mpa": test_predictions.ravel(),
        }
    )
    predictions_df.to_csv(OUTPUT_DIR / "predictions.csv", index=False, encoding="utf-8-sig")

    plot_loss_curve(history, OUTPUT_DIR)
    plot_predictions(y_test.ravel(), test_predictions.ravel(), OUTPUT_DIR)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_columns": feature_cols,
            "feature_mean": feature_scaler.mean,
            "feature_std": feature_scaler.std,
            "target_mean": target_scaler.mean,
            "target_std": target_scaler.std,
        },
        OUTPUT_DIR / "best_model.pt",
    )

    metrics = {
        "dataset_rows": int(len(df)),
        "feature_count": len(feature_cols),
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "validation_size": int(val_count),
        "best_epoch": int(best_epoch),
        "test_mse": float(mse),
        "test_r2": float(r2),
        "linear_regression_test_mse": float(baseline_mse),
        "linear_regression_test_r2": float(baseline_r2),
        "top_correlated_features": [
            {"feature": feature, "correlation": float(value)}
            for feature, value in correlations.items()
        ],
        "notes": [
            "Train/test split follows the homework requirement: first 80% for training and last 20% for testing.",
            "The MLP uses all 8 features; correlation analysis is exported separately for interpretation.",
            "The target uses log1p scaling during training for better numerical stability on this fixed split.",
        ],
    }

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
