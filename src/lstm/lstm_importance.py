"""Rank LSTM predictors with permutation importance."""

from pathlib import Path

import src.config

from src.lstm.lstm_eval import evaluate


def evaluate_feature_importance(
    model_path: str | Path,
    data_folder: str | Path,
) -> tuple[dict[str, float | str], ...]:
    """Return predictors ranked by the increase in evaluation MSE."""
    result = evaluate(model_path, data_folder)
    return result.variable_influence


if __name__ == "__main__":
    model_path = Path(src.config.MODEL_SAVE_PATH) / "lstm" / "vibr2" / "lstm_vibr2.pth"
    data_folder = Path(src.config.SEQUENCE_MODEL_CONFIG.evaluation_data_folder)

    for rank, metrics in enumerate(
        evaluate_feature_importance(model_path, data_folder),
        start=1,
    ):
        print(
            f"{rank:>2}. {metrics['name']}: "
            f"MSE increase={float(metrics['mse_increase']):.6f}"
        )