"""Stage 3 - evaluate

Scores the trained model on the held-out validation split and writes
metrics/scores.json (tracked by `dvc metrics show`) plus a confusion matrix
in metrics/confusion_matrix.json.
"""
import json

import numpy as np
import tensorflow as tf

from src.utils import get_logger, load_params, save_json

log = get_logger("evaluate")


def main() -> None:
    p = load_params()
    tp, mp = p["train"], p["model"]

    model = tf.keras.models.load_model(mp["file"])
    with open(mp["classes_file"], encoding="utf-8") as f:
        classes = json.load(f)["classes"]

    val_ds = tf.keras.utils.image_dataset_from_directory(
        f"{p['data']['dataset_dir']}/validation",
        image_size=(tp["image_size"], tp["image_size"]),
        batch_size=tp["batch_size"],
        label_mode="binary",
        shuffle=False,
    )

    loss, acc = model.evaluate(val_ds, verbose=0)

    y_true = np.concatenate([y.numpy().ravel() for _, y in val_ds]).astype(int)
    y_prob = model.predict(val_ds, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    tp_ = int(((y_pred == 1) & (y_true == 1)).sum())
    tn_ = int(((y_pred == 0) & (y_true == 0)).sum())
    fp_ = int(((y_pred == 1) & (y_true == 0)).sum())
    fn_ = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp_ / (tp_ + fp_) if tp_ + fp_ else 0.0
    recall = tp_ / (tp_ + fn_) if tp_ + fn_ else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    scores = {
        "val_loss": round(float(loss), 4),
        "val_accuracy": round(float(acc), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "n_samples": int(len(y_true)),
    }
    scores["confusion_matrix"] = {"tn": tn_, "fp": fp_, "fn": fn_, "tp": tp_}
    save_json("metrics/scores.json", scores)

    # One record per image -> `dvc plots show` renders it as a confusion matrix
    records = [
        {"actual": classes[t], "predicted": classes[p_]} for t, p_ in zip(y_true, y_pred)
    ]
    with open("metrics/confusion_matrix.json", "w", encoding="utf-8") as f:
        json.dump(records, f)
    log.info("Scores: %s", scores)


if __name__ == "__main__":
    main()
