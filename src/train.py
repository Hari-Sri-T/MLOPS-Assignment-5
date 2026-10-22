"""Stage 2 - train

Transfer learning with MobileNetV2: the ImageNet-pretrained backbone is
frozen and a small classification head is trained on Cats vs Dogs.
Outputs the trained model, the class list, and per-epoch training history.
"""
import json

import tensorflow as tf

from src.utils import ensure_dir, get_logger, load_params, save_json

log = get_logger("train")


def load_datasets(dataset_dir: str, image_size: int, batch_size: int, seed: int):
    common = dict(
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="binary",
        seed=seed,
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(
        f"{dataset_dir}/train", shuffle=True, **common
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        f"{dataset_dir}/validation", shuffle=False, **common
    )
    return train_ds, val_ds


def build_model(image_size: int, weights, learning_rate: float, augmentation: bool):
    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = inputs
    if augmentation:
        x = tf.keras.layers.RandomFlip("horizontal")(x)
        x = tf.keras.layers.RandomRotation(0.1)(x)
        x = tf.keras.layers.RandomZoom(0.1)(x)
    # MobileNetV2 expects pixels scaled to [-1, 1]
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1)(x)

    base = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3), include_top=False, weights=weights
    )
    base.trainable = False

    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs, name="cats_vs_dogs_mobilenetv2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    p = load_params()
    tp, mp = p["train"], p["model"]
    tf.keras.utils.set_random_seed(tp["seed"])

    train_ds, val_ds = load_datasets(
        p["data"]["dataset_dir"], tp["image_size"], tp["batch_size"], tp["seed"]
    )
    class_names = train_ds.class_names
    log.info("Classes: %s", class_names)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    model = build_model(
        tp["image_size"], p["base_model"]["weights"], tp["learning_rate"], tp["augmentation"]
    )
    model.summary(print_fn=log.info)

    history = model.fit(train_ds, validation_data=val_ds, epochs=tp["epochs"])

    ensure_dir(mp["dir"])
    model.save(mp["file"])
    with open(mp["classes_file"], "w", encoding="utf-8") as f:
        json.dump({"classes": class_names, "image_size": tp["image_size"]}, f, indent=2)

    hist = [
        {"epoch": i + 1, **{k: float(v[i]) for k, v in history.history.items()}}
        for i in range(len(history.history["loss"]))
    ]
    save_json("metrics/train_history.json", hist)
    log.info("Model saved to %s", mp["file"])


if __name__ == "__main__":
    main()
