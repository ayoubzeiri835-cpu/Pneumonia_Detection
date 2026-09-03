from pathlib import Path
import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score,
)

warnings.simplefilter("ignore")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
EPOCHS = 30

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = Path(os.getenv("PNEUMONIA_DATASET_DIR", PROJECT_ROOT / "data" / "chest_xray"))
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


@tf.keras.utils.register_keras_serializable()
def focal_loss(gamma=2.0, alpha=0.6):
    def loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        y_true = tf.cast(y_true, tf.float32)
        value = -y_true * alpha * tf.pow(1 - y_pred, gamma) * tf.math.log(y_pred)
        value -= (1 - y_true) * (1 - alpha) * tf.pow(y_pred, gamma) * tf.math.log(1 - y_pred)
        return tf.reduce_mean(value)
    return loss


@tf.keras.utils.register_keras_serializable()
def f1_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.round(y_pred)
    tp = tf.reduce_sum(y_true * y_pred)
    precision = tp / (tf.reduce_sum(y_pred) + 1e-7)
    recall = tp / (tf.reduce_sum(y_true) + 1e-7)
    return 2 * (precision * recall) / (precision + recall + 1e-7)


def build_datasets():
    if not TRAIN_DIR.exists() or not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Dataset not found. Expected: {TRAIN_DIR} and {TEST_DIR}. "
            "Set PNEUMONIA_DATASET_DIR to your chest_xray directory."
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, validation_split=0.2, subset="training", seed=42,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, color_mode="rgb"
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, validation_split=0.2, subset="validation", seed=42,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, color_mode="rgb"
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        shuffle=False, color_mode="rgb"
    )

    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])
    preprocess_input = tf.keras.applications.efficientnet.preprocess_input

    train_ds = train_ds.map(lambda x, y: (preprocess_input(augmentation(x)), y)).prefetch(AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y)).prefetch(AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y)).prefetch(AUTOTUNE)
    return train_ds, val_ds, test_ds


def build_model():
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False, input_shape=(224, 224, 3), weights="imagenet"
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
        loss=focal_loss(gamma=2.0, alpha=0.6),
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
            f1_metric,
        ],
    )
    return model


def select_threshold(y_true, probabilities):
    best_threshold, best_score = 0.0, 0.0
    for threshold in np.arange(0.30, 0.90, 0.01):
        labels = (probabilities > threshold).astype(int)
        r_normal = recall_score(y_true, labels, pos_label=0, zero_division=0)
        r_pneumonia = recall_score(y_true, labels, pos_label=1, zero_division=0)
        score = 0.4 * r_normal + 0.6 * r_pneumonia
        if score > best_score:
            best_score, best_threshold = score, float(threshold)
    return best_threshold, best_score


def main():
    train_ds, val_ds, test_ds = build_datasets()
    model = build_model()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=7, restore_best_weights=True, mode="min"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=3, verbose=1, mode="min"
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_DIR / "best_model_tl.keras", save_best_only=True
        ),
    ]

    history = model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks
    )

    best_model = tf.keras.models.load_model(
        MODEL_DIR / "best_model_tl.keras",
        custom_objects={
            "focal_loss": focal_loss,
            "loss": focal_loss(gamma=2.0, alpha=0.6),
            "f1_metric": f1_metric,
        },
    )

    y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0).astype(int)
    y_prob = best_model.predict(test_ds, verbose=1).reshape(-1)
    threshold, weighted_recall = select_threshold(y_true, y_prob)
    y_pred = (y_prob > threshold).astype(int)

    metrics = {
        "threshold": threshold,
        "weighted_recall_score": float(weighted_recall),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)),
    }

    print("\nBest threshold:", threshold)
    print("Weighted recall score:", weighted_recall)
    print("\nClassification report:")
    print(classification_report(
        y_true, y_pred, target_names=["NORMAL", "PNEUMONIA"], zero_division=0
    ))

    (RESULTS_DIR / "threshold.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["NORMAL", "PNEUMONIA"],
        yticklabels=["NORMAL", "PNEUMONIA"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Focal Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "loss_curve.png", dpi=150)
    plt.close()

    print("\nSaved model:", MODEL_DIR / "best_model_tl.keras")
    print("Saved results:", RESULTS_DIR)


if __name__ == "__main__":
    main()
