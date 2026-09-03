import io
import json
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf

from src.ontology import create_ontology, explain_prediction
from src.xai import detect_zone, make_gradcam_heatmap, overlay_heatmap


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model_tl.keras"
THRESHOLD_PATH = PROJECT_ROOT / "results" / "threshold.json"
ONTOLOGY_PATH = PROJECT_ROOT / "ontology" / "pneumo_ontology.owl"
IMG_SIZE = (224, 224)


@tf.keras.utils.register_keras_serializable()
def focal_loss(gamma=2.0, alpha=0.6):
    def loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        y_true = tf.cast(y_true, tf.float32)
        value = -y_true * alpha * tf.pow(1 - y_pred, gamma) * tf.math.log(y_pred)
        value -= (
            (1 - y_true) * (1 - alpha)
            * tf.pow(y_pred, gamma) * tf.math.log(1 - y_pred)
        )
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


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. "
            "Run training first or place best_model_tl.keras in models/."
        )

    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "focal_loss": focal_loss,
            "loss": focal_loss(gamma=2.0, alpha=0.6),
            "f1_metric": f1_metric,
        },
    )


def load_threshold():
    if THRESHOLD_PATH.exists():
        return float(json.loads(
            THRESHOLD_PATH.read_text(encoding="utf-8")
        ).get("threshold", 0.5))
    return 0.5


model = load_model()
threshold = load_threshold()
create_ontology(ONTOLOGY_PATH)


def analyze_xray(pil_image):
    if pil_image is None:
        return None, "<p>Please upload a chest X-ray.</p>"

    image = pil_image.convert("RGB").resize(IMG_SIZE)
    image_array = np.asarray(image).astype("float32")
    batch = np.expand_dims(image_array, axis=0)
    batch = tf.keras.applications.efficientnet.preprocess_input(batch)

    probability = float(model.predict(batch, verbose=0)[0][0])
    prediction = int(probability > threshold)
    label = "PNEUMONIE" if prediction else "NORMAL"
    confidence = probability if prediction else 1 - probability

    heatmap = make_gradcam_heatmap(batch, model)
    overlay = overlay_heatmap(
        image_array.astype(np.uint8), heatmap
    )
    zone = detect_zone(heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(image_array.astype(np.uint8))
    axes[0].set_title("Radiographie")
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Heatmap")
    axes[2].imshow(overlay)
    axes[2].set_title("Grad-CAM")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"Prédiction : {label} | "
        f"Confiance : {confidence:.1%} | Zone : {zone}"
    )
    plt.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    result = Image.open(buffer)

    explanation = explain_prediction(
        prediction, probability, zone
    )

    html = f"""
    <div style="padding:16px;border:1px solid #30363d;border-radius:10px;">
        <h3>Résultat : {label}</h3>
        <p><b>Confiance :</b> {confidence:.1%}</p>
        <p><b>Seuil utilisé :</b> {threshold:.2f}</p>
        <p><b>Zone d'activation :</b> {zone}</p>
        <h4>Explication XAI + ontologie</h4>
        <p>{explanation}</p>
        <small>
        Démonstration de recherche : ce système ne constitue pas un outil
        de diagnostic médical.
        </small>
    </div>
    """

    return result, html


with gr.Blocks(title="Pneumonia Detection — XAI") as demo:
    gr.Markdown(
        """
        # Pneumonia Detection — XAI Demo

        **EfficientNetB0 · Focal Loss · Grad-CAM · OWL Ontology**

        Upload a chest X-ray to visualize the model prediction and its
        Grad-CAM-based explanation.
        """
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                type="pil",
                label="Radiographie thoracique",
            )
            analyze_button = gr.Button("Analyser", variant="primary")

        with gr.Column():
            image_output = gr.Image(
                label="Original | Heatmap | Grad-CAM"
            )

    html_output = gr.HTML(label="Analyse")

    analyze_button.click(
        fn=analyze_xray,
        inputs=image_input,
        outputs=[image_output, html_output],
    )


if __name__ == "__main__":
    demo.launch()
