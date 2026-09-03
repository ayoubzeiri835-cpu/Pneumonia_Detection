import cv2
import numpy as np
import tensorflow as tf


def make_gradcam_heatmap(img_array, model):
    base_model = model.get_layer("efficientnetb0")
    last_conv_layer = None

    for layer in reversed(base_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break

    if last_conv_layer is None:
        raise ValueError("No Conv2D layer found inside EfficientNetB0.")

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        class_channel = predictions[:, 0]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8
    return heatmap.numpy()


def overlay_heatmap(img, heatmap, alpha=0.6):
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_resized = cv2.resize(
        heatmap_uint8, (img.shape[1], img.shape[0])
    )
    colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    return cv2.addWeighted(
        img.astype(np.uint8), 1.0, colored, alpha, 0
    )


def detect_zone(heatmap):
    _, width = heatmap.shape
    left_sum = np.sum(heatmap[:, :width // 2])
    right_sum = np.sum(heatmap[:, width // 2:])

    if left_sum > right_sum * 1.1:
        return "gauche"
    if right_sum > left_sum * 1.1:
        return "droite"
    return "bilatérale"
