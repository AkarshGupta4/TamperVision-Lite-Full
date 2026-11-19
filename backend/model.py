import os
import pickle
from typing import Any

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "TamperVision_SRM_MobileNet_FineTuned_best.keras")
PKL_PATH = os.path.join(MODEL_DIR, "TamperVision_Final_Metadata.pkl")

IMG_SIZE = 224
# User-requested decision rule:
# - below 20% tampering probability => authentic
# - above 20% tampering probability => tampered
DEFAULT_THRESHOLD = 0.20
MASK_THRESHOLD = 0.60
MIN_CONTOUR_AREA = 250

class SRMFilter(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.srm_filters = tf.constant([
            [[0, 0, 0], [0, -1, 1], [0, 0, 0]],
            [[0, 0, 0], [0, -1, 0], [0, 1, 0]],
            [[0, 0, 0], [-1, 0, 1], [0, 0, 0]],
        ], dtype=tf.float32)

    def build(self, input_shape):
        channels = input_shape[-1]
        self.kernel = tf.stack([self.srm_filters for _ in range(channels)], axis=2)
        super().build(input_shape)

    def call(self, inputs):
        return tf.nn.depthwise_conv2d(
            inputs, self.kernel, strides=[1, 1, 1, 1], padding="SAME"
        )

with open(PKL_PATH, "rb") as f:
    metadata = pickle.load(f)

stored_threshold = float(metadata.get("threshold", DEFAULT_THRESHOLD))
# The saved metadata can be too permissive for real-world authentic images.
# Keep the model itself unchanged, but enforce a more conservative decision boundary.
threshold = max(stored_threshold, DEFAULT_THRESHOLD)

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"SRMFilter": SRMFilter},
    compile=False,
)

mobilenet = None
for layer in model.layers:
    if isinstance(layer, tf.keras.Model) and "mobilenet" in layer.name.lower():
        mobilenet = layer
        break
if mobilenet is None:
    raise RuntimeError("MobileNetV2 branch was not found in the saved model.")

target_layer = None
for layer in reversed(mobilenet.layers):
    try:
        if len(layer.output.shape) == 4:
            target_layer = layer
            break
    except Exception:
        pass
if target_layer is None:
    raise RuntimeError("A 4-D Grad-CAM target layer was not found.")

cam_model = tf.keras.Model(
    inputs=mobilenet.input,
    outputs=[target_layer.output, mobilenet.output],
)


def _read_image(image_bytes: bytes):
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("The uploaded file is not a valid image.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def _preprocess(rgb: np.ndarray):
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    return resized[None, ...]


def _gradcam(tensor, original_shape):
    with tf.GradientTape() as tape:
        conv_outputs, mobile_features = cam_model(tensor, training=False)
        score = tf.reduce_mean(mobile_features)
    grads = tape.gradient(score, conv_outputs)
    if grads is None:
        raise RuntimeError("Gradients could not be calculated for Grad-CAM.")
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(conv_outputs * weights[:, None, None, :], axis=-1)
    cam = tf.maximum(cam, 0.0)[0].numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cv2.resize(cam, (original_shape[1], original_shape[0]))


def analyze_image(image_bytes: bytes, decision_threshold: float | None = None) -> dict[str, Any]:
    rgb = _read_image(image_bytes)
    tensor = _preprocess(rgb)
    probability = float(model.predict(tensor, verbose=0)[0][0])
    used_threshold = threshold if decision_threshold is None else float(decision_threshold)

    # Keep the trained model unchanged, but interpret the output with the requested business rule:
    # below 20% => authentic, 20% and above => tampered.
    if probability < 0.20:
        prediction = "AUTHENTIC"
    else:
        prediction = "TAMPERED"

    heatmap = _gradcam(tensor, rgb.shape)
    mask = (heatmap >= MASK_THRESHOLD).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    localized = rgb.copy()
    regions = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = float(cv2.contourArea(contour))
        if area < MIN_CONTOUR_AREA:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        regions.append({"x": int(x), "y": int(y), "width": int(w), "height": int(h), "area": area})
        cv2.rectangle(localized, (x, y), (x + w, y + h), (255, 0, 0), 3)
        cv2.putText(localized, f"Region {len(regions)}", (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    color = cv2.applyColorMap(np.uint8(heatmap * 255), cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    gradcam = cv2.addWeighted(rgb, 0.55, color, 0.45, 0)

    overlay = rgb.copy()
    overlay[mask == 255] = (255, 0, 0)
    pseudo_overlay = cv2.addWeighted(rgb, 0.65, overlay, 0.35, 0)

    return {
        "prediction": prediction,
        "probability": probability,
        "threshold": used_threshold,
        "image": rgb,
        "heatmap": heatmap,
        "gradcam": gradcam,
        "mask": mask,
        "localized": localized,
        "pseudo_overlay": pseudo_overlay,
        "regions": regions,
    }
