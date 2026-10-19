
"""Compare Model A, Model B, MobileNetV2 and EfficientNet-B0.

Run from the repository root:
    python src/compare_models.py

Outputs:
    reports/final_comparison.json
    reports/compare_log.txt  (if output is redirected in PowerShell)
    reports/figures/final_tradeoff.png
"""

import os
import json
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from train import get_datasets


# ---------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT, "models")
REPORT_DIR = os.path.join(ROOT, "reports")
FIGURE_DIR = os.path.join(REPORT_DIR, "figures")

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

MODEL_FILES = {
    "Model A": "model_a.keras",
    "Model B": "model_b.keras",
    "MobileNetV2": "model_mobilenetv2.keras",
    "EfficientNet-B0": "model_efficientnetb0.keras",
}

# Hand-calculated reference values for sanity checking only.
EXPECTED_MACS = {
    "Model A": 14_876_800,
    "Model B": 8_003_712,
}


# ---------------------------------------------------------
# Load the existing test dataset
# ---------------------------------------------------------

print("\nLoading test dataset...")

_, _, test_ds = get_datasets(os.path.join(ROOT, "data", "processed"))

# Preserve the dataset's class order.
y_true = np.concatenate([
    np.argmax(labels.numpy(), axis=1)
    for _, labels in test_ds
])

# One test image for single-image inference timing.
x_one = next(iter(test_ds.unbatch().batch(1)))[0]

print(f"Test images: {len(y_true)}")
print("Class order: 0 = Demented, 1 = NonDemented")


# ---------------------------------------------------------
# Estimate MACs
# ---------------------------------------------------------

def estimate_macs(model):
    """Estimate MACs for supported convolution and Dense layers.

    Batch normalization, activation, pooling and other operations
    are not included. Unsupported or uninspectable layers are reported.
    """

    total_macs = 0
    issues = []

    def visit(layer):
        nonlocal total_macs

        # Count layers inside nested Keras models.
        if isinstance(layer, tf.keras.Model):
            for child in layer.layers:
                visit(child)
            return

        name = type(layer).__name__

        try:
            if name == "Conv2D":
                _, h, w, _ = layer.output.shape
                kh, kw = layer.kernel_size
                cin = int(layer.input.shape[-1])

                total_macs += int(
                    h * w * kh * kw
                    * (cin // layer.groups)
                    * layer.filters
                )

            elif name == "DepthwiseConv2D":
                _, h, w, _ = layer.output.shape
                kh, kw = layer.kernel_size
                cin = int(layer.input.shape[-1])

                total_macs += int(
                    h * w * kh * kw
                    * cin
                    * layer.depth_multiplier
                )

            elif name == "SeparableConv2D":
                _, h, w, _ = layer.output.shape
                kh, kw = layer.kernel_size
                cin = int(layer.input.shape[-1])

                depthwise_macs = (
                    h * w * kh * kw
                    * cin
                    * layer.depth_multiplier
                )
                pointwise_macs = (
                    h * w * cin
                    * layer.depth_multiplier
                    * layer.filters
                )

                total_macs += int(depthwise_macs + pointwise_macs)

            elif name == "Dense":
                input_units = int(np.prod(layer.input.shape[1:]))
                total_macs += input_units * int(layer.units)

        except Exception as exc:
            issues.append(
                f"{name}: could not estimate MACs ({type(exc).__name__})"
            )

    for layer in model.layers:
        visit(layer)

    return total_macs, issues


# ---------------------------------------------------------
# Measure single-image CPU inference latency
# ---------------------------------------------------------

def median_latency_ms(model, runs=300, warmup=30):
    """Measure median single-image inference latency on this machine."""

    inference_fn = tf.function(
        lambda x: model(x, training=False)
    )

    # Warm-up calls are excluded from the measurement.
    for _ in range(warmup):
        inference_fn(x_one).numpy()

    timings = []

    for _ in range(runs):
        start = time.perf_counter()
        inference_fn(x_one).numpy()
        timings.append(time.perf_counter() - start)

    return float(np.median(timings) * 1000)


# ---------------------------------------------------------
# Evaluate each model
# ---------------------------------------------------------

results = []

for name, filename in MODEL_FILES.items():

    model_path = os.path.join(MODEL_DIR, filename)

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Missing trained model: {model_path}"
        )

    print(f"\n{'=' * 60}")
    print(f"Evaluating {name}")
    print(f"Model file: {filename}")
    print(f"{'=' * 60}")

    model = tf.keras.models.load_model(model_path)

    # Test predictions.
    probabilities = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)

    if len(y_pred) != len(y_true):
        raise ValueError(
            f"{name}: prediction count does not match test labels."
        )

    # Confusion matrix order: Demented, NonDemented.
    cm = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    )

    # Model resource estimates.
    params = int(model.count_params())
    weights_kb = params * 4 / 1024  # Assumes float32 parameters.
    keras_size_mb = os.path.getsize(model_path) / (1024 ** 2)

    macs, mac_issues = estimate_macs(model)

    if name in EXPECTED_MACS and macs != EXPECTED_MACS[name]:
        print(
            f"WARNING: {name} estimated MACs = {macs:,}; "
            f"hand-calculated reference = {EXPECTED_MACS[name]:,}."
        )
        print(
            "Investigate the layer shapes and MAC calculation "
            "before reporting this value."
        )

    if mac_issues:
        print("MAC estimation issues:")
        for issue in mac_issues:
            print(f"  - {issue}")

    # Latency is measured on this computer, normally using CPU.
    print("Measuring latency (300 runs after 30 warm-up runs)...")
    latency = median_latency_ms(model)

    row = {
        "model": name,
        "parameters_total": params,
        "float32_weight_estimate_kb": float(weights_kb),
        "keras_file_size_mb": float(keras_size_mb),
        "estimated_macs": int(macs),
        "estimated_macs_millions": float(macs / 1e6),
        "mac_estimation_issues": mac_issues,
        "inference_latency_median_ms": latency,
        "latency_device": "TensorFlow device used by this run",
        "test_accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "demented_precision": float(
            precision_score(
                y_true, y_pred, pos_label=0, zero_division=0
            )
        ),
        "demented_recall": float(
            recall_score(
                y_true, y_pred, pos_label=0, zero_division=0
            )
        ),
        "nondemented_precision": float(
            precision_score(
                y_true, y_pred, pos_label=1, zero_division=0
            )
        ),
        "nondemented_recall": float(
            recall_score(
                y_true, y_pred, pos_label=1, zero_division=0
            )
        ),
        "confusion_matrix_labels_demented_then_nondemented":
            cm.tolist(),
    }

    results.append(row)

    print(f"Parameters: {params:,}")
    print(f"Estimated float32 weights: {weights_kb:.2f} KB")
    print(f"Actual .keras file size: {keras_size_mb:.2f} MB")
    print(f"Estimated MACs: {macs:,} ({macs / 1e6:.3f} million)")
    print(f"Median inference latency: {latency:.3f} ms")
    print(f"Test accuracy: {row['test_accuracy'] * 100:.2f}%")
    print("Confusion matrix:")
    print(cm)


# ---------------------------------------------------------
# Save results as JSON
# ---------------------------------------------------------

json_path = os.path.join(
    REPORT_DIR, "final_comparison.json"
)

with open(json_path, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=2)

print(f"\nMetrics saved to: {json_path}")


# ---------------------------------------------------------
# Plot accuracy vs float32 weight size
# Bubble area represents estimated MACs.
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

for row in results:
    x = row["float32_weight_estimate_kb"] / 1024
    y = row["test_accuracy"] * 100

    # Add a base size so small models remain visible.
    bubble_size = row["estimated_macs_millions"] * 12 + 40

    plt.scatter(
        x,
        y,
        s=bubble_size,
        alpha=0.6,
    )

    plt.annotate(
        row["model"],
        (x, y),
        xytext=(6, 6),
        textcoords="offset points",
    )

plt.xscale("log")
plt.xlabel("Float32 weights size (MB, log scale)")
plt.ylabel("Test accuracy (%)")
plt.title("Accuracy vs memory (bubble area = estimated MACs)")
plt.grid(alpha=0.3, which="both")
plt.tight_layout()

plot_path = os.path.join(
    FIGURE_DIR, "final_tradeoff.png"
)

plt.savefig(plot_path, dpi=300)
plt.close()

print(f"Graph saved to: {plot_path}")
print("\nComparison complete.")
