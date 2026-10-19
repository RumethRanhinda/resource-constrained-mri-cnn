"""
Section 5: SOTA fine-tuning & evaluation (MobileNetV2, EfficientNet-B0).

Uses the SAME data pipeline, optimizer family (Adam), loss, callbacks and
evaluation metrics as train.py so results are directly comparable with Model A / B.

Two-stage transfer learning:
  Stage 1 - backbone frozen, train only the new classification head (lr 1e-3).
  Stage 2 - unfreeze the whole backbone (BatchNorm kept frozen) and fine-tune (lr 1e-4).

Usage (from the repo root):
  python src/train_sota.py --model mobilenetv2
  python src/train_sota.py --model efficientnetb0
  python src/train_sota.py --model both
  python src/train_sota.py --model mobilenetv2 --epochs-head 1 --epochs-finetune 1   # quick smoke test
"""
import os
import json
import argparse
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             confusion_matrix, classification_report)

# Reuse the teammate's pipeline so data loading, timing and confusion-matrix plots are identical.
from train import get_datasets, EpochTimer, plot_confusion_matrix
from sota_models import (build_mobilenetv2, build_efficientnetb0,
                         set_backbone_trainable, param_summary, get_backbone)

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SRC_DIR, '..')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
FIG_DIR = os.path.join(ROOT_DIR, 'reports', 'figures')
RESULTS_PATH = os.path.join(ROOT_DIR, 'reports', 'sota_results.json')

BUILDERS = {
    'mobilenetv2': build_mobilenetv2,
    'efficientnetb0': build_efficientnetb0,
}
MB = 1024.0 * 1024.0


def compile_model(model, lr):
    """Same optimizer / loss / metric as train.py (Adam, categorical cross-entropy)."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )


def make_callbacks(early_stopping):
    """Same callbacks as train.py: epoch timer, EarlyStopping, ReduceLROnPlateau."""
    timer = EpochTimer()
    callbacks = [timer]
    if early_stopping:
        callbacks.extend([
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=12, restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5, verbose=1),
        ])
    return callbacks, timer


def plot_two_stage_history(h1, h2, model_name, save_dir):
    """Loss/accuracy curves over both stages, with a marker where fine-tuning starts."""
    os.makedirs(save_dir, exist_ok=True)
    hist = {k: h1.history[k] + h2.history[k] for k in h1.history}
    n1 = len(h1.history['loss'])
    epochs = range(1, len(hist['loss']) + 1)

    plt.figure(figsize=(12, 5))
    for idx, (key, title) in enumerate([('loss', 'Loss'), ('accuracy', 'Accuracy')], start=1):
        plt.subplot(1, 2, idx)
        plt.plot(epochs, hist[key], 'b-o', label=f'Train {title}')
        plt.plot(epochs, hist[f'val_{key}'], 'r--s', label=f'Val {title}')
        plt.axvline(n1 + 0.5, color='gray', linestyle='--', label='Fine-tuning starts')
        plt.title(f'{model_name} - {title} Curve')
        plt.xlabel('Epochs')
        plt.ylabel(title)
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    path = os.path.join(save_dir, f"{model_name.lower()}_learning_curves.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved learning curves to: {path}")


def evaluate_on_test(model, test_ds, model_name):
    """Test-set metrics. Positive class = NonDemented (index 1), same as train.py;
    Demented-as-positive precision/recall are also reported."""
    print("\n--- Evaluating on Test Set ---")
    y_true = np.concatenate([np.argmax(labels.numpy(), axis=1) for _, labels in test_ds])
    y_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)
    class_names = ['Demented', 'NonDemented']

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='binary')
    rec = recall_score(y_true, y_pred, average='binary')
    prec_dem = precision_score(y_true, y_pred, pos_label=0)
    rec_dem = recall_score(y_true, y_pred, pos_label=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Test Accuracy:  {acc * 100:.2f}%")
    print(f"Test Precision (NonDemented positive): {prec:.4f}")
    print(f"Test Recall    (NonDemented positive): {rec:.4f}")
    print(f"Test Precision (Demented positive):    {prec_dem:.4f}")
    print(f"Test Recall    (Demented positive):    {rec_dem:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    plot_confusion_matrix(cm, class_names, model_name, FIG_DIR)
    return acc, prec, rec, prec_dem, rec_dem, cm


def run_model(key, args, train_ds, val_ds, test_ds):
    weights = None if args.weights == 'none' else 'imagenet'
    model = BUILDERS[key](input_shape=(64, 64, 1), num_classes=2, weights=weights)
    model_name = model.name
    total, trainable, frozen = param_summary(model)

    print(f"\n=======================================================")
    print(f"  Starting Fine-Tuning: {model_name}")
    print(f"=======================================================")
    print(f"Total params: {total:,} | trainable (stage 1, head only): {trainable:,}")

    # ---- Stage 1: frozen backbone, train the head only ----
    print(f"\n--- Stage 1: head only ({args.epochs_head} epochs, lr={args.lr_head}) ---")
    compile_model(model, args.lr_head)
    cbs1, timer1 = make_callbacks(not args.no_early_stopping)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_head,
                   callbacks=cbs1, verbose=1)

    # ---- Stage 2: unfreeze the backbone (BatchNorm stays frozen) ----
    set_backbone_trainable(model, True, freeze_batchnorm=True)
    if args.unfreeze_top > 0:  # optional partial unfreezing (0 = whole backbone)
        for layer in get_backbone(model).layers[:-args.unfreeze_top]:
            layer.trainable = False
    _, trainable2, _ = param_summary(model)
    scope = 'whole backbone' if args.unfreeze_top == 0 else f'top {args.unfreeze_top} layers'
    print(f"\n--- Stage 2: fine-tune {scope} ({args.epochs_finetune} epochs, lr={args.lr_finetune}) ---")
    print(f"Trainable params in stage 2: {trainable2:,}")
    compile_model(model, args.lr_finetune)  # must recompile after changing trainable flags
    cbs2, timer2 = make_callbacks(not args.no_early_stopping)
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_finetune,
                   callbacks=cbs2, verbose=1)

    all_times = timer1.epoch_times + timer2.epoch_times
    avg_time = float(np.mean(all_times))
    print(f"\nAvg Training Time per Epoch: {avg_time:.2f} s "
          f"(stage 1: {np.mean(timer1.epoch_times):.2f} s, stage 2: {np.mean(timer2.epoch_times):.2f} s)")

    # ---- Model size ----
    os.makedirs(MODELS_DIR, exist_ok=True)
    keras_path = os.path.join(MODELS_DIR, f"{model_name.lower()}.keras")
    weights_path = os.path.join(MODELS_DIR, f"{model_name.lower()}.weights.h5")
    model.save(keras_path)
    # A compiled Keras 3 model also writes its optimizer state (Adam moments, ~3x the weights)
    # into the weights file, so save from a fresh, uncompiled copy to get the true deployable size.
    deploy_model = BUILDERS[key](input_shape=(64, 64, 1), num_classes=2, weights=None)
    deploy_model.set_weights(model.get_weights())
    deploy_model.save_weights(weights_path)
    size_keras_mb = os.path.getsize(keras_path) / MB      # .keras file (same method as train.py; includes optimizer state)
    size_weights_mb = os.path.getsize(weights_path) / MB  # weights only = what would be deployed
    size_fp32_mb = total * 4 / MB                         # theoretical float32 size
    print(f"Model size: weights file {size_weights_mb:.2f} MB | float32 estimate {size_fp32_mb:.2f} MB | "
          f".keras file {size_keras_mb:.2f} MB")

    plot_two_stage_history(h1, h2, model_name, FIG_DIR)
    acc, prec, rec, prec_dem, rec_dem, cm = evaluate_on_test(model, test_ds, model_name)

    return {
        'model_name': model_name,
        'total_params': int(total),
        'trainable_params_stage1': int(trainable),
        'trainable_params_stage2': int(trainable2),
        'size_weights_mb': float(size_weights_mb),
        'size_fp32_estimate_mb': float(size_fp32_mb),
        'size_keras_file_mb': float(size_keras_mb),
        'avg_time_per_epoch_sec': avg_time,
        'epochs_run_stage1': len(h1.history['loss']),
        'epochs_run_stage2': len(h2.history['loss']),
        'test_accuracy': float(acc),
        'precision_nondemented_pos': float(prec),
        'recall_nondemented_pos': float(rec),
        'precision_demented_pos': float(prec_dem),
        'recall_demented_pos': float(rec_dem),
        'confusion_matrix': cm.tolist(),
        'settings': {
            'weights': args.weights, 'batch_size': args.batch_size,
            'epochs_head': args.epochs_head, 'epochs_finetune': args.epochs_finetune,
            'lr_head': args.lr_head, 'lr_finetune': args.lr_finetune,
            'unfreeze_top': args.unfreeze_top, 'early_stopping': not args.no_early_stopping,
        },
    }


def print_comparison_table(results):
    print("\n" + "=" * 80)
    print("                   SOTA FINE-TUNING RESULTS (Section 5)                   ")
    print("=" * 80)
    names = list(results.keys())
    header = f"{'Metric':<34} | " + " | ".join(f"{n:<20}" for n in names)
    print(header)
    print("-" * len(header))
    rows = [
        ('Total Parameter Count', lambda r: f"{r['total_params']:,}"),
        ('Trainable Params (fine-tuning)', lambda r: f"{r['trainable_params_stage2']:,}"),
        ('Model Size - weights (MB)', lambda r: f"{r['size_weights_mb']:.2f}"),
        ('Model Size - .keras file (MB)', lambda r: f"{r['size_keras_file_mb']:.2f}"),
        ('Avg Training Time / Epoch', lambda r: f"{r['avg_time_per_epoch_sec']:.2f}s"),
        ('Test Accuracy', lambda r: f"{r['test_accuracy'] * 100:.2f}%"),
        ('Test Precision (NonDem. pos.)', lambda r: f"{r['precision_nondemented_pos']:.4f}"),
        ('Test Recall (NonDem. pos.)', lambda r: f"{r['recall_nondemented_pos']:.4f}"),
        ('Test Precision (Dem. pos.)', lambda r: f"{r['precision_demented_pos']:.4f}"),
        ('Test Recall (Dem. pos.)', lambda r: f"{r['recall_demented_pos']:.4f}"),
    ]
    for label, fmt in rows:
        print(f"{label:<34} | " + " | ".join(f"{fmt(results[n]):<20}" for n in names))
    print("=" * 80 + "\n")


def save_results(new_results):
    """Merge into reports/sota_results.json so separate runs accumulate (used for Section 6)."""
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    existing = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, 'r') as f:
            existing = json.load(f)
    existing.update(new_results)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f"Saved results to: {RESULTS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate MobileNetV2 / EfficientNet-B0.")
    parser.add_argument('--model', type=str, default='both', choices=['mobilenetv2', 'efficientnetb0', 'both'],
                        help="Which model to fine-tune")
    parser.add_argument('--epochs-head', type=int, default=5, help="Stage 1 epochs (frozen backbone, default: 5)")
    parser.add_argument('--epochs-finetune', type=int, default=15, help="Stage 2 epochs (default: 15; total 20)")
    parser.add_argument('--lr-head', type=float, default=1e-3, help="Stage 1 learning rate (default: 1e-3)")
    parser.add_argument('--lr-finetune', type=float, default=1e-4, help="Stage 2 learning rate (default: 1e-4)")
    parser.add_argument('--unfreeze-top', type=int, default=0,
                        help="Stage 2: unfreeze only the last N backbone layers (default 0 = whole backbone)")
    parser.add_argument('--weights', type=str, default='imagenet', choices=['imagenet', 'none'],
                        help="'none' skips the pretrained-weight download (smoke tests only)")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument('--no-early-stopping', action='store_true',
                        help="Disable early stopping and learning rate scheduling")
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(42)
    data_path = os.path.join(ROOT_DIR, 'data', 'processed')
    train_ds, val_ds, test_ds = get_datasets(data_path, batch_size=args.batch_size)

    keys = list(BUILDERS.keys()) if args.model == 'both' else [args.model]
    results = {}
    for key in keys:
        res = run_model(key, args, train_ds, val_ds, test_ds)
        results[res['model_name']] = res
        save_results({res['model_name']: res})

    print_comparison_table(results)