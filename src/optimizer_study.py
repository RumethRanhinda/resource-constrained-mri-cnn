"""Section 3: optimizer comparison on Model B (Adam vs SGD vs SGD+Momentum).

Run from the repo root:
    python src/optimizer_study.py --epochs 2     # quick smoke test
    python src/optimizer_study.py --epochs 20    # full run

Data loading, loss and model builder match src/train.py and src/model_b.py.
Unlike train.py, this uses NO early stopping / LR scheduler, so every optimizer
runs the same fixed number of epochs and the curves are directly comparable.
"""
import os
import json
import time
import argparse

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from model_b import build_model_b   # <-- adjust to match teammate's function name

SEED, IMG, BATCH = 42, (64, 64), 32


def load(split):
    """Load one split from data/processed/<split> (settings copied from train.py)."""
    ds = keras.utils.image_dataset_from_directory(
        f"data/processed/{split}", labels="inferred", label_mode="categorical",
        color_mode="grayscale", batch_size=BATCH, image_size=IMG,
        shuffle=(split == "train"), seed=SEED)
    return ds.cache().prefetch(tf.data.AUTOTUNE)


def run(name, make_opt, epochs, train, val, test):
    """Train a fresh Model B with the given optimizer and collect results."""
    keras.utils.set_random_seed(SEED)   # same initial weights for every optimizer
    model = build_model_b(input_shape=(64, 64, 1), num_classes=2)
    model.compile(optimizer=make_opt(),
                  loss="categorical_crossentropy",   # same as train.py
                  metrics=["accuracy"])
    t0 = time.time()
    h = model.fit(train, validation_data=val, epochs=epochs, verbose=2)  # no early stopping
    elapsed = time.time() - t0
    test_loss, test_acc = model.evaluate(test, verbose=0)
    hist = h.history
    hit90 = next((i + 1 for i, a in enumerate(hist["val_accuracy"]) if a >= 0.90), None)
    return dict(name=name, history=hist, test_acc=test_acc, test_loss=test_loss,
                best_val_acc=max(hist["val_accuracy"]),
                final_val_loss=hist["val_loss"][-1],
                epochs_to_90=hit90, sec_per_epoch=elapsed / epochs)


def plot(results, fname, title):
    """Plot training and validation loss for every run on one figure."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for r in results:
        ax[0].plot(r["history"]["loss"], label=r["name"])
        ax[1].plot(r["history"]["val_loss"], label=r["name"])
    ax[0].set_title("Training loss")
    ax[1].set_title("Validation loss")
    for a in ax:
        a.set_xlabel("Epoch")
        a.set_ylabel("Loss")
        a.legend()
        a.grid(alpha=.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(f"reports/figures/{fname}", dpi=200)
    plt.close(fig)


def table(results):
    """Print a summary table to the console."""
    print(f"{'Optimizer':34}{'BestValAcc':>11}{'FinalValLoss':>13}"
          f"{'TestAcc':>9}{'Ep>=90%':>9}{'s/ep':>7}")
    for r in results:
        print(f"{r['name']:34}{r['best_val_acc']:>11.4f}{r['final_val_loss']:>13.4f}"
              f"{r['test_acc']:>9.4f}{str(r['epochs_to_90']):>9}{r['sec_per_epoch']:>7.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()
    os.makedirs("reports/figures", exist_ok=True)
    train, val, test = load("train"), load("val"), load("test")

    # Main comparison: Adam vs plain SGD vs SGD with momentum
    main_cfg = {
        "Adam (lr=0.001)": lambda: keras.optimizers.Adam(1e-3),
        "SGD (lr=0.01, m=0)": lambda: keras.optimizers.SGD(0.01, momentum=0.0),
        "SGD+Momentum (lr=0.01, m=0.9)": lambda: keras.optimizers.SGD(0.01, momentum=0.9),
    }
    main = [run(n, f, args.epochs, train, val, test) for n, f in main_cfg.items()]
    plot(main, "optimizer_comparison.png", "Model B: Adam vs SGD vs SGD+Momentum")
    table(main)

    # Momentum sweep: isolates the effect of the momentum parameter
    sweep_cfg = {f"SGD lr=0.01, m={m}": (lambda m=m: keras.optimizers.SGD(0.01, momentum=m))
                 for m in [0.0, 0.5, 0.9, 0.99]}
    sweep = [run(n, f, args.epochs, train, val, test) for n, f in sweep_cfg.items()]
    plot(sweep, "momentum_sweep.png", "Effect of momentum (SGD, lr=0.01)")
    table(sweep)

    with open("reports/optimizer_results.json", "w") as fh:
        json.dump({"main": main, "sweep": sweep}, fh, indent=2, default=float)
