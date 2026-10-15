import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

from model_a import build_model_a
from model_b import build_model_b

class EpochTimer(tf.keras.callbacks.Callback):
    """Callback to measure exact training time per epoch."""
    def on_train_begin(self, logs=None):
        self.epoch_times = []
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        
    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.epoch_start_time
        self.epoch_times.append(elapsed)

def get_datasets(data_dir, batch_size=32, image_size=(64, 64)):
    """
    Creates tf.data.Dataset pipelines for train, validation, and test sets.
    """
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    
    print("\n--- Loading Datasets ---")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='categorical',
        color_mode='grayscale',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=42
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels='inferred',
        label_mode='categorical',
        color_mode='grayscale',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False
    )
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels='inferred',
        label_mode='categorical',
        color_mode='grayscale',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False
    )
    
    # Cache and prefetch for hardware I/O efficiency
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds, test_ds

def plot_history(history, model_name, save_dir):
    """Plots and saves loss and accuracy curves."""
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history.history['loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['loss'], 'b-o', label='Train Loss')
    plt.plot(epochs, history.history['val_loss'], 'r--s', label='Val Loss')
    plt.title(f'{model_name} - Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['accuracy'], 'b-o', label='Train Accuracy')
    plt.plot(epochs, history.history['val_accuracy'], 'r--s', label='Val Accuracy')
    plt.title(f'{model_name} - Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"{model_name.lower()}_learning_curves.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved learning curves to: {plot_path}")

def plot_confusion_matrix(cm, class_names, model_name, save_dir):
    """Plots and saves a confusion matrix visualization."""
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Print numbers inside cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=12, fontweight='bold')
                     
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    
    cm_path = os.path.join(save_dir, f"{model_name.lower()}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix to: {cm_path}")

def train_and_evaluate(model, model_name, train_ds, val_ds, test_ds, epochs=30, lr=0.001, early_stopping=True):
    """
    Compiles, trains, and evaluates a single model.
    """
    print(f"\n=======================================================")
    print(f"  Starting Training: {model_name}")
    print(f"=======================================================")
    
    # Compile with Adam optimizer as primary selection
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    timer_cb = EpochTimer()
    callbacks = [timer_cb]
    
    if early_stopping:
        callbacks.extend([
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=12,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,
                min_lr=1e-5,
                verbose=1
            )
        ])
    
    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Calculate average time per epoch
    avg_time_per_epoch = np.mean(timer_cb.epoch_times)
    print(f"\nAvg Training Time per Epoch: {avg_time_per_epoch:.2f} seconds")
    
    # Save Model Weights and evaluate disk size
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_save_path = os.path.join(models_dir, f"{model_name.lower()}.keras")
    model.save(model_save_path)
    model_size_kb = os.path.getsize(model_save_path) / 1024.0
    print(f"Saved model to: {model_save_path} ({model_size_kb:.2f} KB)")
    
    # Save Loss & Accuracy Curves
    reports_fig_dir = os.path.join(os.path.dirname(__file__), '..', 'reports', 'figures')
    plot_history(history, model_name, reports_fig_dir)
    
    # Evaluation on Test Set
    print(f"\n--- Evaluating on Test Set ---")
    y_true_list = []
    y_pred_list = []
    
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true_list.extend(np.argmax(labels.numpy(), axis=1))
        y_pred_list.extend(np.argmax(preds, axis=1))
        
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    class_names = ['Demented', 'NonDemented']
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='binary')
    rec = recall_score(y_true, y_pred, average='binary')
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"Test Accuracy:  {acc * 100:.2f}%")
    print(f"Test Precision: {prec:.4f}")
    print(f"Test Recall:    {rec:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    plot_confusion_matrix(cm, class_names, model_name, reports_fig_dir)
    
    total_params = model.count_params()
    
    return {
        'model_name': model_name,
        'total_params': total_params,
        'size_kb': model_size_kb,
        'avg_time_per_epoch_sec': avg_time_per_epoch,
        'test_accuracy': acc,
        'precision': prec,
        'recall': rec,
        'history': history,
        'confusion_matrix': cm
    }

def print_comparison_table(results_a, results_b):
    """Prints the required assignment comparison table."""
    print("\n" + "="*75)
    print("                MODEL A vs MODEL B COMPARISON TABLE                ")
    print("="*75)
    header = f"{'Metric':<30} | {'Model A (Standard)':<20} | {'Model B (Lightweight)':<20}"
    print(header)
    print("-" * len(header))
    time_a = f"{results_a['avg_time_per_epoch_sec']:.2f}s"
    time_b = f"{results_b['avg_time_per_epoch_sec']:.2f}s"
    acc_a = f"{results_a['test_accuracy']*100:.2f}%"
    acc_b = f"{results_b['test_accuracy']*100:.2f}%"
    prec_a = f"{results_a['precision']:.4f}"
    prec_b = f"{results_b['precision']:.4f}"
    rec_a = f"{results_a['recall']:.4f}"
    rec_b = f"{results_b['recall']:.4f}"
    
    print(f"{'Total Parameter Count':<30} | {results_a['total_params']:<20,d} | {results_b['total_params']:<20,d}")
    print(f"{'Model Size on Disk (KB)':<30} | {results_a['size_kb']:<20.2f} | {results_b['size_kb']:<20.2f}")
    print(f"{'Avg Training Time / Epoch':<30} | {time_a:<20} | {time_b:<20}")
    print(f"{'Test Accuracy':<30} | {acc_a:<20} | {acc_b:<20}")
    print(f"{'Test Precision':<30} | {prec_a:<20} | {prec_b:<20}")
    print(f"{'Test Recall':<30} | {rec_a:<20} | {rec_b:<20}")
    print("="*75 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate Model A and Model B.")
    parser.add_argument('--model', type=str, default='both', choices=['a', 'b', 'both'], help="Which model to train")
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs (default: 20)")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument('--no-early-stopping', action='store_true', help="Disable early stopping and learning rate scheduling")
    args = parser.parse_args()
    
    use_es = not args.no_early_stopping
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    train_ds, val_ds, test_ds = get_datasets(data_path, batch_size=args.batch_size)
    
    results = {}
    
    if args.model in ['a', 'both']:
        model_a = build_model_a(input_shape=(64, 64, 1), num_classes=2)
        results['a'] = train_and_evaluate(model_a, "Model_A", train_ds, val_ds, test_ds, epochs=args.epochs, early_stopping=use_es)
        
    if args.model in ['b', 'both']:
        model_b = build_model_b(input_shape=(64, 64, 1), num_classes=2)
        results['b'] = train_and_evaluate(model_b, "Model_B", train_ds, val_ds, test_ds, epochs=args.epochs, early_stopping=use_es)
        
    if args.model == 'both':
        print_comparison_table(results['a'], results['b'])
