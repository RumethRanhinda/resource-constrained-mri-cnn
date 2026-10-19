import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def get_backbone(model):
    """
    Returns the nested pretrained backbone (the only nested Model inside the wrapper).

    The backbone is deliberately NOT given a custom name: for EfficientNet, Keras builds the
    ImageNet weight download URL from the model name, so renaming it breaks the download (HTTP 403).
    """
    return next(layer for layer in model.layers if isinstance(layer, tf.keras.Model))


def _build_transfer_model(base_fn, preprocess, name, input_shape, num_classes, weights):
    """
    Shared builder for the pretrained lightweight SOTA models (Section 5).

    Pipeline (mirrors the head used in Model A / Model B):
    - Input is a raw 0-255 grayscale image, exactly what get_datasets() in train.py yields.
    - Grayscale is replicated to 3 channels so ImageNet weights can be used unchanged.
    - Model-specific preprocessing (if any) is applied inside the network.
    - Pretrained backbone (frozen by default), then GlobalAveragePooling2D -> Dropout(0.3) -> Dense(softmax).
    """
    h, w = input_shape[0], input_shape[1]

    inputs = layers.Input(shape=input_shape, name='input')
    x = layers.Concatenate(axis=-1, name='gray_to_rgb')([inputs, inputs, inputs])
    if preprocess is not None:
        x = preprocess(x)

    base = base_fn(include_top=False, weights=weights, input_shape=(h, w, 3))
    base.trainable = False  # stage 1: train only the new classification head
    x = base(x, training=False)  # keep BatchNorm layers in inference mode

    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = layers.Dropout(0.3, name='dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)

    return models.Model(inputs, outputs, name=name)


def build_mobilenetv2(input_shape=(64, 64, 1), num_classes=2, weights='imagenet'):
    """
    MobileNetV2 (ImageNet pretrained), adapted to 64x64 grayscale input.

    - Depthwise separable convolutions + inverted residual blocks.
    - Expects inputs scaled to [-1, 1], done here with a Rescaling layer
      (equivalent to mobilenet_v2.preprocess_input applied to 0-255 inputs).
    """
    preprocess = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name='mobilenet_preprocess')
    return _build_transfer_model(
        tf.keras.applications.MobileNetV2, preprocess,
        "Model_MobileNetV2", input_shape, num_classes, weights)


def build_efficientnetb0(input_shape=(64, 64, 1), num_classes=2, weights='imagenet'):
    """
    EfficientNet-B0 (ImageNet pretrained), adapted to 64x64 grayscale input.

    - MBConv blocks with squeeze-and-excitation, compound-scaled architecture.
    - The Keras implementation already contains its own rescaling/normalization
      and expects raw 0-255 inputs, so no extra preprocessing is added here.
    """
    return _build_transfer_model(
        tf.keras.applications.EfficientNetB0, None,
        "Model_EfficientNetB0", input_shape, num_classes, weights)


def set_backbone_trainable(model, trainable, freeze_batchnorm=True):
    """
    Freeze / unfreeze the pretrained backbone (call model.compile() again afterwards).

    freeze_batchnorm=True keeps BatchNorm layers frozen when unfreezing, which is
    the usual choice for fine-tuning on a small dataset.
    """
    base = get_backbone(model)
    base.trainable = trainable
    if trainable and freeze_batchnorm:
        for layer in base.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False


def param_summary(model):
    """Returns (total, trainable, non_trainable) parameter counts for the report."""
    total = model.count_params()
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    return total, trainable, total - trainable


if __name__ == "__main__":
    # weights=None here only so the parameter counts print without downloading anything.
    for builder in (build_mobilenetv2, build_efficientnetb0):
        m = builder(weights=None)
        total, trainable, frozen = param_summary(m)
        print(f"{m.name}: total={total:,} | trainable (head only)={trainable:,} | frozen={frozen:,}")