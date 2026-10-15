import tensorflow as tf
from tensorflow.keras import layers, models

def build_model_b(input_shape=(64, 64, 1), num_classes=2):
    """
    Model B: Enhanced Lightweight Edge CNN with Depthwise Separable Convolutions
    
    Constraints & Design Decisions:
    - Trainable parameter count: ~63,330 (Strictly under the 100,000 assignment ceiling).
    - Uses Depthwise Separable Convolutions (SeparableConv2D) to minimize MAC operations.
    - Channel progression: 32 -> 64 -> 128 -> 256 for rich multi-scale feature extraction.
    - Uses ReLU activations for single-cycle execution on edge hardware (avoiding exponentials).
    - Edge-optimized head with GlobalAveragePooling2D and Dropout(0.3) to prevent overfitting.
    """
    model = models.Sequential(name="Model_B_Lightweight_CNN", layers=[
        layers.Input(shape=input_shape),
        layers.Rescaling(1.0 / 255.0, name='rescale'),
        
        # Initial standard Conv2D to expand grayscale channel (1 -> 32)
        layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', name='init_conv'),
        layers.BatchNormalization(name='bn0'),
        layers.MaxPooling2D(pool_size=(2, 2), name='pool1'),
        
        # Block 1: Depthwise Separable Convolution (32 -> 64)
        layers.SeparableConv2D(64, kernel_size=(3, 3), padding='same', activation='relu', name='sepconv1'),
        layers.BatchNormalization(name='bn1'),
        layers.MaxPooling2D(pool_size=(2, 2), name='pool2'),
        
        # Block 2: Depthwise Separable Convolution (64 -> 128)
        layers.SeparableConv2D(128, kernel_size=(3, 3), padding='same', activation='relu', name='sepconv2'),
        layers.BatchNormalization(name='bn2'),
        layers.MaxPooling2D(pool_size=(2, 2), name='pool3'),
        
        # Block 3: Depthwise Separable Convolution (128 -> 256)
        layers.SeparableConv2D(256, kernel_size=(3, 3), padding='same', activation='relu', name='sepconv3'),
        layers.BatchNormalization(name='bn3'),
        layers.MaxPooling2D(pool_size=(2, 2), name='pool4'),
        
        # Edge-Optimized Classification Head
        layers.GlobalAveragePooling2D(name='global_avg_pool'),
        layers.Dense(64, activation='relu', name='fc1'),
        layers.Dropout(0.3, name='dropout'),
        layers.Dense(num_classes, activation='softmax', name='output')
    ])
    
    return model

if __name__ == "__main__":
    model = build_model_b()
    model.summary()
