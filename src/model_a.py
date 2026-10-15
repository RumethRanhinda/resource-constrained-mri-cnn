import tensorflow as tf
from tensorflow.keras import layers, models

def build_model_a(input_shape=(64, 64, 1), num_classes=2):
    """
    Model A: Standard Baseline CNN
    
    Architecture:
    - Interleaved standard Conv2D layers and MaxPooling2D layers.
    - Uses ReLU activations for hardware efficiency on edge devices.
    - Flatten and Dense layers for classification.
    """
    model = models.Sequential(name="Model_A_Standard_CNN", layers=[
        layers.Input(shape=input_shape),
        layers.Rescaling(1.0 / 255.0, name='rescale'),
        layers.Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu', name='conv1'),
        layers.MaxPooling2D(pool_size=(2, 2), name='pool1'),
        layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', name='conv2'),
        layers.MaxPooling2D(pool_size=(2, 2), name='pool2'),
        layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', name='conv3'),
        layers.MaxPooling2D(pool_size=(2, 2), name='pool3'),
        layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu', name='conv4'),
        layers.MaxPooling2D(pool_size=(2, 2), name='pool4'),
        
        layers.Flatten(name='flatten'),
        layers.Dense(64, activation='relu', name='fc1'),
        layers.Dropout(0.3, name='dropout'),
        layers.Dense(num_classes, activation='softmax', name='output')
    ])
    
    return model

if __name__ == "__main__":
    model = build_model_a()
    model.summary()
