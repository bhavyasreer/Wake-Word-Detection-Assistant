import tensorflow as tf
from tensorflow.keras import layers, models

def create_model():
    """
    Create a simple CNN model for wake word detection.
    
    Returns:
        model: Compiled Keras model
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=(13, 32, 1)),  # MFCC features shape
        
        # Convolutional layers with 'same' padding
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), padding='same'),
        # Only two conv+pool blocks to avoid shrinking too much
        
        # Flatten and Dense layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def save_model(model, filepath):
    """
    Save the trained model.
    
    Args:
        model: Trained Keras model
        filepath (str): Path to save the model
    """
    model.save(filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """
    Load a trained model.
    
    Args:
        filepath (str): Path to the saved model
        
    Returns:
        tensorflow.keras.Model: Loaded model
    """
    return models.load_model(filepath) 