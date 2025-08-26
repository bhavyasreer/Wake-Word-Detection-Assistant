import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from audio_utils import extract_mfcc, load_audio
import librosa
import matplotlib.pyplot as plt

# Constants
SAMPLE_RATE = 16000
NUM_MFCC = 13
NUM_FRAMES = 32
MIN_WAKE_WORD_SAMPLES = 30  # Minimum wake word samples required
MIN_BACKGROUND_SAMPLES = 20  # Minimum background noise samples required
AUGMENTATION_FACTOR = 15  # Increased augmentation factor for better generalization

def validate_audio(audio_path):
    """Validate audio file quality"""
    try:
        audio = load_audio(audio_path, SAMPLE_RATE)
        # Check if audio is not too quiet
        if np.max(np.abs(audio)) < 0.01:
            return False
        # Check if audio is not too short
        if len(audio) < SAMPLE_RATE * 0.5:  # At least 0.5 seconds
            return False
        # Check if audio is not too long
        if len(audio) > SAMPLE_RATE * 2.0:  # Not more than 2 seconds
            return False
        return True
    except Exception as e:
        print(f"Error validating {audio_path}: {str(e)}")
        return False

def augment_audio(audio, is_wake_word=True):
    """Apply audio augmentation focused on same-voice variations"""
    augmented = audio.copy()
    
    if is_wake_word:
        # Subtle pitch variations (same voice, different intonation)
        if np.random.random() < 0.4:
            steps = np.random.uniform(-1, 1)  # Smaller range for same voice
            augmented = librosa.effects.pitch_shift(augmented, sr=SAMPLE_RATE, n_steps=steps)
        
        # Subtle time stretch (same voice, different speed)
        if np.random.random() < 0.4:
            rate = np.random.uniform(0.9, 1.1)  # Smaller range for same voice
            augmented = librosa.effects.time_stretch(augmented, rate=rate)
        
        # Volume variations (same voice, different loudness)
        if np.random.random() < 0.3:
            volume_factor = np.random.uniform(0.8, 1.2)
            augmented *= volume_factor
        
        # Add slight room effect (same voice, different position)
        if np.random.random() < 0.3:
            room_size = np.random.uniform(0.05, 0.15)  # Subtle room effect
            augmented = librosa.effects.preemphasis(augmented, coef=room_size)
        
        # Add slight noise (same voice, different environment)
        if np.random.random() < 0.3:
            noise_level = np.random.uniform(0.001, 0.005)  # Very subtle noise
            noise = np.random.normal(0, noise_level, len(audio))
            augmented += noise
    
    # Less aggressive augmentation for background
    else:
        if np.random.random() < 0.3:
            noise_level = np.random.uniform(0.001, 0.01)
            noise = np.random.normal(0, noise_level, len(audio))
            augmented += noise
        
        if np.random.random() < 0.3:
            volume_factor = np.random.uniform(0.8, 1.2)
            augmented *= volume_factor
    
    return augmented

def load_and_prepare_data(wake_word_dir='data/wake_word', background_dir='data/background'):
    """Load and prepare the training data"""
    print("Loading wake word samples...")
    wake_word_files = [f for f in os.listdir(wake_word_dir) if f.endswith('.wav')]
    wake_word_mfccs = []
    
    # Validate wake word samples
    valid_wake_word_files = []
    for file in wake_word_files:
        audio_path = os.path.join(wake_word_dir, file)
        if validate_audio(audio_path):
            valid_wake_word_files.append(audio_path)
    
    if len(valid_wake_word_files) < MIN_WAKE_WORD_SAMPLES:
        raise ValueError(f"Not enough valid wake word samples. Need at least {MIN_WAKE_WORD_SAMPLES}, got {len(valid_wake_word_files)}")
    
    print(f"Found {len(valid_wake_word_files)} valid wake word samples")
    
    for audio_path in valid_wake_word_files:
        # Original sample
        mfcc = extract_mfcc(audio_path, SAMPLE_RATE)
        wake_word_mfccs.append(mfcc)
        
        # Create augmented versions
        audio = load_audio(audio_path, SAMPLE_RATE)
        for _ in range(AUGMENTATION_FACTOR):  # Increased augmentation
            aug_audio = augment_audio(audio, is_wake_word=True)
            aug_mfcc = librosa.feature.mfcc(y=aug_audio, sr=SAMPLE_RATE, n_mfcc=NUM_MFCC)
            if aug_mfcc.shape[1] < NUM_FRAMES:
                aug_mfcc = np.pad(aug_mfcc, ((0, 0), (0, NUM_FRAMES - aug_mfcc.shape[1])), mode='constant')
            else:
                aug_mfcc = aug_mfcc[:, :NUM_FRAMES]
            wake_word_mfccs.append(aug_mfcc)
    
    print("\nLoading background noise samples...")
    background_files = [f for f in os.listdir(background_dir) if f.endswith('.wav')]
    background_mfccs = []
    
    # Validate background samples
    valid_background_files = []
    for file in background_files:
        audio_path = os.path.join(background_dir, file)
        if validate_audio(audio_path):
            valid_background_files.append(audio_path)
    
    if len(valid_background_files) < MIN_BACKGROUND_SAMPLES:
        raise ValueError(f"Not enough valid background samples. Need at least {MIN_BACKGROUND_SAMPLES}, got {len(valid_background_files)}")
    
    print(f"Found {len(valid_background_files)} valid background samples")
    
    for audio_path in valid_background_files:
        # Original sample
        mfcc = extract_mfcc(audio_path, SAMPLE_RATE)
        background_mfccs.append(mfcc)
        
        # Create augmented versions for background too
        audio = load_audio(audio_path, SAMPLE_RATE)
        for _ in range(3):  # Fewer augmentations for background
            aug_audio = augment_audio(audio, is_wake_word=False)
            aug_mfcc = librosa.feature.mfcc(y=aug_audio, sr=SAMPLE_RATE, n_mfcc=NUM_MFCC)
            if aug_mfcc.shape[1] < NUM_FRAMES:
                aug_mfcc = np.pad(aug_mfcc, ((0, 0), (0, NUM_FRAMES - aug_mfcc.shape[1])), mode='constant')
            else:
                aug_mfcc = aug_mfcc[:, :NUM_FRAMES]
            background_mfccs.append(aug_mfcc)
    
    print(f"\nTotal wake word samples (including augmented): {len(wake_word_mfccs)}")
    print(f"Total background samples (including augmented): {len(background_mfccs)}")
    
    # Create labels: 1 for wake word, 0 for background
    X = np.array(wake_word_mfccs + background_mfccs)
    y = np.array([1] * len(wake_word_mfccs) + [0] * len(background_mfccs))
    
    # Add channel dimension for CNN
    X = np.expand_dims(X, axis=-1)
    
    # Calculate class weights
    total_samples = len(y)
    wake_word_samples = np.sum(y == 1)
    background_samples = np.sum(y == 0)
    
    class_weights = {
        0: total_samples / (2 * background_samples),  # Weight for background class
        1: total_samples / (2 * wake_word_samples)    # Weight for wake word class
    }
    
    print(f"\nClass weights: {class_weights}")
    
    # Shuffle the data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y, class_weights

def create_model(input_shape):
    """Create a CNN model optimized for same-voice detection"""
    model = models.Sequential([
        # First conv block
        layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),  # Reduced dropout for better feature preservation
        
        # Second conv block
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Third conv block
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    return model

def main():
    print("Loading and preparing data...")
    X, y, class_weights = load_and_prepare_data()
    
    print(f"\nData shape: {X.shape}")
    print(f"Number of wake word samples: {np.sum(y == 1)}")
    print(f"Number of background samples: {np.sum(y == 0)}")
    
    # Create and train the model
    print("\nCreating and training the model...")
    model = create_model(input_shape=(NUM_MFCC, NUM_FRAMES, 1))
    
    # Train the model with early stopping and model checkpoint
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'models/wake_word_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    
    # Add learning rate reduction
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6
    )
    
    history = model.fit(
        X, y,
        epochs=150,
        batch_size=32,
        validation_split=0.2,
        class_weight=class_weights,
        callbacks=[early_stopping, model_checkpoint, reduce_lr]
    )
    
    # Print final accuracy
    final_accuracy = history.history['accuracy'][-1]
    final_val_accuracy = history.history['val_accuracy'][-1]
    print(f"\nFinal training accuracy: {final_accuracy:.2%}")
    print(f"Final validation accuracy: {final_val_accuracy:.2%}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

if __name__ == "__main__":
    main() 