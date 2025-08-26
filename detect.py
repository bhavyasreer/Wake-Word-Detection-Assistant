import numpy as np
from audio_utils import record_audio, extract_mfcc
from model import load_model

def detect_wake_word(model, threshold=0.5):
    """
    Continuously monitor audio for wake word detection.
    
    Args:
        model: Trained wake word detection model
        threshold (float): Probability threshold for wake word detection
    """
    print("Listening for wake word... (Press Ctrl+C to stop)")
    
    try:
        while True:
            # Record audio
            audio = record_audio(duration=1)
            
            # Extract features
            mfcc = extract_mfcc(audio)
            
            # Make prediction
            prediction = model.predict(np.expand_dims(mfcc, axis=0))
            probability = prediction[0][1]  # Probability of wake word
            
            # Check if wake word is detected
            if probability > threshold:
                print(f"Wake word detected! (Confidence: {probability:.2f})")
            else:
                print(f"Listening... (Confidence: {probability:.2f})", end='\r')
    
    except KeyboardInterrupt:
        print("\nStopping wake word detection...")

def main():
    # Load the trained model
    try:
        model = load_model('models/wake_word_model.h5')
    except:
        print("Error: Model not found. Please train the model first using train.py")
        return
    
    # Start wake word detection
    detect_wake_word(model)

if __name__ == "__main__":
    main() 