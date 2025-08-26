import os
import numpy as np
import sounddevice as sd
import soundfile as sf
from audio_utils import load_audio
import librosa
import time

def verify_audio_file(file_path):
    """Verify a single audio file"""
    try:
        # Load audio
        audio = load_audio(file_path)
        
        # Check duration
        duration = len(audio) / 16000  # assuming 16kHz sample rate
        if duration < 0.5 or duration > 2.0:
            print(f"⚠️ {os.path.basename(file_path)}: Duration {duration:.2f}s is outside normal range (0.5-2.0s)")
            return False
            
        # Check audio level
        max_level = np.max(np.abs(audio))
        if max_level < 0.1:
            print(f"⚠️ {os.path.basename(file_path)}: Audio level too low ({max_level:.3f})")
            return False
            
        # Check for silence at start/end
        start_silence = np.mean(np.abs(audio[:1600]))  # first 0.1s
        end_silence = np.mean(np.abs(audio[-1600:]))   # last 0.1s
        if start_silence > 0.1 or end_silence > 0.1:
            print(f"⚠️ {os.path.basename(file_path)}: Possible silence issues at start/end")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ {os.path.basename(file_path)}: Error - {str(e)}")
        return False

def play_sample(file_path):
    """Play a sample and wait for user input"""
    try:
        audio = load_audio(file_path)
        print(f"\nPlaying: {os.path.basename(file_path)}")
        sd.play(audio, 16000)
        sd.wait()
        return input("Is this a good sample? (y/n): ").lower() == 'y'
    except Exception as e:
        print(f"Error playing {file_path}: {str(e)}")
        return False

def main():
    wake_word_dir = 'data/wake_word'
    files = [f for f in os.listdir(wake_word_dir) if f.endswith('.wav')]
    
    print(f"Found {len(files)} wake word samples")
    print("\nVerifying samples...")
    
    valid_files = []
    for file in files:
        file_path = os.path.join(wake_word_dir, file)
        if verify_audio_file(file_path):
            valid_files.append(file_path)
    
    print(f"\nFound {len(valid_files)} valid samples")
    
    # Manual verification of a subset
    print("\nWould you like to manually verify some samples? (y/n)")
    if input().lower() == 'y':
        num_to_check = min(10, len(valid_files))
        print(f"\nLet's check {num_to_check} random samples:")
        
        samples_to_check = np.random.choice(valid_files, num_to_check, replace=False)
        good_samples = 0
        
        for sample in samples_to_check:
            if play_sample(sample):
                good_samples += 1
            time.sleep(0.5)  # Small pause between samples
            
        print(f"\nManual verification results:")
        print(f"Good samples: {good_samples}/{num_to_check}")
        
        if good_samples < num_to_check * 0.8:  # If less than 80% are good
            print("\n⚠️ Warning: Many samples seem to be of poor quality")
            print("Consider reviewing and cleaning the dataset before training")
        else:
            print("\n✅ Sample quality looks good!")
            print("You can proceed with training")

if __name__ == "__main__":
    main() 