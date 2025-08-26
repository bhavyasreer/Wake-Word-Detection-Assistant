import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf

def record_audio(duration=1, sample_rate=16000):
    """
    Record audio from microphone.
    
    Args:
        duration (float): Duration of recording in seconds
        sample_rate (int): Sample rate of recording
        
    Returns:
        numpy.ndarray: Recorded audio data
    """
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    return audio.flatten()

def save_audio(audio, file_path, sample_rate=16000):
    """Save audio to file"""
    sf.write(file_path, audio, sample_rate)

def load_audio(file_path, sample_rate=16000):
    """Load audio file and resample if necessary"""
    if isinstance(file_path, str):
        # If it's a file path, load it
        audio, sr = librosa.load(file_path, sr=sample_rate)
    else:
        # If it's already an audio array, just return it
        audio = file_path
        sr = sample_rate
    return audio

def extract_mfcc(audio_path, sample_rate=16000, n_mfcc=13, n_frames=32):
    """
    Extract MFCC features from audio file or array
    
    Args:
        audio_path: Path to audio file or audio array
        sample_rate: Sample rate of audio
        n_mfcc: Number of MFCC coefficients
        n_frames: Number of frames to extract
        
    Returns:
        MFCC features as numpy array
    """
    # Load audio file or use audio array
    audio = load_audio(audio_path, sample_rate)
    
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    
    # Pad or truncate to desired number of frames
    if mfccs.shape[1] < n_frames:
        # Pad with zeros if shorter
        pad_width = n_frames - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
    else:
        # Truncate if longer
        mfccs = mfccs[:, :n_frames]
    
    return mfccs 