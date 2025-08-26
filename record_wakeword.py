import sounddevice as sd
import soundfile as sf
import os

SAMPLE_RATE = 16000
DURATION = 1.0  # seconds
NUM_SAMPLES = 30  # Number of wake word samples to record
OUTPUT_DIR = 'data/wake_word'

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("You will record your wake word ('Ok Nino')", NUM_SAMPLES, "times.")
print("Press Enter to start each recording. Speak clearly after you press Enter.")
print("Make sure to say 'Ok Nino' clearly and consistently each time.")

for i in range(1, NUM_SAMPLES + 1):
    input(f"\nPress Enter to record sample {i}/{NUM_SAMPLES}...")
    print("Recording... (say 'Ok Nino')")
    audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    filename = os.path.join(OUTPUT_DIR, f"wake_word_{i:02d}.wav")
    sf.write(filename, audio, SAMPLE_RATE)
    print(f"Saved: {filename}")

print("\nAll wake word samples recorded!") 