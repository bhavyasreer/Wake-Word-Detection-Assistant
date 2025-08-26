import os
from pydub import AudioSegment

# Path to your background noise wav files folder
input_folder = "background_raw"
# Folder to save the 3-second clips
output_folder = "background_clips"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Duration of each clip in milliseconds (3 seconds)
clip_duration = 3000

for filename in os.listdir(input_folder):
    if filename.endswith(".wav"):
        audio_path = os.path.join(input_folder, filename)
        sound = AudioSegment.from_wav(audio_path)
        total_length = len(sound)
        print(f"Processing {filename}, length: {total_length} ms")

        # Slice into 3-second clips
        for i in range(0, total_length, clip_duration):
            clip = sound[i:i+clip_duration]
            # If last clip is shorter than 3 seconds, you can skip or save it as is
            if len(clip) < clip_duration:
                print(f"Skipping short clip from {i} ms")
                continue
            output_path = os.path.join(output_folder, f"{filename[:-4]}_clip{i//1000}.wav")
            clip.export(output_path, format="wav")
            print(f"Saved clip: {output_path}")

print("All clips created!")
