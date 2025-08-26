import os
import urllib.request
import zipfile
import shutil

def download_background_samples():
    """
    Download pre-recorded background noise samples.
    """
    # Create background directory if it doesn't exist
    os.makedirs('data/background', exist_ok=True)
    
    # URL for background noise samples (using Freesound dataset)
    url = "https://freesound.org/data/previews/131/131660_2398403-lq.mp3"
    temp_file = "temp_background.zip"
    
    print("\nDownloading background noise samples...")
    try:
        # Download the file
        urllib.request.urlretrieve(url, temp_file)
        
        # Extract the files
        with zipfile.ZipFile(temp_file, 'r') as zip_ref:
            zip_ref.extractall('data/background')
        
        # Clean up
        os.remove(temp_file)
        print("\nBackground noise samples downloaded successfully!")
        
    except Exception as e:
        print(f"\nError downloading background samples: {e}")
        print("\nAlternative: You can manually download background noise samples from:")
        print("1. Freesound.org (https://freesound.org/)")
        print("2. AudioSet (https://research.google.com/audioset/)")
        print("\nSave the .wav files in the 'data/background' directory")

if __name__ == "__main__":
    download_background_samples() 