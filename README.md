# Wake Word Detection System

This project implements a wake word detection system using machine learning. It can detect a specific wake word (e.g., "Hey Assistant") from audio input.

## Features
- Real-time audio recording and processing
- MFCC feature extraction
- Neural network-based wake word detection
- Training and inference scripts

## Setup
1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the training script to train the model:
```bash
python train.py
```

3. Run the inference script to test wake word detection:
```bash
python detect.py
```

## Project Structure
- `train.py`: Script for training the wake word detection model
- `detect.py`: Script for real-time wake word detection
- `model.py`: Neural network model architecture
- `audio_utils.py`: Audio processing utilities
- `data/`: Directory for storing training data
- `models/`: Directory for saving trained models 