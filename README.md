# Sign Language Detection and Translation

This project implements a real-time American Sign Language (ASL) fingerspelling detection and translation pipeline. It leverages a combination of MediaPipe for robust hand tracking and a custom PyTorch-based neural network for accurate classification across a 35-class ASL dataset.

## Features

- **Real-time Inference**: Designed for low-latency live translation using a standard webcam.
- **Robust Preprocessing**: Includes a `BackgroundAwarePreprocessor` that detects hands using MediaPipe, extracts regions of interest (ROI), and normalizes landmarks relative to the wrist.
- **Deep Learning Model**: A PyTorch-based architecture (`DualStreamASLModel`) that uses 63-dimensional hand landmark data for high-accuracy classification.
- **Temporal Debouncing**: Implements a `RealTimeDebouncer` to smooth out frame-by-frame predictions and provide stable, readable output in real-time.

## Project Structure

- `Version1.2/` (Latest Version)
  - `build_dataset.py`: Script to process images and extract landmarks into a PyTorch dataset.
  - `train.py`: Training script for the `DualStreamASLModel` using the extracted dataset.
  - `dual_stream_model.py`: Defines the PyTorch neural network architecture.
  - `background_aware_preprocessor.py`: Handles MediaPipe hand detection, landmark extraction, and bounding box logic.
  - `temporal_head.py`: Contains the `RealTimeDebouncer` for prediction smoothing.
  - `live_inference.py`: The main script to run real-time ASL translation using a webcam.
- `dataset.pt`: Processed dataset containing landmark features and labels.
- `hand_landmarker.task`: MediaPipe model asset for hand detection.

## Setup & Installation

1. **Clone the repository:**
   Ensure you have all the project files locally.

2. **Install Dependencies:**
   The project requires Python and several libraries. Install them via pip:
   ```bash
   pip install torch torchvision torchaudio
   pip install opencv-python numpy mediapipe
   ```

3. **Dataset Download link:**
   The training dataset can be downloaded from `https://github.com/KashyapLanka/Sign-Language-Detection-and-Translation`.

## Usage

### 1. Building the Dataset (Optional)
If you need to process new data or rebuild the dataset, run:
```bash
cd Version1.2
python build_dataset.py
```

### 2. Training the Model (Optional)
To retrain the PyTorch model with the dataset:
```bash
cd Version1.2
python train.py
```

### 3. Live Inference
To start real-time ASL fingerspelling translation, ensure your webcam is connected and run:
```bash
cd Version1.2
python live_inference.py
```
Press `ESC` to exit the live inference window.

## Architecture Highlights
The real-time pipeline operates as follows:
1. **Frame Capture**: Reads RGB frames from the webcam.
2. **Preprocessing**: The `BackgroundAwarePreprocessor` extracts 3D hand landmarks using MediaPipe.
3. **Normalization**: Landmarks are normalized by centering them at the wrist and scaling them to a consistent size.
4. **Classification**: The 63-dimensional normalized landmark vector is passed through the PyTorch model to get class probabilities.
5. **Debouncing**: The `RealTimeDebouncer` filters the raw probabilities over time to output a stable, readable text prediction.

## License
MIT License
