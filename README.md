# Mochi - Real-time ASL-to-Text Chat App

A real-time American Sign Language (ASL) to text chat application with a pink-themed UI, powered by a Convolutional Neural Network (CNN) that recognizes 29 classes (A-Z plus delete, space, and nothing). The app uses MediaPipe for hand landmark detection and TensorFlow for sign classification.

---

## Description

**Mochi** enables users to communicate via ASL by translating hand signs into text messages in real time. Users can toggle between typing and sign mode, where the app captures webcam input, detects hand landmarks, and predicts ASL signs using a trained CNN model. The interface is built with CustomTkinter for a modern, pink-themed GUI.

---

## Features

- Real-time ASL recognition for 29 classes: A-Z, delete, space, and nothing.
- Toggle between typing and sign detection modes.
- Pink-themed, scrollable chat interface.
- Uses MediaPipe for precise hand landmark detection.
- CNN model trained on hand landmarks for accurate sign classification.
- Threaded webcam capture for smooth UI experience.

---

## Requirements

- Python 3.7+
- OpenCV (`opencv-python`)
- MediaPipe
- TensorFlow
- CustomTkinter
- NumPy

Install dependencies via pip:

```bash
pip install opencv-python mediapipe tensorflow customtkinter numpy
```

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/nerdyalgorithm/mochi.git
cd mochi
```

2. Ensure the trained model file `asl_29_model.keras` is in the project directory.

3. Install required Python packages (see above).

---

## Usage

Run the main app:

```bash
python mochi.py
```

- Use the **Sign Mode** toggle button to switch between typing and ASL sign detection.
- When sign mode is on, the webcam opens and detects hand signs in real time.
- Detected signs are automatically inserted into the message input box.
- Press **Send** or hit Enter to send messages to the chat window.
- Press `q` in the webcam window to quit sign detection mode.

---

## Training the CNN Model

The `train_cnn.py` script trains a CNN on hand landmark data extracted from images of ASL signs covering 29 classes (A-Z plus `del`, `space`, and `nothing`).

### Training Steps:

- Collect labeled images of ASL signs organized in folders named after each class.
- Run:

```bash
python train_cnn.py
```

- The script extracts hand landmarks using MediaPipe, preprocesses data, trains a dense neural network, and saves the model as `asl_29_model.keras`.
- Adjust the `data_dir` variable in `train_cnn.py` to point to your dataset location.

---

## Project Structure

```
mochi/
├── mochi.py           # Main app with GUI and real-time sign detection
├── train_cnn.py       # Script to train the CNN model on ASL hand landmarks
└── README.md          # This file
```

---

## Contributing

Contributions and improvements are welcome! Please fork the repo and submit pull requests with enhancements or bug fixes.

---

## License

This project is licensed under the MIT License.

---

## Contact

Created by [Heaven](https://github.com/nerdyalgorithm) (GitHub username: **nerdyalgorithm**).  
Feel free to open issues or reach out on GitHub for questions or feedback.

---

