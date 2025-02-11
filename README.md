# Wildfire Detection AI/ML System

An AI-powered system for detecting wildfires in images and video streams using deep learning.

## Project Overview

This system uses computer vision and deep learning to detect the presence of wildfires in images and video streams. The project is implemented in three phases:

1. **Phase 1:** Core AI/ML model for image-based fire detection
2. **Phase 2:** Real-time webcam integration
3. **Phase 3:** Raspberry Pi deployment (planned)

## Installation

1. Clone this repository:
```bash
git clone [repository-url]
cd firedetect
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
firedetect/
├── data/
│   ├── fire/         # Fire images for training
│   └── no_fire/      # Non-fire images for training
├── src/
│   ├── data.py       # Data loading and preprocessing
│   ├── model.py      # Neural network model definition
│   ├── train.py      # Training script
│   ├── inference.py  # Inference module
│   └── webcam.py     # Webcam integration (Phase 2)
├── models/           # Saved model files
├── logs/            # Training and inference logs
├── requirements.txt
└── README.md
```

## Usage

### Training the Model

```bash
python src/train.py --data_dir data/ --epochs 50 --batch_size 32
```

### Running Inference on Images

```bash
python src/inference.py --image path/to/image.jpg
```

### Real-time Webcam Detection (Phase 2)

```bash
python src/webcam.py
```

## Data Requirements

- Place fire images in `data/fire/`
- Place non-fire (normal forest) images in `data/no_fire/`
- Supported formats: JPG, PNG
- Recommended minimum dataset size: 1000 images per class

## Model Architecture

The system uses a convolutional neural network (CNN) based on TensorFlow/Keras for image classification. The model is designed to be efficient enough for real-time processing while maintaining high accuracy.

## Performance Metrics

- Target accuracy: >80% on validation set
- Real-time processing: ≥1 FPS during webcam detection
- Metrics tracked: Accuracy, Precision, Recall, F1-score

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 