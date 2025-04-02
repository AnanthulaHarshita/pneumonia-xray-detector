# Pneumonia X-ray Detector

This project provides a backend service for detecting pneumonia from chest X-ray images using a trained deep learning model.

## Project Status

- Model: Trained and saved
- Flask API: Completed
- Frontend: Not yet implemented

## Overview

The system uses a convolutional neural network to classify chest X-ray images as:
- Positive (Pneumonia detected)
- Negative (No pneumonia)

The model was trained on a publicly available dataset from Kaggle.

## Getting Started

### 1. Create and activate a virtual environment (Python 3.9+)

python3.9 -m venv venv  
source venv/bin/activate

### 2. Install dependencies

pip install -r requirements.txt

### 3. Run the Flask app

python app.py

### 4. Send a test request

curl -X POST -F "file=@test_xray.jpg" http://127.0.0.1:5000/predict

## API Response Format

The API will return a JSON response like this:

{
  "is_pneumonia": "Yes" or "No",
  "confidence": "87.3%"
}

## Model Info

- Format: .keras
- Epochs: 15
- Final Accuracy: 93.84%
- Final Loss: 0.1625

## Project Structure

XRAY/
├── app.py  
├── pneumonia_model.keras  
├── requirements.txt  
├── README.md  
└── test_xray.jpg (sample input)

## Notes

The frontend (web interface) is not yet implemented and will be added later.

## Contact

Feel free to reach out for help or suggestions.
EOF
