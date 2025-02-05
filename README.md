# Emotion Detection using ResNet34 and OpenCV

## Overview
This project implements an emotion detection system using a ResNet34-based convolutional neural network. The model is trained on grayscale facial images and detects seven emotions: **Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise**. It also includes a real-time webcam-based emotion detection system using OpenCV.

## Features
- Data preprocessing and augmentation using `ImageDataGenerator`
- Custom-built ResNet34 model for emotion classification
- Training with class weights to handle class imbalance
- Model evaluation with confusion matrix and classification report
- Real-time emotion detection using OpenCV and Haarcascade

## Project Structure
```
Emotion-Detection-ResNet34/
│── dataset/                      # Folder to store dataset (add a README for instructions)
│── models/                        # Stores trained models (add .gitignore for large files)
│── notebooks/                     # Jupyter Notebooks for experimentation
│── src/                           # Source code
│   ├── train.py                   # Training script
│   ├── model.py                   # ResNet34 model definition
│   ├── dataset_loader.py          # Data preprocessing & augmentation
│   ├── inference.py               # Script for live webcam inference
│── requirements.txt               # List of dependencies
│── README.md                      # Project documentation
│── .gitignore                     # To ignore unnecessary files (dataset, large model files)
```

## Installation
### 1. Clone the Repository
```sh
git clone https://github.com/your-username/Emotion-Detection-ResNet34.git
cd Emotion-Detection-ResNet34
```

### 2. Install Dependencies
```sh
pip install -r requirements.txt
```

## Training the Model
Run the following command to train the model:
```sh
python src/train.py
```
The trained model will be saved in the `models/` directory.

## Running Real-Time Emotion Detection
After training, you can use the webcam-based emotion detection system:
```sh
python src/inference.py
```
Press **'q'** to exit the webcam feed.

## Dataset
- The dataset should be structured as follows:
  ```
  dataset/
  ├── train/
  │   ├── angry/
  │   ├── disgust/
  │   ├── fear/
  │   ├── happy/
  │   ├── neutral/
  │   ├── sad/
  │   ├── surprise/
  ├── test/
  ```
- Make sure to place your dataset inside the `dataset/` folder.

## Model Performance
- The model is trained for 300 epochs with an early stopping criterion.
- Achieves 64% accuracy on the validation set.
- Provides a confusion matrix and classification report for evaluation.

## Dependencies
- TensorFlow
- OpenCV
- NumPy
- Matplotlib
- Pandas
- Scikit-learn


## License
This project is open-source and available under the MIT License.

---
Developed by **Pradeesh**

