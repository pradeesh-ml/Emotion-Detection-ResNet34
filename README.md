# Emotion Detection using ResNet34 and OpenCV

## Overview
This project implements an emotion detection system using a ResNet34-based convolutional neural network. The model is trained on grayscale facial images from the FER-2013 dataset and detects seven emotions:  **Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise**. It also includes a real-time webcam-based emotion detection system using OpenCV.

## Features
- Data preprocessing and augmentation using `ImageDataGenerator`
- Custom-built ResNet34 model for emotion classification
- Training with class weights to handle class imbalance
- Model evaluation with confusion matrix and classification report
- Real-time emotion detection using OpenCV and Haarcascade

## Project Structure
```
Emotion-Detection-ResNet34/
│── models/                        
│── notebooks/                     
│── src/                           
│   ├── train.py                   
│   ├── model.py                   
│   ├── inference.py
│── requirements.txt
│── .gitignore       
│── README.md                      

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

## Dataset - FER-2013
- The model is trained on the FER-2013 (Facial Expression Recognition 2013) dataset, which consists of 35,887 grayscale images of 48x48 pixels. The dataset is structured as follows:
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

