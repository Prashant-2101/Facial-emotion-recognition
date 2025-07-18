# Emotion Recognition from Images

This repository contains a Jupyter notebook (`g6-old (5).ipynb`) implementing an emotion recognition system using traditional image processing and machine learning techniques. The project analyzes facial images to classify emotions such as Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

## Project Overview

The goal of this project is to detect and classify human emotions from facial images using both traditional image processing techniques and machine learning models. The notebook processes images from the `IMG-FER-SKY` dataset, extracts color channel features (RGB, HSV, LAB), and trains models like k-Nearest Neighbors (kNN), Random Forest, and Convolutional Neural Networks (CNN) for emotion classification.

## Dataset

The dataset used is the `IMG-FER-SKY` dataset, hosted on Kaggle. It contains facial images organized into train and test directories, with subdirectories for each emotion:
- **Train Directory**: `/kaggle/input/smith-image/IMG-FER-SKY/train`
- **Test Directory**: `/kaggle/input/smith-image/IMG-FER-SKY/test`
- **Emotions**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

The dataset includes a count of images per emotion for both training and testing sets, as shown in the notebook's data collection section.

## Methodology

### 1. Data Collection
- The notebook lists the number of images per emotion category in the training and testing datasets using Python's `os` module and `pandas` for visualization.

### 2. Traditional Image Processing
- **Color Channels Analyzed**:
  - RGB: Red, Green, Blue channels
  - HSV: Hue, Saturation, Value channels
  - LAB: Lightness (L), A, and B channels
- For each channel, histograms are generated to compare pixel value distributions between two sample images (one labeled "Angry" and one labeled "Disgust").
- OpenCV (`cv2`) is used for image loading and color space conversion, and Matplotlib is used for plotting histograms.

### 3. Machine Learning Models
The notebook implements and evaluates three machine learning models for emotion classification:
- **k-Nearest Neighbors (kNN)**:
  - Features: Concatenated pixel values from RGB, HSV, and LAB channels.
  - Model: `KNeighborsClassifier` with `n_neighbors=3`.
  - Evaluation: Accuracy and classification report.
- **Random Forest**:
  - Features: Same as kNN.
  - Model: `RandomForestClassifier` with `n_estimators=100`.
  - Evaluation: Accuracy and classification report.
- **Convolutional Neural Network (CNN)**:
  - Architecture: Custom CNN with multiple Conv2D, MaxPooling2D, BatchNormalization, Dropout, and Dense layers.
  - Data Preprocessing: Images are resized to 48x48 pixels and normalized using `ImageDataGenerator`.
  - Training: 50 epochs with early stopping and model checkpointing to save the best weights (`model_weights.h5`).
  - Evaluation: Accuracy, confusion matrix, and classification report.
- **VGG16 (Transfer Learning)**:
  - Architecture: Pre-trained VGG16 model with a custom top layer for 7-class classification.
  - Training: Similar setup as the custom CNN, with weights saved to `vgg16_emodel.h5`.
  - Evaluation: Confusion matrix and classification report on the test set.

### 4. Evaluation
- The notebook evaluates model performance using accuracy, F1 score, classification reports, and confusion matrices.
- For the CNN and VGG16 models, predictions are visualized for sample test images.

## Dependencies

To run the notebook, ensure you have the following Python libraries installed:
```bash
pip install pandas numpy opencv-python matplotlib scikit-learn tensorflow scikit-image seaborn
```

Additionally, the notebook requires a Kaggle environment with GPU support (e.g., NVIDIA Tesla T4) and the `IMG-FER-SKY` dataset available at `/kaggle/input/smith-image/IMG-FER-SKY`.

## Results

- **Traditional Image Processing**: Histograms show differences in pixel value distributions across color channels for different emotions, indicating potential discriminative features.
- **Machine Learning Models**:
  - kNN and Random Forest provide baseline performance using handcrafted features.
  - The custom CNN and VGG16 models achieve better performance due to their ability to learn complex features from raw images.
- **Visualizations**: Confusion matrices and sample predictions are provided for the CNN and VGG16 models, highlighting their performance across emotion classes.

## Future Work

- Explore additional feature extraction techniques (e.g., SIFT, HOG) to enhance traditional models.
- Fine-tune hyperparameters for kNN and Random Forest models.
- Experiment with other pre-trained models (e.g., ResNet, EfficientNet) for improved accuracy.
- Augment the dataset with more diverse images to improve model robustness.

## Acknowledgments

- The `IMG-FER-SKY` dataset is sourced from Kaggle.
- Built with Python, OpenCV, TensorFlow, and scikit-learn.
