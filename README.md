# Counterfeit Money Detection System

## Project Overview

This project develops a machine learning system to detect counterfeit banknotes using image processing and classification techniques. The system extracts features from banknote images and classifies them as either "Real" or "Counterfeit" using a K-Nearest Neighbors (KNN) model. A Streamlit web application is provided for an interactive demonstration.

## Features

* **Image Preprocessing**: Resizing and color space conversion.
* **Feature Extraction**: Utilizes Histogram of Oriented Gradients (HOG) for shape and texture features, and Color Histograms (HSV) for color distribution.
* **Machine Learning Models**: Implements and evaluates both Support Vector Machine (SVM) and K-Nearest Neighbors (KNN) classifiers.
* **Hyperparameter Tuning**: Uses GridSearchCV to find optimal parameters for KNN.
* **Model Persistence**: Saves the trained model and scaler for later use.
* **Interactive Web Application**: A Streamlit app allows users to upload custom images or select sample images for real-time predictions.

## Technologies Used

* **Python 3.x**
* **Machine Learning**: `scikit-learn` (for SVM, KNN, StandardScaler)
* **Image Processing**: `opencv-python` (OpenCV), `scikit-image` (for HOG), `Pillow` (PIL)
* **Data Handling**: `numpy`
* **Model Persistence**: `joblib`
* **Web Application**: `streamlit`

## Project Structure

Your project directory (`cmd-1`) should have a structure similar to this:
