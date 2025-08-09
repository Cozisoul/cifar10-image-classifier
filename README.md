# CIFAR-10 Image Classifier with a Web UI

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.x-orange.svg)](https://www.gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository contains a complete project for building, training, and deploying a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset. The project includes the training notebook and a user-friendly web interface built with Gradio.

## Project Features

*   **Model Training:** A Jupyter Notebook (`train_classifier.ipynb`) that details the entire process of loading, preprocessing, building, training, and evaluating the CNN. The final model achieves **77.77% accuracy** on the test set.
*   **Interactive Web App:** A Gradio application (`app.py`) that loads the pre-trained model and allows users to upload their own images for instant classification.
*   **Robust CNN Architecture:** The model uses multiple convolutional blocks with MaxPooling and Dropout layers to effectively learn features and prevent overfitting.

## Tech Stack

*   **TensorFlow / Keras:** For building and training the deep learning model.
*   **Gradio:** For creating the interactive web UI.
*   **Python**, NumPy, Matplotlib

## How to Run the Application

**1. Get the Code**

First, get the code by cloning the repository or downloading it as a ZIP file.

**2. Create a Virtual Environment**

It is highly recommended to use a virtual environment to keep your project dependencies isolated.

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

**3. Install Dependencies**

Install all the required packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

**4. Run the Application**

Launch the Gradio app by running the following command:

```bash
python app.py
```

The application will be available at a local URL (usually `http://127.0.0.1:7860`).