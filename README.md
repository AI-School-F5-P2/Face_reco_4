# Face_reco_4
Face recognition project, team_4


## Description

This script is designed for a facial recognition application that performs data augmentation, face recognition, and captures positive images (👍) and anchor images (⚓) for training.

## Train_main.py

The primary purpose of Train_main.py is to train the facial recognition model. It leverages the face_recognition library to process images, extract facial features, and train a K-nearest neighbors (KNN) classifier. The trained model is then saved for later use.

## FaceRecogKnn.py

This script focuses on real-time facial recognition using a pre-trained KNN classifier. It utilizes the webcam feed to identify faces, displaying bounding boxes around recognized faces along with their names and accuracy percentages.

## streamlit.py

streamlit.py serves as the entry point for the user interface. It employs the Streamlit framework to create an interactive web application. Users can toggle webcam usage, adjust detection confidence, and upload images for face recognition. The recognized faces are highlighted, and additional information is displayed.

## Supporting Scripts:

- data_augmentation.py: This script augments positive and anchor images for training. It applies various transformations, enhancing the diversity of the dataset.

- webcam.py: Designed for image collection, this script allows users to capture positive and anchor images using a webcam. Positive images are captured by pressing 'p', while anchor images are captured by pressing 'a'.

## Prerequisites

- Python 3.x
- TensorFlow
- OpenCV
- face_recognition
- NumPy
- Matplotlib
- Streamlit
- python-dotenv

## Installation

Clone the repository: https://github.com/AI-School-F5-P2/Face_reco_4.git

Install the required packages:pip install -r requirements.txt

## Usage

Run the data augmentation script: python data_augmentation.py

Train the KNN classifier: python knn_train.py

Run the face recognition app: streamlit run face_recognition_app.py
- Use the sidebar to toggle webcam usage and set detection confidence.
- Upload an image using the file uploader for face recognition.

Capture positive and anchor images using the webcam script: python webcam_capture.py
- Press 'p' to capture positive images.
- Press 'a' to capture anchor images.
- Press 'q' to exit the program.