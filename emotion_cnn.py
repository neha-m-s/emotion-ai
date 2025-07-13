import cv2
import numpy as np
from keras.models import load_model

# Load the pretrained model
model = load_model("emotion_model.hdf5", compile=False)

# FER2013 emotions
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load OpenCV face detector
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_emotion_from_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (64, 64), interpolation=cv2.INTER_AREA)
        
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float32') / 255.0
            roi = np.expand_dims(roi, axis=-1)  # (64, 64, 1)
            roi = np.expand_dims(roi, axis=0)   # (1, 64, 64, 1)

            prediction = model.predict(roi, verbose=0)[0]
            label = emotion_labels[prediction.argmax()]
            return label
    return "No Face"
