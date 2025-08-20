import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, TimeDistributed, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from datetime import datetime
from tqdm import tqdm

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def mediapipe_detection(image, model):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    return image, results


def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             )
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             )
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
    )


def extract_keypoints(results):
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([face, lh, rh])

model = tf.keras.models.load_model('models/main_wsl_model.h5')
label_map = np.load('models/label_map.npy', allow_pickle=True).item()

print("----------------------------Model loaded successfully.")
actions = list(label_map.keys())

# Variables for prediction
sequence = []
sequence_length = 30
threshold = 0.7

# Start webcam / video
cap = cv.VideoCapture('D:\\Intellify_hackathon\\b.webm')

print("Starting inference...--------------------------")

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    print("Mediapipe Holistic model initialized.")
    while cap.isOpened():
        print("Reading frame...")
        ret, frame = cap.read()
        print(ret)
        if not ret:
            break

        # Resize for consistency
        frame = cv.resize(frame, (640, 480))

        # Detection
        image, results = mediapipe_detection(frame, holistic)

        # Draw landmarks
        draw_styled_landmarks(image, results)

        # Extract keypoints (same as training)
        keypoints = extract_keypoints(results)

        # Append to sequence
        sequence.append(keypoints)
        sequence = sequence[-sequence_length:]

        if len(sequence) == sequence_length:
            input_seq = np.expand_dims(sequence, axis=0)   # shape (1, 30, 1530)
            print("Processing sequence...---------------")

            # Predict
            res = model.predict(input_seq, verbose=0)[0]
            predicted_action = actions[np.argmax(res)]
            confidence = np.max(res)

            # Show prediction if above threshold
            if confidence > threshold:
                print(f'Predicted: {predicted_action} with confidence {confidence:.2f}')    
                cv.putText(image, f'{predicted_action}: {confidence:.2f}',
                           (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Show probabilities``
            for i, (action, prob) in enumerate(zip(actions, res)):
                y_pos = 100 + i * 30
                cv.rectangle(image, (10, y_pos), (int(prob * 300) + 10, y_pos + 25), (0, 255, 0), -1)
                cv.putText(image, f'{action}: {prob:.2f}', (15, y_pos + 18),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Show output
        cv.imshow('ASL Inference', image)

        # Quit
        if cv.waitKey(10) & 0xFF == ord('q'):
            break


