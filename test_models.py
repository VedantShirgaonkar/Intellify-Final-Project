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
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             )
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )


def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([ lh, rh])


model = tf.keras.models.load_model('models/main_less_actions_model.h5')
print('model loaded')
label_map = np.load('models/label_map2.npy', allow_pickle=True).item()
print('label map loaded')
actions = list(label_map.keys())

# Variables for prediction
sequence = []
sequence_length = 30
threshold = 0.7

# Start webcam / video
cap = cv.VideoCapture('./college.mp4')

print('Starting inference...')

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for consistency
        frame = cv.resize(frame, (640, 480))

        print('Frame resized')

        # Detection
        image, results = mediapipe_detection(frame, holistic)


        # Draw landmarks
        draw_styled_landmarks(image, results)

        # Extract keypoints (same as training)
        keypoints = extract_keypoints(results)
        print('Keypoints extracted')

        # Append to sequence
        sequence.append(keypoints)
        sequence = sequence[-sequence_length:]

        if len(sequence) == sequence_length:
            input_seq = np.expand_dims(sequence, axis=0)   # shape (1, 30, 1530)

            # Predict
            res = model.predict(input_seq, verbose=0)[0]
            print(f"Prediction result: {res}, Shape: {res.shape}")
            predicted_action = actions[np.argmax(res)]
            confidence = np.max(res)

            # Show prediction if above threshold
            if confidence > threshold:
                cv.putText(image, f'{predicted_action}: {confidence:.2f}',
                           (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Show probabilities
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

cap.release()
cv.destroyAllWindows()
