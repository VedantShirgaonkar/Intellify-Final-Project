import cv2
import numpy as np
import os
import mediapipe as mp

# --- REUSE YOUR MEDIAPIPE FUNCTIONS ---
# Copy the mediapipe_detection and extract_keypoints functions from your
# first notebook and paste them here.
mp_holistic = mp.solutions.holistic

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh   = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh   = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# --- CONFIGURATION ---
VIDEO_FOLDER = 'videos' # The main folder where your class subfolders are

# --- GET CLASS NAMES AND CREATE LABEL MAP ---
# This automatically finds the classes based on the folders you downloaded
class_names = sorted([d for d in os.listdir(VIDEO_FOLDER) if os.path.isdir(os.path.join(VIDEO_FOLDER, d))])
label_map = {name: i for i, name in enumerate(class_names)}

sequences, labels = [], []

# --- MAIN PROCESSING LOOP ---
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # Loop through each class folder (e.g., 'videos/book')
    for class_name in class_names:
        class_path = os.path.join(VIDEO_FOLDER, class_name)
        label = label_map[class_name]
        print(f"Processing class: {class_name}")

        # Loop through each video in the class folder
        for video_file in os.listdir(class_path):
            video_path = os.path.join(class_path, video_file)
            
            cap = cv2.VideoCapture(video_path)
            frame_landmarks = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Make detections
                _, results = mediapipe_detection(frame, holistic)
                
                # Extract and append keypoints
                keypoints = extract_keypoints(results)
                frame_landmarks.append(keypoints)
            
            cap.release()

            # Add the sequence of landmarks and its label to our lists
            sequences.append(frame_landmarks)
            labels.append(label)

print("\n--- Video processing complete. Saving data... ---")

# Save the processed data to .npy files
np.save('sequences.npy', np.array(sequences, dtype=object))
np.save('labels.npy', labels)

print("Data saved successfully as 'sequences.npy' and 'labels.npy'.")
print("You are now ready for Step 4: Padding and Training.")