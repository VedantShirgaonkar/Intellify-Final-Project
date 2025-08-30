import cv2
import mediapipe as mp
import numpy as np
import joblib
import math

# ============== Utility Functions ==============

def normalize_landmarks(landmarks):
    """
    Normalize landmarks like during training:
    - Translate so wrist is at origin.
    - Scale by wrist → middle finger MCP distance.
    """
    wrist = landmarks[0]
    translated = [(lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z) for lm in landmarks]

    scale = math.sqrt(translated[9][0]**2 + translated[9][1]**2 + translated[9][2]**2)
    if scale < 1e-6:
        scale = 1.0

    normalized = [(x/scale, y/scale, z/scale) for (x, y, z) in translated]
    return [coord for triple in normalized for coord in triple]

def empty_hand():
    return [0.0] * (21 * 3)

# ============== Load Model ==============
MODEL_FILE = "gesture_model.pkl"
model = joblib.load(MODEL_FILE)

# ============== Mediapipe Setup ==============
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

# ============== Webcam Loop ==============
cap = cv2.VideoCapture(0)
print("[INFO] Starting webcam... Press 'q' to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    left_hand, right_hand = None, None

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                              results.multi_handedness):
            label = handedness.classification[0].label  # "Left" or "Right"
            score = handedness.classification[0].score

            if score < 0.7:
                continue  # skip low-confidence

            norm = normalize_landmarks(hand_landmarks.landmark)

            if label == "Left":
                left_hand = norm
            else:
                right_hand = norm

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if left_hand is None:
        left_hand = empty_hand()
    if right_hand is None:
        right_hand = empty_hand()

    features = np.array([left_hand + right_hand])

    # Only predict if at least one hand is visible
    if any(v != 0.0 for v in features[0]):
        pred = model.predict(features)[0]

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features).max()
            text = f"Gesture: {pred} ({proba:.2f})"
        else:
            text = f"Gesture: {pred}"

        cv2.putText(frame, text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
