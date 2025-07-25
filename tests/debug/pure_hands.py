# hand_debug.py
# Simple webcam-based hand coordinate extractor using MediaPipe for debugging

# Requirements:
# pip install opencv-python mediapipe

import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open default webcam (device 0)
cap = cv2.VideoCapture(0)

print("Starting hand detection. Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting.")
        break

    # Flip image for mirror view (optional)
    frame = cv2.flip(frame, 1)

    # Convert to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    hand_coords = []  # list of lists: each hand's landmark coords
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = []
            h, w, _ = frame.shape
            for lm in hand_landmarks.landmark:
                # Convert normalized coordinates to pixel values
                x, y = int(lm.x * w), int(lm.y * h)
                lm_list.append((x, y))
            hand_coords.append(lm_list)
            # Draw landmarks on the frame
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Print coordinates for all detected hands
    if hand_coords:
        for idx, coords in enumerate(hand_coords):
            print(f"Hand {idx+1} landmarks:")
            for i, (x, y) in enumerate(coords):
                print(f"  Landmark {i}: (x={x}, y={y})")
    else:
        print("No hands detected")

    # Show annotated frame
    cv2.imshow('Hand Debug', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
