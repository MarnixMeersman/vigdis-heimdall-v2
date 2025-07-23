import cv2
import mediapipe as mp
from typing import List, Tuple

class HandTracker:
    """Lightweight hand tracking wrapper around MediaPipe Hands."""

    def __init__(self) -> None:
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def detect(self, frame) -> List[Tuple[int, int]]:
        """Return (x, y) image coordinates for detected hands."""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        points: List[Tuple[int, int]] = []
        if results.multi_hand_landmarks:
            h, w, _ = frame.shape
            for hand_landmarks in results.multi_hand_landmarks:
                lm = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x = int(lm.x * w)
                y = int(lm.y * h)
                points.append((x, y))
        return points
