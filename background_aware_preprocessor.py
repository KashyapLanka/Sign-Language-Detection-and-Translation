import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from dataclasses import dataclass


@dataclass
class PreprocessResult:
    success: bool
    roi: np.ndarray = None
    landmarks: np.ndarray = None


class BackgroundAwarePreprocessor:
    def __init__(self, target_size=(160,160)):
        self.target_size = target_size

        self.detector = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
                num_hands=1
            )
        )

    def __call__(self, frame_rgb):
        h, w, _ = frame_rgb.shape

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.detector.detect(mp_image)

        if not result.hand_landmarks:
            return PreprocessResult(False)

        hand = result.hand_landmarks[0]

        xs = [lm.x * w for lm in hand]
        ys = [lm.y * h for lm in hand]

        x1, x2 = int(min(xs)), int(max(xs))
        y1, y2 = int(min(ys)), int(max(ys))

        # padding
        pad = int(0.4 * max(x2 - x1, y2 - y1))
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        # safety check
        if x2 <= x1 or y2 <= y1:
            return PreprocessResult(False)

        roi = frame_rgb[y1:y2, x1:x2]

        if roi.size == 0:
            return PreprocessResult(False)

        # make square (important)
        size = max(roi.shape[:2])
        square = np.zeros((size, size, 3), dtype=roi.dtype)
        square[:roi.shape[0], :roi.shape[1]] = roi

        roi = cv2.resize(square, self.target_size)

        landmarks = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand],
            dtype=np.float32
        )

        return PreprocessResult(True, roi, landmarks)