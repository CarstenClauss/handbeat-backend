from typing import Optional, Tuple

import numpy as np
import cv2 as cv

from .constants import HAND_CLASSIFIER_FILE

__all__ = [
    'HandDetector',
    'CascadeClassifierHandDetector'
]


class HandDetector:
    def __call__(self, img: np.ndarray) -> np.ndarray:
        return self.detect(img)

    def detect(self, img: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        raise NotImplementedError


class CascadeClassifierHandDetector(HandDetector):
    def __init__(self, min_area: int = 10000, alpha: float = 0.7, classifier_file: str = HAND_CLASSIFIER_FILE):
        self.classifier = cv.CascadeClassifier(classifier_file)
        self.min_area = min_area
        self.alpha = alpha
        self._hand_left: Optional[np.ndarray] = None
        self._hand_right: Optional[np.ndarray] = None

    def filter(self, hand_left, hand_right):
        if hand_left is None:
            self._hand_left = None
        elif self._hand_left is None:
            self._hand_left = hand_left
        else:
            self._hand_left = self.alpha * hand_left + (1 - self.alpha) * self._hand_left

        if hand_right is None:
            self._hand_right = None
        elif self._hand_right is None:
            self._hand_right = hand_right
        else:
            self._hand_right = self.alpha * hand_right + (1 - self.alpha) * self._hand_right

        return self._hand_left, self._hand_right

    def detect(self, img: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        img_shape = np.array(img.shape)

        rects = self.classifier.detectMultiScale(
            img,
            flags=cv.CASCADE_DO_CANNY_PRUNING | cv.CASCADE_FIND_BIGGEST_OBJECT
        )

        if len(rects) == 0:
            return self.filter(None, None)

        filtered_rects = rects[(rects[:, 2] * rects[:, 3]) >= self.min_area]

        if len(filtered_rects) < 1:
            return self.filter(None, None)

        centers = filtered_rects[:, :2] + filtered_rects[:, 2:] // 2

        horizontally_sorted_rects = centers[np.argsort(centers[:, 1])]

        hand_left = horizontally_sorted_rects[0].astype(np.float32)

        hand_right: np.ndarray = None
        if horizontally_sorted_rects.shape[0] > 1:
            hand_right = horizontally_sorted_rects[-1].astype(np.float32)

        if hand_left is not None:
            hand_left /= img_shape[:2][::-1]
            hand_left[0] = 1.0 - hand_left[0]
        if hand_right is not None:
            hand_right /= img_shape[:2][::-1]
            hand_right[0] = 1.0 - hand_right[0]

        return self.filter(hand_left, hand_right)
