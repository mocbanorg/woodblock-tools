from __future__ import annotations

import cv2
import numpy as np


def _estimate_board_angle(image: np.ndarray, threshold_deskew=170, kernel_size=15, 
                          connectivity=8) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, 0, threshold_deskew)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=connectivity)

    if num_labels <= 1:
        return 0.0

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    board_mask = np.zeros_like(mask)
    board_mask[labels == largest] = 255
    contours, _ = cv2.findContours(board_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0.0

    contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(contour)
    (_, _), (w, h), angle = rect

    if w < h:
        angle += 90.0
    while angle >= 45.0:
        angle -= 90.0
    while angle < -45.0:
        angle += 90.0
    return float(angle)


def _rotate_expand(image: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += new_w / 2.0 - w / 2.0
    M[1, 2] += new_h / 2.0 - h / 2.0

    rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),)
    return rotated, M


def global_deskew(image: np.ndarray, threshold_deskew=170, kernel_size=15,
                  connectivity=8, min_angle=0.3) -> tuple[np.ndarray, float, np.ndarray]:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be np.ndarray")
    if image.ndim not in (2, 3):
        raise ValueError("image must be 2D or 3D")

    angle = _estimate_board_angle(image, threshold_deskew, kernel_size, connectivity)

    if abs(angle) < min_angle:
        identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        return image.copy(), angle, identity
    correction = angle
    rotated, affine_matrix = _rotate_expand(image, correction)
    return rotated, angle, affine_matrix