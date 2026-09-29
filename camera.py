"""Camera management utilities."""

import cv2


def open_camera(idx):
    """Open a camera by index, trying DirectShow first on Windows.

    Returns the VideoCapture object, or None if the camera cannot be opened.
    """
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap
