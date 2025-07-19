"""
Camera capture module for real-time image input.
Handles webcam or video file capture, with basic frame processing.
"""

import cv2
import numpy as np
import time
from .floor_plan_processor import FloorPlanProcessor  # Import for homography

class CameraCapture:
    def __init__(self, source=0, frame_rate=30, resolution=(640, 480)):
        """
        Initialize camera capture.
        source: 0 for webcam, or path to video file.
        """
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.frame_rate = frame_rate
        self.last_frame_time = 0

    def get_frame(self):
        """
        Get a single frame, respecting frame rate.
        Returns: RGB image or None if failed.
        """
        current_time = time.time()
        if current_time - self.last_frame_time < 1.0 / self.frame_rate:
            time.sleep(1.0 / self.frame_rate - (current_time - self.last_frame_time))
        
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to capture frame.")
            return None
        
        # Convert BGR to RGB for consistency
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.last_frame_time = time.time()
        return frame

    def release(self):
        """Release the camera resource."""
        self.cap.release()