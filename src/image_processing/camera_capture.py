"""
Camera capture module for real-time image input.
Handles webcam or video file capture, with basic frame processing.
"""

import cv2
import numpy as np
import time
import threading
import queue  # For thread-safe frame queue

from .floor_plan_processor import FloorPlanProcessor  # For homography if needed

class CameraCapture:
    def __init__(self, source=0, frame_rate=30, resolution=(320, 240)):  # Lower res for speed
        """
        Initialize camera capture.
        source: 0 for webcam, or path to video file.
        """
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera at index {source}. Check permissions or index.")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.frame_rate = frame_rate
        self.last_frame_time = 0
        
        # Threading setup
        self.frame_queue = queue.Queue(maxsize=1)  # Latest frame only
        self.stop_thread = threading.Event()
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()

    def _capture_loop(self):
        """Background thread to grab frames."""
        while not self.stop_thread.is_set():
            ret, frame = self.cap.read()
            if ret:
                # print("Debug: Frame captured successfully.")  # Debug: Confirm read works
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with self.frame_queue.mutex:  # Drop old frames
                    self.frame_queue.queue.clear()
                self.frame_queue.put(frame)
            else:
                print("Debug: cap.read() failed - no frame.")  # Debug: If failing, why?
            time.sleep(1.0 / self.frame_rate)  # Respect rate

    def get_frame(self):
        """
        Get the latest frame from queue (with short timeout to wait for first frame).
        Returns: RGB image or None if failed after wait.
        """
        try:
            frame = self.frame_queue.get(timeout=0.5)  # Increased to 0.5s for startup lag
            print("Debug: Got frame from queue.")  # Debug: Confirm queue pop
            return frame
        except queue.Empty:
            print("Debug: Queue empty - no frame yet.")  # Debug msg
            return None

    def release(self):
        """Release the camera resource."""
        self.stop_thread.set()
        self.capture_thread.join()
        self.cap.release()