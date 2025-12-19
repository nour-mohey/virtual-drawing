"""
Enhanced utility functions for PyQt5
"""
import cv2
import numpy as np
from PyQt6.QtGui import QImage, QPixmap

def cv2_to_qpixmap(frame_bgr):
    """Convert BGR OpenCV image to QPixmap for display."""
    if frame_bgr is None or frame_bgr.size == 0:
        # Create a black placeholder
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        blank[:] = (0, 0, 0)
        frame_bgr = blank
    
    try:
        # Ensure valid image format
        if len(frame_bgr.shape) == 2:
            # Convert grayscale to BGR
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
        elif frame_bgr.shape[2] == 4:
            # Remove alpha channel if present
            frame_bgr = frame_bgr[:, :, :3]
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        
        # Create QImage (PyQt6 format)
        q_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(q_image)
        
    except Exception as e:
        print(f"Error converting image to QPixmap: {e}")
        # Return a black placeholder
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        q_image = QImage(blank.data, 320, 240, 3*320, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(q_image)

def resize_with_aspect_ratio(image, target_width=None, target_height=None):
    """Resize image while maintaining aspect ratio."""
    if image is None or image.size == 0:
        # Create a blank image
        if target_width and target_height:
            return np.zeros((target_height, target_width, 3), dtype=np.uint8)
        else:
            return np.zeros((240, 320, 3), dtype=np.uint8)
    
    try:
        h, w = image.shape[:2]
        
        if target_width and target_height:
            # Both dimensions specified
            return cv2.resize(image, (target_width, target_height))
        elif target_width:
            # Width specified, calculate height
            ratio = target_width / w
            new_height = int(h * ratio)
            return cv2.resize(image, (target_width, new_height))
        elif target_height:
            # Height specified, calculate width
            ratio = target_height / h
            new_width = int(w * ratio)
            return cv2.resize(image, (new_width, target_height))
        else:
            return image
            
    except Exception as e:
        print(f"Error resizing image: {e}")
        return image

def preprocess_for_hand_tracking(frame):
    """Preprocess frame for better hand tracking."""
    if frame is None or frame.size == 0:
        return frame
    
    try:
        # Apply CLAHE for better contrast
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Apply bilateral filter for noise reduction
        filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        return filtered
        
    except Exception as e:
        print(f"Error preprocessing frame: {e}")
        return frame

def draw_hand_info(frame, landmarks, gesture, confidence, fps):
    """Draw hand information on frame."""
    if frame is None:
        return frame
    
    try:
        # Draw gesture information
        if gesture != "no_hand" and confidence > 0:
            cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence:.0%}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Draw hand landmarks if available
        if landmarks and len(landmarks) >= 21:
            # Draw hand center
            center_x = sum(lm[1] for lm in landmarks) // len(landmarks)
            center_y = sum(lm[2] for lm in landmarks) // len(landmarks)
            cv2.circle(frame, (center_x, center_y), 10, (255, 0, 0), -1)
        
        return frame
        
    except Exception as e:
        print(f"Error drawing hand info: {e}")
        return frame

def create_blank_image(width, height, color=(255, 255, 255)):
    """Create a blank image with specified color."""
    try:
        blank = np.ones((height, width, 3), dtype=np.uint8)
        blank[:] = color
        return blank
    except Exception as e:
        print(f"Error creating blank image: {e}")
        return np.ones((720, 1280, 3), dtype=np.uint8) * 255