"""
Professional Hand Tracker - Optimized for Gesture Recognition
High-resolution processing with robust detection and smooth tracking
Compatible with GestureController and GestureCalibrator
"""
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from typing import List, Tuple, Optional

class HandTracker:
    """
    Professional hand tracker optimized for gesture recognition.
    
    Features:
    - High-resolution processing (1920x1080 capture, 1280x720 processing)
    - Robust landmark smoothing and validation
    - Professional detection with adaptive thresholds
    - Seamless integration with GestureController
    - Optimized for real-time performance
    """
    
    def __init__(self, config=None):
        self.config = config
        
        # MediaPipe setup - optimized for single hand, high accuracy
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Focus on single hand for better performance
            min_detection_confidence=0.6,  # Lower threshold for better detection
            min_tracking_confidence=0.6,   # Lower threshold for continuous tracking
            model_complexity=1  # Balance between accuracy and speed
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Enhanced smoothing buffers - larger for professional tracking
        self.smoothing_buffer_size = 7  # Increased for smoother tracking
        self.landmark_buffer = deque(maxlen=self.smoothing_buffer_size)
        self.position_buffer = deque(maxlen=self.smoothing_buffer_size)
        
        # Processing resolution - higher for better detection
        self.processing_width = 1280  # High resolution for accurate detection
        self.processing_height = 720
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.last_time = 0
        
        # Detection state
        self.hand_detected = False
        self.detection_confidence = 0.0
        self.last_valid_landmarks = None
        
        # Tracking stability
        self.stability_threshold = 12
        self.hand_stability_counter = 0
        self.consecutive_failures = 0
        self.max_failures = 3
        
        # Results storage
        self.results = None
        
    def process(self, frame: np.ndarray, draw_landmarks: bool = True, 
                draw_connections: bool = True) -> Optional[mp.solutions.hands.Hands]:
        """
        Process frame with high-resolution hand tracking.
        
        Args:
            frame: BGR frame from camera
            draw_landmarks: Whether to draw landmarks on frame
            draw_connections: Whether to draw hand connections
            
        Returns:
            MediaPipe Hands results or None if error
        """
        if frame is None or frame.size == 0:
            self.hand_detected = False
            return None
        
        # Ensure frame is valid
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            self.hand_detected = False
            return None
        
        try:
            h, w = frame.shape[:2]
            
            # Process at high resolution for better accuracy
            # Maintain aspect ratio
            aspect_ratio = w / h
            if w != self.processing_width:
                # Resize to processing resolution
                new_w = self.processing_width
                new_h = int(new_w / aspect_ratio)
                frame_resized = cv2.resize(frame, (new_w, new_h), 
                                         interpolation=cv2.INTER_LINEAR)
            else:
                frame_resized = frame.copy()
                new_w, new_h = w, h
            
            # Convert to RGB for MediaPipe
            img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img_rgb.flags.writeable = False
            
            # Process with MediaPipe
            self.results = self.hands.process(img_rgb)
            img_rgb.flags.writeable = True
            
            # Check if hand is detected
            self.hand_detected = bool(self.results.multi_hand_landmarks)
            
            if self.hand_detected:
                self.consecutive_failures = 0
            else:
                self.consecutive_failures += 1
            
            # Scale landmarks back to original frame size if needed
            if self.hand_detected and draw_landmarks and self.results.multi_hand_landmarks:
                scale_x = w / new_w
                scale_y = h / new_h
                
                for hand_landmarks in self.results.multi_hand_landmarks:
                    # Scale landmarks to original frame coordinates
                    for landmark in hand_landmarks.landmark:
                        landmark.x *= scale_x
                        landmark.y *= scale_y
                    
                    # Draw landmarks on original frame
                    if draw_landmarks:
                        self.mp_draw.draw_landmarks(
                            frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS if draw_connections else None,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style()
                        )
                        
                        # Draw larger circles on fingertips for visibility
                        for landmark_id in [4, 8, 12, 16, 20]:  # Fingertips
                            landmark = hand_landmarks.landmark[landmark_id]
                            x = int(landmark.x * w)
                            y = int(landmark.y * h)
                            cv2.circle(frame, (x, y), 12, (0, 255, 0), -1)
                            cv2.circle(frame, (x, y), 12, (0, 0, 0), 2)
            
            return self.results
            
        except Exception as e:
            self.hand_detected = False
            self.consecutive_failures += 1
            print(f"Error in hand tracking: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_landmarks_list(self, frame: np.ndarray, hand_index: int = 0) -> List[Tuple[int, int, int]]:
        """
        Get smoothed landmarks for specified hand with professional processing.
        
        Returns landmarks in format: [(index, x, y), ...] where:
        - index: landmark ID (0-20)
        - x: pixel x-coordinate
        - y: pixel y-coordinate (OpenCV convention: increases downward)
        
        This format is compatible with GestureController and GestureCalibrator.
        """
        if (not hasattr(self, 'results') or not self.results or 
            not self.results.multi_hand_landmarks or 
            hand_index >= len(self.results.multi_hand_landmarks)):
            self.hand_stability_counter = max(0, self.hand_stability_counter - 1)
            # Return last valid landmarks if available (temporal continuity)
            if self.last_valid_landmarks and self.consecutive_failures < self.max_failures:
                return self.last_valid_landmarks
            return []
        
        try:
            hand = self.results.multi_hand_landmarks[hand_index]
            h, w = frame.shape[:2]
            
            # Extract current landmarks in (index, x, y) format
            current_landmarks = []
            for i, lm in enumerate(hand.landmark):
                # Convert normalized coordinates to pixel coordinates
                x = max(0, min(int(lm.x * w), w - 1))
                y = max(0, min(int(lm.y * h), h - 1))
                current_landmarks.append((i, x, y))
            
            # Validate landmarks (check for reasonable values)
            if not self._validate_landmarks(current_landmarks, w, h):
                if self.last_valid_landmarks and self.consecutive_failures < self.max_failures:
                    return self.last_valid_landmarks
                return []
            
            # Add to buffer for smoothing
            self.landmark_buffer.append(current_landmarks)
            
            # Apply weighted temporal smoothing (exponential moving average)
            if len(self.landmark_buffer) >= 2:
                smoothed_landmarks = []
                # Exponential weights (more weight to recent frames)
                weights = np.exp(np.linspace(-2, 0, len(self.landmark_buffer)))
                weights = weights / weights.sum()
                
                for i in range(21):  # 21 landmarks per hand
                    avg_x = 0.0
                    avg_y = 0.0
                    
                    for j, landmarks in enumerate(self.landmark_buffer):
                        if i < len(landmarks):
                            weight = weights[j]
                            avg_x += landmarks[i][1] * weight
                            avg_y += landmarks[i][2] * weight
                    
                    avg_x = int(avg_x)
                    avg_y = int(avg_y)
                    
                    # Apply bounds checking
                    avg_x = max(0, min(w - 1, avg_x))
                    avg_y = max(0, min(h - 1, avg_y))
                    
                    smoothed_landmarks.append((i, avg_x, avg_y))
                
                # Check stability
                if len(self.landmark_buffer) >= 3:
                    recent_frames = list(self.landmark_buffer)[-3:]
                    stable = self.check_landmark_stability(recent_frames)
                    if stable:
                        self.hand_stability_counter += 1
                    else:
                        self.hand_stability_counter = max(0, self.hand_stability_counter - 1)
                
                # Store as last valid landmarks
                self.last_valid_landmarks = smoothed_landmarks
                return smoothed_landmarks
            
            # If not enough frames for smoothing, return current
            self.last_valid_landmarks = current_landmarks
            return current_landmarks
            
        except Exception as e:
            print(f"Error getting landmarks: {e}")
            import traceback
            traceback.print_exc()
            # Return last valid if available
            if self.last_valid_landmarks:
                return self.last_valid_landmarks
            return []
    
    def _validate_landmarks(self, landmarks: List[Tuple[int, int, int]], 
                           frame_width: int, frame_height: int) -> bool:
        """
        Validate landmarks for reasonableness.
        Checks for out-of-bounds, NaN, or invalid values.
        """
        if not landmarks or len(landmarks) != 21:
            return False
        
        try:
            for idx, x, y in landmarks:
                # Check index
                if idx < 0 or idx > 20:
                    return False
                
                # Check coordinates are valid numbers
                if not (0 <= x < frame_width) or not (0 <= y < frame_height):
                    return False
                
                # Check for NaN or Inf
                if not (np.isfinite(x) and np.isfinite(y)):
                    return False
            
            # Check that key landmarks are present
            required_indices = [0, 4, 8, 12, 16, 20]  # Wrist and fingertips
            present_indices = {lm[0] for lm in landmarks}
            if not all(idx in present_indices for idx in required_indices):
                return False
            
            return True
            
        except Exception:
            return False
    
    def check_landmark_stability(self, landmark_frames: List[List[Tuple[int, int, int]]], 
                                 threshold: int = 12) -> bool:
        """
        Check if landmarks are stable across frames.
        Lower threshold = more sensitive to movement.
        """
        if len(landmark_frames) < 2:
            return False
        
        try:
            # Compare first and last frame
            frame1 = landmark_frames[0]
            frame2 = landmark_frames[-1]
            
            if len(frame1) != len(frame2) or len(frame1) != 21:
                return False
            
            distances = []
            for (i1, x1, y1), (i2, x2, y2) in zip(frame1, frame2):
                if i1 == i2:
                    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    distances.append(distance)
            
            if distances:
                avg_distance = np.mean(distances)
                max_distance = np.max(distances)
                # Consider stable if average movement is low and max movement is reasonable
                return avg_distance < threshold and max_distance < threshold * 2
            
            return False
            
        except Exception:
            return False
    
    def get_index_tip(self, frame: np.ndarray, hand_index: int = 0) -> Optional[Tuple[int, int]]:
        """
        Get enhanced smoothed position of index finger tip.
        Returns (x, y) pixel coordinates.
        """
        landmarks = self.get_landmarks_list(frame, hand_index)
        
        if not landmarks or len(landmarks) < 9:
            return None
        
        try:
            # Get index tip (landmark 8)
            index_tip = next((p for p in landmarks if p[0] == 8), None)
            
            if not index_tip:
                return None
            
            current_pos = (index_tip[1], index_tip[2])
            
            # Add to position buffer
            self.position_buffer.append(current_pos)
            
            # Apply weighted moving average for smooth tracking
            if len(self.position_buffer) >= 2:
                weights = np.exp(np.linspace(-2, 0, len(self.position_buffer)))
                weights = weights / weights.sum()
                
                avg_x = 0.0
                avg_y = 0.0
                for i, (x, y) in enumerate(self.position_buffer):
                    avg_x += x * weights[i]
                    avg_y += y * weights[i]
                
                return (int(avg_x), int(avg_y))
            
            return current_pos
            
        except Exception as e:
            print(f"Error getting index tip: {e}")
            return None
    
    def get_hand_center(self, frame: np.ndarray, hand_index: int = 0) -> Optional[Tuple[int, int]]:
        """
        Get center of palm for better stability.
        Uses wrist and MCP joints for robust center calculation.
        """
        landmarks = self.get_landmarks_list(frame, hand_index)
        
        if not landmarks or len(landmarks) < 10:
            return None
        
        try:
            # Use wrist (0) and middle MCP (9) for stable center
            wrist = landmarks[0]  # (0, x, y)
            middle_mcp = landmarks[9]  # (9, x, y)
            
            center_x = (wrist[1] + middle_mcp[1]) // 2
            center_y = (wrist[2] + middle_mcp[2]) // 2
            
            return (center_x, center_y)
            
        except Exception as e:
            print(f"Error getting hand center: {e}")
            return None
    
    def is_hand_stable(self, threshold: int = 12) -> bool:
        """
        Check if hand position is stable based on recent movement.
        """
        if len(self.position_buffer) < 3:
            return False
        
        try:
            # Calculate variance of recent positions
            positions = list(self.position_buffer)[-3:]
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            
            var_x = np.var(xs)
            var_y = np.var(ys)
            
            # Stable if variance is low
            return var_x < threshold and var_y < threshold
            
        except Exception:
            return False
    
    def get_hand_stability(self) -> float:
        """
        Get hand stability score (0-100).
        Higher score = more stable tracking.
        """
        if self.hand_stability_counter == 0:
            return 0.0
        
        # Cap at 10 frames for stability measurement
        stability_frames = min(self.hand_stability_counter, 10)
        return (stability_frames / 10.0) * 100.0
    
    def calibrate(self, landmarks: List[Tuple[int, int, int]]):
        """
        Set calibration reference based on current hand size.
        Compatible with GestureCalibrator.
        """
        if landmarks and len(landmarks) >= 21:
            try:
                # Calculate hand size as distance from wrist to middle finger MCP
                wrist = landmarks[0]  # (0, x, y)
                middle_mcp = landmarks[9]  # (9, x, y)
                
                dx = wrist[1] - middle_mcp[1]
                dy = wrist[2] - middle_mcp[2]
                self.hand_size_reference = np.sqrt(dx*dx + dy*dy)
            except Exception as e:
                print(f"Error in calibration: {e}")
    
    def reset_buffers(self):
        """Reset smoothing buffers and tracking state."""
        self.landmark_buffer.clear()
        self.position_buffer.clear()
        self.hand_stability_counter = 0
        self.consecutive_failures = 0
        self.last_valid_landmarks = None
    
    def get_detection_status(self) -> dict:
        """
        Get current detection status for debugging.
        """
        return {
            'hand_detected': self.hand_detected,
            'stability_score': self.get_hand_stability(),
            'consecutive_failures': self.consecutive_failures,
            'buffer_size': len(self.landmark_buffer),
            'has_valid_landmarks': self.last_valid_landmarks is not None
        }
