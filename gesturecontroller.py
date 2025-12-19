"""
Enhanced Gesture Controller with Kalman Filtering and Adaptive Thresholds
"""
import math
import numpy as np
from collections import deque
from typing import List, Tuple, Dict

class KalmanFilter:
    """Simple Kalman filter for smoothing"""
    def __init__(self, process_variance=1e-5, measurement_variance=0.1):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.posteri_estimate = 0.0
        self.posteri_error_estimate = 1.0
        
    def update(self, measurement):
        # Prediction
        priori_estimate = self.posteri_estimate
        priori_error_estimate = self.posteri_error_estimate + self.process_variance
        
        # Kalman gain
        kalman_gain = priori_error_estimate / (priori_error_estimate + self.measurement_variance)
        
        # Correction
        self.posteri_estimate = priori_estimate + kalman_gain * (measurement - priori_estimate)
        self.posteri_error_estimate = (1 - kalman_gain) * priori_error_estimate
        
        return self.posteri_estimate

class GestureController:
    def __init__(self, config=None, calibrator=None):
        self.config = config
        self.calibrator = calibrator
        
        # Kalman filters for smoothing
        self.kalman_x = KalmanFilter(process_variance=1e-5, measurement_variance=0.05)
        self.kalman_y = KalmanFilter(process_variance=1e-5, measurement_variance=0.05)
        
        # Enhanced gesture history for consistency
        self.gesture_history = deque(maxlen=7)
        self.finger_state_history = deque(maxlen=5)
        self.position_history = deque(maxlen=15)
        
        # Dynamic thresholds with adaptive learning
        self.dynamic_thresholds = {
            'thumb': 0.15,
            'index': 0.12,
            'middle': 0.12,
            'ring': 0.13,
            'pinky': 0.14,
            'pinch': 25,
            'fist': 0.7
        }
        
        # Hand size normalization
        self.base_hand_size = 100
        self.normalized_hand_size = 100
        self.use_hand_normalization = True
        self.hand_size_history = deque(maxlen=10)
        
        # Gesture state tracking
        self.current_gesture = "none"
        self.previous_gesture = "none"
        self.gesture_confidence = 0.0
        self.gesture_stable_frames = 0
        self.gesture_hold_frames = 0
        self.gesture_hold_threshold = 3
        
        # Position history for velocity calculation
        self.velocity = (0, 0)
        self.acceleration = (0, 0)
        
        # Continuous tracking
        self.tracking_active = False
        self.last_finger_positions = {}
        
        # Performance metrics
        self.detection_count = 0
        self.successful_detections = 0
        
        # Adaptive learning rates
        self.learning_rate = 0.1
        self.confidence_threshold = 0.65
        self.min_gesture_frames = 2
    
    def set_calibration_data(self, thresholds, base_hand_size):
        """Set calibration data from calibrator"""
        if thresholds:
            self.dynamic_thresholds.update(thresholds)
        self.base_hand_size = base_hand_size
        self.normalized_hand_size = base_hand_size
    
    def update_hand_normalization(self, landmarks):
        """Update hand size normalization dynamically"""
        if landmarks and len(landmarks) >= 10:
            wrist = landmarks[0]
            middle_mcp = landmarks[9]
            
            dx = wrist[1] - middle_mcp[1]
            dy = wrist[2] - middle_mcp[2]
            current_size = math.sqrt(dx*dx + dy*dy)
            
            # GUARD: prevent zero or negative hand size
            if current_size <= 0:
                current_size = max(self.normalized_hand_size, 1.0)
            
            # Add to history
            self.hand_size_history.append(current_size)
            
            # Calculate exponential moving average
            if len(self.hand_size_history) >= 3:
                recent_sizes = list(self.hand_size_history)
                weights = np.linspace(0.3, 1.0, len(recent_sizes))
                weights = weights / weights.sum()
                
                weighted_size = 0
                for i, size in enumerate(recent_sizes):
                    weighted_size += size * weights[i]
                
                # GUARD: ensure normalized_hand_size is never zero
                self.normalized_hand_size = max(weighted_size, 1.0)
            else:
                self.normalized_hand_size = max(current_size, 1.0)
    
    def calculate_hand_size(self, landmarks):
        """Calculate normalized hand size"""
        if len(landmarks) < 10:
            return self.normalized_hand_size
        
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        
        dx = wrist[1] - middle_mcp[1]
        dy = wrist[2] - middle_mcp[2]
        
        hand_size = math.sqrt(dx*dx + dy*dy)
        
        # Update normalization
        self.update_hand_normalization(landmarks)
        
        # Calibrator removed - use dynamic normalization only
        return hand_size
    
    def get_normalized_distance(self, distance):
        """Normalize distance based on hand size"""
        # GUARD: prevent division by zero
        if self.use_hand_normalization and self.normalized_hand_size > 0:
            return distance / self.normalized_hand_size * 100
        # Fallback: if no hand size, return raw distance
        return distance
    
    def get_smooth_position(self, landmarks):
        """Get smoothed position of index finger tip"""
        if not landmarks or len(landmarks) < 9:
            return None
        
        index_tip = next((p for p in landmarks if p[0] == 8), None)
        if not index_tip:
            return None
        
        raw_x, raw_y = index_tip[1], index_tip[2]
        
        # GUARD: validate coordinates are finite numbers
        if not (np.isfinite(raw_x) and np.isfinite(raw_y)):
            return None
        
        # Apply Kalman filtering with adaptive measurement variance
        # Adaptive: trust recent frames more if many are available
        adaptive_variance = max(0.01, min(0.2, 1.0 / max(len(self.position_history), 1)))
        self.kalman_x.measurement_variance = adaptive_variance
        self.kalman_y.measurement_variance = adaptive_variance
        
        smooth_x = self.kalman_x.update(raw_x)
        smooth_y = self.kalman_y.update(raw_y)
        
        # GUARD: validate smoothed output
        if not (np.isfinite(smooth_x) and np.isfinite(smooth_y)):
            return None
        
        # Store position for velocity calculation
        current_time = len(self.position_history)
        self.position_history.append((smooth_x, smooth_y, current_time))
        
        # Calculate velocity and acceleration if we have enough history
        if len(self.position_history) >= 3:
            positions = list(self.position_history)[-3:]
            
            # Velocity (pixels per frame)
            dt = positions[2][2] - positions[1][2]
            if dt > 0:
                vx = (positions[2][0] - positions[1][0]) / dt
                vy = (positions[2][1] - positions[1][1]) / dt
                self.velocity = (vx, vy)
            
            # Acceleration
            if len(self.position_history) >= 4:
                positions = list(self.position_history)[-4:]
                dt1 = positions[2][2] - positions[1][2]
                dt2 = positions[3][2] - positions[2][2]
                
                if dt1 > 0 and dt2 > 0:
                    vx1 = (positions[2][0] - positions[1][0]) / dt1
                    vy1 = (positions[2][1] - positions[1][1]) / dt1
                    vx2 = (positions[3][0] - positions[2][0]) / dt2
                    vy2 = (positions[3][1] - positions[2][1]) / dt2
                    
                    ax = (vx2 - vx1) / (dt1 + dt2) * 2 if (dt1 + dt2) > 0 else 0
                    ay = (vy2 - vy1) / (dt1 + dt2) * 2 if (dt1 + dt2) > 0 else 0
                    self.acceleration = (ax, ay)
        
        return (int(smooth_x), int(smooth_y))
    
    def is_finger_up(self, landmarks, finger_name, tip_id, pip_id, mcp_id=None):
        """Enhanced finger detection with adaptive thresholds"""
        if len(landmarks) <= max(tip_id, pip_id, mcp_id if mcp_id else 0):
            return False
        
        tip = landmarks[tip_id]
        pip = landmarks[pip_id]
        
        # Calculate vertical difference (inverted for OpenCV coordinates)
        y_diff = pip[2] - tip[2]  # Higher Y = lower on screen
        
        # Get dynamic threshold for this finger
        threshold_ratio = self.dynamic_thresholds.get(finger_name, 0.12)
        
        # Calculate threshold based on current hand size
        hand_size = self.calculate_hand_size(landmarks)
        threshold = hand_size * threshold_ratio
        
        # Check if tip is significantly above PIP
        if y_diff > threshold * 0.8:  # 80% of threshold is enough
            # Additional check for angle if MCP is available
            if mcp_id and mcp_id < len(landmarks):
                mcp = landmarks[mcp_id]
                
                # Calculate vectors
                tip_to_pip = np.array([tip[1] - pip[1], tip[2] - pip[2]])
                mcp_to_pip = np.array([mcp[1] - pip[1], mcp[2] - pip[2]])
                
                # Calculate angle
                dot_product = tip_to_pip[0] * mcp_to_pip[0] + tip_to_pip[1] * mcp_to_pip[1]
                mag_tip_pip = np.linalg.norm(tip_to_pip)
                mag_mcp_pip = np.linalg.norm(mcp_to_pip)
                
                if mag_tip_pip > 0 and mag_mcp_pip > 0:
                    cos_angle = dot_product / (mag_tip_pip * mag_mcp_pip)
                    cos_angle = max(-1.0, min(1.0, cos_angle))
                    angle = np.arccos(cos_angle)
                    
                    # Finger is up if angle > 30 degrees and tip is above PIP
                    return angle > np.pi/6 and y_diff > threshold * 0.5
            
            return y_diff > threshold
        
        return False
    
    def fingers_state(self, landmarks):
        """Return enhanced finger states with confidence"""
        if not landmarks or len(landmarks) < 21:
            return {}
        
        finger_configs = [
            ('thumb', 4, 3, 2),
            ('index', 8, 6, 5),
            ('middle', 12, 10, 9),
            ('ring', 16, 14, 13),
            ('pinky', 20, 18, 17)
        ]
        
        states = {}
        confidences = {}
        
        for name, tip_id, pip_id, mcp_id in finger_configs:
            is_up = self.is_finger_up(landmarks, name, tip_id, pip_id, mcp_id)
            states[name] = is_up
            
            # Calculate confidence based on distance from threshold
            if len(landmarks) > max(tip_id, pip_id):
                tip = landmarks[tip_id]
                pip = landmarks[pip_id]
                y_diff = pip[2] - tip[2]
                hand_size = self.calculate_hand_size(landmarks)
                threshold = hand_size * self.dynamic_thresholds.get(name, 0.12)
                
                # Confidence is higher when further from threshold
                if is_up:
                    confidence = min(1.0, y_diff / (threshold * 1.5))
                else:
                    confidence = min(1.0, (threshold - y_diff) / threshold)
                
                confidences[name] = confidence
        
        # Store in history
        self.finger_state_history.append((states, confidences))
        
        # Apply consistency filter over multiple frames
        if len(self.finger_state_history) >= 3:
            recent_frames = list(self.finger_state_history)[-3:]
            
            # Count occurrences of each state
            state_counts = {}
            confidence_sum = {}
            
            for frame_states, frame_confidences in recent_frames:
                for finger in states.keys():
                    state = frame_states.get(finger, False)
                    confidence = frame_confidences.get(finger, 0.0)
                    
                    if finger not in state_counts:
                        state_counts[finger] = {'up': 0, 'down': 0}
                        confidence_sum[finger] = 0.0
                    
                    if state:
                        state_counts[finger]['up'] += 1
                    else:
                        state_counts[finger]['down'] += 1
                    
                    confidence_sum[finger] += confidence
            
            # Apply majority voting with confidence weighting
            consistent_states = {}
            for finger in states.keys():
                up_count = state_counts[finger]['up']
                down_count = state_counts[finger]['down']
                avg_confidence = confidence_sum[finger] / 3
                
                # Require at least 2 frames agreement
                if up_count >= 2 and avg_confidence > 0.6:
                    consistent_states[finger] = True
                elif down_count >= 2 and avg_confidence > 0.6:
                    consistent_states[finger] = False
                else:
                    # Keep original if not consistent
                    consistent_states[finger] = states[finger]
            
            return consistent_states
        
        return states
    
    def is_pinch(self, landmarks, threshold_multiplier=1.0):
        """Detect pinch with adaptive threshold"""
        if len(landmarks) < 9:
            return False
        
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        # Calculate Euclidean distance
        dx = thumb_tip[1] - index_tip[1]
        dy = thumb_tip[2] - index_tip[2]
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Get normalized distance
        normalized_distance = self.get_normalized_distance(distance)
        
        # Dynamic threshold
        base_threshold = self.dynamic_thresholds.get('pinch', 25)
        threshold = base_threshold * threshold_multiplier
        
        # Additional check: thumb and index should be the only extended fingers
        states = self.fingers_state(landmarks)
        thumb_up = states.get('thumb', False)
        index_up = states.get('index', False)
        other_fingers_down = not states.get('middle', False) and \
                            not states.get('ring', False) and \
                            not states.get('pinky', False)
        
        return normalized_distance < threshold and thumb_up and index_up and other_fingers_down
    
    def is_fist(self, landmarks):
        """Enhanced fist detection"""
        if len(landmarks) < 21:
            return False
        
        states = self.fingers_state(landmarks)
        
        # All fingers should be down
        fingers_down = all(not state for state in states.values())
        
        if fingers_down:
            # Additional check: fingers should be close to palm
            wrist = landmarks[0]
            
            # Calculate average distance of fingertips from wrist
            tip_distances = []
            for tip_id in [4, 8, 12, 16, 20]:
                if tip_id < len(landmarks):
                    tip = landmarks[tip_id]
                    dist = math.sqrt(
                        (tip[1] - wrist[1])**2 + 
                        (tip[2] - wrist[2])**2
                    )
                    tip_distances.append(dist)
            
            if tip_distances:
                avg_distance = sum(tip_distances) / len(tip_distances)
                
                # Compare with reference hand size
                hand_size = self.calculate_hand_size(landmarks)
                
                # Fist ratio threshold from dynamic thresholds
                fist_ratio = self.dynamic_thresholds.get('fist', 0.7)
                
                return avg_distance < hand_size * fist_ratio
        
        return False
    
    def is_pointing(self, landmarks):
        """Detect pointing gesture (only index finger up)"""
        states = self.fingers_state(landmarks)
        
        index_up = states.get('index', False)
        others_down = not states.get('thumb', False) and \
                     not states.get('middle', False) and \
                     not states.get('ring', False) and \
                     not states.get('pinky', False)
        
        return index_up and others_down
    
    def is_open_hand(self, landmarks):
        """Detect open hand (all fingers up)"""
        states = self.fingers_state(landmarks)
        return all(states.values())
    
    def is_scissors(self, landmarks):
        """Detect scissors gesture (index and middle up, others down)"""
        states = self.fingers_state(landmarks)
        
        index_up = states.get('index', False)
        middle_up = states.get('middle', False)
        others_down = not states.get('thumb', False) and \
                     not states.get('ring', False) and \
                     not states.get('pinky', False)
        
        return index_up and middle_up and others_down
    
    def get_hand_gesture(self, landmarks):
        """Get comprehensive hand gesture with confidence"""
        self.detection_count += 1
        
        if not landmarks or len(landmarks) < 21:
            self.gesture_history.append("no_hand")
            return "no_hand", 0.0
        
        # Update hand normalization
        self.update_hand_normalization(landmarks)
        
        # Get finger states
        states = self.fingers_state(landmarks)
        
        # Check for specific gestures in priority order
        gestures = []
        
        # 1. Fist (highest priority - overrides other gestures)
        if self.is_fist(landmarks):
            gestures.append(("fist", 0.95))
        
        # 2. Open hand
        if all(states.values()):
            gestures.append(("open_hand", 0.9))
        
        # 3. Pointing
        if self.is_pointing(landmarks):
            gestures.append(("pointing", 0.85))
        
        # 4. Pinch
        if self.is_pinch(landmarks, 0.8):
            gestures.append(("pinch", 0.8))
        
        # 5. Scissors
        if self.is_scissors(landmarks):
            gestures.append(("scissors", 0.75))
        
        # Select gesture with highest confidence
        if gestures:
            gesture, confidence = max(gestures, key=lambda x: x[1])
            self.successful_detections += 1
        else:
            gesture, confidence = "other", 0.3
        
        # Apply temporal consistency
        self.gesture_history.append(gesture)
        
        if len(self.gesture_history) >= self.min_gesture_frames:
            recent_gestures = list(self.gesture_history)[-self.min_gesture_frames:]
            
            # Count occurrences
            gesture_counts = {}
            for g in recent_gestures:
                gesture_counts[g] = gesture_counts.get(g, 0) + 1
            
            # Find most common gesture
            most_common = max(gesture_counts.items(), key=lambda x: x[1])
            
            # If consistent for required frames
            if most_common[1] >= self.min_gesture_frames - 1:
                if most_common[0] == gesture:
                    self.gesture_stable_frames += 1
                    confidence = min(confidence + 0.1 * self.gesture_stable_frames, 1.0)
                else:
                    self.gesture_stable_frames = 0
        
        # Update current gesture if confidence is high enough
        if confidence >= self.confidence_threshold:
            self.previous_gesture = self.current_gesture
            self.current_gesture = gesture
            self.gesture_confidence = confidence
            
            # Check for gesture hold
            if self.previous_gesture == self.current_gesture:
                self.gesture_hold_frames += 1
            else:
                self.gesture_hold_frames = 1
        else:
            self.gesture_hold_frames = 0
        
        return gesture, confidence
    
    def get_index_tip_position(self, landmarks):
        """Get position of index finger tip"""
        if not landmarks or len(landmarks) < 9:
            return None
        
        index_tip = landmarks[8]
        return (index_tip[1], index_tip[2])
    
    def get_hand_center(self, landmarks):
        """Get center of hand (average of key points)"""
        if not landmarks or len(landmarks) < 10:
            return None
        
        # Use wrist and palm points for stability
        points = [
            landmarks[0],  # Wrist
            landmarks[5],  # Index MCP
            landmarks[9],  # Middle MCP
            landmarks[13], # Ring MCP
            landmarks[17]  # Pinky MCP
        ]
        
        avg_x = sum(p[1] for p in points) / len(points)
        avg_y = sum(p[2] for p in points) / len(points)
        
        return (int(avg_x), int(avg_y))
    
    def get_gesture_for_drawing(self, landmarks):
        """Get gesture specifically for drawing control"""
        gesture, confidence = self.get_hand_gesture(landmarks)
        
        # Map to drawing actions
        drawing_gestures = {
            "pointing": "draw",
            "pinch": "draw_small",
            "fist": "stop",
            "open_hand": "clear",
            "scissors": "erase",
            "other": "none"
        }
        
        action = drawing_gestures.get(gesture, "none")
        
        # Only return if gesture is stable
        if confidence > self.confidence_threshold and self.gesture_hold_frames >= self.gesture_hold_threshold:
            return action, confidence
        else:
            return "none", confidence
    
    def is_gesture_stable(self, gesture_name=None, min_frames=3):
        """Check if gesture has been stable for minimum frames"""
        if gesture_name is None:
            gesture_name = self.current_gesture
        
        if len(self.gesture_history) < min_frames:
            return False
        
        recent = list(self.gesture_history)[-min_frames:]
        return all(g == gesture_name for g in recent) and gesture_name != "no_hand"
    
    def get_detection_accuracy(self):
        """Get detection accuracy percentage"""
        if self.detection_count == 0:
            return 0.0
        return (self.successful_detections / self.detection_count) * 100
    
    def reset(self):
        """Reset all filters and history"""
        self.kalman_x = KalmanFilter()
        self.kalman_y = KalmanFilter()
        self.gesture_history.clear()
        self.finger_state_history.clear()
        self.position_history.clear()
        self.hand_size_history.clear()
        self.current_gesture = "none"
        self.previous_gesture = "none"
        self.gesture_confidence = 0.0
        self.gesture_stable_frames = 0
        self.gesture_hold_frames = 0
        self.velocity = (0, 0)
        self.acceleration = (0, 0)
        self.detection_count = 0
        self.successful_detections = 0