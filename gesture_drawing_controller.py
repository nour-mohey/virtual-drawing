"""
Gesture-to-Drawing Bridge Controller
Robust state machine that connects hand gestures to drawing engine with:
- Temporal smoothing to eliminate gesture flicker
- Persistent drawing state (stroke_active)
- Clear gesture → drawing action mapping
- Safe coordinate handling
"""
import numpy as np
from collections import deque
from typing import Optional, Tuple


class GestureDrawingController:
    """
    Reliable bridge between GestureController and DrawingEngine.
    Implements temporal smoothing and persistent drawing state.
    """
    
    def __init__(self, gesture_controller, drawing_engine):
        """
        Initialize the gesture-drawing bridge.
        
        Args:
            gesture_controller: GestureController instance for gesture classification
            drawing_engine: DrawingEngine instance for drawing operations
        """
        self.gesture_controller = gesture_controller
        self.drawing_engine = drawing_engine
        
        # === PERSISTENT DRAWING STATE ===
        self.stroke_active = False  # Core flag: is an active stroke in progress?
        self.last_draw_position = None  # Position of last draw call
        self.last_draw_time = 0
        self.stroke_start_time = 0
        self.min_stroke_duration = 0.05  # Min 50ms before ending stroke
        
        # === GESTURE SMOOTHING ===
        # Prevent gesture flickering by requiring temporal consensus
        self.gesture_history = deque(maxlen=5)  # Last 5 frames
        self.confidence_history = deque(maxlen=5)
        self.drawing_action_history = deque(maxlen=3)  # More aggressive smoothing
        
        # === DRAWING MODE TRACKING ===
        self.current_mode = 'brush'  # 'brush', 'eraser', 'clear'
        self.mode_confirmed = False  # Mode only changes after multi-frame consensus
        
        # === SAFETY THRESHOLDS ===
        self.min_confidence_threshold = 0.70  # Require 70%+ confidence
        self.min_frames_for_action = 2  # Require 2 consecutive frames of same gesture
        self.gesture_history_size = 5
        
        # === POSITION SMOOTHING ===
        # Even with stable gestures, position can jitter
        self.position_buffer = deque(maxlen=3)  # Last 3 positions
        
        # === DEBUG/STATS ===
        self.total_strokes_drawn = 0
        self.total_frames_processed = 0
        self.flickering_detected = 0
    
    def process_frame(self, landmarks: list, frame_time: float = 0.0) -> dict:
        """
        Process landmarks to decide drawing action.
        
        Returns:
            {
                'action': 'draw'|'stop'|'clear'|'none',
                'position': (x, y) or None,
                'mode': 'brush'|'eraser'|'clear',
                'confidence': float,
                'stroke_active': bool
            }
        """
        self.total_frames_processed += 1
        
        if not landmarks or len(landmarks) < 21:
            # No hand detected - end any active stroke
            if self.stroke_active:
                self._end_stroke()
            return {
                'action': 'none',
                'position': None,
                'mode': self.current_mode,
                'confidence': 0.0,
                'stroke_active': False
            }
        
        # ========== STEP 1: CLASSIFY GESTURE ==========
        gesture, confidence = self.gesture_controller.get_hand_gesture(landmarks)
        self.gesture_history.append(gesture)
        self.confidence_history.append(confidence)
        
        # ========== STEP 2: GET DRAWING POSITION ==========
        # Use smooth position from GestureController (already Kalman-filtered)
        position = self.gesture_controller.get_smooth_position(landmarks)
        
        if position is None:
            # Fallback to index tip
            try:
                index_tip = self.gesture_controller.get_index_tip_position(landmarks)
                position = index_tip
            except Exception:
                position = None
        
        # GUARD: validate position
        if position:
            try:
                pos_x, pos_y = int(position[0]), int(position[1])
                # Check bounds (assuming 1280x720 canvas, but be safe)
                if pos_x < 0 or pos_y < 0:
                    position = None
            except (TypeError, ValueError):
                position = None
        
        # ========== STEP 3: TEMPORAL SMOOTHING - ELIMINATE FLICKERING ==========
        # Require gesture to be stable across multiple frames
        consensus_gesture = self._get_consensus_gesture()
        confidence_score = self._get_average_confidence()
        
        # ========== STEP 4: MAP GESTURE TO DRAWING ACTION ==========
        drawing_action = self._gesture_to_action(consensus_gesture, confidence_score)
        self.drawing_action_history.append(drawing_action)
        
        # ========== STEP 5: EXECUTE DRAWING LOGIC ==========
        action_taken = self._execute_drawing_action(
            drawing_action, 
            position, 
            confidence_score,
            frame_time
        )
        
        return {
            'action': action_taken,
            'position': position,
            'mode': self.current_mode,
            'confidence': confidence_score,
            'stroke_active': self.stroke_active
        }
    
    def _get_consensus_gesture(self) -> str:
        """
        Get gesture with temporal smoothing.
        Requires at least 2 of last 3 frames to match.
        """
        if len(self.gesture_history) < 2:
            return "none"
        
        # Count occurrences in recent history
        recent = list(self.gesture_history)[-3:]
        gesture_counts = {}
        for g in recent:
            gesture_counts[g] = gesture_counts.get(g, 0) + 1
        
        # Find most common gesture
        most_common_gesture = max(gesture_counts.items(), key=lambda x: x[1])
        gesture, count = most_common_gesture
        
        # Only return if appears in at least 2 of 3 frames (66% consensus)
        if count >= 2:
            return gesture
        
        # Fallback to most recent gesture if no consensus
        return recent[-1] if recent else "none"
    
    def _get_average_confidence(self) -> float:
        """Get average confidence over recent frames"""
        if not self.confidence_history:
            return 0.0
        
        # Exponential weighted average (more weight to recent)
        weights = np.exp(np.linspace(-1, 0, len(self.confidence_history)))
        weights = weights / weights.sum()
        
        avg = sum(c * w for c, w in zip(self.confidence_history, weights))
        return float(avg)
    
    def _gesture_to_action(self, gesture: str, confidence: float) -> str:
        """
        Map gesture to drawing action with confidence threshold.
        
        Gesture → Action mapping:
        - 'pointing' → 'draw'       (index extended)
        - 'pinch'    → 'draw_small' (precision drawing)
        - 'open_hand' → 'clear'     (all fingers up)
        - 'fist'     → 'stop'       (close hand)
        - others     → 'none'
        """
        # Only proceed if confidence is high enough
        if confidence < self.min_confidence_threshold:
            return 'none'
        
        # Map gesture → action
        gesture_action_map = {
            'pointing': 'draw',
            'pinch': 'draw_small',
            'open_hand': 'clear',
            'fist': 'stop',
            'scissors': 'erase',
            'no_hand': 'none',
            'other': 'none'
        }
        
        action = gesture_action_map.get(gesture, 'none')
        return action
    
    def _execute_drawing_action(self, action: str, position: Optional[Tuple[int, int]], 
                                confidence: float, frame_time: float) -> str:
        """
        Execute drawing engine commands based on action.
        Manages stroke lifecycle: start → continue → end
        """
        current_time = frame_time if frame_time > 0 else 0
        
        if action == 'draw' or action == 'draw_small':
            # ===== DRAWING MODE =====
            if not self.stroke_active:
                # START NEW STROKE
                if position:
                    self._start_stroke(position)
            else:
                # CONTINUE EXISTING STROKE
                if position and self.last_draw_position:
                    # Only update if position has moved (avoid redundant draws)
                    dx = abs(position[0] - self.last_draw_position[0])
                    dy = abs(position[1] - self.last_draw_position[1])
                    
                    if dx > 1 or dy > 1:  # Moved more than 1 pixel
                        self.drawing_engine.continue_stroke(position[0], position[1])
                        self.last_draw_position = position
            
            return 'draw'
        
        elif action == 'stop':
            # ===== STOP / END STROKE =====
            if self.stroke_active:
                self._end_stroke()
            
            return 'stop'
        
        elif action == 'erase':
            # ===== ERASER MODE =====
            # Set eraser mode in drawing engine if not already
            if self.current_mode != 'eraser':
                self.drawing_engine.set_mode('eraser')
                self.current_mode = 'eraser'
            
            # Draw with eraser
            if not self.stroke_active and position:
                self._start_stroke(position)
            elif self.stroke_active and position:
                self.drawing_engine.continue_stroke(position[0], position[1])
                self.last_draw_position = position
            
            return 'erase'
        
        elif action == 'clear':
            # ===== CLEAR CANVAS =====
            if self.stroke_active:
                self._end_stroke()
            
            # Clear only if gesture is held stable
            if self._get_consensus_gesture() == 'open_hand':
                self.drawing_engine.clear_active_layer()
            
            return 'clear'
        
        else:  # action == 'none'
            # ===== NO ACTION =====
            if self.stroke_active:
                # End stroke if confidence drops
                self._end_stroke()
            
            return 'none'
    
    def _start_stroke(self, position: Tuple[int, int]):
        """Start a new drawing stroke"""
        if not position:
            return
        
        try:
            x, y = int(position[0]), int(position[1])
            
            # Ensure brush mode is set
            if self.current_mode != 'brush':
                self.drawing_engine.set_mode('brush')
                self.current_mode = 'brush'
            
            # Start stroke in drawing engine
            self.drawing_engine.start_stroke(x, y)
            
            # Update state
            self.stroke_active = True
            self.last_draw_position = (x, y)
            self.stroke_start_time = 0  # Would be set to current time if timing is available
            self.total_strokes_drawn += 1
            
            print(f"[Gesture] Stroke started at ({x}, {y})")
            
        except Exception as e:
            print(f"[ERROR] Failed to start stroke: {e}")
            self.stroke_active = False
    
    def _end_stroke(self):
        """End current drawing stroke"""
        try:
            if self.stroke_active:
                # End stroke in drawing engine
                self.drawing_engine.end_stroke()
                self.stroke_active = False
                self.last_draw_position = None
                
                print(f"[Gesture] Stroke ended. Total strokes: {self.total_strokes_drawn}")
        
        except Exception as e:
            print(f"[ERROR] Failed to end stroke: {e}")
            self.stroke_active = False
    
    def detect_flicker(self) -> bool:
        """Detect if gesture is flickering (changing every frame)"""
        if len(self.gesture_history) < 3:
            return False
        
        recent = list(self.gesture_history)[-3:]
        # Flickering if all 3 gestures are different
        if len(set(recent)) == 3:
            self.flickering_detected += 1
            return True
        
        return False
    
    def get_status(self) -> dict:
        """Get controller status for debugging"""
        return {
            'frames_processed': self.total_frames_processed,
            'total_strokes': self.total_strokes_drawn,
            'stroke_active': self.stroke_active,
            'current_mode': self.current_mode,
            'current_gesture': self._get_consensus_gesture(),
            'average_confidence': self._get_average_confidence(),
            'flicker_count': self.flickering_detected,
            'gesture_history': list(self.gesture_history),
            'position': self.last_draw_position
        }
    
    def reset(self):
        """Reset all state"""
        self.stroke_active = False
        self.last_draw_position = None
        self.gesture_history.clear()
        self.confidence_history.clear()
        self.drawing_action_history.clear()
        self.position_buffer.clear()
        self.current_mode = 'brush'
        self.mode_confirmed = False
