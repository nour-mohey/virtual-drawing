"""
QUICK START: Minimal Example of Fixed Gesture-Drawing Pipeline

This example demonstrates the corrected pipeline with GestureDrawingController.
It's a minimal, self-contained script that shows how to use all components together.
"""

import cv2
import numpy as np
from handtracker import HandTracker
from gesturecontroller import GestureController
from drawingengine import DrawingEngine
from gesture_drawing_controller import GestureDrawingController


def main():
    """
    Minimal gesture-drawing demonstration.
    Run this to verify the fixes work end-to-end.
    """
    
    # Initialize components
    print("[Init] Creating components...")
    tracker = HandTracker()
    gesture_controller = GestureController()
    drawing_engine = DrawingEngine(h=720, w=1280)
    
    # Create the gesture-drawing bridge (THE KEY FIX)
    gesture_drawing_controller = GestureDrawingController(
        gesture_controller=gesture_controller,
        drawing_engine=drawing_engine
    )
    
    print("[Init] Opening camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Camera not available")
        return
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame")
                break
            
            frame_count += 1
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # 1. HAND TRACKING
            # ================
            results = tracker.process(frame, draw_landmarks=True)
            landmarks = tracker.get_landmarks_list(frame, 0)
            
            if not landmarks or len(landmarks) < 21:
                # No hand detected
                cv2.putText(frame, "No hand detected", (20, 50),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                # 2. GESTURE CLASSIFICATION + DRAWING (UNIFIED)
                # ============================================
                # Instead of:
                #   gesture, conf = gesture_controller.get_hand_gesture(landmarks)
                #   # ... manual stroke handling ...
                #
                # We do:
                result = gesture_drawing_controller.process_frame(landmarks)
                
                # result contains:
                # {
                #     'action': 'draw'|'stop'|'clear'|'erase'|'none',
                #     'position': (x, y) or None,
                #     'mode': 'brush'|'eraser'|'clear',
                #     'confidence': float,
                #     'stroke_active': bool
                # }
                
                gesture = gesture_drawing_controller._get_consensus_gesture()
                confidence = result['confidence']
                position = result['position']
                stroke_active = result['stroke_active']
                
                # 3. DISPLAY GESTURE INFO
                # ======================
                cv2.putText(frame, f"Gesture: {gesture.upper()}", (20, 50),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {confidence:.0%}", (20, 90),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(frame, f"Action: {result['action']}", (20, 130),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Draw status bar
                if stroke_active:
                    cv2.rectangle(frame, (20, 160), (120, 180), (0, 255, 0), -1)
                    cv2.putText(frame, "DRAWING", (30, 175),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    cv2.rectangle(frame, (20, 160), (120, 180), (100, 100, 100), 2)
                
                # Draw position cursor
                if position:
                    x, y = position
                    cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
                    cv2.circle(frame, (x, y), 8, (255, 255, 255), 2)
            
            # 4. DISPLAY CANVAS
            # =================
            # Merge drawing canvas with camera frame for visualization
            canvas = drawing_engine.merge_layers()
            
            # Side-by-side display
            h, w = frame.shape[:2]
            display = np.hstack([frame, canvas[:h, :w]])
            
            # Add labels
            cv2.putText(display, "Camera", (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, "Canvas", (w + 10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show display
            cv2.imshow("Gesture Drawing - Fixed Pipeline", display)
            
            # 5. KEYBOARD CONTROLS
            # ====================
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                print("[Exit] Quit requested")
                break
            elif key == ord('c'):
                print("[Action] Clearing canvas")
                drawing_engine.clear_all()
            elif key == ord('u'):
                print("[Action] Undo")
                drawing_engine.undo()
            elif key == ord('r'):
                print("[Action] Reset gesture controller")
                gesture_drawing_controller.reset()
            elif key == ord('s'):
                # Save drawing
                path = f"drawing_{frame_count}.png"
                if drawing_engine.save_image(path):
                    print(f"[Save] Drawing saved to {path}")
            elif key == ord('d'):
                # Print debug info
                status = gesture_drawing_controller.get_status()
                print(f"\n[Debug] Gesture Drawing Controller Status:")
                print(f"  Frames processed: {status['frames_processed']}")
                print(f"  Total strokes: {status['total_strokes']}")
                print(f"  Stroke active: {status['stroke_active']}")
                print(f"  Current gesture: {status['current_gesture']}")
                print(f"  Confidence: {status['average_confidence']:.2f}")
                print(f"  Flicker count: {status['flicker_count']}")
                print()
            
            # Print occasional stats
            if frame_count % 100 == 0:
                status = gesture_drawing_controller.get_status()
                print(f"[Stats] Frame {frame_count}: "
                      f"Strokes={status['total_strokes']}, "
                      f"Gesture={status['current_gesture']}, "
                      f"Conf={status['average_confidence']:.2f}, "
                      f"Active={status['stroke_active']}")
    
    except KeyboardInterrupt:
        print("\n[Exit] Interrupted by user")
    
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("[Cleanup] Releasing resources...")
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final stats
        if gesture_drawing_controller:
            status = gesture_drawing_controller.get_status()
            print(f"\n[Final Stats]")
            print(f"  Total frames: {status['frames_processed']}")
            print(f"  Total strokes: {status['total_strokes']}")
            print(f"  Flicker events: {status['flicker_count']}")


if __name__ == "__main__":
    print("=" * 60)
    print("Gesture-Drawing Pipeline - Fixed Version")
    print("=" * 60)
    print("\nControls:")
    print("  q - Quit")
    print("  c - Clear canvas")
    print("  u - Undo last action")
    print("  s - Save drawing")
    print("  r - Reset gesture state")
    print("  d - Print debug info")
    print("\nGestures:")
    print("  Pointing (index) → Draw")
    print("  Pinch (thumb+index) → Draw small")
    print("  Open hand (all fingers) → Clear")
    print("  Fist (closed hand) → Stop")
    print("=" * 60 + "\n")
    
    main()
