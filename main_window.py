"""
Enhanced Main Window - Virtual Drawing Studio Pro
Complete working version with improved hand tracking and modern UI
"""
import sys
import os
import cv2
import numpy as np
import time
import json
import traceback
from datetime import datetime
from collections import deque

from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

# Import custom modules
try:
    from config_manager import ConfigManager
    from drawingengine import DrawingEngine
    from gesturecontroller import GestureController
    from handtracker import HandTracker
    from viewport_manager import ViewportManager
    from utils import cv2_to_qpixmap, resize_with_aspect_ratio
    
    print("✅ All custom modules imported successfully")
    
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print(traceback.format_exc())
    QMessageBox.critical(None, "Import Error", 
                       f"Failed to import required modules:\n{str(e)}\n\nPlease ensure all required files are in the same directory.")
    sys.exit(1)

# ============================================================================
# WORKER THREADS
# ============================================================================

class CameraWorker(QThread):
    """Enhanced camera worker for better hand tracking and FPS"""
    frame_ready = pyqtSignal(np.ndarray)
    landmarks_ready = pyqtSignal(list, str, float)
    error_occurred = pyqtSignal(str)
    fps_updated = pyqtSignal(float)
    
    def __init__(self, tracker, gesture_controller):
        super().__init__()
        self.tracker = tracker
        self.gesture_controller = gesture_controller
        self.running = True
        self.mutex = QMutex()
        self.paused = False
        self.cap = None
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        
        # Store landmarks for external access
        self.landmarks = []
        self.current_gesture = "none"
        self.confidence = 0.0
        
        # Performance optimization
        self.processing_interval = 1  # Process every frame
        self.frame_skip = 0
        
    def preprocess_frame(self, frame):
        """Preprocess frame for better hand detection"""
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
            print(f"Preprocessing error: {e}")
            return frame
    
    def run(self):
        """Main thread loop with optimized hand tracking"""
        try:
            # Initialize camera with optimized settings
            self.cap = cv2.VideoCapture(0)
            
            # Try multiple camera indices if needed
            if not self.cap.isOpened():
                for i in range(1, 5):
                    self.cap = cv2.VideoCapture(i)
                    if self.cap.isOpened():
                        break
            
            if not self.cap or not self.cap.isOpened():
                self.error_occurred.emit("No camera found")
                return
            
            # Set camera properties for MAXIMUM resolution and better detection
            # Try to set highest supported resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Maximum resolution
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            
            # Verify actual resolution (camera may not support requested)
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # If camera doesn't support 1920x1080, try 1280x720
            if actual_width < 1280:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Optimize camera settings
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus for consistency
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Enable auto exposure
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
            self.cap.set(cv2.CAP_PROP_CONTRAST, 0.5)
            self.cap.set(cv2.CAP_PROP_SATURATION, 0.5)
            
            print(f"Camera initialized at: {actual_width}x{actual_height}")
            
            print(f"Camera initialized: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            
            last_process_time = time.time()
            
            while self.running:
                self.mutex.lock()
                if self.paused:
                    self.mutex.unlock()
                    self.msleep(100)
                    continue
                self.mutex.unlock()
                
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                # Flip horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Calculate FPS
                current_time = time.time()
                elapsed = current_time - self.last_fps_time
                if elapsed > 1.0:
                    self.current_fps = self.frame_count / elapsed
                    self.fps_updated.emit(self.current_fps)
                    self.frame_count = 0
                    self.last_fps_time = current_time
                
                self.frame_count += 1
                
                # Process hand tracking with professional detection (every frame)
                try:
                    if self.tracker:
                        # Process frame with high-resolution tracker
                        results = self.tracker.process(frame, draw_landmarks=True)
                        landmarks = self.tracker.get_landmarks_list(frame, 0)
                        
                        if landmarks and len(landmarks) >= 21:
                            # Validate landmarks before processing
                            if self._validate_landmarks_for_gesture(landmarks):
                                # Get gesture with confidence using GestureController
                                gesture, confidence = self.gesture_controller.get_hand_gesture(landmarks)
                                
                                # Store for external access
                                self.landmarks = landmarks
                                self.current_gesture = gesture
                                self.confidence = confidence
                                
                                # Emit signal for gesture recognition
                                self.landmarks_ready.emit(landmarks, gesture, confidence)
                                
                                # Draw professional visual feedback on frame
                                if gesture != "no_hand" and confidence > 0.5:
                                    # Draw gesture name with larger font
                                    cv2.putText(frame, f"{gesture.upper()}", (20, 50), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                                    
                                    # Draw confidence percentage
                                    conf_text = f"{confidence:.0%}"
                                    cv2.putText(frame, conf_text, (20, 90), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                                    
                                    # Draw confidence bar (wider)
                                    bar_width = int(confidence * 150)
                                    cv2.rectangle(frame, (20, 100), (20 + bar_width, 115), 
                                                (0, 255, 0), -1)
                                    cv2.rectangle(frame, (20, 100), (170, 115), 
                                                (255, 255, 255), 2)
                                    
                                    # Draw hand center with larger circle
                                    hand_center = self.gesture_controller.get_hand_center(landmarks)
                                    if hand_center:
                                        cv2.circle(frame, hand_center, 12, (0, 0, 255), -1)
                                        cv2.circle(frame, hand_center, 12, (255, 255, 255), 2)
                                    
                                    # Highlight index finger tip for pointing gesture
                                    if gesture == "pointing" and len(landmarks) > 8:
                                        index_tip = landmarks[8]  # (8, x, y)
                                        cv2.circle(frame, (index_tip[1], index_tip[2]), 15, 
                                                  (255, 0, 255), 3)
                                        cv2.circle(frame, (index_tip[1], index_tip[2]), 8, 
                                                  (255, 255, 0), -1)
                        else:
                            # No hand detected - clear state
                            self.landmarks = []
                            self.current_gesture = "no_hand"
                            self.confidence = 0.0
                            
                except Exception as e:
                    print(f"Hand tracking error: {e}")
                    traceback.print_exc()
                    # Clear state on error
                    self.landmarks = []
                    self.current_gesture = "no_hand"
                    self.confidence = 0.0
                
                # Draw FPS counter
                cv2.putText(frame, f"FPS: {self.current_fps:.1f}", 
                          (frame.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.7, (0, 255, 255), 2)
                
                # Emit frame for display
                self.frame_ready.emit(frame)
                
                # Control frame rate to ~30 FPS
                elapsed_frame = time.time() - last_process_time
                sleep_time = max(0, 0.033 - elapsed_frame)
                if sleep_time > 0:
                    self.msleep(int(sleep_time * 1000))
                last_process_time = time.time()
                
        except Exception as e:
            self.error_occurred.emit(str(e))
            print(f"Camera worker error: {traceback.format_exc()}")
        finally:
            if self.cap:
                self.cap.release()
            print("Camera worker stopped")
    
    def pause(self):
        self.mutex.lock()
        self.paused = True
        self.mutex.unlock()
    
    def resume(self):
        self.mutex.lock()
        self.paused = False
        self.mutex.unlock()
    
    def stop(self):
        self.running = False
        self.wait()
    
    def _validate_landmarks_for_gesture(self, landmarks):
        """Validate landmarks are suitable for gesture recognition"""
        if not landmarks or len(landmarks) < 21:
            return False
        
        try:
            # Check that key landmarks are present and valid
            required_landmarks = [0, 4, 8, 12, 16, 20]  # Wrist and fingertips
            for idx in required_landmarks:
                if idx >= len(landmarks):
                    return False
                lm = landmarks[idx]
                if len(lm) != 3 or lm[0] != idx:
                    return False
                # Check coordinates are reasonable
                if not (0 <= lm[1] < 5000 and 0 <= lm[2] < 5000):  # Reasonable bounds
                    return False
            return True
        except Exception:
            return False

# ============================================================================
# CUSTOM WIDGETS (PyQt5 Version)
# ============================================================================

class ToolButton(QPushButton):
    """Custom tool button for PyQt5"""
    def __init__(self, text, icon=None, parent=None):
        super().__init__(text, parent)
        if icon:
            self.setIcon(icon)
        self.setCheckable(True)
        self.setStyleSheet("""
            QPushButton {
                padding: 8px;
                border: 2px solid #2a3b5c;
                border-radius: 4px;
                background-color: #1a1a2e;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2a3b5c;
                border-color: #3a5b8c;
            }
            QPushButton:checked {
                background-color: #4CAF50;
                border-color: #4CAF50;
            }
        """)

class ColorButton(QPushButton):
    """Color selection button"""
    def __init__(self, color, parent=None):
        super().__init__(parent)
        self.color = color
        self.setFixedSize(30, 30)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                border: 2px solid #2a3b5c;
                border-radius: 15px;
            }}
            QPushButton:hover {{
                border: 3px solid white;
            }}
        """)

class StatusBar(QWidget):
    """Custom status bar"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(30)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 0)
        
        self.status_label = QLabel("Ready")
        self.fps_label = QLabel("FPS: --")
        self.gesture_label = QLabel("Gesture: --")
        
        layout.addWidget(self.status_label)
        layout.addStretch()
        layout.addWidget(self.gesture_label)
        layout.addWidget(self.fps_label)
        
        self.setStyleSheet("""
            QWidget {
                background-color: #2c3e50;
            }
            QLabel {
                color: white;
            }
        """)
    
    def set_status(self, message):
        self.status_label.setText(message)
    
    def set_fps(self, fps):
        self.fps_label.setText(f"FPS: {fps:.1f}")
    
    def set_gesture(self, gesture, confidence=0.0):
        if confidence > 0.7:
            self.gesture_label.setText(f"Gesture: {gesture} ({confidence:.0%})")
        else:
            self.gesture_label.setText(f"Gesture: {gesture}")

class ToolPanel(QGroupBox):
    """Tool panel widget"""
    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #34495e;
                border-radius: 5px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)

class ZoomSlider(QSlider):
    """Zoom slider widget"""
    def __init__(self, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.setRange(10, 500)
        self.setValue(100)
        self.setFixedWidth(200)

class CanvasDisplay(QWidget):
    """Canvas display widget"""
    def __init__(self, width=1280, height=720, parent=None):
        super().__init__(parent)
        self.width = width
        self.height = height
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        
        self.canvas_label = QLabel()
        self.canvas_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.canvas_label.setStyleSheet("background-color: white;")
        
        self.scroll_area.setWidget(self.canvas_label)
        
        layout = QVBoxLayout(self)
        layout.addWidget(self.scroll_area)
        
        self.image = None
    
    def display_image(self, qimage):
        """Display QImage on canvas"""
        if qimage:
            pixmap = QPixmap.fromImage(qimage)
            self.canvas_label.setPixmap(pixmap)
            self.image = qimage
    
    def clear(self):
        """Clear canvas display"""
        self.canvas_label.clear()
        self.image = None
    
    def get_canvas_coords(self, event):
        """Get canvas coordinates from event"""
        label_pos = self.canvas_label.mapFromParent(event.pos())
        return label_pos.x(), label_pos.y()

class ColorPickerDialog(QDialog):
    """Color picker dialog"""
    def __init__(self, current_color=(0, 0, 0), parent=None):
        super().__init__(parent)
        self.setWindowTitle("Color Picker")
        self.setFixedSize(400, 500)
        self.current_color = QColor(*current_color[::-1])  # Convert BGR to RGB
        self.result = None
        
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Current color display
        self.color_display = QLabel()
        self.color_display.setFixedHeight(60)
        self.color_display.setStyleSheet(f"background-color: {self.current_color.name()}; border: 2px solid black;")
        layout.addWidget(self.color_display)
        
        # Color presets
        presets_group = QGroupBox("Color Presets")
        presets_layout = QGridLayout()
        
        colors = [
            ("Black", "#000000"),
            ("White", "#FFFFFF"),
            ("Red", "#FF0000"),
            ("Green", "#00FF00"),
            ("Blue", "#0000FF"),
            ("Yellow", "#FFFF00"),
            ("Magenta", "#FF00FF"),
            ("Cyan", "#00FFFF")
        ]
        
        for i, (name, color) in enumerate(colors):
            btn = ColorButton(color)
            btn.setToolTip(name)
            btn.clicked.connect(lambda checked, c=color: self.select_preset(c))
            presets_layout.addWidget(btn, i // 4, i % 4)
        
        presets_group.setLayout(presets_layout)
        layout.addWidget(presets_group)
        
        # RGB sliders
        sliders_group = QGroupBox("RGB Sliders")
        sliders_layout = QVBoxLayout()
        
        self.red_slider = QSlider(Qt.Horizontal)
        self.red_slider.setRange(0, 255)
        self.red_slider.setValue(self.current_color.red())
        self.red_slider.valueChanged.connect(self.update_from_sliders)
        
        self.green_slider = QSlider(Qt.Horizontal)
        self.green_slider.setRange(0, 255)
        self.green_slider.setValue(self.current_color.green())
        self.green_slider.valueChanged.connect(self.update_from_sliders)
        
        self.blue_slider = QSlider(Qt.Horizontal)
        self.blue_slider.setRange(0, 255)
        self.blue_slider.setValue(self.current_color.blue())
        self.blue_slider.valueChanged.connect(self.update_from_sliders)
        
        sliders_layout.addWidget(QLabel("Red:"))
        sliders_layout.addWidget(self.red_slider)
        sliders_layout.addWidget(QLabel("Green:"))
        sliders_layout.addWidget(self.green_slider)
        sliders_layout.addWidget(QLabel("Blue:"))
        sliders_layout.addWidget(self.blue_slider)
        
        sliders_group.setLayout(sliders_layout)
        layout.addWidget(sliders_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        self.system_button = QPushButton("System Picker")
        self.system_button.clicked.connect(self.system_picker)
        
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.system_button)
        
        layout.addLayout(button_layout)
    
    def select_preset(self, color):
        """Select preset color"""
        self.current_color = QColor(color)
        self.update_display()
        self.update_sliders()
    
    def update_from_sliders(self):
        """Update color from sliders"""
        self.current_color = QColor(
            self.red_slider.value(),
            self.green_slider.value(),
            self.blue_slider.value()
        )
        self.update_display()
    
    def update_sliders(self):
        """Update sliders from current color"""
        self.red_slider.setValue(self.current_color.red())
        self.green_slider.setValue(self.current_color.green())
        self.blue_slider.setValue(self.current_color.blue())
    
    def update_display(self):
        """Update color display"""
        self.color_display.setStyleSheet(f"background-color: {self.current_color.name()}; border: 2px solid black;")
    
    def system_picker(self):
        """Open system color picker"""
        color = QColorDialog.getColor(self.current_color, self, "Choose Color")
        if color.isValid():
            self.current_color = color
            self.update_display()
            self.update_sliders()
    
    def accept(self):
        """Accept and close dialog"""
        self.result = (self.current_color.blue(), self.current_color.green(), self.current_color.red())  # BGR
        super().accept()

# ============================================================================
# CANVAS WIDGET
# ============================================================================

class CanvasWidget(QWidget):
    """Modern canvas widget with enhanced visual feedback"""
    
    # Signals
    canvasClicked = pyqtSignal(QPoint, Qt.MouseButton)
    canvasMoved = pyqtSignal(QPoint)
    canvasReleased = pyqtSignal(QPoint, Qt.MouseButton)
    viewportChanged = pyqtSignal()
    
    def __init__(self, engine, viewport, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.viewport = viewport
        
        # Navigation state only (no mouse drawing)
        self.panning = False
        self.pan_start = None
        
        # Visual enhancements
        self.grid_enabled = False
        self.grid_size = 50
        self.grid_color = QColor(100, 100, 100, 80)
        
        self.show_crosshair = True
        self.crosshair_pos = None
        self.crosshair_color = QColor(255, 0, 0, 180)
        
        # Performance
        self.cached_pixmap = None
        self.needs_redraw = True
        
        # Setup - Gesture-only mode (no mouse drawing)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        # No cursor change - gestures only
        self.setCursor(Qt.CursorShape.ArrowCursor)
    
    def paintEvent(self, event):
        """Enhanced paint event with visual feedback"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        
        # Save painter state
        painter.save()
        
        # Apply viewport transformation
        painter.translate(self.viewport.offset_x, self.viewport.offset_y)
        painter.scale(self.viewport.zoom, self.viewport.zoom)
        
        # Get visible area
        visible_rect = self.get_visible_rect()
        
        # Draw background
        if self.grid_enabled:
            painter.fillRect(visible_rect, QBrush(QColor(240, 240, 240)))
        else:
            painter.fillRect(visible_rect, QColor(Qt.GlobalColor.white))
        
        # Draw grid if enabled
        if self.grid_enabled:
            self.draw_grid(painter, visible_rect)
        
        # Update cached pixmap if needed
        if self.needs_redraw or not self.cached_pixmap:
            self.cached_pixmap = self.canvas_to_pixmap()
            self.needs_redraw = False
        
        # Draw canvas content
        if self.cached_pixmap:
            painter.drawPixmap(visible_rect.topLeft(), 
                             self.cached_pixmap, 
                             visible_rect)
        
        # Draw selection
        if self.engine.selection_rect and self.engine.selection_active:
            self.draw_selection(painter)
        
        # Restore painter state
        painter.restore()
        
        # Draw crosshair (in screen coordinates)
        if self.show_crosshair and self.crosshair_pos:
            self.draw_crosshair(painter)
    
    def draw_grid(self, painter, rect):
        """Draw grid"""
        painter.save()
        painter.setPen(QPen(self.grid_color, 1, Qt.PenStyle.SolidLine))
        
        actual_grid = max(10, self.grid_size)
        start_x = rect.left() - (rect.left() % actual_grid)
        start_y = rect.top() - (rect.top() % actual_grid)
        
        # Vertical lines
        x = start_x
        while x <= rect.right():
            painter.drawLine(x, rect.top(), x, rect.bottom())
            x += actual_grid
        
        # Horizontal lines
        y = start_y
        while y <= rect.bottom():
            painter.drawLine(rect.left(), y, rect.right(), y)
            y += actual_grid
        
        painter.restore()
    
    def draw_selection(self, painter):
        """Draw selection rectangle"""
        try:
            if not self.engine.selection_rect:
                return
                
            x, y, w, h = self.engine.selection_rect
            
            painter.save()
            
            # Draw translucent overlay
            overlay = QColor(0, 120, 255, 30)
            painter.fillRect(x, y, w, h, overlay)
            
            # Draw border
            painter.setPen(QPen(QColor(0, 120, 255), 2, Qt.PenStyle.DashLine))
            painter.drawRect(x, y, w, h)
            
            painter.restore()
            
        except Exception as e:
            print(f"Error drawing selection: {e}")
    
    def draw_crosshair(self, painter):
        """Draw crosshair"""
        if not self.crosshair_pos:
            return
            
        # Convert canvas coordinates to screen
        screen_pos = self.canvas_to_screen(self.crosshair_pos)
        
        painter.save()
        
        # Draw cross lines
        painter.setPen(QPen(self.crosshair_color, 1, Qt.PenStyle.DashLine))
        painter.drawLine(screen_pos.x() - 15, screen_pos.y(),
                        screen_pos.x() + 15, screen_pos.y())
        painter.drawLine(screen_pos.x(), screen_pos.y() - 15,
                        screen_pos.x(), screen_pos.y() + 15)
        
        painter.restore()
    
    def canvas_to_pixmap(self):
        """Convert OpenCV canvas to QPixmap"""
        try:
            canvas = self.engine.merge_layers()
            if canvas is None or canvas.size == 0:
                return QPixmap(self.engine.w, self.engine.h)
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            
            # Create QImage (PyQt6 format)
            image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            return QPixmap.fromImage(image)
            
        except Exception as e:
            print(f"Error converting canvas to pixmap: {e}")
            return QPixmap(self.engine.w, self.engine.h)
    
    def get_visible_rect(self):
        """Get visible portion of canvas in canvas coordinates"""
        try:
            top_left = self.screen_to_canvas(QPoint(0, 0))
            bottom_right = self.screen_to_canvas(QPoint(self.width(), self.height()))
            
            return QRect(top_left, bottom_right)
        except:
            return QRect(0, 0, self.engine.w, self.engine.h)
    
    def screen_to_canvas(self, point):
        """Convert screen coordinates to canvas coordinates"""
        try:
            x = (point.x() - self.viewport.offset_x) / self.viewport.zoom
            y = (point.y() - self.viewport.offset_y) / self.viewport.zoom
            
            # Clamp to canvas bounds
            x = max(0, min(self.engine.w - 1, x))
            y = max(0, min(self.engine.h - 1, y))
            
            return QPoint(int(x), int(y))
        except:
            return QPoint(point.x(), point.y())
    
    def canvas_to_screen(self, point):
        """Convert canvas coordinates to screen coordinates"""
        try:
            x = point.x() * self.viewport.zoom + self.viewport.offset_x
            y = point.y() * self.viewport.zoom + self.viewport.offset_y
            return QPoint(int(x), int(y))
        except:
            return QPoint(point.x(), point.y())
    
    def update_canvas(self):
        """Update canvas display"""
        self.needs_redraw = True
        self.update()
    
    # Mouse event handlers - DISABLED FOR GESTURE-ONLY MODE
    # Only panning and zoom are allowed, no drawing with mouse
    def mousePressEvent(self, event):
        try:
            # Only allow panning with middle button, no drawing
            if event.button() == Qt.MouseButton.MiddleButton:
                self.panning = True
                self.pan_start = event.pos()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            
        except Exception as e:
            print(f"Mouse press error: {e}")
    
    def mouseMoveEvent(self, event):
        try:
            canvas_pos = self.screen_to_canvas(event.pos())
            self.crosshair_pos = canvas_pos
            
            # Only panning allowed, no drawing
            if self.panning:
                delta = event.pos() - self.pan_start
                self.viewport.offset_x += delta.x()
                self.viewport.offset_y += delta.y()
                self.pan_start = event.pos()
                self.update()
                self.viewportChanged.emit()
            
            self.canvasMoved.emit(canvas_pos)
            
        except Exception as e:
            print(f"Mouse move error: {e}")
    
    def mouseReleaseEvent(self, event):
        try:
            # Only handle panning release
            if event.button() == Qt.MouseButton.MiddleButton:
                self.panning = False
                self.setCursor(Qt.CursorShape.CrossCursor)
            
        except Exception as e:
            print(f"Mouse release error: {e}")
    
    def wheelEvent(self, event):
        try:
            delta = event.angleDelta().y()
            pos = event.pos()
            
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                # Zoom with Ctrl
                if delta > 0:
                    self.viewport.zoom_in(pos.x(), pos.y())
                else:
                    self.viewport.zoom_out(pos.x(), pos.y())
                self.viewportChanged.emit()
            else:
                # Pan vertically
                self.viewport.offset_y -= delta / 12
                self.viewportChanged.emit()
            
            self.update()
            
        except Exception as e:
            print(f"Wheel event error: {e}")

# ============================================================================
# MAIN WINDOW CLASS
# ============================================================================

class MainWindow(QMainWindow):
    """Main window with improved hand tracking and modern UI"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize configuration
        self.config = ConfigManager()
        
        # State variables
        self.current_file = None
        self.gesture_mode = False
        self.webcam_visible = True
        self.grid_enabled = False
        
        # FPS tracking
        self.current_fps = 0
        self.fps_history = deque(maxlen=60)  # Keep last 60 FPS readings
        
        # Gesture state tracking
        self.gesture_active = False
        self.last_gesture = "none"
        self.gesture_stable_count = 0
        self.last_hand_position = None
        self.shape_drawing_state = 'idle'  # 'idle', 'start_point_set', 'drawing'
        self.shape_start_point = None
        
        # Initialize components
        self.init_components()
        self.init_ui()
        self.setup_connections()
        self.start_services()
        self.apply_styles()
        
        print("🚀 Virtual Drawing Studio Pro - Enhanced Edition Initialized")
    
    def init_components(self):
        """Initialize all components with error handling"""
        try:
            print("Initializing components...")
            
            # Core components
            self.gesture_controller = GestureController(self.config, None)  # No calibrator
            
            self.engine = DrawingEngine(
                h=self.config.ui.canvas_height,
                w=self.config.ui.canvas_width
            )
            
            # Set update callback for the engine
            self.engine.set_update_callback(self.update_canvas_display)
            
            self.viewport = ViewportManager(
                self.config.ui.canvas_width,
                self.config.ui.canvas_height
            )
            
            # Enhanced hand tracker
            self.tracker = HandTracker(self.config)
            
            # Camera worker
            self.camera_worker = CameraWorker(self.tracker, self.gesture_controller)
            
            # UI state
            self.current_color = QColor(Qt.GlobalColor.black)
            self.current_tool = 'brush'
            self.brush_size = self.config.drawing.default_brush_size
            self.eraser_size = 30
            self.current_shape = 'line'
            self.fill_shape = False
            
            print("✅ Components initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing components: {e}")
            traceback.print_exc()
            QMessageBox.critical(self, "Initialization Error", 
                               f"Failed to initialize components:\n{str(e)}")
    
    def init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle("🎨 Virtual Drawing Studio Pro - Enhanced")
        
        # Set window properties
        self.setGeometry(100, 100, 1400, 800)
        self.center_on_screen()
        
        # Create central widget
        central_widget = QWidget()
        central_widget.setObjectName("centralWidget")
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # Left toolbar
        left_toolbar = self.create_toolbar()
        main_layout.addWidget(left_toolbar)
        
        # Central canvas area
        central_area = self.create_canvas_area()
        main_layout.addWidget(central_area, 1)
        
        # Right sidebar
        right_sidebar = self.create_sidebar()
        main_layout.addWidget(right_sidebar)
        
        # Create menu bar
        self.create_menu()
        
        # Create status bar
        self.create_status_bar()
        
        print("✅ UI initialized successfully")
    
    def center_on_screen(self):
        """Center window on screen"""
        frame = self.frameGeometry()
        screen_center = QApplication.primaryScreen().availableGeometry().center()
        frame.moveCenter(screen_center)
        self.move(frame.topLeft())
    
    def create_toolbar(self):
        """Create left toolbar with improved webcam display"""
        toolbar = QFrame()
        toolbar.setFixedWidth(280)  # Increased width for shape controls
        toolbar.setObjectName("leftToolbar")
        
        layout = QVBoxLayout(toolbar)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Title
        title = QLabel("🎨 Tools")
        title.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #4CAF50;
                padding-bottom: 10px;
                border-bottom: 2px solid #2a3b5c;
            }
        """)
        layout.addWidget(title)
        
        # Tools section
        tools_group = QGroupBox("Drawing Tools")
        tools_group.setStyleSheet(self.get_group_style())
        tools_layout = QGridLayout()
        
        tools = [
            ("✏️", "brush", "Brush"),
            ("🧹", "eraser", "Eraser"),
            ("🖍️", "highlighter", "Highlighter"),
            ("●", "dot", "Dot"),
            ("⬛", "shape", "Shapes"),
            ("📐", "select", "Select"),
            ("---", "dashline", "Dash Line")
        ]
        
        self.tool_button_group = QButtonGroup()
        
        for i, (icon, tool_id, tooltip) in enumerate(tools):
            if tool_id == "---":
                # Add separator
                separator = QFrame()
                separator.setFrameShape(QFrame.Shape.HLine)
                separator.setFrameShadow(QFrame.Shadow.Sunken)
                separator.setStyleSheet("background-color: #2a3b5c;")
                tools_layout.addWidget(separator, i, 0, 1, 2)
            else:
                btn = self.create_tool_button(icon, tool_id, tooltip)
                self.tool_button_group.addButton(btn)
                
                row = i // 2
                col = i % 2
                tools_layout.addWidget(btn, row, col)
                
                if tool_id == 'brush':
                    btn.setChecked(True)
        
        tools_group.setLayout(tools_layout)
        layout.addWidget(tools_group)
        
        # Shape controls (visible when shape tool is selected)
        self.shape_group = QGroupBox("Shape Settings")
        self.shape_group.setStyleSheet(self.get_group_style())
        self.shape_group.setVisible(False)
        shape_layout = QVBoxLayout()
        
        # Shape type combo
        shape_type_layout = QHBoxLayout()
        shape_type_label = QLabel("Shape:")
        shape_type_layout.addWidget(shape_type_label)
        
        self.shape_combo = QComboBox()
        self.shape_combo.addItems(["Line", "Rectangle", "Circle", "Triangle", "Arrow", "Dash Line"])
        self.shape_combo.currentTextChanged.connect(self.update_shape_type)
        shape_type_layout.addWidget(self.shape_combo)
        shape_layout.addLayout(shape_type_layout)
        
        # Fill checkbox
        self.fill_checkbox = QCheckBox("Fill Shape")
        self.fill_checkbox.stateChanged.connect(self.toggle_shape_fill)
        shape_layout.addWidget(self.fill_checkbox)
        
        self.shape_group.setLayout(shape_layout)
        layout.addWidget(self.shape_group)
        
        # Brush controls
        brush_group = QGroupBox("Brush Settings")
        brush_group.setStyleSheet(self.get_group_style())
        brush_layout = QVBoxLayout()
        
        # Brush size
        size_layout = QHBoxLayout()
        size_label = QLabel("Size:")
        size_layout.addWidget(size_label)
        
        self.brush_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_slider.setRange(1, 50)
        self.brush_slider.setValue(self.brush_size)
        self.brush_slider.valueChanged.connect(self.update_brush_size)
        size_layout.addWidget(self.brush_slider)
        
        self.brush_size_label = QLabel(f"{self.brush_size}px")
        size_layout.addWidget(self.brush_size_label)
        
        brush_layout.addLayout(size_layout)
        
        # Color preview
        color_layout = QHBoxLayout()
        color_label = QLabel("Color:")
        color_layout.addWidget(color_label)
        
        self.color_preview = QLabel()
        self.color_preview.setFixedSize(30, 30)
        self.update_color_preview()
        color_layout.addWidget(self.color_preview)
        
        color_button = QPushButton("Pick Color")
        color_button.clicked.connect(self.open_color_picker)
        color_layout.addWidget(color_button)
        
        brush_layout.addLayout(color_layout)
        
        brush_group.setLayout(brush_layout)
        layout.addWidget(brush_group)
        
        # View controls
        view_group = QGroupBox("View")
        view_group.setStyleSheet(self.get_group_style())
        view_layout = QVBoxLayout()
        
        self.grid_checkbox = QCheckBox("Show Grid")
        self.grid_checkbox.setChecked(self.grid_enabled)
        self.grid_checkbox.stateChanged.connect(self.toggle_grid)
        view_layout.addWidget(self.grid_checkbox)
        
        # Zoom controls
        zoom_layout = QHBoxLayout()
        zoom_in_btn = QPushButton("+")
        zoom_in_btn.clicked.connect(self.zoom_in)
        zoom_out_btn = QPushButton("-")
        zoom_out_btn.clicked.connect(self.zoom_out)
        reset_view_btn = QPushButton("Reset")
        reset_view_btn.clicked.connect(self.reset_view)
        
        zoom_layout.addWidget(zoom_in_btn)
        zoom_layout.addWidget(zoom_out_btn)
        zoom_layout.addWidget(reset_view_btn)
        view_layout.addLayout(zoom_layout)
        
        view_group.setLayout(view_layout)
        layout.addWidget(view_group)
        
        # Test button for debugging
        test_layout = QHBoxLayout()
        test_btn = QPushButton("🛠️ Test Gestures")
        test_btn.clicked.connect(self.test_gesture_drawing)
        test_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF5722;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #E64A19;
            }
        """)
        test_layout.addWidget(test_btn)
        layout.addLayout(test_layout)
        
        # Gesture mode checkbox (moved from webcam section)
        self.gesture_checkbox = QCheckBox("Gesture Mode")
        self.gesture_checkbox.setChecked(self.gesture_mode)
        self.gesture_checkbox.stateChanged.connect(self.toggle_gesture_mode)
        layout.addWidget(self.gesture_checkbox)
        
        layout.addStretch()
        
        return toolbar
    
    def create_tool_button(self, icon, tool_id, tooltip):
        """Create individual tool button"""
        btn = QPushButton(icon)
        btn.setCheckable(True)
        btn.setObjectName(f"tool_{tool_id}")
        btn.setToolTip(tooltip)
        btn.setFixedSize(60, 60)
        btn.setStyleSheet("""
            QPushButton {
                font-size: 24px;
                border: 2px solid #2a3b5c;
                border-radius: 8px;
                background-color: #1a1a2e;
                color: white;
            }
            QPushButton:hover {
                background-color: #2a3b5c;
                border-color: #3a5b8c;
            }
            QPushButton:checked {
                background-color: #4CAF50;
                border-color: #4CAF50;
                color: white;
            }
        """)
        btn.clicked.connect(lambda checked, t=tool_id: self.select_tool(t))
        return btn
    
    def create_canvas_area(self):
        """Create canvas area"""
        container = QWidget()
        container.setObjectName("canvasContainer")
        
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Canvas widget
        self.canvas_widget = CanvasWidget(self.engine, self.viewport)
        layout.addWidget(self.canvas_widget, 1)
        
        # Canvas status bar
        status_frame = QFrame()
        status_frame.setFixedHeight(30)
        status_frame.setStyleSheet("""
            QFrame {
                background-color: #1a1a2e;
                border-top: 1px solid #2a3b5c;
            }
        """)
        
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(10, 0, 10, 0)
        
        # Coordinates
        self.coord_label = QLabel("X: 0, Y: 0")
        self.coord_label.setStyleSheet("color: white;")
        status_layout.addWidget(self.coord_label)
        
        status_layout.addStretch()
        
        # Tool info
        self.tool_info_label = QLabel("Tool: Brush | Shape: Line")
        self.tool_info_label.setStyleSheet("color: #4CAF50;")
        status_layout.addWidget(self.tool_info_label)
        
        status_layout.addStretch()
        
        # Zoom level
        self.zoom_label = QLabel("100%")
        self.zoom_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        status_layout.addWidget(self.zoom_label)
        
        layout.addWidget(status_frame)
        
        return container
    
    def create_sidebar(self):
        """Create right sidebar with webcam at top-right"""
        sidebar = QFrame()
        sidebar.setFixedWidth(280)  # Increased width for webcam
        sidebar.setObjectName("rightSidebar")
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Webcam preview - HIGH RESOLUTION at top-right
        webcam_group = QGroupBox("📹 Webcam Feed")
        webcam_group.setStyleSheet(self.get_group_style())
        webcam_layout = QVBoxLayout()
        
        self.webcam_checkbox = QCheckBox("Show Webcam")
        self.webcam_checkbox.setChecked(self.webcam_visible)
        self.webcam_checkbox.stateChanged.connect(self.toggle_webcam)
        webcam_layout.addWidget(self.webcam_checkbox)
        
        # High-resolution webcam display (640x480 for better quality)
        self.webcam_label = QLabel()
        self.webcam_label.setFixedSize(640, 480)  # High resolution
        self.webcam_label.setMinimumSize(320, 240)  # Minimum size
        self.webcam_label.setScaledContents(True)  # Scale to fit
        self.webcam_label.setStyleSheet("""
            QLabel {
                border: 3px solid #2a3b5c;
                border-radius: 10px;
                background-color: #1a1a2e;
                padding: 5px;
            }
        """)
        self.webcam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.webcam_label.setText("Webcam\nPreview\n(640x480)")
        
        # Add FPS overlay label
        self.webcam_fps_label = QLabel("")
        self.webcam_fps_label.setStyleSheet("""
            QLabel {
                color: #00FF00;
                background-color: rgba(0, 0, 0, 180);
                padding: 3px 8px;
                border-radius: 3px;
                font-weight: bold;
                font-size: 12px;
            }
        """)
        self.webcam_fps_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        
        webcam_layout.addWidget(self.webcam_label)
        
        webcam_group.setLayout(webcam_layout)
        layout.addWidget(webcam_group)  # Add at top
        
        # Color palette
        colors_group = QGroupBox("Color Palette")
        colors_group.setStyleSheet(self.get_group_style())
        colors_layout = QGridLayout()
        
        colors = [
            ("#000000", "Black"),
            ("#FFFFFF", "White"),
            ("#FF0000", "Red"),
            ("#00FF00", "Green"),
            ("#0000FF", "Blue"),
            ("#FFFF00", "Yellow"),
            ("#FF00FF", "Magenta"),
            ("#00FFFF", "Cyan"),
            ("#FFA500", "Orange"),
            ("#800080", "Purple"),
            ("#A52A2A", "Brown"),
            ("#FFC0CB", "Pink")
        ]
        
        for i, (hex_color, name) in enumerate(colors):
            row = i // 4
            col = i % 4
            
            btn = self.create_color_button(hex_color, name)
            colors_layout.addWidget(btn, row, col)
        
        colors_group.setLayout(colors_layout)
        layout.addWidget(colors_group)
        
        # Quick actions
        actions_group = QGroupBox("Quick Actions")
        actions_group.setStyleSheet(self.get_group_style())
        actions_layout = QVBoxLayout()
        
        actions = [
            ("📋 Copy", self.copy_selection, "Ctrl+C"),
            ("✂️ Cut", self.cut_selection, "Ctrl+X"),
            ("📎 Paste", self.paste_selection, "Ctrl+V"),
            ("↶ Undo", self.undo, "Ctrl+Z"),
            ("↷ Redo", self.redo, "Ctrl+Y"),
            ("🗑️ Clear", self.clear_canvas, "Del"),
            ("💾 Save", self.save_image, "Ctrl+S"),
            ("📤 Export", lambda: self.export_image('png'), "")
        ]
        
        for icon_text, callback, shortcut in actions:
            btn = QPushButton(icon_text)
            if shortcut:
                btn.setToolTip(shortcut)
            btn.clicked.connect(callback)
            btn.setStyleSheet("""
                QPushButton {
                    text-align: left;
                    padding: 8px;
                    border-radius: 4px;
                    background-color: #2a3b5c;
                    color: white;
                }
                QPushButton:hover {
                    background-color: #3a5b8c;
                }
            """)
            actions_layout.addWidget(btn)
        
        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)
        
        # Gesture info
        gesture_group = QGroupBox("Gesture Control")
        gesture_group.setStyleSheet(self.get_group_style())
        gesture_layout = QVBoxLayout()
        
        self.gesture_label = QLabel("Gesture: --")
        self.gesture_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        gesture_layout.addWidget(self.gesture_label)
        
        self.gesture_confidence_label = QLabel("Confidence: --")
        self.gesture_confidence_label.setStyleSheet("color: #FFC107;")
        gesture_layout.addWidget(self.gesture_confidence_label)
        
        calibrate_btn = QPushButton("🎯 Gesture Info")
        calibrate_btn.clicked.connect(self.show_gesture_info)
        calibrate_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        gesture_layout.addWidget(calibrate_btn)
        
        gesture_group.setLayout(gesture_layout)
        layout.addWidget(gesture_group)
        
        layout.addStretch()
        
        scroll.setWidget(content)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.addWidget(scroll)
        
        return sidebar
    
    def create_color_button(self, hex_color, name):
        """Create color button"""
        btn = QPushButton()
        btn.setFixedSize(40, 40)
        btn.setToolTip(name)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {hex_color};
                border: 2px solid #2a3b5c;
                border-radius: 20px;
            }}
            QPushButton:hover {{
                border: 3px solid #4CAF50;
            }}
        """)
        btn.clicked.connect(lambda: self.select_color(QColor(hex_color)))
        return btn
    
    def create_menu(self):
        """Create menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        new_action = QAction("&New", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_drawing)
        file_menu.addAction(new_action)
        
        open_action = QAction("&Open...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_image)
        file_menu.addAction(open_action)
        
        save_action = QAction("&Save", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_image)
        file_menu.addAction(save_action)
        
        save_as_action = QAction("Save &As...", self)
        save_as_action.triggered.connect(self.save_image_as)
        file_menu.addAction(save_as_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Alt+F4")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        
        undo_action = QAction("&Undo", self)
        undo_action.setShortcut("Ctrl+Z")
        undo_action.triggered.connect(self.undo)
        edit_menu.addAction(undo_action)
        
        redo_action = QAction("&Redo", self)
        redo_action.setShortcut("Ctrl+Y")
        redo_action.triggered.connect(self.redo)
        edit_menu.addAction(redo_action)
        
        edit_menu.addSeparator()
        
        cut_action = QAction("Cu&t", self)
        cut_action.setShortcut("Ctrl+X")
        cut_action.triggered.connect(self.cut_selection)
        edit_menu.addAction(cut_action)
        
        copy_action = QAction("&Copy", self)
        copy_action.setShortcut("Ctrl+C")
        copy_action.triggered.connect(self.copy_selection)
        edit_menu.addAction(copy_action)
        
        paste_action = QAction("&Paste", self)
        paste_action.setShortcut("Ctrl+V")
        paste_action.triggered.connect(self.paste_selection)
        edit_menu.addAction(paste_action)
        
        edit_menu.addSeparator()
        
        clear_action = QAction("&Clear Canvas", self)
        clear_action.triggered.connect(self.clear_canvas)
        edit_menu.addAction(clear_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        zoom_in_action = QAction("Zoom &In", self)
        zoom_in_action.setShortcut("Ctrl++")
        zoom_in_action.triggered.connect(self.zoom_in)
        view_menu.addAction(zoom_in_action)
        
        zoom_out_action = QAction("Zoom &Out", self)
        zoom_out_action.setShortcut("Ctrl+-")
        zoom_out_action.triggered.connect(self.zoom_out)
        view_menu.addAction(zoom_out_action)
        
        reset_view_action = QAction("&Reset View", self)
        reset_view_action.setShortcut("Ctrl+0")
        reset_view_action.triggered.connect(self.reset_view)
        view_menu.addAction(reset_view_action)
        
        view_menu.addSeparator()
        
        grid_action = QAction("Show &Grid", self, checkable=True)
        grid_action.setChecked(self.grid_enabled)
        grid_action.triggered.connect(lambda checked: self.toggle_grid(2 if checked else 0))
        view_menu.addAction(grid_action)
    
    def create_status_bar(self):
        """Create status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Mode indicator
        self.mode_label = QLabel("Mode: Mouse")
        self.status_bar.addWidget(self.mode_label)
        
        # Tool indicator
        self.tool_label = QLabel("Tool: Brush")
        self.status_bar.addWidget(self.tool_label)
        
        # FPS indicator
        self.fps_label = QLabel("FPS: --")
        self.status_bar.addWidget(self.fps_label)
        
        # Gesture indicator
        self.status_gesture_label = QLabel("Gesture: --")
        self.status_bar.addWidget(self.status_gesture_label)
        
        # Initial message
        self.status_bar.showMessage("Ready - Virtual Drawing Studio Pro", 5000)
    
    def apply_styles(self):
        """Apply styles"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a2e;
            }
            
            QFrame#leftToolbar, QFrame#rightSidebar {
                background-color: #16213e;
                border-radius: 8px;
            }
            
            QFrame#canvasContainer {
                background-color: #0f3460;
                border-radius: 8px;
            }
            
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                color: white;
                border: 2px solid #2a3b5c;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
            }
            
            QLabel {
                color: white;
            }
            
            QStatusBar {
                background-color: #16213e;
                color: white;
            }
            
            QMenuBar {
                background-color: #16213e;
                color: white;
            }
            
            QMenuBar::item:selected {
                background-color: #0f3460;
            }
            
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            
            QComboBox {
                background-color: #2a3b5c;
                color: white;
                border: 1px solid #3a5b8c;
                border-radius: 4px;
                padding: 3px;
            }
            
            QComboBox:hover {
                border: 1px solid #4CAF50;
            }
        """)
    
    def get_group_style(self):
        """Get group box style"""
        return """
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                color: white;
                border: 2px solid #2a3b5c;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
            }
        """
    
    # ============================================================================
    # SETUP AND CONNECTIONS
    # ============================================================================
    
    def setup_connections(self):
        """Setup signal connections"""
        # Canvas signals (coordinate tracking only, no mouse drawing)
        self.canvas_widget.canvasMoved.connect(self.update_coordinates)
        self.canvas_widget.viewportChanged.connect(self.update_zoom_display)
        # Note: canvasClicked and canvasReleased signals disabled for gesture-only mode
        
        # Camera worker signals
        self.camera_worker.frame_ready.connect(self.update_webcam_display)
        self.camera_worker.landmarks_ready.connect(self.handle_gesture_data)
        self.camera_worker.fps_updated.connect(self.update_fps_display)
        self.camera_worker.error_occurred.connect(self.handle_camera_error)
        
        # Update timer for canvas
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_canvas_display)
        self.update_timer.start(33)  # ~30 FPS
        
        # Gesture processing timer
        self.gesture_timer = QTimer()
        self.gesture_timer.timeout.connect(self.process_gesture_commands)
        self.gesture_timer.start(16)  # 60 Hz for smoother tracking
    
    def start_services(self):
        """Start background services"""
        try:
            # Start camera worker
            self.camera_worker.start()
            
            # Start with brush tool
            self.select_tool('brush')
            
            print("✅ Services started successfully")
            
        except Exception as e:
            print(f"❌ Error starting services: {e}")
    
    # ============================================================================
    # SLOTS AND EVENT HANDLERS
    # ============================================================================
    
    @pyqtSlot(QPoint)
    def update_coordinates(self, point):
        """Update coordinate display"""
        self.coord_label.setText(f"X: {point.x()}, Y: {point.y()}")
    
    @pyqtSlot()
    def update_zoom_display(self):
        """Update zoom display"""
        zoom_percent = int(self.viewport.zoom * 100)
        self.zoom_label.setText(f"{zoom_percent}%")
    
    @pyqtSlot(np.ndarray)
    def update_webcam_display(self, frame):
        """Update webcam display with FPS overlay"""
        if not self.webcam_visible:
            return
        
        try:
            # Keep original high resolution frame for better quality
            # The label will scale it down if needed, but we preserve quality
            display_frame = frame.copy()
            
            # Add FPS text to frame (larger text for high-res display)
            fps_text = f"FPS: {self.current_fps:.1f}"
            cv2.putText(display_frame, fps_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Add gesture text if available
            if hasattr(self.camera_worker, 'current_gesture'):
                gesture = self.camera_worker.current_gesture
                confidence = self.camera_worker.confidence
                if gesture != "no_hand" and confidence > 0.5:
                    gesture_text = f"{gesture} ({confidence:.0%})"
                    cv2.putText(display_frame, gesture_text, (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Highlight index finger tip for better visibility (improved single finger detection)
            if hasattr(self.camera_worker, 'landmarks') and self.camera_worker.landmarks:
                landmarks = self.camera_worker.landmarks
                if len(landmarks) >= 9:
                    # Get index finger tip (landmark 8)
                    index_tip = next((p for p in landmarks if p[0] == 8), None)
                    if index_tip:
                        x, y = index_tip[1], index_tip[2]
                        # Draw larger circle for visibility
                        cv2.circle(display_frame, (x, y), 15, (0, 255, 255), 3)
                        cv2.circle(display_frame, (x, y), 5, (255, 0, 0), -1)
            
            # Convert to QPixmap and display (high quality)
            pixmap = cv2_to_qpixmap(display_frame)
            self.webcam_label.setPixmap(pixmap)
            
            # Update FPS overlay label
            if hasattr(self, 'webcam_fps_label'):
                self.webcam_fps_label.setText(f"FPS: {self.current_fps:.1f}")
                
        except Exception as e:
            print(f"Error updating webcam display: {e}")
            import traceback
            traceback.print_exc()
    
    @pyqtSlot(list, str, float)
    def handle_gesture_data(self, landmarks, gesture, confidence):
        """Handle gesture data from camera"""
        if not self.gesture_mode:
            return
        
        try:
            # Update gesture display
            self.gesture_label.setText(f"Gesture: {gesture}")
            self.gesture_confidence_label.setText(f"Confidence: {confidence:.0%}")
            self.status_gesture_label.setText(f"Gesture: {gesture}")
            
        except Exception as e:
            print(f"Error handling gesture data: {e}")
    
    def process_gesture_commands(self):
        """Process gesture commands for drawing with improved stability"""
        if not self.gesture_mode:
            return
        
        try:
            # Get landmarks from camera worker
            if hasattr(self.camera_worker, 'landmarks') and self.camera_worker.landmarks:
                landmarks = self.camera_worker.landmarks
                gesture = self.camera_worker.current_gesture
                confidence = self.camera_worker.confidence
                
                if not landmarks or len(landmarks) < 21:
                    # No hand detected, stop drawing if active
                    if self.gesture_active:
                        self.stop_gesture_drawing()
                    self.shape_drawing_state = 'idle'
                    return
                
                # Only process if we have reasonable confidence
                if confidence < 0.6:
                    if self.gesture_active:
                        self.stop_gesture_drawing()
                    self.shape_drawing_state = 'idle'
                    return
                
                # Check for gesture stability (avoid flickering)
                if gesture != self.last_gesture:
                    self.gesture_stable_count = 0
                    self.last_gesture = gesture
                else:
                    self.gesture_stable_count += 1
                
                # Require gesture to be stable for at least 2 frames
                if self.gesture_stable_count < 2:
                    return
                
                # Get hand position
                hand_position = self.gesture_controller.get_smooth_position(landmarks)
                if not hand_position:
                    if self.gesture_active:
                        self.stop_gesture_drawing()
                    self.shape_drawing_state = 'idle'
                    return
                
                # Professional index finger detection with validation
                index_tip = self.gesture_controller.get_index_tip_position(landmarks)
                
                # Enhanced single finger detection - verify only index finger is extended
                finger_states = self.gesture_controller.fingers_state(landmarks)
                index_extended = finger_states.get('index', False)
                other_fingers_down = (not finger_states.get('thumb', False) and 
                                     not finger_states.get('middle', False) and 
                                     not finger_states.get('ring', False) and 
                                     not finger_states.get('pinky', False))
                
                # Professional validation: only use index tip if truly pointing
                if not index_tip or not (index_extended and other_fingers_down):
                    # Fallback to hand center if index tip invalid or not pointing
                    index_tip = hand_position
                    # Lower confidence for non-pointing gestures
                    if not index_extended:
                        confidence = confidence * 0.7
                
                # Map webcam coordinates to canvas coordinates with high precision
                # Get actual camera resolution
                actual_cam_width = 1920  # Try maximum first
                actual_cam_height = 1080
                
                if hasattr(self.camera_worker, 'cap') and self.camera_worker.cap:
                    actual_cam_width = int(self.camera_worker.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_cam_height = int(self.camera_worker.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # If camera resolution not available, use processing resolution
                if actual_cam_width == 0 or actual_cam_height == 0:
                    actual_cam_width = 1280
                    actual_cam_height = 720
                
                # Precise coordinate mapping
                webcam_to_canvas_x = self.engine.w / actual_cam_width
                webcam_to_canvas_y = self.engine.h / actual_cam_height
                
                canvas_x = int(index_tip[0] * webcam_to_canvas_x)
                canvas_y = int(index_tip[1] * webcam_to_canvas_y)
                
                # Clamp to canvas bounds
                canvas_x = max(0, min(self.engine.w - 1, canvas_x))
                canvas_y = max(0, min(self.engine.h - 1, canvas_y))
                
                # Process based on gesture and current tool
                if gesture == "pointing" and confidence > 0.7:
                    # Handle pointing gesture based on current tool
                    if self.current_tool == 'shape':
                        self.handle_shape_drawing(canvas_x, canvas_y, gesture, 'drawing')
                    elif self.current_tool == 'select':
                        self.handle_selection(canvas_x, canvas_y, gesture, 'drawing')
                    elif self.current_tool == 'dashline':
                        self.handle_dashline_drawing(canvas_x, canvas_y, gesture, 'drawing')
                    else:
                        # For brush, eraser, highlighter, dot
                        self.handle_freehand_drawing(canvas_x, canvas_y, gesture, 'drawing')
                
                elif gesture == "pinch" and confidence > 0.7:
                    # Pinch gesture for erasing (overrides current tool)
                    if not self.gesture_active:
                        self.engine.set_mode('eraser')
                        self.engine.start_stroke(canvas_x, canvas_y)
                        self.gesture_active = True
                        self.last_hand_position = (canvas_x, canvas_y)
                        self.status_bar.showMessage("Erasing with pinch gesture", 1000)
                    else:
                        if self.last_hand_position:
                            self.engine.continue_stroke(canvas_x, canvas_y)
                            self.canvas_widget.update_canvas()
                        self.last_hand_position = (canvas_x, canvas_y)
                
                elif gesture == "fist" and confidence > 0.7:
                    # Stop drawing/erasing with fist gesture
                    if self.gesture_active:
                        self.stop_gesture_drawing()
                        self.status_bar.showMessage("Drawing stopped with fist gesture", 1000)
                    
                    # For shape tool, complete the shape
                    if self.shape_drawing_state == 'drawing' and self.shape_start_point:
                        self.complete_shape_drawing(canvas_x, canvas_y)
                
                elif gesture == "open_hand" and confidence > 0.7:
                    # Clear canvas or selection
                    if self.engine.selection_active:
                        self.engine.clear_selection()
                        self.canvas_widget.update_canvas()
                        self.status_bar.showMessage("Selection cleared", 1000)
                    else:
                        # Just show message, don't auto-clear canvas
                        self.status_bar.showMessage("Open hand detected", 1000)
                
                elif gesture == "scissors" and confidence > 0.7:
                    # Scissors gesture for copy/cut operations
                    if self.current_tool == 'select' and self.engine.selection_active:
                        # Copy selection
                        if self.copy_selection():
                            self.status_bar.showMessage("Selection copied via gesture", 2000)
                    else:
                        # Switch to eraser tool if no selection
                        self.select_tool('eraser')
                        self.status_bar.showMessage("Switched to eraser", 1000)
                
                elif gesture == "victory" and confidence > 0.7:
                    # Victory gesture (two fingers) for cut operation
                    if self.current_tool == 'select' and self.engine.selection_active:
                        # Cut selection
                        if self.cut_selection():
                            self.status_bar.showMessage("Selection cut via gesture", 2000)
                            self.canvas_widget.update_canvas()
                
                # Handle gesture ending (hand lost or low confidence)
                elif gesture == "no_hand" or confidence < 0.6:
                    if self.gesture_active:
                        self.stop_gesture_drawing()
                    
        except Exception as e:
            print(f"Error processing gesture commands: {e}")
            traceback.print_exc()
    
    def handle_freehand_drawing(self, canvas_x, canvas_y, gesture, action):
        """Handle freehand drawing tools (brush, eraser, highlighter, dot)"""
        if not self.gesture_active:
            # Start drawing
            self.engine.set_mode(self.current_tool)
            self.engine.start_stroke(canvas_x, canvas_y)
            self.gesture_active = True
            self.last_hand_position = (canvas_x, canvas_y)
            tool_name = self.current_tool.capitalize()
            self.status_bar.showMessage(f"{tool_name} drawing started", 1000)
        else:
            # Continue drawing
            if self.last_hand_position:
                self.engine.continue_stroke(canvas_x, canvas_y)
                self.canvas_widget.update_canvas()
            self.last_hand_position = (canvas_x, canvas_y)
    
    def handle_shape_drawing(self, canvas_x, canvas_y, gesture, action):
        """Handle shape drawing"""
        if self.shape_drawing_state == 'idle':
            # Start shape drawing
            self.shape_start_point = (canvas_x, canvas_y)
            self.shape_drawing_state = 'start_point_set'
            self.engine.set_mode('shape')
            self.engine.start_stroke(canvas_x, canvas_y)
            self.gesture_active = True
            self.status_bar.showMessage(f"Shape started at ({canvas_x}, {canvas_y})", 1000)
        
        elif self.shape_drawing_state == 'start_point_set' or self.shape_drawing_state == 'drawing':
            # Update shape preview
            self.shape_drawing_state = 'drawing'
            self.engine.continue_stroke(canvas_x, canvas_y)
            self.canvas_widget.update_canvas()
    
    def handle_dashline_drawing(self, canvas_x, canvas_y, gesture, action):
        """Handle dash line drawing"""
        if not self.gesture_active:
            # Start dash line drawing
            self.engine.set_mode('dashline')
            self.engine.start_stroke(canvas_x, canvas_y)
            self.gesture_active = True
            self.last_hand_position = (canvas_x, canvas_y)
            self.status_bar.showMessage("Dash line drawing started", 1000)
        else:
            # Continue dash line drawing
            if self.last_hand_position:
                self.engine.continue_stroke(canvas_x, canvas_y)
                self.canvas_widget.update_canvas()
            self.last_hand_position = (canvas_x, canvas_y)
    
    def handle_selection(self, canvas_x, canvas_y, gesture, action):
        """Handle selection tool"""
        if not self.gesture_active:
            # Start selection
            self.engine.set_mode('select')
            self.engine.start_stroke(canvas_x, canvas_y)
            self.gesture_active = True
            self.last_hand_position = (canvas_x, canvas_y)
            self.status_bar.showMessage("Selection started", 1000)
        else:
            # Update selection
            if self.last_hand_position:
                self.engine.continue_stroke(canvas_x, canvas_y)
                self.canvas_widget.update_canvas()
            self.last_hand_position = (canvas_x, canvas_y)
    
    def complete_shape_drawing(self, canvas_x, canvas_y):
        """Complete shape drawing"""
        if self.shape_start_point:
            self.engine.end_stroke()
            self.shape_drawing_state = 'idle'
            self.shape_start_point = None
            self.gesture_active = False
            self.canvas_widget.update_canvas()
            self.status_bar.showMessage("Shape completed", 1000)
    
    def stop_gesture_drawing(self):
        """Stop gesture-based drawing"""
        if self.gesture_active:
            self.engine.end_stroke()
            self.gesture_active = False
            self.last_hand_position = None
            
            # Reset shape drawing state
            if self.shape_drawing_state != 'idle':
                self.shape_drawing_state = 'idle'
                self.shape_start_point = None
            
            self.canvas_widget.update_canvas()
    
    @pyqtSlot(float)
    def update_fps_display(self, fps):
        """Update FPS display"""
        self.current_fps = fps
        self.fps_history.append(fps)
        self.fps_label.setText(f"FPS: {fps:.1f}")
    
    @pyqtSlot(str)
    def handle_camera_error(self, error):
        """Handle camera errors"""
        self.status_bar.showMessage(f"Camera error: {error}", 5000)
        self.gesture_checkbox.setChecked(False)
        self.gesture_mode = False
    
    def update_canvas_display(self):
        """Update canvas display - called by DrawingEngine callback"""
        # Mark canvas widget for redraw
        self.canvas_widget.needs_redraw = True
        # Trigger Qt update
        self.canvas_widget.update()
    
    # ============================================================================
    # TOOL AND UI METHODS
    # ============================================================================
    
    def select_tool(self, tool):
        """Select drawing tool"""
        try:
            self.current_tool = tool
            self.engine.set_mode(tool)
            
            # Update UI
            self.tool_label.setText(f"Tool: {tool.capitalize()}")
            
            # Show/hide shape controls
            if tool == 'shape':
                self.shape_group.setVisible(True)
                # Update engine shape
                self.engine.shape = self.current_shape
                self.engine.fill_shape = self.fill_shape
            else:
                self.shape_group.setVisible(False)
            
            # Update brush if needed
            if tool in ['brush', 'highlighter', 'dot', 'dashline']:
                self.update_brush_engine()
            
            # Update tool info label
            shape_text = f" | Shape: {self.current_shape.capitalize()}" if tool == 'shape' else ""
            self.tool_info_label.setText(f"Tool: {tool.capitalize()}{shape_text}")
            
            self.status_bar.showMessage(f"Selected {tool} tool", 2000)
            
            # Reset shape drawing state
            self.shape_drawing_state = 'idle'
            self.shape_start_point = None
            
        except Exception as e:
            print(f"Error selecting tool: {e}")
    
    def update_shape_type(self, shape_text):
        """Update selected shape type"""
        shape_map = {
            "Line": "line",
            "Rectangle": "rect",
            "Circle": "circle",
            "Triangle": "triangle",
            "Arrow": "arrow",
            "Dash Line": "dashline"
        }
        
        self.current_shape = shape_map.get(shape_text, "line")
        self.engine.shape = self.current_shape
        
        # Update tool info label
        self.tool_info_label.setText(f"Tool: Shape | Shape: {self.current_shape.capitalize()}")
        
        self.status_bar.showMessage(f"Shape changed to {shape_text}", 1000)
    
    def toggle_shape_fill(self, state):
        """Toggle shape fill"""
        self.fill_shape = (state == Qt.CheckState.Checked)
        self.engine.fill_shape = self.fill_shape
        self.status_bar.showMessage(f"Shape fill {'enabled' if self.fill_shape else 'disabled'}", 1000)
    
    def select_color(self, color):
        """Select color"""
        self.current_color = color
        self.update_brush_engine()
        self.update_color_preview()
    
    def open_color_picker(self):
        """Open color picker dialog"""
        dialog = ColorPickerDialog(
            (self.current_color.blue(), self.current_color.green(), self.current_color.red()),
            self
        )
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.result:
            b, g, r = dialog.result
            self.select_color(QColor(r, g, b))
    
    def update_color_preview(self):
        """Update color preview"""
        self.color_preview.setStyleSheet(f"""
            QLabel {{
                background-color: {self.current_color.name()};
                border: 2px solid white;
                border-radius: 15px;
            }}
        """)
    
    def update_brush_engine(self):
        """Update brush in engine"""
        try:
            color_tuple = (self.current_color.blue(), 
                          self.current_color.green(), 
                          self.current_color.red())
            self.engine.set_brush(color_tuple, self.brush_size)
        except Exception as e:
            print(f"Error updating brush engine: {e}")
    
    def update_brush_size(self, value):
        """Update brush size dynamically"""
        self.brush_size = value
        self.brush_size_label.setText(f"{value}px")
        
        # Update eraser size proportionally
        self.eraser_size = max(10, int(value * 2.5))
        
        # Update brush in engine
        self.update_brush_engine()
        
        # Update status
        self.status_bar.showMessage(f"Brush size: {value}px | Eraser size: {self.eraser_size}px", 1500)
    
    def toggle_grid(self, state):
        """Toggle grid display"""
        self.grid_enabled = (state == Qt.CheckState.Checked)
        self.canvas_widget.grid_enabled = self.grid_enabled
        self.canvas_widget.update_canvas()
    
    def toggle_webcam(self, state):
        """Toggle webcam display"""
        self.webcam_visible = (state == Qt.CheckState.Checked)
        if not self.webcam_visible:
            self.webcam_label.setText("Webcam\nDisabled")
            self.webcam_label.setPixmap(QPixmap())
    
    def toggle_gesture_mode(self, state):
        """Toggle gesture mode"""
        self.gesture_mode = (state == Qt.CheckState.Checked)
        
        if self.gesture_mode:
            self.mode_label.setText("Mode: Gesture")
            self.status_bar.showMessage("Gesture mode enabled - Use pointing to draw, fist to stop", 3000)
            
            # Ensure we're in a drawing mode for gestures
            if self.current_tool not in ['brush', 'eraser', 'highlighter', 'dot', 'dashline', 'shape', 'select']:
                self.select_tool('brush')
        else:
            self.mode_label.setText("Mode: Gesture Only (Mouse Disabled)")
            self.status_bar.showMessage("Gesture-only mode - Mouse drawing is disabled. Enable gesture mode to draw.", 2000)
            
            # Stop any active gesture drawing
            if self.gesture_active:
                self.stop_gesture_drawing()
    
    def zoom_in(self):
        """Zoom in"""
        center_x = self.canvas_widget.width() // 2
        center_y = self.canvas_widget.height() // 2
        self.viewport.zoom_in(center_x, center_y)
        self.update_zoom_display()
        self.canvas_widget.update_canvas()
    
    def zoom_out(self):
        """Zoom out"""
        center_x = self.canvas_widget.width() // 2
        center_y = self.canvas_widget.height() // 2
        self.viewport.zoom_out(center_x, center_y)
        self.update_zoom_display()
        self.canvas_widget.update_canvas()
    
    def reset_view(self):
        """Reset view"""
        self.viewport.reset_view()
        self.update_zoom_display()
        self.canvas_widget.update_canvas()
    
    def test_gesture_drawing(self):
        """Simple test method to verify gesture drawing works"""
        # Enable gesture mode
        self.gesture_mode = True
        self.gesture_checkbox.setChecked(True)
        
        # Set to brush mode
        self.select_tool('brush')
        
        # Set a bright color for visibility
        self.select_color(QColor(255, 0, 0))  # Red
        
        # Set larger brush for visibility
        self.brush_slider.setValue(15)
        
        QMessageBox.information(self, "Gesture Test",
                              "Gesture Drawing Test Started:\n\n"
                              "1. Make sure your hand is clearly visible in the webcam\n"
                              "2. Use POINTING gesture to draw (index finger extended)\n"
                              "3. Use FIST gesture to stop drawing\n"
                              "4. Use PINCH gesture to erase\n"
                              "5. Try different tools: Brush, Shapes, Dash Line, etc.\n"
                              "6. For shapes: Point to start, move to adjust, fist to complete\n"
                              "7. Make sure your hand is clearly visible in the webcam feed\n\n"
                              "The webcam feed will show recognized gestures with confidence.")
        
        # Update status
        self.status_bar.showMessage("Gesture test mode active - Point with index finger to draw", 5000)
    
    # ============================================================================
    # FILE OPERATIONS
    # ============================================================================
    
    def new_drawing(self):
        """Create new drawing"""
        reply = QMessageBox.question(self, "New Drawing",
                                   "Create new drawing?",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.engine.clear_all()
                self.canvas_widget.update_canvas()
                self.current_file = None
                self.status_bar.showMessage("New drawing created", 2000)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to create new drawing: {e}")
    
    def open_image(self):
        """Open image file"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Open Image", "",
                "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)"
            )
            
            if file_path:
                img = cv2.imread(file_path)
                if img is not None:
                    if self.engine.import_background(img):
                        self.canvas_widget.update_canvas()
                        self.current_file = file_path
                        self.status_bar.showMessage(f"Loaded: {os.path.basename(file_path)}", 3000)
                    else:
                        QMessageBox.warning(self, "Error", "Failed to load image")
                else:
                    QMessageBox.warning(self, "Error", "Could not read image file")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading image: {str(e)}")
    
    def save_image(self):
        """Save current drawing"""
        if not self.current_file:
            self.save_image_as()
        else:
            try:
                success = self.engine.save_image(self.current_file)
                if success:
                    self.status_bar.showMessage(f"Saved: {os.path.basename(self.current_file)}", 3000)
                else:
                    QMessageBox.warning(self, "Error", "Failed to save image")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error saving image: {str(e)}")
    
    def save_image_as(self):
        """Save drawing with new filename"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Drawing", "",
                "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
            )
            
            if file_path:
                self.current_file = file_path
                success = self.engine.save_image(file_path)
                
                if success:
                    self.status_bar.showMessage(f"Saved: {os.path.basename(file_path)}", 3000)
                else:
                    QMessageBox.warning(self, "Error", "Failed to save image")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving image: {str(e)}")
    
    def export_image(self, format_type):
        """Export image in specific format"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, f"Export Image", "",
                f"{format_type.upper()} Files (*.{format_type})"
            )
            
            if file_path:
                success = self.engine.save_image(file_path)
                if success:
                    self.status_bar.showMessage(f"Exported as {format_type.upper()}", 3000)
                else:
                    QMessageBox.warning(self, "Error", f"Failed to export image")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error exporting image: {str(e)}")
    
    # ============================================================================
    # EDIT OPERATIONS
    # ============================================================================
    
    def undo(self):
        """Undo last action"""
        if self.engine.undo():
            self.canvas_widget.update_canvas()
            self.status_bar.showMessage("Undo successful", 2000)
        else:
            self.status_bar.showMessage("Nothing to undo", 2000)
    
    def redo(self):
        """Redo last undone action"""
        if self.engine.redo():
            self.canvas_widget.update_canvas()
            self.status_bar.showMessage("Redo successful", 2000)
        else:
            self.status_bar.showMessage("Nothing to redo", 2000)
    
    def copy_selection(self):
        """Copy selected area"""
        if self.engine.copy_selection():
            self.status_bar.showMessage("Selection copied", 2000)
        else:
            self.status_bar.showMessage("No selection to copy", 2000)
    
    def cut_selection(self):
        """Cut selected area"""
        if self.engine.cut_selection():
            self.canvas_widget.update_canvas()
            self.status_bar.showMessage("Selection cut", 2000)
        else:
            self.status_bar.showMessage("No selection to cut", 2000)
    
    def paste_selection(self):
        """Paste from clipboard"""
        x = self.canvas_widget.width() // 2
        y = self.canvas_widget.height() // 2
        
        if self.engine.paste_from_clipboard(x, y):
            self.canvas_widget.update_canvas()
            self.status_bar.showMessage("Pasted from clipboard", 2000)
        else:
            self.status_bar.showMessage("Clipboard is empty", 2000)
    
    def clear_selection(self):
        """Clear current selection"""
        self.engine.clear_selection()
        self.canvas_widget.update_canvas()
        self.status_bar.showMessage("Selection cleared", 2000)
    
    def clear_canvas(self):
        """Clear entire canvas"""
        reply = QMessageBox.question(self, "Clear Canvas",
                                   "Clear the entire canvas?",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.engine.clear_all()
                self.canvas_widget.update_canvas()
                self.status_bar.showMessage("Canvas cleared", 2000)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to clear canvas: {e}")
    
    # ============================================================================
    # DIALOG METHODS
    # ============================================================================
    
    def show_gesture_info(self):
        """Show gesture information"""
        QMessageBox.information(self, "Gesture Controls",
                              "Available Gestures:\n\n"
                              "• POINTING (index finger) - Draw/Select\n"
                              "• FIST - Stop drawing\n"
                              "• PINCH - Erase\n"
                              "• VICTORY (V) - Cut selection\n"
                              "• SCISSORS - Copy selection\n"
                              "• OPEN HAND - No action\n\n"
                              "The system automatically adapts to your hand size.")
    
    # ============================================================================
    # EVENT HANDLING
    # ============================================================================
    
    def closeEvent(self, event):
        """Handle window close event"""
        reply = QMessageBox.question(self, "Exit",
                                   "Are you sure you want to exit?",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            # Stop camera worker
            if hasattr(self, 'camera_worker') and self.camera_worker.isRunning():
                self.camera_worker.stop()
                self.camera_worker.wait()
            
            event.accept()
        else:
            event.ignore()