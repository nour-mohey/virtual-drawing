from PyQt6.QtWidgets import (
    QPushButton, QFrame, QVBoxLayout, QHBoxLayout, QLabel, QSlider, 
    QColorDialog, QDialog, QGridLayout, QWidget
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QRect, QPoint
from PyQt6.QtGui import QColor, QPixmap, QFont, QImage
import cv2
import numpy as np
from PIL import Image

class ToolButton(QPushButton):
    def __init__(self, parent, text, command, icon=None, **kwargs):
        super().__init__(text, parent)
        self.clicked.connect(command)
        self.icon = icon
        
        # Set styling
        self.setStyleSheet("""
            QPushButton {
                background-color: #34495e;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 4px;
                font-size: 10px;
                min-height: 30px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #2c3e50;
            }
            QPushButton:pressed {
                background-color: #1a252f;
            }
        """)
        
        if icon:
            self.setIcon(icon)

class ColorButton(QPushButton):
    clicked_with_color = pyqtSignal(tuple)
    
    def __init__(self, parent, color_bgr, name, command):
        super().__init__(parent)
        self.color_bgr = color_bgr
        self.name = name
        self.command = command
        
        # Set color
        hex_color = self._bgr_to_hex(color_bgr)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {hex_color};
                border: 1px solid #555;
                padding: 5px;
                min-width: 40px;
                min-height: 40px;
                border-radius: 3px;
            }}
            QPushButton:hover {{
                border: 2px solid #fff;
            }}
        """)
        
        self.clicked.connect(self.on_clicked)
        self.setToolTip(name)
    
    def _bgr_to_hex(self, bgr):
        b, g, r = bgr
        return f'#{r:02x}{g:02x}{b:02x}'
    
    def on_clicked(self):
        self.command(self.color_bgr)


class StatusBar(QFrame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent)
        self.setStyleSheet("background-color: #2c3e50;")
        self.setMaximumHeight(30)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        
        # Status message
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: white; font-size: 10px;")
        layout.addWidget(self.status_label, 1)
        
        # Gesture info
        self.gesture_label = QLabel("Gesture: --")
        self.gesture_label.setStyleSheet("color: #95a5a6; font-size: 9px;")
        layout.addWidget(self.gesture_label)
        
        # FPS counter
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setStyleSheet("color: #95a5a6; font-size: 9px;")
        layout.addWidget(self.fps_label)
    
    def set_status(self, message):
        self.status_label.setText(message)
    
    def set_fps(self, fps):
        self.fps_label.setText(f"FPS: {fps}")
    
    def set_gesture(self, gesture, confidence=0.0):
        if confidence > 0.7:
            self.gesture_label.setText(f"Gesture: {gesture} ({confidence:.0%})")
        else:
            self.gesture_label.setText(f"Gesture: {gesture}")

class ToolPanel(QFrame):
    def __init__(self, parent, title, **kwargs):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background-color: #34495e;
                border-radius: 4px;
                padding: 10px;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        
        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet("color: white; font-size: 11px; font-weight: bold;")
        layout.addWidget(title_label)
        
        # Content frame
        self.content_frame = QFrame()
        self.content_layout = QVBoxLayout(self.content_frame)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.content_frame)
    
    def add_widget(self, widget, **kwargs):
        self.content_layout.addWidget(widget, **kwargs)

class ZoomSlider(QSlider):
    def __init__(self, parent, command, **kwargs):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.setMinimum(10)
        self.setMaximum(500)
        self.setValue(100)
        self.setMaximumWidth(200)
        self.valueChanged.connect(command)
        
        self.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #2c3e50;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3498db;
                width: 12px;
                margin: -2px 0;
                border-radius: 6px;
            }
            QSlider::handle:horizontal:hover {
                background: #2980b9;
            }
        """)


class CanvasDisplay(QFrame):
    def __init__(self, parent, width, height, **kwargs):
        super().__init__(parent)
        self.setStyleSheet("background-color: white;")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create scroll area
        from PyQt6.QtWidgets import QScrollArea
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        
        # Canvas widget
        self.canvas_widget = QLabel()
        self.canvas_widget.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.canvas_widget.setStyleSheet("background-color: white;")
        self.scroll_area.setWidget(self.canvas_widget)
        
        layout.addWidget(self.scroll_area)
        
        self.pixmap = None
        self.zoom_level = 100
    
    def display_image(self, pil_image):
        """Display PIL Image on canvas."""
        try:
            if pil_image is None:
                return
            
            # Convert PIL image to QPixmap
            import io
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            buffer.seek(0)
            
            pixmap = QPixmap()
            pixmap.loadFromData(buffer.getvalue())
            
            # Apply zoom
            scaled_pixmap = pixmap.scaledToWidth(
                int(pixmap.width() * self.zoom_level / 100),
                Qt.TransformationMode.SmoothTransformation
            )
            
            self.pixmap = pixmap
            self.canvas_widget.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"Error displaying image on canvas: {e}")
    
    def clear(self):
        """Clear canvas display."""
        try:
            self.canvas_widget.clear()
            self.pixmap = None
        except Exception as e:
            print(f"Error clearing canvas: {e}")
    
    def set_zoom(self, zoom_level):
        """Set zoom level."""
        self.zoom_level = zoom_level
        if self.pixmap:
            scaled_pixmap = self.pixmap.scaledToWidth(
                int(self.pixmap.width() * zoom_level / 100),
                Qt.TransformationMode.SmoothTransformation
            )
            self.canvas_widget.setPixmap(scaled_pixmap)
    
    def get_canvas_coords(self, event):
        """Get canvas coordinates from event."""
        try:
            pos = self.scroll_area.mapFromGlobal(event.globalPos())
            return pos.x(), pos.y()
        except:
            return event.pos().x(), event.pos().y()

class ColorPickerDialog(QDialog):
    def __init__(self, parent, current_color=(0, 0, 0)):
        super().__init__(parent)
        self.setWindowTitle("Color Picker")
        self.current_color = current_color
        self.result = None
        
        self.setGeometry(100, 100, 400, 500)
        self.setModal(True)
        
        self.build_ui()
    
    def build_ui(self):
        layout = QVBoxLayout(self)
        
        # Current color display
        hex_color = self._bgr_to_hex(self.current_color)
        self.current_display = QFrame()
        self.current_display.setStyleSheet(f"background-color: {hex_color};")
        self.current_display.setMinimumHeight(60)
        layout.addWidget(self.current_display)
        
        # Color presets
        presets_layout = QGridLayout()
        presets = [
            ("Black", (0, 0, 0)),
            ("White", (255, 255, 255)),
            ("Red", (0, 0, 255)),
            ("Green", (0, 255, 0)),
            ("Blue", (255, 0, 0)),
            ("Yellow", (0, 255, 255)),
            ("Magenta", (255, 0, 255)),
            ("Cyan", (255, 255, 0))
        ]
        
        for i, (name, color) in enumerate(presets):
            btn = ColorButton(self, color, name, self.on_preset_selected)
            presets_layout.addWidget(btn, i // 4, i % 4)
        
        layout.addLayout(presets_layout)
        
        # RGB sliders
        sliders_frame = QWidget()
        sliders_layout = QGridLayout(sliders_frame)
        
        self.red_slider = QSlider(Qt.Orientation.Horizontal)
        self.red_slider.setMinimum(0)
        self.red_slider.setMaximum(255)
        self.red_slider.setValue(self.current_color[2])
        self.red_slider.valueChanged.connect(self.on_slider_change)
        sliders_layout.addWidget(QLabel("Red:"), 0, 0)
        sliders_layout.addWidget(self.red_slider, 0, 1)
        
        self.green_slider = QSlider(Qt.Orientation.Horizontal)
        self.green_slider.setMinimum(0)
        self.green_slider.setMaximum(255)
        self.green_slider.setValue(self.current_color[1])
        self.green_slider.valueChanged.connect(self.on_slider_change)
        sliders_layout.addWidget(QLabel("Green:"), 1, 0)
        sliders_layout.addWidget(self.green_slider, 1, 1)
        
        self.blue_slider = QSlider(Qt.Orientation.Horizontal)
        self.blue_slider.setMinimum(0)
        self.blue_slider.setMaximum(255)
        self.blue_slider.setValue(self.current_color[0])
        self.blue_slider.valueChanged.connect(self.on_slider_change)
        sliders_layout.addWidget(QLabel("Blue:"), 2, 0)
        sliders_layout.addWidget(self.blue_slider, 2, 1)
        
        layout.addWidget(sliders_frame)
        
        # Buttons
        buttons_layout = QHBoxLayout()
        
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.on_ok)
        buttons_layout.addWidget(ok_button)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.on_cancel)
        buttons_layout.addWidget(cancel_button)
        
        system_picker_button = QPushButton("System Picker")
        system_picker_button.clicked.connect(self.on_system_picker)
        buttons_layout.addWidget(system_picker_button)
        
        layout.addLayout(buttons_layout)
    
    def _bgr_to_hex(self, bgr):
        b, g, r = bgr
        return f'#{r:02x}{g:02x}{b:02x}'
    
    def on_preset_selected(self, color):
        self.current_color = color
        self.red_slider.setValue(color[2])
        self.green_slider.setValue(color[1])
        self.blue_slider.setValue(color[0])
        self.update_display()
    
    def on_slider_change(self):
        self.current_color = (
            self.blue_slider.value(),
            self.green_slider.value(),
            self.red_slider.value()
        )
        self.update_display()
    
    def update_display(self):
        hex_color = self._bgr_to_hex(self.current_color)
        self.current_display.setStyleSheet(f"background-color: {hex_color};")
    
    def on_system_picker(self):
        color = QColorDialog.getColor(
            QColor(*self._bgr_to_rgb(self.current_color)),
            self,
            "Choose color"
        )
        if color.isValid():
            r, g, b = color.red(), color.green(), color.blue()
            self.current_color = (b, g, r)
            self.red_slider.setValue(r)
            self.green_slider.setValue(g)
            self.blue_slider.setValue(b)
            self.update_display()
    
    def _bgr_to_rgb(self, bgr):
        b, g, r = bgr
        return r, g, b
    
    def on_ok(self):
        self.result = self.current_color
        self.accept()
    
    def on_cancel(self):
        self.result = None
        self.reject()
    
    def show_dialog(self):
        self.exec()
        return self.result