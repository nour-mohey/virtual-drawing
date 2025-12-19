import json
import os
from dataclasses import dataclass, asdict

@dataclass
class GestureConfig:
    sensitivity: float = 0.5
    pinch_threshold: float = 30
    fist_threshold: float = 50
    hand_size_normalization: bool = True
    require_consistent_gestures: int = 3

@dataclass
class DrawingConfig:
    default_brush_size: int = 8
    smooth_strokes: bool = True
    pressure_sensitivity: bool = False
    auto_save_interval: int = 300  # seconds
    max_undo_steps: int = 50

@dataclass
class TrackingConfig:
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.7
    smoothing_buffer_size: int = 5
    enable_background_subtraction: bool = True

# Update ConfigManager to include tracking
class ConfigManager:
    def __init__(self, config_file='config.json'):
        self.config_file = config_file
        self.gesture = GestureConfig()
        self.drawing = DrawingConfig()
        self.ui = UIConfig()
        self.tracking = TrackingConfig()  # Add this
        self.load_config()
        
@dataclass
class UIConfig:
    theme: str = 'dark'
    show_fps: bool = True
    show_gesture_hints: bool = True
    canvas_width: int = 1280
    canvas_height: int = 720

class ConfigManager:
    def __init__(self, config_file='config.json'):
        self.config_file = config_file
        self.gesture = GestureConfig()
        self.drawing = DrawingConfig()
        self.ui = UIConfig()
        self.load_config()
    
    def load_config(self):
        """Load configuration from file with error handling."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    
                    # Load with validation
                    if 'gesture' in data:
                        gesture_data = data.get('gesture', {})
                        self.gesture = GestureConfig(
                            sensitivity=float(gesture_data.get('sensitivity', 0.5)),
                            pinch_threshold=float(gesture_data.get('pinch_threshold', 30)),
                            fist_threshold=float(gesture_data.get('fist_threshold', 50)),
                            hand_size_normalization=bool(gesture_data.get('hand_size_normalization', True)),
                            require_consistent_gestures=int(gesture_data.get('require_consistent_gestures', 3))
                        )
                    
                    if 'drawing' in data:
                        drawing_data = data.get('drawing', {})
                        self.drawing = DrawingConfig(
                            default_brush_size=int(drawing_data.get('default_brush_size', 8)),
                            smooth_strokes=bool(drawing_data.get('smooth_strokes', True)),
                            pressure_sensitivity=bool(drawing_data.get('pressure_sensitivity', False)),
                            auto_save_interval=int(drawing_data.get('auto_save_interval', 300)),
                            max_undo_steps=int(drawing_data.get('max_undo_steps', 50))
                        )
                    
                    if 'ui' in data:
                        ui_data = data.get('ui', {})
                        self.ui = UIConfig(
                            theme=str(ui_data.get('theme', 'dark')),
                            show_fps=bool(ui_data.get('show_fps', True)),
                            show_gesture_hints=bool(ui_data.get('show_gesture_hints', True)),
                            canvas_width=int(ui_data.get('canvas_width', 1280)),
                            canvas_height=int(ui_data.get('canvas_height', 720))
                        )
                        
                print(f"✅ Config loaded from {self.config_file}")
            except Exception as e:
                print(f"❌ Error loading config: {e}")
                print("⚠️ Using default configuration")
                self._create_default_config()
        else:
            print("⚠️ Config file not found, creating default")
            self._create_default_config()
            self.save_config()
    
    def _create_default_config(self):
        """Create default configuration."""
        self.gesture = GestureConfig()
        self.drawing = DrawingConfig()
        self.ui = UIConfig()
    
    def save_config(self):
        """Save configuration to file with error handling."""
        try:
            config_data = {
                'gesture': asdict(self.gesture),
                'drawing': asdict(self.drawing),
                'ui': asdict(self.ui)
            }
            
            # Create directory if it doesn't exist
            config_dir = os.path.dirname(self.config_file)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir)
            
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=4, ensure_ascii=False)
            
            print(f"✅ Config saved to {self.config_file}")
            return True
        except Exception as e:
            print(f"❌ Error saving config: {e}")
            return False
    
    def update_gesture_config(self, **kwargs):
        """Update gesture configuration with validation."""
        try:
            for key, value in kwargs.items():
                if hasattr(self.gesture, key):
                    setattr(self.gesture, key, value)
            return True
        except Exception as e:
            print(f"Error updating gesture config: {e}")
            return False
    
    def update_drawing_config(self, **kwargs):
        """Update drawing configuration with validation."""
        try:
            for key, value in kwargs.items():
                if hasattr(self.drawing, key):
                    setattr(self.drawing, key, value)
            return True
        except Exception as e:
            print(f"Error updating drawing config: {e}")
            return False
    
    def update_ui_config(self, **kwargs):
        """Update UI configuration with validation."""
        try:
            for key, value in kwargs.items():
                if hasattr(self.ui, key):
                    setattr(self.ui, key, value)
            return True
        except Exception as e:
            print(f"Error updating UI config: {e}")
            return False
    
    def validate_config(self):
        """Validate all configuration values."""
        try:
            # Validate gesture config
            assert 0 <= self.gesture.sensitivity <= 1, "Sensitivity must be between 0 and 1"
            assert self.gesture.pinch_threshold > 0, "Pinch threshold must be positive"
            assert self.gesture.fist_threshold > 0, "Fist threshold must be positive"
            assert self.gesture.require_consistent_gestures >= 1, "Must require at least 1 consistent gesture"
            
            # Validate drawing config
            assert self.drawing.default_brush_size > 0, "Brush size must be positive"
            assert self.drawing.auto_save_interval >= 0, "Auto-save interval cannot be negative"
            assert self.drawing.max_undo_steps >= 0, "Max undo steps cannot be negative"
            
            # Validate UI config
            assert self.ui.theme in ['dark', 'light'], "Theme must be 'dark' or 'light'"
            assert self.ui.canvas_width > 0, "Canvas width must be positive"
            assert self.ui.canvas_height > 0, "Canvas height must be positive"
            
            return True
        except AssertionError as e:
            print(f"Config validation error: {e}")
            return False
        except Exception as e:
            print(f"Unexpected config validation error: {e}")
            return False
        