import numpy as np

class ViewportManager:
    def __init__(self, canvas_width, canvas_height):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        
        # Viewport state
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        # Viewport bounds
        self.min_zoom = 0.1
        self.max_zoom = 5.0
        
        # Panning state
        self.is_panning = False
        self.pan_start = (0, 0)
        
        # Zoom center
        self.zoom_center = (canvas_width // 2, canvas_height // 2)
    
    def screen_to_canvas(self, screen_x, screen_y):
        """Convert screen coordinates to canvas coordinates."""
        canvas_x = (screen_x - self.offset_x) / self.zoom
        canvas_y = (screen_y - self.offset_y) / self.zoom
        
        # Clamp to canvas bounds
        canvas_x = max(0, min(self.canvas_width, canvas_x))
        canvas_y = max(0, min(self.canvas_height, canvas_y))
        
        return canvas_x, canvas_y
    
    def canvas_to_screen(self, canvas_x, canvas_y):
        """Convert canvas coordinates to screen coordinates."""
        screen_x = canvas_x * self.zoom + self.offset_x
        screen_y = canvas_y * self.zoom + self.offset_y
        return screen_x, screen_y
    
    def zoom_to_point(self, point_x, point_y, zoom_delta):
        """Zoom in/out keeping point under cursor fixed."""
        old_zoom = self.zoom
        
        # Update zoom level
        self.zoom = max(self.min_zoom, min(self.max_zoom, self.zoom + zoom_delta))
        
        if old_zoom != self.zoom:
            # Calculate scaling factor
            scale_change = self.zoom / old_zoom
            
            # Adjust offset to keep point fixed
            self.offset_x = point_x - (point_x - self.offset_x) * scale_change
            self.offset_y = point_y - (point_y - self.offset_y) * scale_change
            
            # Update zoom center
            self.zoom_center = (point_x, point_y)
            
            return True
        return False
    
    def zoom_in(self, center_x=None, center_y=None):
        """Zoom in by fixed amount."""
        center_x = center_x or self.zoom_center[0]
        center_y = center_y or self.zoom_center[1]
        return self.zoom_to_point(center_x, center_y, 0.2)
    
    def zoom_out(self, center_x=None, center_y=None):
        """Zoom out by fixed amount."""
        center_x = center_x or self.zoom_center[0]
        center_y = center_y or self.zoom_center[1]
        return self.zoom_to_point(center_x, center_y, -0.2)
    
    def reset_view(self):
        """Reset viewport to default."""
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.zoom_center = (self.canvas_width // 2, self.canvas_height // 2)
    
    def start_pan(self, start_x, start_y):
        """Start panning operation."""
        self.is_panning = True
        self.pan_start = (start_x - self.offset_x, start_y - self.offset_y)
    
    def update_pan(self, current_x, current_y):
        """Update panning position."""
        if self.is_panning:
            self.offset_x = current_x - self.pan_start[0]
            self.offset_y = current_y - self.pan_start[1]
            return True
        return False
    
    def end_pan(self):
        """End panning operation."""
        self.is_panning = False
    
    def get_viewport_rect(self):
        """Get the visible portion of canvas in screen coordinates."""
        screen_tl = self.canvas_to_screen(0, 0)
        screen_br = self.canvas_to_screen(self.canvas_width, self.canvas_height)
        
        return (
            min(screen_tl[0], screen_br[0]),
            min(screen_tl[1], screen_br[1]),
            abs(screen_br[0] - screen_tl[0]),
            abs(screen_br[1] - screen_tl[1])
        )
    
    def is_point_visible(self, canvas_x, canvas_y):
        """Check if a canvas point is visible in viewport."""
        screen_x, screen_y = self.canvas_to_screen(canvas_x, canvas_y)
        
        viewport_rect = self.get_viewport_rect()
        
        return (viewport_rect[0] <= screen_x <= viewport_rect[0] + viewport_rect[2] and
                viewport_rect[1] <= screen_y <= viewport_rect[1] + viewport_rect[3])
    
    def fit_to_canvas(self, screen_width, screen_height):
        """Fit canvas to screen."""
        scale_x = screen_width / self.canvas_width
        scale_y = screen_height / self.canvas_height
        self.zoom = min(scale_x, scale_y) * 0.95  # 95% to add some margin
        
        # Center canvas
        self.offset_x = (screen_width - self.canvas_width * self.zoom) / 2
        self.offset_y = (screen_height - self.canvas_height * self.zoom) / 2
        
        self.zoom_center = (screen_width // 2, screen_height // 2)
    
    def get_transform_matrix(self):
        """Get transformation matrix for rendering."""
        return np.array([
            [self.zoom, 0, self.offset_x],
            [0, self.zoom, self.offset_y],
            [0, 0, 1]
        ])