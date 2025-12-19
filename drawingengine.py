"""
Drawing Engine - Enhanced for PyQt5
"""
import numpy as np
import cv2
from collections import deque
import time
import math

class DrawingEngine:
    """Enhanced drawing engine with PyQt5 optimization."""
    
    def __init__(self, h=720, w=1280):
        self.h = h
        self.w = w
        
        # Create canvas with white background
        self.canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # Brush settings
        self.brush_color = (0, 0, 0)  # BGR
        self.brush_size = 8
        self.eraser_size = 30
        self.mode = 'brush'
        self.shape = 'line'
        self.fill_shape = False
        
        # Stroke tracking
        self.last_point = None
        self.drawing = False
        self.shape_start = None
        self.shape_end = None
        self.preview_canvas = None
        
        # Selection state
        self.selection_rect = None
        self.selection_active = False
        self.is_selecting = False
        
        # Clipboard for cut/copy/paste
        self.clipboard = None
        self.clipboard_rect = None
        
        # History with incremental saves for PyQt5 optimization
        self.history = deque(maxlen=50)
        self.redo_stack = deque(maxlen=50)
        self.save_state()
        
        # Performance optimization
        self.brush_cache = {}
        self._generate_brush_cache()
        self.dirty_regions = []
        
        # For PyQt5 integration
        self.needs_update = False
        self.update_callback = None
    
    def set_update_callback(self, callback):
        """Set callback for UI updates"""
        self.update_callback = callback
    
    def _generate_brush_cache(self):
        """Generate brush shapes for common sizes"""
        for size in range(1, 51):
            diameter = size * 2 + 1
            kernel = np.zeros((diameter, diameter), dtype=np.uint8)
            center = size
            
            # Create circular brush
            y, x = np.ogrid[-size:size+1, -size:size+1]
            mask = x*x + y*y <= size*size
            kernel[mask] = 255
            
            self.brush_cache[size] = kernel
    
    def save_state(self):
        """Save current canvas state."""
        try:
            if len(self.canvas.shape) == 3:
                # Save only if changed
                if not self.history or not np.array_equal(self.history[-1], self.canvas):
                    self.history.append(self.canvas.copy())
                    self.redo_stack.clear()
                    self.needs_update = True
        except Exception as e:
            print(f"Error saving state: {e}")
    
    def undo(self):
        """Undo last action."""
        try:
            if len(self.history) > 1:
                current = self.history.pop()
                self.redo_stack.append(current)
                self.canvas = self.history[-1].copy()
                self.needs_update = True
                if self.update_callback:
                    self.update_callback()
                return True
            return False
        except Exception as e:
            print(f"Error in undo: {e}")
            return False
    
    def redo(self):
        """Redo last undone action."""
        try:
            if self.redo_stack:
                next_state = self.redo_stack.pop()
                self.history.append(next_state)
                self.canvas = next_state.copy()
                self.needs_update = True
                if self.update_callback:
                    self.update_callback()
                return True
            return False
        except Exception as e:
            print(f"Error in redo: {e}")
            return False
    
    def set_brush(self, bgr_tuple, size):
        """Set brush color and size."""
        try:
            self.brush_color = tuple(int(c) for c in bgr_tuple)
            self.brush_size = max(1, int(size))
        except Exception as e:
            print(f"Error setting brush: {e}")
    
    def set_mode(self, mode):
        """Set drawing mode."""
        self.mode = mode
        if mode == 'select':
            self.selection_active = True
        else:
            self.selection_active = False
    
    def start_stroke(self, x, y):
        """Start a new stroke."""
        try:
            x, y = int(x), int(y)
            
            if self.mode == 'shape' or self.mode == 'select':
                self.shape_start = (x, y)
                self.preview_canvas = self.canvas.copy()
                self.is_selecting = True
            else:
                self.save_state()
                self.drawing = True
                self.last_point = (x, y)
                self._draw_point(x, y)
                
        except Exception as e:
            print(f"Error starting stroke: {e}")
    
    def continue_stroke(self, x, y):
        """Continue the stroke with performance optimization."""
        try:
            x, y = int(x), int(y)
            
            if self.mode == 'shape' or self.mode == 'select':
                self.shape_end = (x, y)
                self._draw_shape_preview()
                return
                
            if not self.drawing or self.last_point is None:
                self.start_stroke(x, y)
                return
            
            x2, y2 = x, y
            x1, y1 = self.last_point
            
            # Optimized drawing based on mode
            if self.mode == 'eraser':
                self._draw_line_optimized(x1, y1, x2, y2, 
                                        (255, 255, 255), self.eraser_size)
            elif self.mode == 'highlighter':
                self._draw_highlighter_line(x1, y1, x2, y2)
            elif self.mode == 'dot':
                self._draw_point(x, y)
            elif self.mode == 'dashline':
                self._draw_dashed_line(x1, y1, x2, y2, 
                                     self.brush_color, self.brush_size)
            else:
                self._draw_line_optimized(x1, y1, x2, y2, 
                                        self.brush_color, self.brush_size)
            
            self.last_point = (x2, y2)
            
            # Notify UI of update
            self.needs_update = True
            if self.update_callback:
                self.update_callback()
            
        except Exception as e:
            print(f"Error continuing stroke: {e}")
    
    def _draw_line_optimized(self, x1, y1, x2, y2, color, thickness):
        """Optimized line drawing with batch processing."""
        # Calculate line points using Bresenham's algorithm
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        if dx == 0 and dy == 0:
            points.append((x1, y1))
        else:
            if dx > dy:
                # Horizontal line
                if x1 > x2:
                    x1, x2 = x2, x1
                    y1, y2 = y2, y1
                
                for x in range(x1, x2 + 1):
                    y = int(y1 + (x - x1) * (y2 - y1) / (x2 - x1))
                    points.append((x, y))
            else:
                # Vertical line
                if y1 > y2:
                    x1, x2 = x2, x1
                    y1, y2 = y2, y1
                
                for y in range(y1, y2 + 1):
                    x = int(x1 + (y - y1) * (x2 - x1) / (y2 - y1))
                    points.append((x, y))
        
        # Draw all points at once
        for px, py in points:
            self._draw_point_optimized(px, py, color, thickness)
        
        # Notify UI of update after line is drawn
        self.needs_update = True
        if self.update_callback:
            self.update_callback()
    
    def _draw_point_optimized(self, x, y, color, size):
        """Optimized point drawing with brush cache."""
        if size in self.brush_cache:
            kernel = self.brush_cache[size]
            half_size = size
            
            # Calculate bounds
            x_start = max(0, x - half_size)
            x_end = min(self.w, x + half_size + 1)
            y_start = max(0, y - half_size)
            y_end = min(self.h, y + half_size + 1)
            
            # Calculate kernel slice
            kx_start = half_size - (x - x_start)
            kx_end = kx_start + (x_end - x_start)
            ky_start = half_size - (y - y_start)
            ky_end = ky_start + (y_end - y_start)
            
            kernel_slice = kernel[ky_start:ky_end, kx_start:kx_end]
            mask = kernel_slice > 0
            
            # Apply color efficiently
            roi = self.canvas[y_start:y_end, x_start:x_end]
            for c in range(3):
                roi[:, :, c][mask] = color[c]
        else:
            # Fallback to circle
            cv2.circle(self.canvas, (x, y), size//2, color, -1)
    
    def _draw_highlighter_line(self, x1, y1, x2, y2):
        """Draw highlighter line with transparency."""
        # Create temporary overlay
        overlay = self.canvas.copy()
        cv2.line(overlay, (x1, y1), (x2, y2), 
                (0, 255, 255), self.brush_size, lineType=cv2.LINE_AA)
        
        # Blend with canvas
        cv2.addWeighted(overlay, 0.4, self.canvas, 0.6, 0, self.canvas)
        
        # Notify UI of update
        self.needs_update = True
        if self.update_callback:
            self.update_callback()
    
    def _draw_dashed_line(self, x1, y1, x2, y2, color, thickness, dash_length=10, gap_length=5):
        """Draw a dashed line between two points."""
        try:
            # Calculate line length and direction
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx*dx + dy*dy)
            
            if length == 0:
                return
                
            # Normalize direction vector
            dx /= length
            dy /= length
            
            # Draw dashes
            total_dash = dash_length + gap_length
            num_dashes = int(length / total_dash)
            
            for i in range(num_dashes):
                start_x = int(x1 + i * total_dash * dx)
                start_y = int(y1 + i * total_dash * dy)
                end_x = int(start_x + dash_length * dx)
                end_y = int(start_y + dash_length * dy)
                
                cv2.line(self.canvas, (start_x, start_y), (end_x, end_y), 
                        color, thickness, lineType=cv2.LINE_AA)
            
            # Draw remaining segment if any
            remaining = length - num_dashes * total_dash
            if remaining > dash_length:
                start_x = int(x1 + num_dashes * total_dash * dx)
                start_y = int(y1 + num_dashes * total_dash * dy)
                cv2.line(self.canvas, (start_x, start_y), (x2, y2), 
                        color, thickness, lineType=cv2.LINE_AA)
            
            # Notify UI of update
            self.needs_update = True
            if self.update_callback:
                self.update_callback()
                        
        except Exception as e:
            print(f"Error drawing dashed line: {e}")
    
    def _draw_shape_preview(self):
        """Draw shape preview on temporary canvas."""
        if not self.shape_start or not self.shape_end:
            return
            
        try:
            # Restore canvas to state before preview
            self.canvas = self.preview_canvas.copy()
            
            # Draw preview with temporary color
            preview_color = (100, 100, 100)
            self._draw_shape_internal(self.shape_start, self.shape_end, preview_color, preview=True)
            
            self.needs_update = True
            if self.update_callback:
                self.update_callback()
                
        except Exception as e:
            print(f"Error drawing shape preview: {e}")
    
    def _draw_shape_final(self):
        """Draw final shape on canvas."""
        if not self.shape_start or not self.shape_end:
            return
            
        try:
            self.save_state()
            if self.mode == 'select':
                # For selection, just save rectangle
                self._draw_shape_internal(self.shape_start, self.shape_end, (0, 0, 255), preview=True)
            else:
                self._draw_shape_internal(self.shape_start, self.shape_end, self.brush_color, preview=False)
            
            # Notify UI of update
            self.needs_update = True
            if self.update_callback:
                self.update_callback()
                
        except Exception as e:
            print(f"Error drawing final shape: {e}")
    
    def _draw_shape_internal(self, start, end, color, preview=False):
        """Internal method to draw shapes."""
        try:
            thickness = self.brush_size if not preview else 2
            
            if self.shape == 'line':
                if self.mode == 'dashline' or self.shape == 'dashline':
                    self._draw_dashed_line(start[0], start[1], end[0], end[1],
                                         color, thickness)
                else:
                    cv2.line(self.canvas, start, end, color, thickness, lineType=cv2.LINE_AA)
            
            elif self.shape == 'rect':
                if self.fill_shape and not preview:
                    cv2.rectangle(self.canvas, start, end, color, -1, lineType=cv2.LINE_AA)
                else:
                    cv2.rectangle(self.canvas, start, end, color, thickness, lineType=cv2.LINE_AA)
            
            elif self.shape == 'circle':
                center_x = (start[0] + end[0]) // 2
                center_y = (start[1] + end[1]) // 2
                radius = int(((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5) // 2
                
                if self.fill_shape and not preview:
                    cv2.circle(self.canvas, (center_x, center_y), radius, color, -1, lineType=cv2.LINE_AA)
                else:
                    cv2.circle(self.canvas, (center_x, center_y), radius, color, thickness, lineType=cv2.LINE_AA)
            
            elif self.shape == 'triangle':
                x1, y1 = start
                x2, y2 = end
                
                # Calculate third point for triangle
                x3 = x1
                y3 = y2
                
                pts = np.array([[x1, y1], [x2, y2], [x3, y3]], np.int32)
                
                if self.fill_shape and not preview:
                    cv2.fillPoly(self.canvas, [pts], color)
                else:
                    cv2.polylines(self.canvas, [pts], True, color, thickness, lineType=cv2.LINE_AA)
            
            elif self.shape == 'arrow':
                x1, y1 = start
                x2, y2 = end
                cv2.arrowedLine(self.canvas, (x1, y1), (x2, y2), color, thickness, 
                              tipLength=0.3, line_type=cv2.LINE_AA)
        
        except Exception as e:
            print(f"Error in _draw_shape_internal: {e}")
    
    def end_stroke(self):
        """End the current stroke."""
        try:
            if (self.mode == 'shape' or self.mode == 'select') and self.shape_start and self.shape_end:
                self._draw_shape_final()
                
                if self.mode == 'select':
                    x1, y1 = self.shape_start
                    x2, y2 = self.shape_end
                    self.selection_rect = (
                        min(x1, x2), min(y1, y2),
                        abs(x2 - x1), abs(y2 - y1)
                    )
                    self.selection_active = True
                
                self.shape_start = None
                self.shape_end = None
                self.preview_canvas = None
                self.is_selecting = False
            else:
                self.drawing = False
                self.last_point = None
            
            self.needs_update = True
            if self.update_callback:
                self.update_callback()
                
        except Exception as e:
            print(f"Error ending stroke: {e}")
    
    def _draw_point(self, x, y):
        """Draw a single point."""
        try:
            x, y = int(x), int(y)
            
            if self.mode == 'eraser':
                cv2.circle(self.canvas, (x, y), self.eraser_size//2, 
                          (255, 255, 255), -1)
            elif self.mode == 'highlighter':
                overlay = self.canvas.copy()
                cv2.circle(overlay, (x, y), self.brush_size//2, 
                          (0, 255, 255), -1)
                cv2.addWeighted(overlay, 0.4, self.canvas, 0.6, 0, self.canvas)
            else:
                self._draw_point_optimized(x, y, self.brush_color, self.brush_size)
                
        except Exception as e:
            print(f"Error drawing point: {e}")
    
    # Clipboard Operations
    def copy_selection(self):
        """Copy selected area to clipboard."""
        try:
            if self.selection_rect:
                x, y, w, h = self.selection_rect
                
                if (x >= 0 and y >= 0 and w > 0 and h > 0 and 
                    x + w <= self.w and y + h <= self.h):
                    self.clipboard = self.canvas[y:y+h, x:x+w].copy()
                    self.clipboard_rect = self.selection_rect
                    return True
            return False
        except Exception as e:
            print(f"Error copying selection: {e}")
            return False
    
    def cut_selection(self):
        """Cut selected area to clipboard and remove from canvas."""
        try:
            if self.selection_rect and self.copy_selection():
                self.save_state()
                x, y, w, h = self.selection_rect
                
                if (x >= 0 and y >= 0 and w > 0 and h > 0 and 
                    x + w <= self.w and y + h <= self.h):
                    self.canvas[y:y+h, x:x+w] = (255, 255, 255)
                    self.clear_selection()
                    self.needs_update = True
                    if self.update_callback:
                        self.update_callback()
                    return True
            return False
        except Exception as e:
            print(f"Error cutting selection: {e}")
            return False
    
    def paste_from_clipboard(self, x, y):
        """Paste clipboard content at specified position."""
        try:
            if self.clipboard is not None:
                self.save_state()
                h, w = self.clipboard.shape[:2]
                
                # Ensure paste is within bounds
                paste_x = max(0, min(int(x), self.w - w))
                paste_y = max(0, min(int(y), self.h - h))
                
                if w > 0 and h > 0:
                    roi = self.canvas[paste_y:paste_y+h, paste_x:paste_x+w]
                    if roi.shape == self.clipboard.shape:
                        self.canvas[paste_y:paste_y+h, paste_x:paste_x+w] = self.clipboard
                        self.needs_update = True
                        if self.update_callback:
                            self.update_callback()
                        return True
            return False
        except Exception as e:
            print(f"Error pasting from clipboard: {e}")
            return False
    
    def clear_selection(self):
        """Clear current selection."""
        self.selection_rect = None
        self.selection_active = False
    
    def merge_layers(self):
        """Return current canvas."""
        try:
            return self.canvas.copy()
        except Exception as e:
            print(f"Error merging layers: {e}")
            return np.ones((self.h, self.w, 3), dtype=np.uint8) * 255
    
    def import_background(self, bgr_img):
        """Import background image."""
        try:
            self.save_state()
            
            # Resize to fit
            h, w = bgr_img.shape[:2]
            if (h, w) != (self.h, self.w):
                bgr_img = cv2.resize(bgr_img, (self.w, self.h))
            
            self.canvas = bgr_img.copy()
            self.needs_update = True
            if self.update_callback:
                self.update_callback()
            return True
        except Exception as e:
            print(f"Error importing background: {e}")
            return False
    
    def import_picture(self, bgr_img, x, y):
        """Import a picture at specific location."""
        try:
            self.save_state()
            
            h, w = bgr_img.shape[:2]
            x, y = int(x), int(y)
            
            # Ensure picture fits within canvas
            w = min(w, self.w - x) if x + w > self.w else w
            h = min(h, self.h - y) if y + h > self.h else h
            
            if w > 0 and h > 0 and x >= 0 and y >= 0:
                roi = self.canvas[y:y+h, x:x+w]
                if roi.shape == bgr_img[:h, :w].shape:
                    self.canvas[y:y+h, x:x+w] = bgr_img[:h, :w]
                    self.needs_update = True
                    if self.update_callback:
                        self.update_callback()
                    return True
            return False
        except Exception as e:
            print(f"Error importing picture: {e}")
            return False
    
    def save_image(self, path, as_jpeg=False):
        """Save image to file with high fidelity."""
        try:
            # Get merged canvas (all layers)
            canvas_to_save = self.merge_layers()
            
            if as_jpeg:
                # High quality JPEG
                cv2.imwrite(path, canvas_to_save, [cv2.IMWRITE_JPEG_QUALITY, 100])
            else:
                # High fidelity PNG (compression level 0 = no compression, maximum quality)
                # Also ensure we're saving in BGR format correctly
                if path.lower().endswith('.png'):
                    # PNG with maximum quality (compression level 0)
                    cv2.imwrite(path, canvas_to_save, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                else:
                    # Other formats
                    cv2.imwrite(path, canvas_to_save)
            return True
        except Exception as e:
            print(f"Error saving image: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def clear_active_layer(self):
        """Clear canvas."""
        try:
            self.save_state()
            self.canvas = np.ones((self.h, self.w, 3), dtype=np.uint8) * 255
            self.clear_selection()
            self.needs_update = True
            if self.update_callback:
                self.update_callback()
        except Exception as e:
            print(f"Error clearing active layer: {e}")
    
    def clear_all(self):
        """Clear everything and reset."""
        try:
            self.save_state()
            self.canvas = np.ones((self.h, self.w, 3), dtype=np.uint8) * 255
            self.clipboard = None
            self.clear_selection()
            self.history.clear()
            self.redo_stack.clear()
            self.save_state()
            self.needs_update = True
            if self.update_callback:
                self.update_callback()
        except Exception as e:
            print(f"Error clearing all: {e}")
    
    def get_dirty_regions(self):
        """Get regions that need updating."""
        if self.dirty_regions:
            regions = self.dirty_regions.copy()
            self.dirty_regions.clear()
            return regions
        return None