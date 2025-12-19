import numpy as np
import cv2
from collections import deque
from datetime import datetime

class ClipboardManager:
    def __init__(self, max_clipboard_items=10):
        self.clipboard_stack = deque(maxlen=max_clipboard_items)
        self.selection_rect = None
        self.selection_content = None
        self.cut_mode = False
        
        # Visual feedback
        self.selection_visible = False
        self.selection_alpha = 0.3
        self.marching_ants_offset = 0
        self.marching_ants_speed = 0.1
        
        # Selection handles for resizing
        self.resize_handles = []
        self.active_handle = None
        self.handle_size = 8
        
    def select_area(self, start_point, end_point):
        """Select rectangular area on canvas."""
        x1, y1 = start_point
        x2, y2 = end_point
        
        # Ensure proper rectangle coordinates
        x = min(x1, x2)
        y = min(y1, y2)
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        
        self.selection_rect = (x, y, width, height)
        self.selection_visible = True
        
        # Create resize handles
        self._create_resize_handles()
        
        return self.selection_rect
    
    def get_selection_content(self, canvas):
        """Extract content from selected area."""
        if not self.selection_rect:
            return None
        
        x, y, w, h = self.selection_rect
        
        # Ensure bounds are within canvas
        x = max(0, min(x, canvas.shape[1] - 1))
        y = max(0, min(y, canvas.shape[0] - 1))
        w = min(w, canvas.shape[1] - x)
        h = min(h, canvas.shape[0] - y)
        
        if w > 0 and h > 0:
            self.selection_content = canvas[y:y+h, x:x+w].copy()
            return self.selection_content
        
        return None
    
    def copy(self, canvas):
        """Copy selected area to clipboard."""
        content = self.get_selection_content(canvas)
        if content is not None:
            clipboard_item = {
                'content': content,
                'timestamp': datetime.now(),
                'source_rect': self.selection_rect,
                'operation': 'copy'
            }
            self.clipboard_stack.append(clipboard_item)
            return True
        return False
    
    def cut(self, canvas):
        """Cut selected area to clipboard."""
        content = self.get_selection_content(canvas)
        if content is not None:
            clipboard_item = {
                'content': content,
                'timestamp': datetime.now(),
                'source_rect': self.selection_rect,
                'operation': 'cut'
            }
            self.clipboard_stack.append(clipboard_item)
            
            # Mark for deletion
            self.cut_mode = True
            return True
        return False
    
    def paste(self, canvas, position, paste_mode='normal'):
        """Paste clipboard content onto canvas."""
        if not self.clipboard_stack:
            return False
        
        # Get latest clipboard item
        clipboard_item = self.clipboard_stack[-1]
        content = clipboard_item['content']
        
        if content is None:
            return False
        
        h, w = content.shape[:2]
        paste_x, paste_y = position
        
        # Adjust position to center content on cursor
        if paste_mode == 'centered':
            paste_x -= w // 2
            paste_y -= h // 2
        
        # Ensure paste is within canvas bounds
        paste_x = max(0, min(paste_x, canvas.shape[1] - w))
        paste_y = max(0, min(paste_y, canvas.shape[0] - h))
        
        # Create a new selection at paste location
        self.selection_rect = (paste_x, paste_y, w, h)
        self.selection_visible = True
        self._create_resize_handles()
        
        # Paste the content
        if content.shape[2] == 4:  # Has alpha channel
            alpha = content[:, :, 3] / 255.0
            for c in range(3):
                canvas[paste_y:paste_y+h, paste_x:paste_x+w, c] = \
                    content[:, :, c] * alpha + \
                    canvas[paste_y:paste_y+h, paste_x:paste_x+w, c] * (1 - alpha)
        else:
            canvas[paste_y:paste_y+h, paste_x:paste_x+w] = content
        
        # If this was a cut operation, clear the source area
        if self.cut_mode and clipboard_item['operation'] == 'cut':
            src_x, src_y, src_w, src_h = clipboard_item['source_rect']
            
            # Ensure source bounds are valid
            src_x = max(0, min(src_x, canvas.shape[1] - 1))
            src_y = max(0, min(src_y, canvas.shape[0] - 1))
            src_w = min(src_w, canvas.shape[1] - src_x)
            src_h = min(src_h, canvas.shape[0] - src_y)
            
            if src_w > 0 and src_h > 0:
                canvas[src_y:src_y+src_h, src_x:src_x+src_w] = (255, 255, 255)
            
            self.cut_mode = False
        
        return True
    
    def clear_selection(self):
        """Clear current selection."""
        self.selection_rect = None
        self.selection_content = None
        self.selection_visible = False
        self.resize_handles = []
        self.active_handle = None
        self.cut_mode = False
    
    def get_clipboard_preview(self, max_size=(100, 100)):
        """Get preview image of clipboard content."""
        if not self.clipboard_stack:
            return None
        
        content = self.clipboard_stack[-1]['content']
        if content is None:
            return None
        
        # Resize for preview
        preview = cv2.resize(content, max_size)
        
        # Add border
        preview = cv2.copyMakeBorder(preview, 2, 2, 2, 2, 
                                   cv2.BORDER_CONSTANT, value=(0, 255, 0))
        
        return preview
    
    def update_marching_ants(self, delta_time):
        """Update marching ants animation offset."""
        self.marching_ants_offset += self.marching_ants_speed * delta_time
        if self.marching_ants_offset >= 20:
            self.marching_ants_offset = 0
    
    def draw_selection(self, canvas):
        """Draw selection rectangle with marching ants."""
        if not self.selection_visible or not self.selection_rect:
            return canvas
        
        x, y, w, h = self.selection_rect
        
        # Draw translucent overlay
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), -1)
        canvas = cv2.addWeighted(overlay, self.selection_alpha, canvas, 1 - self.selection_alpha, 0)
        
        # Draw marching ants border
        dash_len = 10
        gap_len = 10
        
        # Top edge
        for i in range(0, w, dash_len + gap_len):
            start = (x + i + int(self.marching_ants_offset), y)
            end = (min(x + i + dash_len + int(self.marching_ants_offset), x + w), y)
            if end[0] > start[0]:
                cv2.line(canvas, start, end, (255, 255, 255), 2)
        
        # Right edge
        for i in range(0, h, dash_len + gap_len):
            start = (x + w, y + i + int(self.marching_ants_offset))
            end = (x + w, min(y + i + dash_len + int(self.marching_ants_offset), y + h))
            if end[1] > start[1]:
                cv2.line(canvas, start, end, (255, 255, 255), 2)
        
        # Bottom edge
        for i in range(0, w, dash_len + gap_len):
            start = (x + w - i - int(self.marching_ants_offset), y + h)
            end = (max(x + w - i - dash_len - int(self.marching_ants_offset), x), y + h)
            if end[0] < start[0]:
                cv2.line(canvas, start, end, (255, 255, 255), 2)
        
        # Left edge
        for i in range(0, h, dash_len + gap_len):
            start = (x, y + h - i - int(self.marching_ants_offset))
            end = (x, max(y + h - i - dash_len - int(self.marching_ants_offset), y))
            if end[1] < start[1]:
                cv2.line(canvas, start, end, (255, 255, 255), 2)
        
        # Draw resize handles
        for handle in self.resize_handles:
            hx, hy, htype = handle
            color = (0, 255, 0) if htype == 'corner' else (255, 255, 0)
            cv2.rectangle(canvas, 
                         (hx - self.handle_size, hy - self.handle_size),
                         (hx + self.handle_size, hy + self.handle_size),
                         color, -1)
            cv2.rectangle(canvas,
                         (hx - self.handle_size, hy - self.handle_size),
                         (hx + self.handle_size, hy + self.handle_size),
                         (255, 255, 255), 2)
        
        return canvas
    
    def _create_resize_handles(self):
        """Create resize handles for selection rectangle."""
        if not self.selection_rect:
            self.resize_handles = []
            return
        
        x, y, w, h = self.selection_rect
        
        # Corner handles
        handles = [
            (x, y, 'corner'),  # Top-left
            (x + w, y, 'corner'),  # Top-right
            (x, y + h, 'corner'),  # Bottom-left
            (x + w, y + h, 'corner')  # Bottom-right
        ]
        
        # Edge handles (optional)
        if w > 50 and h > 50:
            handles.extend([
                (x + w // 2, y, 'edge'),  # Top
                (x + w, y + h // 2, 'edge'),  # Right
                (x + w // 2, y + h, 'edge'),  # Bottom
                (x, y + h // 2, 'edge')  # Left
            ])
        
        self.resize_handles = handles
    
    def get_handle_at_point(self, point_x, point_y):
        """Get resize handle at specified point."""
        for i, (hx, hy, htype) in enumerate(self.resize_handles):
            if (abs(point_x - hx) <= self.handle_size * 2 and 
                abs(point_y - hy) <= self.handle_size * 2):
                return i, (hx, hy, htype)
        return None
    
    def resize_selection(self, handle_index, new_x, new_y):
        """Resize selection using specified handle."""
        if not self.selection_rect or handle_index >= len(self.resize_handles):
            return False
        
        x, y, w, h = self.selection_rect
        handle_x, handle_y, handle_type = self.resize_handles[handle_index]
        
        # Update rectangle based on handle position
        new_rect = list(self.selection_rect)
        
        if handle_index == 0:  # Top-left
            new_w = w + (x - new_x)
            new_h = h + (y - new_y)
            if new_w > 10 and new_h > 10:
                new_rect[0] = new_x
                new_rect[1] = new_y
                new_rect[2] = new_w
                new_rect[3] = new_h
        elif handle_index == 1:  # Top-right
            new_w = new_x - x
            new_h = h + (y - new_y)
            if new_w > 10 and new_h > 10:
                new_rect[1] = new_y
                new_rect[2] = new_w
                new_rect[3] = new_h
        elif handle_index == 2:  # Bottom-left
            new_w = w + (x - new_x)
            new_h = new_y - y
            if new_w > 10 and new_h > 10:
                new_rect[0] = new_x
                new_rect[2] = new_w
                new_rect[3] = new_h
        elif handle_index == 3:  # Bottom-right
            new_w = new_x - x
            new_h = new_y - y
            if new_w > 10 and new_h > 10:
                new_rect[2] = new_w
                new_rect[3] = new_h
        
        # Update selection
        self.selection_rect = tuple(new_rect)
        self._create_resize_handles()
        
        return True