#!/usr/bin/env python3
"""
Virtual Drawing Studio Pro - Enhanced Edition
Main Entry Point
"""
import sys
import os
import io
import contextlib

# Suppress ALL warnings before importing heavy modules
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '0'

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('mediapipe').setLevel(logging.ERROR)

# Suppress absl logging
logging.getLogger('absl').setLevel(logging.ERROR)

# Set HighDPI attributes BEFORE creating QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

def main():
    # Create application (High DPI scaling is enabled by default in PyQt6)
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Set application name
    app.setApplicationName("Virtual Drawing Studio Pro - Enhanced")
    app.setApplicationDisplayName("Virtual Drawing Studio Pro - Enhanced")
    
    # Ensure directories exist
    for directory in ['auto_saves', 'exports', 'config']:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
    
    try:
        # Import the enhanced main window
        from main_window import MainWindow
        # Create and show main window
        window = MainWindow()
        window.show()
        
        # Start application
        sys.exit(app.exec())
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    main()