import cv2
import os
import time
from pathlib import Path
import threading
from collections import deque

class LatestImageDisplay:
    def __init__(self, folder_path, fps=30):
        self.folder_path = Path(folder_path)
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.current_image = None
        self.current_image_path = None
        self.running = True
        self.lock = threading.Lock()
        
        # Image processing queue to avoid overloading
        self.update_queue = deque(maxlen=2)
        
        # Verify folder exists
        if not self.folder_path.exists():
            raise ValueError(f"Folder {folder_path} does not exist")
    
    def get_latest_jpg(self):
        """Get the most recently modified JPG file"""
        try:
            # Get all jpg/jpeg files
            jpg_files = list(self.folder_path.glob("*.jpg")) + \
                       list(self.folder_path.glob("*.jpeg")) + \
                       list(self.folder_path.glob("*.JPG"))
            
            if not jpg_files:
                return None, None
            
            # Get the most recently modified file
            latest_file = max(jpg_files, key=lambda f: f.stat().st_mtime)
            return latest_file, latest_file.stat().st_mtime
        except Exception as e:
            print(f"Error scanning folder: {e}")
            return None, None
    
    def monitor_folder(self):
        """Background thread to monitor folder for new images"""
        last_modified_time = 0
        last_checked_path = None
        
        while self.running:
            try:
                latest_file, mod_time = self.get_latest_jpg()
                
                # Check if there's a new file or updated file
                if latest_file and (latest_file != last_checked_path or mod_time > last_modified_time):
                    # Load the new image
                    img = cv2.imread(str(latest_file))
                    if img is not None:
                        with self.lock:
                            self.current_image = img
                            self.current_image_path = latest_file
                            last_checked_path = latest_file
                            last_modified_time = mod_time
                            
                        # Optional: print update info
                        print(f"Loaded: {latest_file.name}")
                
                # Small sleep to prevent excessive CPU usage
                time.sleep(0.05)  # 50ms between folder checks
                
            except Exception as e:
                print(f"Monitor error: {e}")
                time.sleep(0.5)
    
    def display_loop(self):
        """Main display loop running at specified FPS"""
        last_frame_time = time.time()
        frame_count = 0
        fps_display = self.fps
        
        cv2.namedWindow('Latest Image Display', cv2.WINDOW_NORMAL)
        
        while self.running:
            current_time = time.time()
            elapsed = current_time - last_frame_time
            
            # Check if it's time to display next frame
            if elapsed >= self.frame_interval:
                # Get current image (thread-safe)
                with self.lock:
                    display_img = self.current_image.copy() if self.current_image is not None else None
                    current_path = self.current_image_path
                
                # Create display frame
                if display_img is not None:
                    # Add information overlay
                    height, width = display_img.shape[:2]
                    
                    # Scale image if too large (max 1200px width)
                    max_width = 1200
                    if width > max_width:
                        scale = max_width / width
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        display_img = cv2.resize(display_img, (new_width, new_height))
                    
                    # Add text overlay with filename and FPS
                    if current_path:
                        cv2.putText(display_img, f"File: {current_path.name}", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.7, (255, 255, 255), 2)
                    
                    cv2.putText(display_img, f"FPS: {fps_display:.1f}", 
                              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.7, (255, 255, 255), 2)
                    
                    cv2.imshow('Latest Image Display', display_img)
                else:
                    # Show waiting message if no images yet
                    waiting_img = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(waiting_img, "Waiting for JPG images...", 
                              (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 
                              1, (255, 255, 255), 2)
                    cv2.putText(waiting_img, f"Monitoring: {self.folder_path}", 
                              (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.6, (200, 200, 200), 1)
                    cv2.imshow('Latest Image Display', waiting_img)
                
                # Update frame timing
                frame_count += 1
                if frame_count >= 30:  # Update FPS display every 30 frames
                    fps_display = 1.0 / (time.time() - last_frame_time) * frame_count
                    frame_count = 0
                
                last_frame_time = current_time
            
            # Handle keyboard input (non-blocking)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC to quit
                self.running = False
                break
            
            # Small delay to prevent 100% CPU usage
            time.sleep(0.001)
    
    def start(self):
        """Start monitoring and display"""
        print(f"Monitoring folder: {self.folder_path}")
        print("Press 'q' or ESC to quit")
        print("Waiting for JPG images...")
        
        # Start monitor thread
        monitor_thread = threading.Thread(target=self.monitor_folder, daemon=True)
        monitor_thread.start()
        
        # Run display loop (main thread)
        try:
            self.display_loop()
        finally:
            cv2.destroyAllWindows()
            self.running = False
            print("Display closed")

if __name__ == "__main__":
    import sys
    import numpy as np
    
    # Use command line argument or current directory
    if len(sys.argv) > 1:
        folder_to_watch = sys.argv[1]
    else:
        folder_to_watch = r"C:\Users\johncklau\Desktop\FX3_PCApp\FX3DisplayImage\Release_Internal_V1A\output"  # Current directory
    
    try:
        # Create and run the display
        display = LatestImageDisplay(folder_to_watch, fps=30)
        display.start()
    except Exception as e:
        print(f"Error: {e}")
        print("Usage: python script.py [folder_path]")