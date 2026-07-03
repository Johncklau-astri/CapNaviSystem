import sys
import os
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont
import pyqtgraph.opengl as gl
from scipy.spatial.transform import Rotation as R
# from stl import mesh as stl_mesh

# =====================================================================
# FILE TELEMETRY RECEIVER ENGINE (SHARED TEXT FILE POLLING)
# =====================================================================
class TelemetryReceiver(QThread):
    data_received = pyqtSignal(np.ndarray, np.ndarray)

    def __init__(self, file_path=r"C:\Users\johncklau\Desktop\CapNaviSystem\telemetry.txt"):
        super().__init__()
        self.file_path = file_path
        self.running = True

    def run(self):
        print(f"Watching shared telemetry source file: {self.file_path}")
        last_modified_time = 0

        while self.running:
            try:
                if os.path.exists(self.file_path):
                    current_mtime = os.path.getmtime(self.file_path)
                    if current_mtime != last_modified_time:
                        last_modified_time = current_mtime
                        
                        with open(self.file_path, "r") as f:
                            line = f.readline().strip()
                        
                        if line:
                            values = [float(val) for val in line.split(",")]
                            if len(values) == 7:
                                pos = np.array(values[0:3], dtype=np.float32)
                                quat = np.array(values[3:7], dtype=np.float32)
                                
                                # Apply the position scaling used in your local file receiver setup
                                scale = 0.7
                                pos *= scale
                                
                                self.data_received.emit(pos, quat)
            except Exception:
                pass
            
            self.msleep(15)

    def stop(self):
        self.running = False
        self.wait()

# =====================================================================
# MAIN 3D VISUALIZER WINDOW
# =====================================================================
class RealTime3DVisualizer(QMainWindow):
    def __init__(self, stationary_stl=None, moving_stl=None, file_path=r"C:\Users\johncklau\Desktop\CapNaviSystem\telemetry.txt"):
        super().__init__()
        self.setWindowTitle("Dual STL Real-Time Tracker (Solid File Stream)")
        self.setGeometry(100, 100, 800, 600)

# 1. Initialize 3D View Widget
        self.view = gl.GLViewWidget()
        self.setCentralWidget(self.view)
        self.view.setCameraPosition(distance=150, elevation=30, azimuth=45)

        # --- ENHANCE MESH LIGHTING AND BRIGHTNESS ---
        # Enable OpenGL's built-in lighting system globally
        self.view.opts['lighting'] = True 
        
        # Boost Ambient Light (R, G, B, A) - acts as an overall brightness floor
        self.view.opts['lights'] = [
            {
                'id': 0,
                'ambient': (0.5, 0.5, 0.5, 1.0),   # Higher ambient stops shadows from being completely pitch black
                'diffuse': (0.8, 0.8, 0.8, 1.0),   # Main structural light intensity
                'specular': (0.3, 0.3, 0.3, 1.0),  # Highlights/reflections reflection intensity
                'direction': (0.0, 0.0, -1.0)      # Direction vector (shining straight down from above)
            },
            {
                'id': 1,
                'ambient': (0.2, 0.2, 0.2, 1.0),
                'diffuse': (0.5, 0.5, 0.5, 1.0),
                'specular': (0.1, 0.1, 0.1, 1.0),
                'direction': (1.0, 1.0, 1.0)       # Secondary fill light hitting the side faces
            }
        ]

        # # Telemetry Display Overlay
        # self.telemetry_label = QLabel(self.view)
        # self.telemetry_label.setStyleSheet("color: white; background-color: rgba(0, 0, 0, 120); padding: 10px; border-radius: 5px;")
        # self.telemetry_label.setFont(QFont("Consolas", 11, QFont.Bold))
        # self.telemetry_label.setGeometry(15, 15, 280, 110)
        # self.telemetry_label.setText("Awaiting Data Stream from Shared File...") 

        # 2. Add Trajectory Trail Item
        self.trail_points = []
        self.trail = gl.GLLinePlotItem(color=(0.0, 1.0, 1.0, 0.6), width=2, antialias=True)
        self.view.addItem(self.trail)

        # 3. Load Meshes 
        self.load_stationary_mesh(stationary_stl)
        self.load_moving_mesh(moving_stl)

        # Initialize internal storage for incoming states
        self.current_pos = np.array([0, 0, 0], dtype=np.float32)
        self.current_quat = np.array([0, 0, 0, 1], dtype=np.float32)

        # 4. Initialize and start background file polling worker
        self.receiver = TelemetryReceiver(file_path=file_path)
        self.receiver.data_received.connect(self.store_incoming_data)
        self.receiver.start()

        # 5. Render Synchronizer clock (~50 FPS UI updates)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(20)

    def parse_stl_edges(self, file_path, default_type="box"):
        if file_path and os.path.exists(file_path):
            mesh_data = stl_mesh.Mesh.from_file(file_path)
            vectors = mesh_data.vectors.astype(np.float32)
            if default_type == "environment":
                vectors = vectors[::2]
            lines = []
            for tri in vectors:
                lines.extend([tri[0], tri[1], tri[1], tri[2], tri[2], tri[0]])
            return np.array(lines, dtype=np.float32)
        else:
            return np.array([[[-4,-4,-2], [4,-4,-2]], [[4,-4,-2], [4,4,-2]]], dtype=np.float32).reshape(-1, 3)

    def parse_stl_faces(self, file_path):
        if file_path and os.path.exists(file_path):
            print(f"Loading Solid Moving Mesh STL: {file_path}")
            mesh_data = stl_mesh.Mesh.from_file(file_path)
            verts = mesh_data.vectors.reshape(-1, 3).astype(np.float32)
            verts -= verts.mean(axis=0) 
            faces = np.arange(len(verts)).reshape(-1, 3)
            return verts, faces
        else:
            verts = np.array([[-5,-5,-5], [5,-5,-5], [5,5,-5], [-5,5,-5], [-5,-5,5], [5,-5,5], [5,5,5], [-5,5,5]], dtype=np.float32)
            faces = np.array([[0,1,2], [0,2,3], [4,5,6], [4,6,7], [0,1,5], [0,5,4], [2,3,7], [2,7,6], [1,2,6], [1,6,5], [0,3,7], [0,7,4]])
            return verts, faces

    def load_stationary_mesh(self, file_path):
        stl_lines = self.parse_stl_edges(file_path, "environment")
        stl_lines -= stl_lines.mean(axis=0)

        tilt_rotation = R.from_euler('x', -225, degrees=True)
        tilt_matrix = tilt_rotation.as_matrix().astype(np.float32)
        stl_lines = np.dot(stl_lines, tilt_matrix.T)
        stl_lines[:, 2] += 50.0
        stl_lines[:, 1] -= 180.0

        stl_lines = np.ascontiguousarray(stl_lines, dtype=np.float32)
        self.stationary_mesh = gl.GLLinePlotItem(
            pos=stl_lines, color=(0.5, 0.5, 0.5, 0.3), width=1, mode='lines', antialias=True
        )
        self.view.addItem(self.stationary_mesh)

    def load_moving_mesh(self, file_path):
        self.base_moving_vertices, self.moving_faces = self.parse_stl_faces(file_path)

        num_faces = len(self.moving_faces)
        face_colors = np.zeros((num_faces, 4), dtype=np.float32)
        
        for i, face in enumerate(self.moving_faces):
            face_verts = self.base_moving_vertices[face]
            face_center_x = face_verts[:, 0].mean()
            if face_center_x >= 0.0 and face_center_x <= 15.0:
                face_colors[i] = [1.0, 0.0, 0.0, 1.0] # Opaque Solid Red
            elif face_center_x > 15.0 and face_center_x <= 40.0:
                face_colors[i] = [1.0, 1.0, 0.0, 1.0] # Opaque Solid Yellow
            else:
                face_colors[i] = [0.0, 1.0, 0.0, 1.0] # Opaque Solid Green

        self.moving_mesh_colors = face_colors

        self.moving_mesh = gl.GLMeshItem(
                    vertexes=self.base_moving_vertices,
                    faces=self.moving_faces,
                    faceColors=self.moving_mesh_colors,
                    smooth=False,
                    shader='balloon'  # 'balloon' provides a bright, self-illuminating look that eliminates dark spots
                )
        self.view.addItem(self.moving_mesh)

    def store_incoming_data(self, pos, quat):
        self.current_pos = pos
        self.current_quat = quat

    def update_frame(self):
        try:
            pos = self.current_pos
            quat = self.current_quat

            text = (
                f"=== CAPSULE TELEMETRY ===\n"
                f"X: {pos[0]:7.2f} mm\n"
                f"Y: {pos[1]:7.2f} mm\n"
                f"Z: {pos[2]:7.2f} mm\n"
                f"Quat [x,y,z,w]:\n"
                f"[{quat[0]:.2f}, {quat[1]:.2f}, {quat[2]:.2f}, {quat[3]:.2f}]"
            )
            # self.telemetry_label.setText(text)

            self.trail_points.append(pos)
            if len(self.trail_points) > 150:
                self.trail_points.pop(0)
            self.trail.setData(pos=np.array(self.trail_points, dtype=np.float32))

            r = R.from_quat(quat)
            rot_matrix = r.as_matrix().astype(np.float32)

            transformed_verts = np.dot(self.base_moving_vertices, rot_matrix.T) + pos
            transformed_verts = np.ascontiguousarray(transformed_verts, dtype=np.float32)
            
            self.moving_mesh.setMeshData(vertexes=transformed_verts, faces=self.moving_faces, faceColors=self.moving_mesh_colors)

        except Exception as e:
            print(f"Pipeline error: {e}")

    def closeEvent(self, event):
        self.receiver.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    stationary_path = r"C:\Users\johncklau\Desktop\CapNaviSystem\stomach.stl"
    moving_path     = r"C:\Users\johncklau\Desktop\CapNaviSystem\capsule.stl"
    shared_txt_path = r"C:\Users\johncklau\Desktop\CapNaviSystem\telemetry.txt"
    
    visualizer = RealTime3DVisualizer(
        stationary_stl=stationary_path, 
        moving_stl=moving_path,
        file_path=shared_txt_path
    )
    visualizer.show()
    sys.exit(app.exec_())