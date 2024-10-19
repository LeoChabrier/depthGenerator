from pathlib import Path
import os
from PIL import Image
import torch
import numpy as np
import OpenEXR
import Imath
from torchvision.transforms import Compose, Resize, ToTensor
import cv2
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QProgressBar
from PySide6.QtGui import QIcon, QPalette, QColor
from PySide6.QtCore import QThread, Qt, Signal
import sys

class DepthGenerationThread(QThread):
    update_progress = Signal(float)

    def __init__(self, input_dir):
        super().__init__()
        self.input_dir = input_dir
        self.previous_depth_map = None  # To store the previous depth map for smoothing

    def run(self):
        output_dir = Path(self.input_dir) / "depth"
        output_dir.mkdir(parents=True, exist_ok=True)

        image_files = os.listdir(self.input_dir)
        total_images = len(image_files)
        processed_images = 0

        # Load MiDaS model
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
        midas.eval()

        # Use a GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        midas.to(device)

        # MiDaS transformations
        transform = Compose([
            Resize((384, 384)),  # MiDaS v3 input size
            ToTensor(),
        ])

        alpha = 0.5  # Smoothing factor for temporal filtering

        for image_file in image_files:
            if image_file != "depth":
                input_path = os.path.join(self.input_dir, image_file)
                image_name, image_number = os.path.splitext(image_file)

                output_path = os.path.join(output_dir, f"{image_name}_depth.exr")
                
                # Open image
                image = Image.open(input_path).convert("RGB")
                input_tensor = transform(image).unsqueeze(0).to(device)

                # Perform depth estimation
                with torch.no_grad():
                    prediction = midas(input_tensor)
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=image.size[::-1],
                        mode="bicubic",
                        align_corners=False
                    ).squeeze()

                # Normalize depth map to range [0, 1]
                depth_map = prediction.cpu().numpy()
                depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

                # Temporal filtering
                if self.previous_depth_map is not None:
                    depth_map = cv2.addWeighted(depth_map, alpha, self.previous_depth_map, 1 - alpha, 0)
                
                self.previous_depth_map = depth_map  # Update the previous depth map

                # Save depth map as EXR
                self.save_exr(output_path, depth_map)

                processed_images += 1
                progress = processed_images / total_images * 100
                self.update_progress.emit(progress)

    @staticmethod
    def save_exr(file_path, depth_map):
        """Save the depth map as an EXR file."""
        h, w = depth_map.shape
        header = OpenEXR.Header(w, h)
        # Specify channels for the EXR file
        header['channels'] = {
            'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
            'G': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
            'B': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        }

        # Prepare EXR output
        depth_data = (depth_map.astype(np.float32)).tobytes()
        channels = {'R': depth_data, 'G': depth_data, 'B': depth_data}

        # Write to EXR file without using the 'with' statement
        exr_file = OpenEXR.OutputFile(file_path, header)
        exr_file.writePixels(channels)
        exr_file.close()

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Depth Generation")
        self.setWindowIcon(QIcon("icon/icon.png"))

        self.input_dir = None

        layout = QVBoxLayout()

        self.label = QLabel("Drag and drop folder here")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("color: white;")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setStyleSheet("""
        QProgressBar {
            border: 2px solid grey;
            border-radius: 5px;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #05B8CC;
            width: 20px;
        }
        """)

        layout.addWidget(self.label)
        layout.addWidget(self.progress_bar)
        self.setLayout(layout)

        self.setAcceptDrops(True)

        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        self.setPalette(palette)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            self.input_dir = urls[0].toLocalFile()
            self.label.setText(f"Input Directory: {self.input_dir}")
            self.start_depth_generation()

    def start_depth_generation(self):
        self.depth_thread = DepthGenerationThread(self.input_dir)
        self.depth_thread.update_progress.connect(self.update_progress)
        self.depth_thread.start()

    def update_progress(self, progress):
        self.progress_bar.setValue(progress)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setMinimumSize(350, 200)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
