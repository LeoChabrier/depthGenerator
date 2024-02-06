from pathlib import Path
import os 
from PIL import Image
from transformers import pipeline, AutoImageProcessor, AutoModelForDepthEstimation
import torch
import OpenEXR
from PySide2.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QProgressBar
from PySide2.QtGui import QIcon, QPalette, QColor
from PySide2.QtCore import QThread, Qt, Signal
from PySide2 import QtCore
import sys

class DepthGenerationThread(QThread):
    update_progress = Signal(float)

    def __init__(self, input_dir):
        super().__init__()
        self.input_dir = input_dir
    
    def run(self):
        output_dir = Path(self.input_dir) / "depth"
        output_dir.mkdir(parents=True, exist_ok=True)

        image_files = os.listdir(self.input_dir)
        total_images = len(image_files)
        processed_images = 0

        for image_file in image_files:
            if image_file != "depth":
                input_path = os.path.join(self.input_dir, image_file)
                image_name, image_number = image_file.split('.')[0], image_file.split('.')[1]
                
                output_path = os.path.join(output_dir, f"{image_name}_depth.{image_number}.exr")
                
                image = Image.open(input_path)

                pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-base-hf")
                result = pipe(image)
                predicted_depth = result["depth"]

                processor = AutoImageProcessor.from_pretrained("nielsr/depth-anything-small")
                model = AutoModelForDepthEstimation.from_pretrained("nielsr/depth-anything-small")
                pixel_values = processor(images=image, return_tensors="pt").pixel_values

                with torch.no_grad():
                    outputs = model(pixel_values)
                    predicted_depth = outputs.predicted_depth

                h, w = image.size[::-1]

                depth = torch.nn.functional.interpolate(predicted_depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
                depth_image = depth.cpu().numpy().astype('float32')

                depth_exr = OpenEXR.OutputFile(output_path, OpenEXR.Header(w, h))
                depth_exr.writePixels({'R': depth_image, 'G': depth_image, 'B': depth_image})

                processed_images += 1
                progress = processed_images / total_images * 100
                self.update_progress.emit(progress)

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
    window.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
    window.setMinimumSize(350, 200)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

# activation env virtuel et compilation .exe :
# .\venv\Scripts\activate
# pyinstaller --onefile --icon=C:\Users\Utilisateur\Desktop\depthGenerator\icon\icon.png --hidden-import=Imath src\main.py
