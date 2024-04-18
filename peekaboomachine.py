import sys
import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QPushButton
from tensorflow.keras.models import load_model
import numpy as np

class WebcamWidget(QWidget):
    def __init__(self):
        super().__init__()

        # Load the model
        self.model = load_model("keras_Model.h5", compile=False)

        # Load the labels
        self.class_names = open("labels.txt", "r", encoding="utf-8").readlines()

        # Set up the layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Set up the label to display the webcam image
        self.image_label = QLabel()
        self.layout.addWidget(self.image_label)

        # Set up the button to close the application
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close_application)
        self.layout.addWidget(self.close_button)

        # Start the webcam
        self.webcam = cv2.VideoCapture(0)

        # Start the timer to read frames from the webcam
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(10)  # Update every 10 milliseconds

    def update_frame(self):
        # Read frame from webcam
        ret, frame = self.webcam.read()
        if not ret:
            return

        # Preprocess the frame for prediction
        frame_preprocessed = cv2.resize(frame, (224, 224))
        frame_preprocessed = (frame_preprocessed / 127.5) - 1
        frame_preprocessed = np.expand_dims(frame_preprocessed, axis=0)

        # Make prediction
        prediction = self.model.predict(frame_preprocessed)
        index = np.argmax(prediction)
        class_name = self.class_names[index]
        confidence_score = prediction[0][index]

        # Add prediction text to the frame
        text = f"Class: {class_name.strip()}, Confidence: {str(np.round(confidence_score * 100))[:-2]}%"
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Convert the frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the frame to QImage
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Update the image label
        self.image_label.setPixmap(QPixmap.fromImage(qt_image))

    def close_application(self):
        # Release webcam and stop the timer when closing the application
        self.webcam.release()
        self.timer.stop()
        sys.exit()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WebcamWidget()
    window.setWindowTitle("Webcam Classifier")
    window.show()
    sys.exit(app.exec_())
