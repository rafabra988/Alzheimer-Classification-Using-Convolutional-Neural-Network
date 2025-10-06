import sys
import numpy as np
import cv2
import tensorflow as tf

from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, 
                             QWidget, QPushButton, QLabel, QFileDialog, 
                             QFrame)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt

# --- Carregue seu modelo aqui ---
# Certificar de que o caminho 'saved_model/gpu_model.keras' está correto
try:
    gpu_model = tf.keras.models.load_model('saved_model/gpu_model.keras')
    MODEL_LOADED = True
except Exception as e:
    print(f"Error loading model: {e}")
    MODEL_LOADED = False

# --- Classe da Interface Principal ---
class ClassifierWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Classifier - Dementia Prediction")
        self.setGeometry(100, 100, 500, 600)
        self.setStyleSheet("background-color: #f0f0f0;")

        self.initUI()

    def initUI(self):
        # Widget Central e Layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Label para exibir a imagem
        self.image_label = QLabel("Please select an image to classify", self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(800, 800)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #cccccc;
                border-radius: 15px;
                background-color: #ffffff;
                color: #888888;
                font-size: 16px;
            }
        """)
        # Centraliza o label da imagem no layout
        layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        # Label para exibir a predição
        self.prediction_label = QLabel("Prediction: N/A", self)
        self.prediction_label.setAlignment(Qt.AlignCenter)
        self.prediction_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.prediction_label.setStyleSheet("color: #333333;")
        layout.addWidget(self.prediction_label, alignment=Qt.AlignCenter)
        
        # Botão para selecionar e classificar
        self.classify_btn = QPushButton("Select Image", self)
        self.classify_btn.setMinimumHeight(45)
        self.classify_btn.setFont(QFont("Arial", 12))
        self.classify_btn.setStyleSheet("""
            QPushButton {
                background-color: #007BFF;
                color: white;
                border: none;
                border-radius: 15px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:pressed {
                background-color: #004494;
            }
        """)
        self.classify_btn.clicked.connect(self.classify_image)
        layout.addWidget(self.classify_btn)
        
        # Mensagem de status do modelo
        if not MODEL_LOADED:
            self.prediction_label.setText("Error: Model could not be loaded.")
            self.prediction_label.setStyleSheet("color: red;")
            self.classify_btn.setEnabled(False)

    def classify_image(self):
        # Abre o diálogo para selecionar o arquivo
        fname, _ = QFileDialog.getOpenFileName(self, 'Select Image', '', "Image files (*.jpg *.jpeg *.png)")
        
        if fname:
            # Exibe a imagem selecionada na interface
            pixmap = QPixmap(fname)
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.width(), 
                self.image_label.height(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            ))
            self.image_label.setStyleSheet("border: none;") # Remove a borda após selecionar

            try:
                # Prepara a imagem para o modelo
                img = cv2.imread(fname)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (180, 180))
                img = np.expand_dims(img, axis=0)
                
                # Faz a predição
                preds = gpu_model.predict(img)
                class_idx = np.argmax(preds[0])
                classes = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']
                
                predicted_class = classes[class_idx]
                
                # Exibe o resultado
                self.prediction_label.setText(f"Prediction: {predicted_class}")
                print(f"Predicted Class: {predicted_class}")

            except Exception as e:
                self.prediction_label.setText("Error during prediction.")
                print(f"An error occurred: {e}")

# --- Função Principal ---
def main():
    app = QApplication(sys.argv)
    window = ClassifierWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
