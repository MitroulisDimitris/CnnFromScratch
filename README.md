# CNN MNIST Image Classifier (From Scratch)

This project implements a Convolutional Neural Network (CNN) from scratch to classify handwritten digits from the MNIST dataset. The model is designed to be flexible, allowing customization of layers, kernel sizes, and training parameters.

---

## 📌 Features

- **Handwritten Digit Classification**: Classifies images into 10 categories (digits 0-9).
- **Real-time Visualization**:
  - Displays model predictions.
  - Visualizes kernels and convolution operations.
  - Shows feedforward model activations.
- **Comprehensive Logging**:
  - Logs forward pass performance metrics.
  - Tracks backpropagation calculations.
  - Monitors loss changes throughout training.
- **Fully Customizable**:
  - Adjustable number of layers for both feedforward and convolutional networks.
  - Customizable kernel sizes for convolution layers.
  - Supports dropout for all layers.
  - Includes data augmentation (rotation, shifting).
- **Supports Model Saving**: Trained models can be saved and used for later inference.
- **Evaluation Metrics**: Provides accuracy, loss tracking, and additional performance insights.
- **Dataset Shuffling**: Ensures diverse training batches for better generalization.
- **Object-Oriented Design**: Each layer is implemented as a callable method in a class.

---

## 🏗 Model Architecture

The CNN follows a typical architecture for MNIST classification:

| Layer | Output Shape |
|--------|------------|
| **Input** (28×28×1) | (28,28,1) |
| **Conv2D** (3×3, 32 filters, ReLU) | (26,26,32) |
| **MaxPooling** (2×2) | (13,13,32) |
| **Conv2D** (3×3, 64 filters, ReLU) | (11,11,64) |
| **MaxPooling** (2×2) | (5,5,64) |
| **Flatten** | (1600) |
| **Dense** (128 neurons, ReLU) | (128) |
| **Dropout** (0.2 probability) | (128) |
| **Dense** (10 neurons, softmax) | (10) |

---

## 📂 Installation

### **Step 1: Clone the Repository**
```bash
git clone https://github.com/your-username/mnist-cnn-from-scratch.git
cd mnist-cnn-from-scratch
```
### Step 2: Install Dependencies
Install all required libraries:
```bash
pip install -r requirements.txt
```
### Step 3: Run the Training Script
```bash
python train.py

```
