# Handwritten-Digits-Classifier
# 🧠 Handwritten Digit Recognition using CNN (MNIST)

This project demonstrates how to classify handwritten digits (0–9) using a **Convolutional Neural Network (CNN)** built with **TensorFlow/Keras**. It leverages the **MNIST dataset** to train a deep learning model capable of recognizing digit images with high accuracy.

---

## 📌 Project Overview

- **Dataset**: [MNIST](http://yann.lecun.com/exdb/mnist/) (from `tensorflow.keras.datasets`)
- **Task**: Multi-class image classification
- **Image Size**: 28x28 grayscale images
- **Model Type**: Convolutional Neural Network (CNN)

---

## 📊 Workflow Summary

1. **Data Loading**  
   - Loaded MNIST dataset using `tf.keras.datasets.mnist`

2. **Visualization**  
   - Displayed sample digits using `matplotlib`

3. **Data Preprocessing**  
   - Normalized pixel values (0–255 → 0–1)
   - Reshaped data to fit CNN input requirements

4. **Model Building**  
   - Used 3 convolutional layers + max pooling
   - Added fully connected (dense) layers
   - Final layer uses **softmax** activation (for 10 classes)

5. **Model Compilation & Training**  
   - Loss function: `sparse_categorical_crossentropy`
   - Optimizer: `Adam`
   - Metric: `accuracy`
   - Trained for 5 epochs with 30% validation split

6. **Model Evaluation**  
   - Evaluated accuracy on 10,000 test samples

7. **Predictions**  
   - Made predictions and visualized test images with predicted labels

---

## 🔧 Technologies Used

- Python
- NumPy, Matplotlib, Seaborn
- TensorFlow & Keras (deep learning framework)
- Jupyter Notebook

---

## 📈 Results

- Achieved **high validation and test accuracy**
- Model capable of identifying handwritten digits with strong confidence

---

## 📷 Sample Output

Sample predictions on unseen data:

```python
plt.imshow(x_test[0])  # Sample digit image
print(np.argmax(predictions[0]))  # Predicted digit
