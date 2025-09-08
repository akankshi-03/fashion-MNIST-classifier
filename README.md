# 👗 Fashion MNIST Classifier

## 📌 Project Overview
This project implements a *Deep Learning Classifier* on the *Fashion-MNIST dataset*, which contains 28×28 grayscale images of clothing items (10 categories).  
The project demonstrates key deep learning concepts such as *overfitting, regularization, activation functions, and CNNs*.

---

## 🎯 Goals
- Understand the difference between *simple (MNIST)* and *slightly complex (Fashion-MNIST)* datasets.
- Learn about *overfitting* and how to handle it using *Dropout* and *L2 Regularization*.
- Explore the role of *Activation Functions* (ReLU, Softmax).
- Compare a simple *Dense Neural Network* with a *Convolutional Neural Network (CNN)* for image classification.

---

## 📂 Project Pipeline
1. *Data Loading & Preprocessing*
   - Load Fashion-MNIST dataset from Keras.
   - Normalize pixel values (0–1).
   - Visualize sample images with labels.

2. *Baseline Model (Dense Neural Network)*
   - Flatten → Dense(128, ReLU) → Dense(10, Softmax).
   - Accuracy: ~85%.
   - Observed *overfitting*.

3. *Regularization*
   - Added *Dropout layers*.
   - Added *L2 Regularization* to control large weights.
   - Improved validation accuracy & generalization.

4. *CNN Model (Improved Performance)*
   - Conv2D(32) → MaxPooling → Conv2D(64) → MaxPooling → Flatten → Dense(128) → Output.
   - Accuracy: ~90–92%.
   - CNN outperforms dense model due to feature extraction.

5. *Evaluation & Predictions*
   - Compared training vs validation accuracy.
   - Tested on unseen data.
   - Visualized predictions vs true labels.

6. *Summary & Learnings*
   - Overfitting is reduced with Dropout & L2 Regularization.
   - ReLU works best in hidden layers, Softmax in output layer.
   - CNNs are superior for image data.
   - MNIST = simple, Fashion-MNIST = more complex.

---

## 📊 Results
- *Dense Model Accuracy:* ~85%  
- *Regularized Dense Model Accuracy:* ~87%  
- *CNN Model Accuracy:* ~90–92%  

---

## 🛠 Tech Stack
- Python  
- TensorFlow / Keras  
- Matplotlib (for visualization)  
- NumPy  

---

## 🚀 How to Run
1. Clone this repository:
   bash
   git clone https://github.com/akankshi-03//fashion-mnist-classifier.git
   cd fashion-mnist-classifier
   

2. Install dependencies:
   bash
   pip install -r requirements.txt
   

3. Run the Jupyter Notebook / Python script in Colab.

---

## 👩‍💻 Author
a
Akankshi dubey

---

## 📌 Future Work
- Experiment with *Batch Normalization*.
- Train on *CIFAR-10 dataset* for more complex images.
- Deploy the model using *Streamlit / Flask*.
