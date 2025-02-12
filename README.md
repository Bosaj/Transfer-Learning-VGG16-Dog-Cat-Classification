# 🐶🐱 Transfer Learning with VGG16 for Dog vs. Cat Classification

This repository contains a practical implementation of Transfer Learning using the VGG16 model to classify images of dogs and cats. The code is written in Python using TensorFlow and Keras.

## 📌 Overview
### Objective
To demonstrate how transfer learning can be applied to a pre-trained model (VGG16) for binary image classification (dogs vs. cats).

### Dataset
The dataset consists of images of dogs and cats, split into training and validation sets.

### Model
VGG16 pre-trained on ImageNet, with custom dense layers added for binary classification.

## 📁 Code Structure
- **Importing Libraries:** Essential libraries like TensorFlow, NumPy, and Matplotlib are imported.
- **Data Preprocessing:** Images are resized, normalized, and augmented using `ImageDataGenerator`.
- **Model Building:** The VGG16 model is loaded, and custom layers are added for classification.
- **Training:** The model is trained using the Adam optimizer.
- **Evaluation:** Performance is visualized using accuracy and loss plots.

## 🚀 How to Use
### Clone the repository:
```bash
git clone https://github.com/your-username/Transfer-Learning-VGG16-Dog-Cat-Classification.git  
```

### Install the required libraries:
```bash
pip install -r requirements.txt  
```

### Run the Jupyter Notebook or Python script:
```bash
python dog_cat_classification.py  
```

## 📊 Results
- **Training Accuracy:** 91.60%
- **Example Predictions:** 

## 🔧 Dependencies
- Python 3.13.1
- TensorFlow 2.16.1
- NumPy
- Matplotlib

## 🤝 Contributing
Feel free to contribute by opening issues or submitting pull requests. Feedback and suggestions are always welcome!



