# 🩺 Breast Cancer Diagnosis with MLP 🧠

This project aims to develop a machine learning model for the diagnosis of breast cancer using a Multi-Layer Perceptron (MLP) classifier. The dataset utilized for this task is the Breast Cancer Wisconsin dataset, obtained from the UCI Machine Learning Repository.

## 🎯 Purpose

The main purpose of this code is to demonstrate the application of a MLP classifier in diagnosing breast cancer based on features extracted from fine needle aspiration (FNA) images of breast masses. The features describe various characteristics of cell nuclei present in the images. By analyzing these features, the MLP classifier can learn to differentiate between benign and malignant tumors, thereby aiding in the early detection and diagnosis of breast cancer.

## 🛠️ Decision Making

1. **Model Selection**: The MLP classifier from the scikit-learn library is chosen for its ability to capture complex patterns in the data. MLPs are capable of learning non-linear relationships between features, making them suitable for this classification task.

2. **Data Preprocessing**: Prior to training the model, the dataset is divided into training and testing sets using a standard 80-20 split. Additionally, feature scaling is performed using StandardScaler to standardize the feature values, ensuring that each feature contributes equally to the learning process and improving convergence speed.

3. **Model Training**: The MLP model is instantiated with two hidden layers containing 100 and 50 neurons respectively. The 'adam' solver is chosen for optimization, and the model is trained with a maximum of 500 iterations. The 'verbose' parameter is set to 10 to display progress during training.

4. **Evaluation**: After training, the model is evaluated using the testing set. Accuracy score and confusion matrix are computed to assess the performance of the model in diagnosing breast cancer.

## 🚀 Usage

To run the code, ensure that you have the required dependencies installed. You can execute the script in any Python environment that supports scikit-learn. Simply clone the repository and execute the `breast_cancer_diagnosis_mlp.py` script. Make sure to adjust any parameters or configurations as needed for your specific use case.

