# Breast Cancer Diagnosis with MLP 🩺💻

[![GitHub](https://img.shields.io/badge/GitHub-View%20on%20GitHub-blue?style=flat-square)](https://github.com/sergio11/breast_cancer_diagnosis_mlp)
[![PyPI](https://img.shields.io/pypi/v/BreastCancerMLPModel.svg?style=flat-square)](https://pypi.org/project/BreastCancerMLPModel/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://github.com/sergio11/breast_cancer_diagnosis_mlp/blob/main/LICENSE)

🧠 This project harnesses the power of a Multi-Layer Perceptron (MLP) neural network, implemented with scikit-learn, to perform breast cancer diagnosis based on tumor characteristics extracted from biopsy samples. The MLP model is a type of artificial neural network designed to learn complex patterns in data, making it well-suited for tasks like medical diagnosis.

🔬 The MLP model is trained on a comprehensive dataset containing various features derived from digital images of breast tissue samples. These features include mean radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension. Each feature provides valuable information about the physical properties and spatial arrangements of cells within the tissue, enabling the model to learn to distinguish between benign and malignant tumors.

💡 By analyzing these features, the MLP model can effectively classify breast tumors as either benign or malignant, providing valuable diagnostic information to healthcare professionals. This approach offers a non-invasive and automated method for cancer detection, potentially improving patient outcomes through earlier detection and treatment.

<p align="center">
  <img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue" />
  <img src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white" />
</p>

🙏 I would like to extend my heartfelt gratitude to [Santiago Hernández, an expert in Cybersecurity and Artificial Intelligence](https://www.udemy.com/user/shramos/). His incredible course on Deep Learning, available at Udemy, was instrumental in shaping the development of this project. The insights and techniques learned from his course were crucial in crafting the neural network architecture used in this classifier.

## 🌟 Explore My Other Cutting-Edge AI Projects! 🌟

If you found this project intriguing, I invite you to check out my other AI and machine learning initiatives, where I tackle real-world challenges across various domains:

+ [🌍 Advanced Classification of Disaster-Related Tweets Using Deep Learning 🚨](https://github.com/sergio11/disasters_prediction)  
Uncover how social media responds to crises in real time using **deep learning** to classify tweets related to disasters.

+ [📰 Fighting Misinformation: Source-Based Fake News Classification 🕵️‍♂️](https://github.com/sergio11/fake_news_classifier)  
Combat misinformation by classifying news articles as real or fake based on their source using **machine learning** techniques.

+ [🛡️ IoT Network Malware Classifier with Deep Learning Neural Network Architecture 🚀](https://github.com/sergio11/iot_network_malware_classifier)  
Detect malware in IoT network traffic using **Deep Learning Neural Networks**, offering proactive cybersecurity solutions.

+ [📧 Spam Email Classification using LSTM 🤖](https://github.com/sergio11/spam_email_classifier_lstm)  
Classify emails as spam or legitimate using a **Bi-directional LSTM** model, implementing NLP techniques like tokenization and stopword removal.

+ [💳 Fraud Detection Model with Deep Neural Networks (DNN)](https://github.com/sergio11?tab=repositories)  
Detect fraudulent transactions in financial data with **Deep Neural Networks**, addressing imbalanced datasets and offering scalable solutions.

+ [🧠🚀 AI-Powered Brain Tumor Classification](https://github.com/sergio11/brain_tumor_classification_cnn)  
Classify brain tumors from MRI scans using **Deep Learning**, CNNs, and Transfer Learning for fast and accurate diagnostics.

+ [📊💉 Predicting Diabetes Diagnosis Using Machine Learning](https://github.com/sergio11/diabetes_prediction_ml)  
Create a machine learning model to predict the likelihood of diabetes using medical data, helping with early diagnosis.

+ [🚀🔍 LLM Fine-Tuning and Evaluation](https://github.com/sergio11/llm_finetuning_and_evaluation)  
Fine-tune large language models like **FLAN-T5**, **TinyLLAMA**, and **Aguila7B** for various NLP tasks, including summarization and question answering.

+ [📰 Headline Generation Models: LSTM vs. Transformers](https://github.com/sergio11/headline_generation_lstm_transformers)  
Compare **LSTM** and **Transformer** models for generating contextually relevant headlines, leveraging their strengths in sequence modeling.

+ [🩺💻 Breast Cancer Diagnosis with MLP](https://github.com/sergio11/breast_cancer_diagnosis_mlp)  
Automate breast cancer diagnosis using a **Multi-Layer Perceptron (MLP)** model to classify tumors as benign or malignant based on biopsy data.

+ [Deep Learning for Safer Roads 🚗 Exploring CNN-Based and YOLOv11 Driver Drowsiness Detection 💤](https://github.com/sergio11/safedrive_drowsiness_detection)
Comparing driver drowsiness detection with CNN + MobileNetV2 vs YOLOv11 for real-time accuracy and efficiency 🧠🚗. Exploring both deep learning models to prevent fatigue-related accidents 😴💡.

## Purpose 🎯

The primary objective of this project is to develop an accurate and reliable system for diagnosing breast cancer based on quantitative analysis of cell nuclei characteristics. By leveraging machine learning techniques, specifically MLP neural networks, we aim to create a predictive model capable of classifying tumors as either malignant (cancerous) or benign (non-cancerous) with high accuracy. Early and accurate diagnosis of breast cancer can significantly improve patient outcomes by enabling timely treatment and intervention.

## Key Features 🔑

- Utilizes an MLP neural network for breast cancer diagnosis. 🤖
- Preprocesses input data using feature scaling with StandardScaler. 📊
- Implements training and evaluation functionalities. 📈
- Provides prediction capabilities for new biopsy samples. ⚡
- Offers detailed model evaluation metrics, including accuracy and confusion matrix. 📊
- Supports easy integration into Python applications for breast cancer diagnosis tasks. 🐍

## Installation 🚀

You can easily install BreastCancerMLPModel using pip:

```bash
pip install BreastCancerMLPModel
```
## How BreastCancerMLPModel Works 🤖

BreastCancerMLPModel utilizes an MLP neural network for breast cancer diagnosis. Here's how it works:

1. **Initializing the Model** 🛠️:
   - The model is initialized using the `BreastCancerMLPModel` class from the package.
   - This class encapsulates an MLPClassifier from scikit-learn with predefined parameters.

2. **Preprocessing Input Data** 📊:
   - Input data undergoes preprocessing using feature scaling with StandardScaler.
   - Scaling ensures that features are on the same scale, improving model performance.

3. **Training and Evaluation** 📈:
   - The model is trained using the `fit()` method, which splits the dataset, scales features, and trains the MLP model.
   - Evaluation metrics, including accuracy and confusion matrix, are provided to assess model performance.

4. **Making Predictions** ⚡:
   - The `predict()` method enables prediction capabilities for new biopsy samples.
   - Input data, such as tumor characteristics, is provided to the model for prediction.

5. **Integration with Python Applications** 🐍:
   - BreastCancerMLPModel supports easy integration into Python applications for breast cancer diagnosis tasks.
   - This allows seamless incorporation of the model into existing workflows for efficient diagnosis.

This approach ensures accurate and reliable breast cancer diagnosis based on tumor characteristics, enabling better patient care and treatment decisions.

```python
from BreastCancerMLPModel.BreastCancerMLPModel import BreastCancerMLPModel

# Example usage

# Initialize the model
model = BreastCancerMLPModel()

# Train the model
model.fit()

# Make predictions

# Data for prediction 1
data1 = "mean_radius: 17.99, mean_texture: 10.38, mean_perimeter: 122.8, mean_area: 1001, mean_smoothness: 0.1184, mean_compactness: 0.2776, mean_concavity: 0.3001, mean_concave_points: 0.1471, mean_symmetry: 0.2419, mean_fractal_dimension: 0.07871, se_radius: 1.095, se_texture: 0.9053, se_perimeter: 8.589, se_area: 153.4, se_smoothness: 0.006399, se_compactness: 0.04904, se_concavity: 0.05373, se_concave_points: 0.01587, se_symmetry: 0.03003, se_fractal_dimension: 0.006193, worst_radius: 25.38, worst_texture: 17.33, worst_perimeter: 184.6, worst_area: 2019, worst_smoothness: 0.1622, worst_compactness: 0.6656, worst_concavity: 0.7119, worst_concave_points: 0.2654, worst_symmetry: 0.4601, worst_fractal_dimension: 0.1189"
prediction1 = model.predict(data1)
print("Predicted diagnosis for data 1:", prediction1) ## ('Maligno', 1.0)

# Data for prediction 2
data2 = "mean_radius: 13.08, mean_texture: 15.71, mean_perimeter: 85.63, mean_area: 520, mean_smoothness: 0.1075, mean_compactness: 0.127, mean_concavity: 0.04568, mean_concave_points: 0.0311, mean_symmetry: 0.1967, mean_fractal_dimension: 0.06811, se_radius: 0.1852, se_texture: 0.7477, se_perimeter: 1.383, se_area: 14.67, se_smoothness: 0.004097, se_compactness: 0.01898, se_concavity: 0.01698, se_concave_points: 0.00649, se_symmetry: 0.01678, se_fractal_dimension: 0.002425, worst_radius: 14.5, worst_texture: 20.49, worst_perimeter: 96.09, worst_area: 630.5, worst_smoothness: 0.1312, worst_compactness: 0.2776, worst_concavity: 0.189, worst_concave_points: 0.07283, worst_symmetry: 0.3184, worst_fractal_dimension: 0.08183"
prediction2 = model.predict(data2)
print("Predicted diagnosis for data 2:", prediction2) ##('Benigno', 0.9999982189891156)
```

## Dataset 📊

The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) dataset, available in scikit-learn's built-in datasets module. It consists of features computed from digital images of fine needle aspirate (FNA) of breast masses. Each feature represents various characteristics of cell nuclei present in the images. The dataset contains both malignant and benign tumor samples, making it suitable for binary classification tasks.

### Features and Descriptions

| Label                     | Meaning                                   | Weight in Diagnosis | Description                                           |
|---------------------------|-------------------------------------------|---------------------|-------------------------------------------------------|
| Diagnosis                 | Diagnosis (M = malignant, B = benign)    | Not used            | Result of breast cancer diagnosis                     |
| mean_radius               | Mean radius of cell nuclei               | High                | Average distance from the center to the points on the perimeter of cell nuclei                    |
| mean_texture              | Mean texture of cell nuclei              | Low                 | Standard deviation of gray-scale values in the image of cell nuclei                             |
| mean_perimeter            | Mean perimeter of cell nuclei            | High                | Average lengths of perimeters of cell nuclei                                  |
| mean_area                 | Mean area of cell nuclei                 | Very High           | Average areas of cell nuclei                                        |
| mean_smoothness           | Mean smoothness of cell nuclei           | Low                 | Local variation in lengths of cell nuclei radii                                |
| mean_compactness          | Mean compactness of cell nuclei          | High                | (Perimeter^2 / area) - 1.0                                                 |
| mean_concavity            | Mean concavity of cell nuclei            | Very High           | Severity of concave portions of cell nuclei contour                                       |
| mean_concave_points       | Mean concave points of cell nuclei       | Very High           | Number of concave portions of cell nuclei contour                              |
| mean_symmetry             | Mean symmetry of cell nuclei             | Low                 | Symmetry of cell nuclei                                               |
| mean_fractal_dimension    | Mean fractal dimension of cell nuclei    | Low                 | Coastline approximation of cell nuclei                                |
| se_radius                 | Standard error of radius                 | Medium              | Standard error of cell nuclei radius                                   |
| se_texture                | Standard error of texture                | Low                 | Standard error of cell nuclei texture                                   |
| se_perimeter              | Standard error of perimeter              | Medium              | Standard error of cell nuclei perimeter                               |
| se_area                   | Standard error of area                   | Medium              | Standard error of cell nuclei area                                      |
| se_smoothness             | Standard error of smoothness             | Low                 | Standard error of cell nuclei smoothness                              |
| se_compactness            | Standard error of compactness            | Medium              | Standard error of cell nuclei compactness                            |
| se_concavity              | Standard error of concavity              | High                | Standard error of cell nuclei concavity                            |
| se_concave_points         | Standard error of concave points         | High                | Standard error of cell nuclei concave points                       |
| se_symmetry               | Standard error of symmetry               | Low                 | Standard error of cell nuclei symmetry                              |
| se_fractal_dimension      | Standard error of fractal dimension      | Low                 | Standard error of cell nuclei fractal dimension                     |
| worst_radius              | Worst value of radius                    | High                | Worst value of cell nuclei radius                                       |
| worst_texture             | Worst value of texture                   | Low                 | Worst value of cell nuclei texture                                   |
| worst_perimeter           | Worst value of perimeter                 | High                | Worst value of cell nuclei perimeter                                   |
| worst_area                | Worst value of area                      | Very High           | Worst value of cell nuclei area                                        |
| worst_smoothness          | Worst value of smoothness                | Low                 | Worst value of cell nuclei smoothness                                   |
| worst_compactness         | Worst value of compactness               | High                | Worst value of cell nuclei compactness                                 |
| worst_concavity           | Worst value of concavity                 | Very High           | Worst value of cell nuclei concavity                                   |
| worst_concave_points      | Worst value of concave points            | Very High           | Worst value of cell nuclei concave points                              |
| worst_symmetry            | Worst value of symmetry                  | Low                 | Worst value of cell nuclei symmetry                                    |
| worst_fractal_dimension   | Worst value of fractal dimension         | Low                 | Worst value of cell nuclei fractal dimension                           |

## Usage 🚀

1. **Training the Model**: The model is trained using the `fit` method, which loads the dataset, preprocesses the input features, and trains the MLP classifier.

2. **Making Predictions**: After training, the model can be used to make predictions on new biopsy samples using the `predict` method. The input data should be provided in a specific format, including features such as mean radius, texture, perimeter, etc.

3. **Evaluation**: The model's performance can be evaluated using various metrics, including accuracy and confusion matrix, to assess its diagnostic capabilities.

## Dependencies 🛠️

- scikit-learn
- numpy

## License 📜

This project is licensed under the MIT License - see the [LICENSE](https://github.com/sergio11/breast_cancer_diagnosis_mlp/blob/main/LICENSE) file for details.

## Acknowledgments 🙏

This dataset is a copy of the UCI ML Breast Cancer Wisconsin (Diagnostic) datasets. [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)).

The input features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.

The separation plane described above was obtained using the Multiple Surface Method Tree (MSM-T) [K. P. Bennett, "Constructing a Decision Tree by Linear Programming". Proceedings of the 4th Midwest Artificial Intelligence and Cognitive Science Society, pp. 97-101, 1992], a classification method that uses linear programming to build a decision tree. Relevant features were selected through an exhaustive search in the space of 1-4 features and 1-3 separation planes.

The actual linear program used to obtain the separation plane in the three-dimensional space is described in: [K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].

This database is also available through the UW CS ftp server:

ftp ftp.cs.wisc.edu cd math-prog/cpo-dataset/machine-learn/WDBC/

References:

1. W.N. Street, W.H. Wolberg and O.L. Mangasarian. Nuclear feature extraction for breast tumor diagnosis. IS&T/SPIE 1993 International Symposium on Electronic Imaging: Science and Technology, volume 1905, pages 861-870, San Jose, CA, 1993.
2. O.L. Mangasarian, W.N. Street and W.H. Wolberg. Breast cancer diagnosis and prognosis via linear programming. Operations Research, 43(4), pages 570-577, July-August 1995.
3. W.H. Wolberg, W.N. Street, and O.L. Mangasarian. Machine learning techniques to diagnose breast cancer from fine-needle aspirates. Cancer Letters 77 (1994) 163-171.

## Contribution

Contributions to BreastCancerMLPModel are highly encouraged! If you're interested in adding new features, resolving bugs, or enhancing the project's functionality, please feel free to submit pull requests.

## Get in Touch 📬

BreastCancerMLPModel is developed and maintained by **Sergio Sánchez Sánchez** (Dream Software). Special thanks to the open-source community and the contributors who have made this project possible. If you have any questions, feedback, or suggestions, feel free to reach out at  [dreamsoftware92@gmail.com](mailto:dreamsoftware92@gmail.com).

## Visitors Count

<img width="auto" src="https://profile-counter.glitch.me/breast_cancer_diagnosis_mlp/count.svg" />

## Please Share & Star the repository to keep me motivated.
  <a href = "https://github.com/sergio11/breast_cancer_diagnosis_mlp/stargazers">
     <img src = "https://img.shields.io/github/stars/sergio11/breast_cancer_diagnosis_mlp" />
  </a>


