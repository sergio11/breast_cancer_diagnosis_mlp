# Breast Cancer Diagnosis with MLP 🩺💻

This project utilizes a Multi-Layer Perceptron (MLP) neural network implemented with scikit-learn to perform breast cancer diagnosis based on tumor characteristics extracted from biopsy samples. The MLP model is trained on a dataset containing various features derived from digital images of breast tissue samples, such as mean radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.

[![GitHub](https://img.shields.io/badge/GitHub-View%20on%20GitHub-blue?style=flat-square)](https://github.com/sergio11/breast_cancer_diagnosis_mlp)
[![PyPI](https://img.shields.io/pypi/v/BreastCancerMLPModel.svg?style=flat-square)](https://pypi.org/project/BreastCancerMLPModel/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://github.com/sergio11/breast_cancer_diagnosis_mlp/blob/main/LICENSE)

## Purpose 🎯

The primary objective of this project is to develop an accurate and reliable system for diagnosing breast cancer based on quantitative analysis of cell nuclei characteristics. By leveraging machine learning techniques, specifically MLP neural networks, we aim to create a predictive model capable of classifying tumors as either malignant (cancerous) or benign (non-cancerous) with high accuracy. Early and accurate diagnosis of breast cancer can significantly improve patient outcomes by enabling timely treatment and intervention.

## Key Features 🔑

- Utilizes an MLP neural network for breast cancer diagnosis.
- Preprocesses input data using feature scaling with StandardScaler.
- Implements training and evaluation functionalities.
- Provides prediction capabilities for new biopsy samples.
- Offers detailed model evaluation metrics, including accuracy and confusion matrix.
- Supports easy integration into Python applications for breast cancer diagnosis tasks.

## Installation 🚀

You can easily install BreastCancerMLPModel using pip:

```bash
pip install BreastCancerMLPModel
```
## How BreastCancerMLPModel  Works

BreastCancerMLPModel leverages a Multi-Layer Perceptron (MLP) neural network to diagnose breast cancer based on tumor characteristics. Here's how it works:

1. **Initializing the Model**: 
   - The model is initialized using the `BreastCancerMLPModel` class from the package.
   - This class encapsulates an MLPClassifier from scikit-learn with predefined parameters.

2. **Training the Model**:
   - The `fit()` method is called to train the model.
   - The breast cancer dataset is loaded, split into training and test sets, and scaled using a `StandardScaler`.
   - The scaled features are used to train the MLP model.

3. **Making Predictions**:
   - Once the model is trained, predictions can be made using the `predict()` method.
   - Input data, such as tumor characteristics, is provided to the model in the form of feature values.
   - The model predicts the diagnosis (Malignant or Benign) based on the input data.

4. **Example Usage**:
   - In the provided example, two sets of tumor characteristics (`data` and `data2`) are used as input for prediction.
   - The `predict()` method is called with each set of characteristics, and the predicted diagnosis is printed to the console.

This process enables quick and accurate diagnosis of breast cancer based on tumor characteristics extracted from biopsy samples.

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

- Special thanks to the open-source community for their contributions.

## Contribution

Contributions to BreastCancerMLPModel are highly encouraged! If you're interested in adding new features, resolving bugs, or enhancing the project's functionality, please feel free to submit pull requests.

## Get in Touch 📬

BreastCancerMLPModel is developed and maintained by **Sergio Sánchez Sánchez** (Dream Software). Special thanks to the open-source community and the contributors who have made this project possible. If you have any questions, feedback, or suggestions, feel free to reach out at  [dreamsoftware92@gmail.com](mailto:dreamsoftware92@gmail.com).

## Please Share & Star the repository to keep me motivated.
  <a href = "https://github.com/sergio11/breast_cancer_diagnosis_mlp/stargazers">
     <img src = "https://img.shields.io/github/stars/sergio11/breast_cancer_diagnosis_mlp" />
  </a>


