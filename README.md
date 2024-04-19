# Breast Cancer Diagnosis with MLP 🩺💻

This project utilizes a Multi-Layer Perceptron (MLP) neural network implemented with scikit-learn to perform breast cancer diagnosis based on tumor characteristics extracted from biopsy samples. The MLP model is trained on a dataset containing various features derived from digital images of breast tissue samples, such as mean radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.

## Purpose 🎯

The primary objective of this project is to develop an accurate and reliable system for diagnosing breast cancer based on quantitative analysis of cell nuclei characteristics. By leveraging machine learning techniques, specifically MLP neural networks, we aim to create a predictive model capable of classifying tumors as either malignant (cancerous) or benign (non-cancerous) with high accuracy. Early and accurate diagnosis of breast cancer can significantly improve patient outcomes by enabling timely treatment and intervention.

## Key Features 🔑

- Utilizes an MLP neural network for breast cancer diagnosis.
- Preprocesses input data using feature scaling with StandardScaler.
- Implements training and evaluation functionalities.
- Provides prediction capabilities for new biopsy samples.
- Offers detailed model evaluation metrics, including accuracy and confusion matrix.
- Supports easy integration into Python applications for breast cancer diagnosis tasks.

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


