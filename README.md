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


