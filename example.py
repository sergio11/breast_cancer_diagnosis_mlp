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
print("Predicted diagnosis for data 1:", prediction1)

# Data for prediction 2
data2 = "mean_radius: 13.08, mean_texture: 15.71, mean_perimeter: 85.63, mean_area: 520, mean_smoothness: 0.1075, mean_compactness: 0.127, mean_concavity: 0.04568, mean_concave_points: 0.0311, mean_symmetry: 0.1967, mean_fractal_dimension: 0.06811, se_radius: 0.1852, se_texture: 0.7477, se_perimeter: 1.383, se_area: 14.67, se_smoothness: 0.004097, se_compactness: 0.01898, se_concavity: 0.01698, se_concave_points: 0.00649, se_symmetry: 0.01678, se_fractal_dimension: 0.002425, worst_radius: 14.5, worst_texture: 20.49, worst_perimeter: 96.09, worst_area: 630.5, worst_smoothness: 0.1312, worst_compactness: 0.2776, worst_concavity: 0.189, worst_concave_points: 0.07283, worst_symmetry: 0.3184, worst_fractal_dimension: 0.08183"
prediction2 = model.predict(data2)
print("Predicted diagnosis for data 2:", prediction2)