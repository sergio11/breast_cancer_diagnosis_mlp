import unittest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from BreastCancerMLPModel.BreastCancerMLPModel import BreastCancerMLPModel

class TestMLPModel(unittest.TestCase):
    def setUp(self):
        self.model = BreastCancerMLPModel()

        # Load breast cancer dataset
        breast_cancer = load_breast_cancer()
        X = breast_cancer.data
        y = breast_cancer.target

        # Split dataset into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def test_fit(self):
        self.model.fit()

        # Ensure that the model is trained successfully
        self.assertIsNotNone(self.model.model)

    def test_predict(self):
        # Train the model
        self.model.fit()

        # Create a sample input data
        data = ("mean_radius: 13.08, mean_texture: 15.71, mean_perimeter: 85.63, mean_area: 520, "
                "mean_smoothness: 0.1075, mean_compactness: 0.127, mean_concavity: 0.04568, "
                "mean_concave_points: 0.0311, mean_symmetry: 0.1967, mean_fractal_dimension: 0.06811, "
                "se_radius: 0.1852, se_texture: 0.7477, se_perimeter: 1.383, se_area: 14.67, "
                "se_smoothness: 0.004097, se_compactness: 0.01898, se_concavity: 0.01698, "
                "se_concave_points: 0.00649, se_symmetry: 0.01678, se_fractal_dimension: 0.002425, "
                "worst_radius: 14.5, worst_texture: 20.49, worst_perimeter: 96.09, worst_area: 630.5, "
                "worst_smoothness: 0.1312, worst_compactness: 0.2776, worst_concavity: 0.189, "
                "worst_concave_points: 0.07283, worst_symmetry: 0.3184, worst_fractal_dimension: 0.08183")

        # Make a prediction
        predicted_label, probability = self.model.predict(data)

        # Ensure that the predicted label is either "Maligno" or "Benigno"
        self.assertIn(predicted_label, ["Maligno", "Benigno"])

        # Ensure that the probability is a float
        self.assertTrue(isinstance(probability, float))

if __name__ == '__main__':
    unittest.main()