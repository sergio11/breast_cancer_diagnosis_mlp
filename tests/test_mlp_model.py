import unittest
import numpy as np
from mlp_breast_cancer_diagnosis.mlp_model import BreastCancerMLPModel

class TestMLPModel(unittest.TestCase):
    def setUp(self):
        self.model = BreastCancerMLPModel()
        self.X_train = np.array([[1, 2], [3, 4]])
        self.y_train = np.array([0, 1])
        self.X_test = np.array([[5, 6], [7, 8]])
        self.y_test = np.array([1, 0])

    def test_train(self):
        self.model.train(self.X_train, self.y_train)
        self.assertIsNotNone(self.model.model)

    def test_evaluate(self):
        self.model.train(self.X_train, self.y_train)
        accuracy, _ = self.model.evaluate(self.X_test, self.y_test)
        self.assertTrue(isinstance(accuracy, float))

if __name__ == '__main__':
    unittest.main()