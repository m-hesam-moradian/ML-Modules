import unittest
import numpy as np
from src.model_runner import train_and_evaluate_model

class TestModelRunner(unittest.TestCase):
    def setUp(self):
        # Create sample data
        np.random.seed(42)
        self.x_train = np.random.rand(100, 2)
        self.x_test = np.random.rand(20, 2)
        self.y_train = np.random.randint(0, 2, 100)
        self.y_test = np.random.randint(0, 2, 20)

    def test_different_models(self):
        # Test Random Forest
        rf_results = train_and_evaluate_model(
            'ensemble', 'RandomForestClassifier',
            self.x_train, self.x_test, self.y_train, self.y_test,
            {'n_estimators': 10}
        )
        self.assertIn('model', rf_results)
        self.assertIn('train', rf_results)
        self.assertIn('test', rf_results)
        
        # Test Logistic Regression
        lr_results = train_and_evaluate_model(
            'linear_model', 'LogisticRegression',
            self.x_train, self.x_test, self.y_train, self.y_test
        )
        self.assertIn('model', lr_results)
        self.assertIn('train', lr_results)
        self.assertIn('test', lr_results)

if __name__ == '__main__':
    unittest.main()