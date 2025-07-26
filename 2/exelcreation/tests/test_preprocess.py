# filepath: d:\ML\project\tests\test_preprocess.py
import unittest
import numpy as np
from src.preprocess import split_and_scale

class TestPreprocess(unittest.TestCase):
    def setUp(self):
        # Create sample data
        self.X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        self.y = np.array([0, 1, 0, 1])

    def test_split_and_scale(self):
        x_train, x_test, y_train, y_test = split_and_scale(self.X, self.y)
        
        # Check shapes
        self.assertEqual(x_train.shape[1], self.X.shape[1])
        self.assertEqual(x_test.shape[1], self.X.shape[1])
        
        # Check scaling (should be between 0 and 1)
        self.assertTrue(np.all(x_train >= 0) and np.all(x_train <= 1))
        self.assertTrue(np.all(x_test >= 0) and np.all(x_test <= 1))

if __name__ == '__main__':
    unittest.main()