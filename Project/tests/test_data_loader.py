# filepath: d:\ML\project\tests\test_data_loader.py
import unittest
from src.data_loader import load_data
import pandas as pd
import os

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        # Create sample data
        self.test_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [5, 6, 7, 8],
            'target': [0, 1, 0, 1]
        })
        # Save test data
        self.test_file = 'test_data.xlsx'
        self.test_data.to_excel(self.test_file, index=False)

    def test_load_data(self):
        X, y = load_data(self.test_file, 'target')
        self.assertEqual(X.shape, (4, 2))
        self.assertEqual(y.shape, (4,))
        
    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

if __name__ == '__main__':
    unittest.main()