import unittest
import pandas as pd
from src.preprocessing import preprocess_data

class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.sample_data = pd.DataFrame({
            'process_value_1': [1.0, 2.0, 3.0, None, 5.0],
            'process_value_2': [5.0, 4.0, None, 2.0, 1.0],
            'vibration_value': [0.1, 0.2, 0.3, 0.4, 0.5]
        })

    def test_preprocess_data(self):
        processed_data = preprocess_data(self.sample_data)
        self.assertEqual(processed_data.isnull().sum().sum(), 0)
        self.assertTrue('normalized_process_value_1' in processed_data.columns)
        self.assertTrue('normalized_process_value_2' in processed_data.columns)

    def test_preprocess_data_empty(self):
        empty_data = pd.DataFrame()
        processed_data = preprocess_data(empty_data)
        self.assertTrue(processed_data.empty)

if __name__ == '__main__':
    unittest.main()