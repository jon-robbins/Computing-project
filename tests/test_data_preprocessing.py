import unittest
import pandas as pd

from oldfartsfinalproject import data_preprocessing as dp


class TestDataPreprocessing(unittest.TestCase):

    def test_impute_na(self):
        df_a = pd.DataFrame({"a": [None, None]})
        df_b = pd.DataFrame({"a": [1, 1]})
        df = dp.impute_na(df_a, [("a", 1)])
        self.assertEqual(df, df_b)

    def test_to_num(self):
        df_test = pd.DataFrame({"a": ["1", "1"]})
        df_result = pd.DataFrame({"a": [1.00, 1.00]})
        self.assertEqual(dp.to_num(df_test, ["a"]), df_result)
