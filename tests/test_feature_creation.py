import unittest
import pandas as pd
from oldfartsfinalproject import feature_creation as fc


class TestFeatureCreation(unittest.TestCase):

    def test_impute_na(self):
        df_test = pd.DataFrame({"a": ["1", "0"]})
        df_result = pd.DataFrame({"a_0": ["0", "1"], "a_1": ["1", "0"]})
        self.assertEqual(fc.create_dummies(df_test, ["a"]), df_result)
