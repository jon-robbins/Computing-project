import unittest
import pandas as pd
from oldfartsfinalproject import feature_creation as fc


class TestFeatureCreation(unittest.TestCase):

    def test_create_dummies(self):
        """
        Test for the functionality of the create_dummies method of the feature creation module
        """
        df_test = pd.DataFrame({"a": ["1", "0"]})
        df_result = pd.DataFrame({"a_0": [0, 1], "a_1": [1, 0]})
        pd.testing.assert_frame_equal(fc.create_dummies(df_test, ["a"]), df_result, check_dtype=False)
