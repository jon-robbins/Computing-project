import unittest
import pandas as pd

from oldfartsfinalproject import data_preprocessing as dp


class TestDataPreprocessing(unittest.TestCase):

    def test_impute_na(self):
        """
        Test for the functionality of the impute_na method of data pre_processing module
        """
        df_a = pd.DataFrame({"a": [None, None]})
        df_b = pd.DataFrame({"a": [1, 1]})
        df = dp.impute_na(df_a, [("a", 1)])
        pd.testing.assert_frame_equal(df, df_b)

    def test_to_num(self):
        """
        Test for the functionality of the impute_to_num method of data pre_processing module
        """
        df_test = pd.DataFrame({"a": ["1", "1"]})
        df_result = pd.DataFrame({"a": [1, 1]})
        pd.testing.assert_frame_equal(dp.to_num(df_test, ["a"]), df_result)

    def test_delete_unnec_cols(self):
        """
        Test for the functionality of the test_delete_unnec method of data pre_processing module
        """
        df_test = pd.DataFrame({"totaltaxvalue": ["1"], "a": ["1"]})
        df_result = pd.DataFrame({"a": ["1"]})
        pd.testing.assert_frame_equal(dp.delete_unnec_cols(df_test, 0.4, ["totaltaxvalue"]), df_result)
