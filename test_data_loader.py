from email import header
import unittest
import numpy as np
import pandas as pd

from data_loader import BaseDataLoader

class TestDataLoader(unittest.TestCase):
    def create_unsupervised_dataset(self):
        M = np.random.random([10, 3])
        test_df = pd.DataFrame(M, columns = ["A", "B", "C"])
        test_df.to_csv("test.csv", index = False)

    def create_supervised_dataset(self):
        M = np.random.random([10, 2])
        Ind = np.random.random([10, 1])
        Ind[Ind < 0.5] = 0
        Ind[Ind > 0.5] = 1
        test_df = pd.DataFrame(M, columns = ["A", "B"])
        test_df["C"] = Ind
        test_df.to_csv("test.csv", index = False)

    def test_supervised_dataset_no_split(self):
        self.create_supervised_dataset()
        dl = BaseDataLoader("test.csv",  ["A", "B"],  "C", False)
        df = pd.read_csv("test.csv")
        X = df[["A", "B"]]
        y = df[["C"]]

        return self.assertEqual((X.to_dict(), y.to_dict()), (dl.get_data()[0].to_dict(), dl.get_data()[1].to_dict()))

if __name__ == "__main__":
    unittest.main()
