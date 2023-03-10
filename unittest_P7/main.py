import unittest
import pandas as pd

def load_data_train(path):
    df = pd.read_csv(path)
    return df, len(df)
data, len_data = load_data_train(r"C:\Users\Shata\P7\data.csv")


class Test(unittest.TestCase):
    def test_load_data_train(self):
        expected = 307507
        df, len_ = load_data_train(r"C:\Users\Shata\P7\data.csv")
        self.assertEqual(len_, expected)


if __name__ == '__main__':
    unittest.main()