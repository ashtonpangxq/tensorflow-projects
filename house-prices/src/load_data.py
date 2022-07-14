import pandas as pd
from typing import Tuple


def get_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Get train data
    train_data_path = train_path
    train = pd.read_csv(train_data_path)

    # Get test data
    test_data_path = test_path
    test = pd.read_csv(test_data_path)

    return train, test
