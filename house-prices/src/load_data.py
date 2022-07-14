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


def get_combined_data(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series]:

    target = train_df.SalePrice
    train_df.drop(["SalePrice"], axis=1, inplace=True)

    combined = train_df.append(test_df)
    combined.reset_index(inplace=True)
    combined.drop(["index", "Id"], inplace=True, axis=1)

    return combined, target

