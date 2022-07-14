import pandas as pd
import numpy as np
from typing import List


def get_cols_with_no_nans(df: pd.DataFrame, col_type: str) -> List:
    """
    Arguments :
    df : The dataframe to process
    col_type : 
          num : to only get numerical columns with no nans
          no_num : to only get nun-numerical columns with no nans
          all : to get any columns with no nans    
    """

    if col_type == "num":
        predictors = df.select_dtypes(exclude=["object"])
    elif col_type == "no_num":
        predictors = df.select_dtypes(include=["object"])
    elif col_type == "all":
        predictors = df
    else:
        print("Error : choose a type (num, no_num, all)")
        return 0
    cols_with_no_nans = []

    for col in predictors.columns:
        if not df[col].isnull().any():
            cols_with_no_nans.append(col)

    return cols_with_no_nans


def oneHotEncode(df: pd.DataFrame, colNames: List[str]):
    for col in colNames:
        if df[col].dtype == np.dtype("object"):
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)

            # drop the encoded column
            df.drop([col], axis=1, inplace=True)

    return df
