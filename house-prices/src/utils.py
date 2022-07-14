import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List


def filling_missing_values(df: pd.DataFrame) -> pd.DataFrame:

    df["PoolQC"] = df["PoolQC"].fillna("no")
    df["MiscFeature"] = df["MiscFeature"].fillna("no")
    df["Alley"] = df["Alley"].fillna("no")
    df["Fence"] = df["Fence"].fillna("no")
    df["FireplaceQu"] = df["FireplaceQu"].fillna("no")
    df["GarageCond"] = df["GarageCond"].fillna("no")
    df["GarageQual"] = df["GarageQual"].fillna("no")
    df["GarageFinish"] = df["GarageFinish"].fillna("no")
    df["BsmtExposure"] = df["BsmtExposure"].fillna("no")
    df["BsmtCond"] = df["BsmtCond"].fillna("no")
    df["BsmtQual"] = df["BsmtQual"].fillna("no")
    df["BsmtFinType2"] = df["BsmtFinType2"].fillna("no")
    df["BsmtFinType1"] = df["BsmtFinType1"].fillna("no")
    df["Fence"] = df["Fence"].fillna("no")
    df["MasVnrType"] = df["MasVnrType"].fillna("no")
    df["GarageYrBlt"] = df["GarageYrBlt"].fillna(0)
    df["GarageType"] = df["GarageType"].fillna(0)
    df["GarageArea"] = df["GarageArea"].fillna(0)
    df["GarageCars"] = df["GarageCars"].fillna(0)
    df["BsmtFinSF1"] = df["BsmtFinSF1"].fillna(0)
    df["BsmtFinSF2"] = df["BsmtFinSF2"].fillna(0)
    df["MasVnrArea"] = df["MasVnrArea"].fillna(0)
    df["BsmtFullBath"] = df["BsmtFullBath"].fillna(0)
    df["BsmtHalfBath"] = df["BsmtHalfBath"].fillna(0)
    df["BsmtUnfSF"] = df["BsmtUnfSF"].fillna(0)
    df["TotalBsmtSF"] = df["TotalBsmtSF"].fillna(0)

    return df


def mae(y_test, y_pred):
    """
  Calculates mean absolute error between y_test and y_preds.
  """
    return tf.metrics.mean_absolute_error(y_test, y_pred)


def mse(y_test, y_pred):
    """
  Calculates mean squared error between y_test and y_preds.
  """
    return tf.metrics.mean_squared_error(y_test, y_pred)

