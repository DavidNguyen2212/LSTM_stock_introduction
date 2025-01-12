from typing import List, Tuple
import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


def convert_df(csv_path: str) -> Tuple[DataFrame, MinMaxScaler]:
    df: DataFrame = pd.read_csv(csv_path)
    # Convert the datetime column to real datetime type
    df['Date/Time'] = pd.to_datetime(df.iloc[:, 1])
    # Extract date
    df['Date'] = df['Date/Time'].dt.date
    # Create Date collumn
    df = df.groupby("Date")[df.columns].apply(
        lambda x: x.iloc[-1],
        include_groups = True
    ).reset_index(drop=True)

    scaler = MinMaxScaler()
    df['Close'] = scaler.fit_transform(df.iloc[:, 5:6].astype('float32'))

    df = df.drop(columns=['Open Interest', 'Date/Time'])
    df.head()

    return df, scaler

def extract_features(df: DataFrame, columns: List[str] = ["Close"]):
    if columns == ["Close"]:
        return df[columns]


