from typing import List, Tuple
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler


def convert_df(csv_path: str, attribute_to_convert: str) -> Tuple[DataFrame, DataFrame, MinMaxScaler]:
    """
    Do some modification with the csv file.

    @params:
    - csv_path: Path to the csv

    @returns: 
    - origin df
    - adjusted df
    - a scaler of the adjusted columns
    """
    df: DataFrame = pd.read_csv(csv_path)
    # Convert the datetime column to real datetime type
    df['Date/Time'] = pd.to_datetime(df.iloc[:, 1])
    # Extract date
    df['Date'] = df['Date/Time'].dt.date
    # Group Date collumn
    daily_df = df.groupby("Date")[df.columns].apply(
        lambda x: x.iloc[-1],
        include_groups = True
    ).reset_index(drop=True)

    scaler = MinMaxScaler()
    daily_df['Close_Transform'] = scaler.fit_transform(daily_df.iloc[:, 5:6].astype('float32'))

    daily_df = daily_df.drop(columns=['Open Interest', 'Date/Time'])
    daily_df.head()

    return df, daily_df, scaler

def extract_features(df: DataFrame, columns: List[str] = ["Close_Transform"]) -> (DataFrame | None):
    """
    Extract the necessary columns from the Dataframe.

    @params:
    - df: DataFrame 
    - columns: features to be extracted

    @returns:
    - a dataframe includes needed features
    """ 
    if columns == ["Close_Transform"]:
        return df[columns]


