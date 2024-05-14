import pandas as pd
from .config import get_env_params
import typing

ROOT = "../.."

read_functions = {
    "csv": pd.read_csv,
    "parquet": pd.read_parquet,
    "excel": pd.read_excel,
    "json": pd.read_json,
}


def load_data(env="main") -> pd.DataFrame:
    """Loads dataset using configuration variables
    :param env: environment config file (conf/<env>.yml), defaults to 'main'
    :type env: str, optional
    :return: DataFrame loaded
    :rtype: pd.DataFrame
    """
    env_params = get_env_params(env=env)

    read_func = read_functions[env_params.train_data.format]
    path = env_params.train_data.path

    data = read_func(f"{ROOT}/{path}", **env_params.train_data.kwargs)
    return data


def split_timeseries(
    data: pd.DataFrame, periods: int
) -> typing.Tuple[typing.Union[pd.DataFrame, pd.Series]]:
    """Splits the series for training/testing

    :param data: series to be split
    :type data: pd.DataFrame
    :param periods: amount of periods in the test dataset
    :type periods: int
    :return: set of series split
    :rtype: typing.Tuple[typing.Union[pd.DataFrame, pd.Series]]
    """

    test_series = data[-periods:]
    train_series = data[0:-periods]

    return train_series, test_series
