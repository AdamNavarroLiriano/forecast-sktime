import pandas as pd
import numpy as np
import typing
from sktime.split import ExpandingWindowSplitter, SlidingWindowSplitter

cv_strategy = {"expanding": ExpandingWindowSplitter, "sliding": SlidingWindowSplitter}


# TODO: create class for splitter
def create_cv_splitter(
    strategy: str,
    forecasting_horizon: typing.Union[int, list, np.ndarray, pd.Index],
    **kwargs
):
    """Creates a split for cross validation

    :param strategy: string representing the strategy for cross val.
    Can be 'expanding' or 'sliding'
    :type strategy: str
    :param forecasting_horizon: forecasting horizong for cross val
    :type forecasting_horizon: typing.Union[int, list, np.ndarray, pd.Index]
    :return: crossvalidation splitter
    :rtype: Splitter
    """
    splitter = cv_strategy[strategy]
    return splitter(fh=forecasting_horizon, **kwargs)
