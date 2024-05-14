import argparse
import pandas as pd
import typing

from models.base import Estimator
from models.registry import MODELS, TRANSFORMATIONS
import typing

from sktime.forecasting.model_selection import (
    ForecastingGridSearchCV,
)

from sktime.forecasting.base import ForecastingHorizon
from utils.data import load_data, split_timeseries
from utils.config import get_env_params, get_model_params

from utils.cross_val import create_cv_splitter
import pickle
import logging

import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_estimator(
    model_name: str, model_params: dict, transforms: typing.Union[list, None] = None
) -> Estimator:
    """Creates an Estimator with possibly some data transformations

    :param model_name: model name, as in registry.py
    :type model_name: str
    :param model_params: dictionary containing parameters required to fit model
    :type model_params: dict
    :param transforms: list of transformations to be applied to Series, as in registry.py.
    Defaults to None
    :type transforms: typing.Union[list, None], optional
    :return: Estimator object
    :rtype: Estimator
    """

    model_type = MODELS[model_name]
    model = model_type(**model_params)

    if transforms is not None:
        transformations = [TRANSFORMATIONS[transform] for transform in transforms]
    else:
        transformations = None

    estimator = Estimator(model=model, transformations=transformations)

    return estimator


def fit_estimator(y: pd.Series, fh, estimator: Estimator) -> Estimator:
    """Takes an estimator and fits the data to it

    :param y: series properly parsed to be used with sktime
    :type y: pd.Series
    :param fh: forecasting horizon
    :type fh: _type_
    :param estimator: estimator instance
    :type estimator: Estimator
    :return: estimator after being fitted with data
    :rtype: Estimator
    """
    estimator.fit(y, fh=fh)
    return estimator


def find_gridsearch_params(
    y: pd.Series, forecaster, param_grid: dict, fh, cv, **kwargs
):
    """Finds the best set of estimators using GridSearch

    :param y: series to be fitted
    :type y: pd.Series
    :param forecaster: forecaster from Sktime or valid Estimator
    :type forecaster: Estimator or Sktime estimator
    :param param_grid: dictionary containing parameters to test
    :type param_grid: dict
    :param fh: forecasting hroizon
    :type fh: valid forecasting horizon instances
    :param cv: cross validation object to test the series
    :type cv: valid cross validation instance
    :return: result from gridsearch cv
    """

    if isinstance(forecaster, Estimator):
        gridsearch_cv = ForecastingGridSearchCV(
            forecaster.pipeline, cv=cv, param_grid=param_grid, **kwargs
        )

    else:
        gridsearch_cv = ForecastingGridSearchCV(
            forecaster, cv=cv, param_grid=param_grid, **kwargs
        )

    gridsearch_cv.fit(y, fh=fh)
    return gridsearch_cv


def save_forecaster(forecaster, path):
    dir_name = os.path.dirname(path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(forecaster, f)


def load_forecaster(path):
    return pickle.load(open(path, "rb"))


def save_best_forecaster(cv_results, path: str) -> None:
    """Stores result from best model

    :param cv_results: result from grid search or randomized search
    :param path: path to save pickle file
    :type path: str
    """
    best_forecaster = cv_results.best_forecaster_
    save_forecaster(best_forecaster, path)
    logger.info(f"Best model stored in {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Script for training models")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--env", type=str, default="main")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--use-model-params", action="store_true", default=True)
    group.add_argument("--tune-hyperparams", action="store_true")

    parser.add_argument("--save_model", action="store_true", default=False)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    env = args.env
    model = args.model

    logger.info(f"Training {model}")

    # Read environment params and model parpams
    env_params = get_env_params(env)
    model_params = get_model_params(model)

    # Load timeseries
    y = load_data(env)["y"].reset_index(drop=True)
    fh = ForecastingHorizon(env_params.fh)

    # Create estimator
    estimator = create_estimator(
        model_name=model,
        model_params=model_params.params,
        transforms=model_params.transforms,
    )

    if args.use_model_params:
        estimator = fit_estimator(y=y, fh=fh, estimator=estimator)

        # Save estimator
        path_output = f"../../{model_params.output_path}"
        save_forecaster(estimator, path=path_output)

        logger.info(
            f"Finished training model. Pickle file at {model_params.output_path}"
        )

        return None

    if args.tune_hyperparams:
        cv = create_cv_splitter(
            strategy=env_params.cross_val.strategy,
            forecasting_horizon=fh,
            **env_params.cross_val.args,
        )

        ytrain, ytest = split_timeseries(y, periods=env_params.test_periods)

        gridsearch_result = find_gridsearch_params(
            y=ytrain,
            forecaster=estimator,
            param_grid=dict(model_params.grid_search.params_grid),
            fh=fh,
            cv=cv,
            **dict(model_params.grid_search.kwargs),
        )
        save_best_forecaster(
            gridsearch_result, path=model_params.grid_search.model_path
        )


if __name__ == "__main__":
    main()
