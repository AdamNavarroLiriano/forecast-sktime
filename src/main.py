import argparse
import pandas as pd
from models.train import fit_estimator, create_estimator
from sktime.forecasting.base import ForecastingHorizon
from utils.data import load_data
from utils.config import get_env_params, get_model_params
import pickle
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Script for training models")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--use-best-model", action="store_true", default=True)
    group.add_argument(
        "--model",
        type=str,
    )

    # parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--env", type=str, default="main")
    args = parser.parse_args()
    return args


def setup_environment(env, model_name, use_best_model=False) -> tuple:
    env_params = get_env_params(env)

    if use_best_model:
        model_name = env_params.best_model

    model_params = get_model_params(model_name)

    # Read data
    train = load_data(env=env)
    y = train["y"].reset_index(drop=True)
    fh = ForecastingHorizon(env_params.fh)

    # Get relevant values
    last_value = pd.to_datetime(train.index, format=env_params.date_format)[
        -1
    ].strftime("%Y-%m")
    cutoff = pd.Period(last_value, freq="M")

    # Try to read model and if not, train
    try:
        model = pickle.load(open(f"../../{model_params.output_path}", "rb"))
        logging.info(f"Model {model_name} loaded from data/models.")
    except:
        logging.info(
            f"Model {model_name} not found in data/models. Fitting it with params"
        )
        # Create estimator
        estimator = create_estimator(
            model_name=model_name,
            model_params=model_params.params,
            transforms=model_params.transforms,
        )
        model = fit_estimator(y=y, fh=fh, estimator=estimator)

    return (model, fh, cutoff, train)


def format_output(prediction: pd.Series, cutoff, fh) -> pd.DataFrame:
    prediction = prediction.copy()
    prediction.index = list(fh.to_absolute(cutoff))
    prediction_df = pd.DataFrame(prediction)
    prediction_df.index = [
        period.strftime("01.%m.%y") for period in prediction_df.index
    ]

    return prediction_df


def main():
    try:
        os.chdir("models/")
    except:
        pass

    args = parse_args()

    if args.model is not None:
        model, fh, cutoff, train = setup_environment(
            env=args.env, model_name=args.model, use_best_model=False
        )
    else:
        model, fh, cutoff, train = setup_environment(
            env=args.env, model_name=args.model, use_best_model=args.use_best_model
        )

    prediction = model.predict(fh)
    prediction_df = format_output(prediction, cutoff, fh)

    # Store in data
    prediction_df.to_csv("../../data/test.csv")
    logger.info("Prediction finished and stored in data/test.csv")


if __name__ == "__main__":
    main()
