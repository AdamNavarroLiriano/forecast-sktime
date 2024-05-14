from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.arima import ARIMA, AutoARIMA
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sktime.forecasting.compose import make_reduction
from sktime.transformations.series.exponent import ExponentTransformer
from sktime.transformations.series.boxcox import LogTransformer

MODELS = {
    "arima": ARIMA,
    "autoarima": AutoARIMA,
    "naive": NaiveForecaster,
    "rf": lambda window_length, **params: make_reduction(
        RandomForestRegressor(**params), window_length=window_length
    ),
    "xgb": lambda window_length, **params: make_reduction(
        XGBRegressor(**params), window_length=window_length
    ),
}

TRANSFORMATIONS = {"log": LogTransformer(), "sqrt": ExponentTransformer(power=0.5)}
