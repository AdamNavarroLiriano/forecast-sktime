import typing
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.base import BaseEstimator


class Estimator(BaseEstimator):
    """
    Fits an estimator after making transformation to series
    """

    def __init__(self, model, transformations: typing.Union[list, str, None] = None):
        super().__init__()
        self.model = model
        self.transformations = transformations
        self._pipeline = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            if self.transformations is None:
                self._pipeline = self.model

            elif isinstance(self.transformations, str):
                self._pipeline = self.transformations * self.model

            else:
                self._pipeline = TransformedTargetForecaster(
                    steps=self.transformations + [self.model]
                )

        return self._pipeline

    @property
    def is_fitted(self):
        return self.pipeline.is_fitted

    def fit(self, y, fh):
        self.pipeline.fit(y, fh=fh)

    def predict(self, fh):
        return self.pipeline.predict(fh=fh)
