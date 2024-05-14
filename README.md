# Time series forecasting

## Introduction

Welcome to the Time Series Forecasting Project! This project aims to provide a flexible library for fitting univariate time series models. It comes with a range of models out-of-the-box, including naive approaches, ARIMA, RandomForest, and XGBoost. The codebase is designed to be easily extended to support more sophisticated models and different types of time series analysis.

## Getting Started

### Environment Setup

There are two ways to set up your environment for this project:

#### 1. Conda Environment

If you prefer using Conda, create an environment with Python 3.8 and install the dependencies listed in `env/requirements.txt`.

```shell
conda create --name time_series_forecasting python=3.8
conda activate time_series_forecasting
pip install -r env/requirements.txt
```

#### 2. Docker Setup

Alternatively, you can build a Docker image that includes Python 3.8 and all necessary dependencies.

```shell
docker-compose up --build
```

### Library Installation

Once your environment is set up, you can install the library by running:

```shell
python setup.py bdist_wheel && pip install dist/*
```

## Usage

### Running Models

To run the pre-configured best model and output a `test.csv` with the forecasts, execute the following commands:

```shell
cd src
python main.py --use-best-model
```

If you want to experiment with different models, you can specify the model using the `--model` flag:

```shell
python main.py --model [arima | xgb | rf]
```

### Notebooks

The `notebook` directory contains Jupyter notebooks that are useful for different stages of the project:

- `1-EDA.ipynb`: Exploratory Data Analysis
- `2-Models.ipynb`: Model Fitting
- `3-Output.ipynb`: Final Output Checking

### Model Training

The best estimator selected is an ARIMA(0, 0, 1) (1, 0, 0, 6) after performing a logarithmic transformation to the series, as per a set of metrics (MAE, MAPE, MDAPE). Other models such as XGBoost with logarithmic transformation work interesting. 

The initial dataset was divided into a training time series with 62 data points and a test series with the last 12 observations.

- **ARIMA**: The ARIMA model was fitted using the Box-Jenkins methodology. The residuals of the ARIMA model were analyzed, and the final model showed non-autocorrelated residuals that were centered at 0 and normally distributed without heteroskedasticity.
- **Tree-based Models**: RandomForest and XGBoost models were fine-tuned using `GridSearchCV`. No residuals analysis was done, but it can prove to be future work.

For more advanced models like LSTM, additional interface adaptations are required to make them compatible with the current library setup.

## Codebase Structure

The project is designed to be configuration-based. It allows for easy experimentation, model adjustments, and integration of new models.

### Environment Variables

General environment variables and configurations for training and inference can be found in `conf/environments/`. Each environment is represented by a YAML file, such as `main.yml`, which contains the configuration values.

### Model Registry

Models and transformations can be added to the registry at `src/models/registry.py`, with their configuration files located in `conf/models/<model_name>.yml`.

### Training a Model

To train a specific model with specified parameters, run:

```shell
cd src/models
python train.py --model arima --use-model-params
```

This command will train an ARIMA model using the parameters defined in `conf/models/arima.yml`. To save the trained model, use `--save-model`. To perform a grid search over the hyperparameter grid defined in the model's config file, use `--tune-hyperparams`. (Note: ARIMA hyperparameter tuning is not supported yet.)

## Future Work

Future work could expand the library to handle multivariate time series and integrate models beyond the sklearn API. The goal is to create a robust and versatile tool for time series forecasting.

---
