# import fundamental libraries
import zipfile
import cProfile
import multiprocessing
from itertools import combinations

# time libraries
import time
from datetime import date
from datetime import datetime
from datetime import timedelta
from dateutil import relativedelta

# Big Data libraries
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd
from pandas.api.types import CategoricalDtype
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

# stats libraries
from scipy import interpolate
from sklearn.metrics import mean_squared_error, mean_absolute_error

# visuals and EDA libraries
import matplotlib
from matplotlib import cm

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.legend_handler import HandlerBase
from matplotlib.text import Text

# http://stanford.edu/~mwaskom/software/seaborn/
import seaborn as sns

import joypy  # ridgeline plot

import plotly.graph_objects as go

# machine learning libraries
from prophet import Prophet

# my libraries
import wrangle as wr
import preprocessing as pr
import explore as ex

# calc last month
last_month = pd.to_datetime(datetime.today().strftime("%Y-%m-%d")) + MonthEnd(-1)


# this function creates a model evaluation metric that measures the average forecast variance to actuals for the timeframe of the model
def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def forecast_trainerize_mrr(
    df,
    variable_to_forecast="mrr",
    train_test_split_months=6,
    forecast_time_horizon="2026-12-31",
    forecast_frequency="d",
):
    """
    This function creates a machine learning model using Facebook's Prophet model.

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
        df:
    """

    # prep data in the exact variable names that Prophet needs
    prophet_df = pd.DataFrame(
        df.groupby("date")[variable_to_forecast].sum()
    ).reset_index()
    prophet_df = prophet_df.rename(columns={"date": "ds", variable_to_forecast: "y"})
    prophet_df = prophet_df.set_index("ds")

    # train/test split is equal to:
    tz_train = prophet_df.loc[
        prophet_df.index <= (last_month + MonthEnd(-(train_test_split_months)))
    ]  # everything prior to 6 months ago
    tz_test = prophet_df.loc[
        prophet_df.index > (last_month + MonthEnd(-(train_test_split_months)))
    ]  # everything after to 6 months ago

    #  A Simple Model
    # Format training data for prophet model using ds and y
    tz_train_prophet = tz_train.reset_index()

    # Create a Prophet model
    model = Prophet()

    # Fit the model
    model.fit(tz_train_prophet)

    # Format test data for prophet model using ds and y
    tz_test_prophet = tz_test.reset_index()

    # Predict using the timeframe data from the test split above so that we can evaluate the model against actual data
    tz_test_fcst = model.predict(tz_test_prophet)

    # Add Holidays - US Federal holidays
    cal = calendar()

    holidays = cal.holidays(
        start=prophet_df.index.min(), end=prophet_df.index.max(), return_name=True
    )
    holiday_df = pd.DataFrame(data=holidays, columns=["holiday"])
    holiday_df = holiday_df.reset_index().rename(columns={"index": "ds"})

    # Create a Prophet model with holidays
    model_with_holidays = Prophet(holidays=holiday_df)

    # Fit the model with holidays on the train data
    model_with_holidays.fit(tz_train_prophet)

    # Predict on training set with model
    tz_test_fcst_with_holidays = model_with_holidays.predict(df=tz_test_prophet)

    # Predict into the Future
    d0 = date(
        (last_month + MonthEnd(-(train_test_split_months))).year,
        (last_month + MonthEnd(-(train_test_split_months))).month,
        (last_month + MonthEnd(-(train_test_split_months))).day,
    )
    d1 = date(
        datetime.strptime(forecast_time_horizon, "%Y-%m-%d").year,
        datetime.strptime(forecast_time_horizon, "%Y-%m-%d").month,
        datetime.strptime(forecast_time_horizon, "%Y-%m-%d").day,
    )
    delta = d1 - d0

    # using baseline model
    future = model.make_future_dataframe(
        periods=delta.days, freq=forecast_frequency, include_history=False
    )

    # Prescience
    forecast = model.predict(future)

    # using model with Holidays
    future_with_holidays = model_with_holidays.make_future_dataframe(
        periods=delta.days, freq=forecast_frequency, include_history=False
    )

    # Prescience
    forecast_with_holidays = model_with_holidays.predict(future_with_holidays)

    return (
        df,
        prophet_df,
        tz_train,
        tz_test,
        model,
        tz_test_fcst,
        holiday_df,
        model_with_holidays,
        tz_test_fcst_with_holidays,
        forecast,
        forecast_with_holidays,
    )


cat_type = CategoricalDtype(
    categories=[
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
    ordered=True,
)


def create_features(df, label=None):
    """
    Creates time series features from datetime index.
    """
    df = df.copy()
    df["date"] = df.index
    df["hour"] = df["date"].dt.hour
    df["dayofweek"] = df["date"].dt.dayofweek
    df["weekday"] = df["date"].dt.day_name()
    df["weekday"] = df["weekday"].astype(cat_type)
    df["quarter"] = df["date"].dt.quarter
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["dayofyear"] = df["date"].dt.dayofyear
    df["dayofmonth"] = df["date"].dt.day
    # df['weekofyear'] = df['date'].dt.weekofyear
    df["date_offset"] = (df.date.dt.month * 100 + df.date.dt.day - 320) % 1300

    df["season"] = pd.cut(
        df["date_offset"],
        [0, 300, 602, 900, 1300],
        labels=["Spring", "Summer", "Fall", "Winter"],
    )
    X = df[
        [
            "hour",
            "dayofweek",
            "quarter",
            "month",
            "year",
            "dayofyear",
            "dayofmonth",
            # "weekofyear",
            "weekday",
            "season",
        ]
    ]
    if label:
        y = df[label]
        return X, y
    return X


def plot_train_test_split(train, test, chart_title="Trainerize MRR Train / Test Split"):

    # Plot train and test so you can see where we have split
    test.rename(columns={"y": "Test Set"}).join(
        train.rename(columns={"y": "Training Set"}), how="outer"
    ).plot(style=".", ms=5)

    # chart formatting
    plt.title(
        chart_title,
        fontsize=18,
        fontweight="bold",
        y=1.02,
    )

    plt.xticks(rotation=45)
    plt.xlabel("Month", fontsize=14, fontweight="bold")

    plt.ylim(
        0,
    )
    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: "${:,.0f}K".format(x / 1000))
    )  # Format as integer with comma separator
    plt.ylabel("MRR", fontsize=14, fontweight="bold")

    plt.tight_layout()  # Adjusts the plot to ensure everything fits without overlap
    plt.show()


def plot_prophet_forecast(model, test_forecast, chart_title="Prophet Forecast"):

    fig, ax = plt.subplots()
    fig = model.plot(test_forecast, ax=ax)

    # chart formatting
    plt.title(
        chart_title,
        fontsize=18,
        fontweight="bold",
        y=1.02,
    )

    plt.xticks(rotation=45)
    plt.xlabel("Month", fontsize=14, fontweight="bold")

    plt.ylim(
        0,
    )
    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: "${:,.0f}K".format(x / 1000))
    )  # Format as integer with comma separator
    plt.ylabel("MRR", fontsize=14, fontweight="bold")

    plt.tight_layout()  # Adjusts the plot to ensure everything fits without overlap
    plt.show()


def plot_prophet_forecast_vs_actuals(
    model, tz_test, tz_test_fcst, chart_title="Prophet Forecast vs. Actuals"
):

    # Plot the forecast with the actuals
    f, ax = plt.subplots()
    ax.scatter(tz_test.index, tz_test["y"], color="r")  # actuals
    fig = model.plot(tz_test_fcst, ax=ax)  # forecast

    # chart formatting
    plt.title(
        chart_title,
        fontsize=18,
        fontweight="bold",
        y=1.02,
    )

    plt.xticks(rotation=45)
    plt.xlabel("Month", fontsize=14, fontweight="bold")

    plt.ylim(
        0,
    )
    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: "${:,.0f}K".format(x / 1000))
    )  # Format as integer with comma separator
    plt.ylabel("MRR", fontsize=14, fontweight="bold")

    plt.tight_layout()  # Adjusts the plot to ensure everything fits without overlap
    plt.show()


def plot_prophet_forecast_vs_actuals_for_forecast_timeframe(
    model,
    tz_test,
    tz_test_fcst,
    chart_title="Trailing 6 Months Forecast vs Actuals",
    lower_bound=datetime(2023, 9, 1),
    upper_bound=datetime(2024, 2, 29),
):

    fig, ax = plt.subplots()
    ax.scatter(tz_test.index, tz_test["y"], color="r")
    fig = model.plot(tz_test_fcst, ax=ax)

    plot = plt.title(
        chart_title,
        fontsize=18,
        fontweight="bold",
        y=1.02,
    )

    ax.set_xbound(lower=lower_bound, upper=upper_bound)
    plt.xticks(rotation=45)
    plt.xlabel("Month", fontsize=14, fontweight="bold")

    # plt.ylim(
    #     0,
    # )
    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: "${:,.0f}K".format(x / 1000))
    )  # Format as integer with comma separator
    plt.ylabel("MRR", fontsize=14, fontweight="bold")

    plt.tight_layout()  # Adjusts the plot to ensure everything fits without overlap
    plt.show()
