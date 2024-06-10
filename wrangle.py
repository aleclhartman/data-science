# import necessary libraries
import time

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd

from datetime import date


def melt_recurly_mrr_data(df):
    """
    This function reads in the Recurly 'Master Trainerize Revenue Model (Interval Unit Month) (Recurly Team copy) v2' data captured in the 'df' variable,
    and returns a melted dataframe
    """
    # melting mrr data
    df = pd.melt(
        df,
        id_vars=[
            "account_code",
            "company_name",
            "product_code",
            "charge_description",
            "currency",
        ],
        var_name="month",
        value_name="mrr",
    )

    return df


def replace_null_values(df, dictionary={}):
    """
    This function iterates through the dictionary and fills null values found in the keys of the dictionary with the values of the dictionary
    """

    for i in range(len(dictionary)):
        df[list(dictionary.keys())[i]].fillna(
            list(dictionary.values())[i], inplace=True
        )

    return df


def misc_data_prep(df):
    """
    Docstring
    """

    # converting date features from an object to a datetime64[ns]
    df["month"] = pd.DatetimeIndex(df.month + "-01")

    # reordering features
    df = df[
        [
            "month",
            "account_code",
            "company_name",
            "product_code",
            "charge_description",
            "currency",
            "mrr",
        ]
    ]

    df = df[
        ~(  # filter out all credits and refunds from the data using `~`
            df.charge_description.str.contains("Credit")
            | df.charge_description.str.contains("Refund")
        )
    ]

    # resetting index because we dropped rows in using the mask above
    df.reset_index(inplace=True, drop=True)

    # are there any duplicated rows?
    # asset test will not return any output if code passes test
    assert df[df.duplicated() == True].shape[0] == 0

    return df


def arr_df_misc_data_prep(arr_df):
    """
    Docstring
    """

    # misc_data_prep
    # converting date features from an object to a datetime64[ns]
    arr_df["month"] = pd.DatetimeIndex(arr_df.month + "-01")

    # reordering features
    arr_df = arr_df[
        [
            "month",
            "account_code",
            "company_name",
            "product_code",
            "charge_description",
            "currency",
            "plan_interval",
            "arr",
        ]
    ]

    arr_df = arr_df[
        ~(  # filter out all credits and refunds from the data using `~`
            arr_df.charge_description.str.contains("Credit")
            | arr_df.charge_description.str.contains("Refund")
        )
    ]

    # resetting index because we dropped rows in using the mask above
    arr_df.reset_index(inplace=True, drop=True)

    arr_df["arr"] = arr_df.arr.fillna(0)

    # are there any duplicated rows?
    # assert test will not return any output if code passes test
    assert arr_df[arr_df.duplicated() == True].shape[0] == 0

    return arr_df


def convert_arr_to_mrr(arr_df):
    """
    This function converts Recurly ARR to MRR
    """

    # plan interval field represents how frequently the customer will be charged for their subscription, for annual plans this is once every year; a plan interval value of 1
    arr_df["plan_interval_times_twelve"] = arr_df.plan_interval * 12

    # calc mrr
    arr_df["mrr"] = arr_df.arr / arr_df.plan_interval_times_twelve

    arr_df = arr_df.sort_values(
        by=[
            "account_code",
            "company_name",
            "product_code",
            "charge_description",
            "currency",
            "month",
        ]
    ).reset_index(drop=True)

    # this loop creates 12 mrr features each shifted one month forward chronologically to be later summed into one mrr feature
    for i in range(arr_df.shape[0]):  # for the whole dataframe

        if arr_df.iloc[i].mrr > 0:  # where mrr is greater than zero

            for n in range(arr_df.iloc[i].plan_interval_times_twelve.astype(int)):

                arr_df[f"mrr_m{n+1}"] = arr_df.groupby(
                    [
                        "account_code",
                        "company_name",
                        "product_code",
                        "charge_description",
                        "currency",
                    ]
                ).mrr.shift(n)
                arr_df[f"mrr_m{n+1}"] = arr_df[f"mrr_m{n+1}"].fillna(0)

    # overwriting the mrr feature with the tweleve features created in the loop above
    arr_df["mrr"] = (
        arr_df.mrr_m1
        + arr_df.mrr_m2
        + arr_df.mrr_m3
        + arr_df.mrr_m4
        + arr_df.mrr_m5
        + arr_df.mrr_m6
        + arr_df.mrr_m7
        + arr_df.mrr_m8
        + arr_df.mrr_m9
        + arr_df.mrr_m10
        + arr_df.mrr_m11
        + arr_df.mrr_m12
    )

    # reorganize features
    arr_df = arr_df[
        [
            "month",
            "account_code",
            "company_name",
            "product_code",
            "charge_description",
            "currency",
            "mrr",
        ]
    ]

    # are there any duplicated rows?
    # assert test will not return any output if code passes test
    assert arr_df[arr_df.duplicated() == True].shape[0] == 0

    return arr_df


def cba_df_misc_data_prep(cba_df):
    """
    Docstring
    """

    # converting date features from an object to a datetime64[ns]
    cba_df["month"] = pd.DatetimeIndex(cba_df.month + "-01")

    # reordering features
    cba_df = cba_df[
        [
            "month",
            "account_code",
            "company_name",
            "product_code",
            "charge_description",
            "currency",
            "mrr",
        ]
    ]

    # I don't think there's any need for data where CBA `mrr` is null
    # below I am making sure that filtering out all null values in the `mrr` field will still calculate to the unfiltered df's mrr value
    assert cba_df[~cba_df.mrr.isna()].mrr.sum() == cba_df.mrr.sum()

    # filtering df for only values where mrr is not null
    cba_df = cba_df[~cba_df.mrr.isna()]

    # resetting index because we dropped rows in using the mask above
    cba_df.reset_index(inplace=True, drop=True)

    # get the last date of the month for the current month
    cba_df["month"] = cba_df.month + MonthEnd(0)

    # sorting the df
    cba_df = cba_df.sort_values(
        by=[
            "account_code",
            "company_name",
            "product_code",
            "charge_description",
            "currency",
            "month",
        ]
    ).reset_index()

    # drop old index
    cba_df.drop(columns=["index"], inplace=True)

    return cba_df
