# import necessary libraries
import time

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd

from datetime import date


def join_plan_codes_and_names_to_mrr_df(df, plan_codes_and_names):
    """
    Docstring
    """

    # joing the main dataframe together with the master plan codes and names
    df = df.merge(
        plan_codes_and_names,
        how="left",
        # main df (left) has a different variable name than the plan code df (right)
        left_on=[
            "product_code",
        ],
        right_on=[
            "plan_code",
        ],
    )

    # reordering features
    df = df[
        [
            "month",
            "account_code",
            "company_name",
            "product_code",
            "plan_name",
            "plan_group",
            "charge_description",
            "segment",
            "plan_type",
            "currency",
            "mrr",
        ]
    ]

    # get the last date of the month for the current month
    df["month"] = df.month + MonthEnd(0)

    # sorting the df
    df = df.sort_values(
        by=[
            "account_code",
            "company_name",
            "product_code",
            "plan_name",
            "plan_group",
            "charge_description",
            "segment",
            "plan_type",
            "currency",
            "month",
        ]
    ).reset_index()

    # drop old index
    df.drop(columns=["index"], inplace=True)

    return df


def create_mrr_features(df):
    """
    This function creates the MRR features for Trainerize revenue where plan-to-plan movement is captured as New Business/Churn
    """

    # this groupby is necessary to aggregate rows to a specific month if there are multiple rows that are the same combination of "month", "account_code", "company_name", "product_code",
    # "plan_name", "plan_group", "charge_description", "segment", "plan_type", "currency",
    df = pd.DataFrame(
        df.groupby(
            [
                "month",
                "account_code",
                "company_name",
                "product_code",
                "plan_name",
                "plan_group",
                "charge_description",
                "segment",
                "plan_type",
                "currency",
            ]
        ).mrr.sum()
    ).reset_index()

    # for each customer shift the mrr values forward one
    # groupby all but the month feature in the base schedule df
    df["prior_period_mrr"] = df.groupby(
        [
            "account_code",
            "company_name",
            "product_code",
            "plan_name",
            "plan_group",
            "charge_description",
            "segment",
            "plan_type",
            "currency",
        ]
    ).mrr.shift()

    # filling null values with 0
    df.mrr.fillna(0, inplace=True)
    df.prior_period_mrr.fillna(0, inplace=True)

    # churn
    df["churn"] = np.where(
        # where current period mrr is 0
        (
            df.groupby(
                [
                    "month",
                    "account_code",
                    "company_name",
                    "product_code",
                    "plan_name",
                    "plan_group",
                    "charge_description",
                    "segment",
                    "plan_type",
                    "currency",
                ]
            ).mrr.transform(sum)
            == 0
        )
        &
        # and prior period mrr is greater than 0
        (
            df.groupby(
                [
                    "month",
                    "account_code",
                    "company_name",
                    "product_code",
                    "plan_name",
                    "plan_group",
                    "charge_description",
                    "segment",
                    "plan_type",
                    "currency",
                ]
            ).prior_period_mrr.transform(sum)
            > 0
        ),
        # give us the difference between the current and prior_period_mrr
        (df.mrr - df.prior_period_mrr),
        0,
    )

    # expansion
    df["expansion"] = np.where(
        # current period mrr is greater than prior period mrr
        (
            df.groupby(
                [
                    "month",
                    "account_code",
                    "company_name",
                    "product_code",
                    "plan_name",
                    "plan_group",
                    "charge_description",
                    "segment",
                    "plan_type",
                    "currency",
                ]
            ).mrr.transform(sum)
            > df.groupby(
                [
                    "month",
                    "account_code",
                    "company_name",
                    "product_code",
                    "plan_name",
                    "plan_group",
                    "charge_description",
                    "segment",
                    "plan_type",
                    "currency",
                ]
            ).prior_period_mrr.transform(sum)
        )
        &
        # where current period mrr greater than 0
        (
            df.groupby(
                [
                    "month",
                    "account_code",
                    "company_name",
                    "product_code",
                    "plan_name",
                    "plan_group",
                    "charge_description",
                    "segment",
                    "plan_type",
                    "currency",
                ]
            ).mrr.transform(sum)
            > 0
        )
        &
        # and prior period mrr is greater than 0
        (
            df.groupby(
                [
                    "month",
                    "account_code",
                    "company_name",
                    "product_code",
                    "plan_name",
                    "plan_group",
                    "charge_description",
                    "segment",
                    "plan_type",
                    "currency",
                ]
            ).prior_period_mrr.transform(sum)
            > 0
        ),
        # give us the difference between the current and prior_period_mrr
        (df.mrr - df.prior_period_mrr),
        0,
    )

    # contraction
    df["contraction"] = np.where(
        # current period mrr is less than prior period mrr
        (
            df.groupby(
                [
                    "month",
                    "account_code",
                    "company_name",
                    "product_code",
                    "plan_name",
                    "plan_group",
                    "charge_description",
                    "segment",
                    "plan_type",
                    "currency",
                ]
            ).mrr.transform(sum)
            < df.groupby(
                [
                    "month",
                    "account_code",
                    "company_name",
                    "product_code",
                    "plan_name",
                    "plan_group",
                    "charge_description",
                    "segment",
                    "plan_type",
                    "currency",
                ]
            ).prior_period_mrr.transform(sum)
        )
        &
        # where current period mrr greater than 0
        (
            df.groupby(
                [
                    "month",
                    "account_code",
                    "company_name",
                    "product_code",
                    "plan_name",
                    "plan_group",
                    "charge_description",
                    "segment",
                    "plan_type",
                    "currency",
                ]
            ).mrr.transform(sum)
            > 0
        )
        &
        # and prior period mrr is greater than 0
        (
            df.groupby(
                [
                    "month",
                    "account_code",
                    "company_name",
                    "product_code",
                    "plan_name",
                    "plan_group",
                    "charge_description",
                    "segment",
                    "plan_type",
                    "currency",
                ]
            ).prior_period_mrr.transform(sum)
            > 0
        ),
        # give us the difference between the current and prior_period_mrr
        (df.mrr - df.prior_period_mrr),
        0,
    )

    # cumulative_sum for all dimensions other than month
    df["cumulative_sum"] = df.groupby(
        [
            "account_code",
            "company_name",
            "product_code",
            "plan_name",
            "plan_group",
            "charge_description",
            "segment",
            "plan_type",
            "currency",
        ]
    ).prior_period_mrr.cumsum()

    # reactivation

    # if the current value for mrr is not 0
    # and if any of the previous values for a specific customer's mrr were not 0
    # and if the prior_period_mrr value was 0
    df["reactivation"] = np.where(
        # where current period mrr greater than 0
        (
            df.groupby(
                [
                    "month",
                    "account_code",
                    "company_name",
                    "product_code",
                    "plan_name",
                    "plan_group",
                    "charge_description",
                    "segment",
                    "plan_type",
                    "currency",
                ]
            ).mrr.transform(sum)
            > 0
        )
        &
        # and prior period mrr is 0
        (
            df.groupby(
                [
                    "month",
                    "account_code",
                    "company_name",
                    "product_code",
                    "plan_name",
                    "plan_group",
                    "charge_description",
                    "segment",
                    "plan_type",
                    "currency",
                ]
            ).prior_period_mrr.transform(sum)
            == 0
        )
        &
        # and this customer has churned at some point in the past
        (
            df.groupby(
                [
                    "month",
                    "account_code",
                    "company_name",
                    "product_code",
                    "plan_name",
                    "plan_group",
                    "charge_description",
                    "segment",
                    "plan_type",
                    "currency",
                ]
            ).cumulative_sum.transform(sum)
            > 0
        ),
        # give us the difference between the current and prior_period_mrr
        (df.mrr - df.prior_period_mrr),
        0,
    )

    # new_business
    df["new_business"] = np.where(
        # where current period mrr greater than 0
        (
            df.groupby(
                [
                    "month",
                    "account_code",
                    "company_name",
                    "product_code",
                    "plan_name",
                    "plan_group",
                    "charge_description",
                    "segment",
                    "plan_type",
                    "currency",
                ]
            ).mrr.transform(sum)
            > 0
        )
        &
        # and prior period mrr is 0
        (
            df.groupby(
                [
                    "month",
                    "account_code",
                    "company_name",
                    "product_code",
                    "plan_name",
                    "plan_group",
                    "charge_description",
                    "segment",
                    "plan_type",
                    "currency",
                ]
            ).prior_period_mrr.transform(sum)
            == 0
        )
        &
        # and this customer has never churned
        (
            df.groupby(
                [
                    "month",
                    "account_code",
                    "company_name",
                    "product_code",
                    "plan_name",
                    "plan_group",
                    "charge_description",
                    "segment",
                    "plan_type",
                    "currency",
                ]
            ).cumulative_sum.transform(sum)
            == 0
        ),
        # give us the difference between the current and prior_period_mrr
        (df.mrr - df.prior_period_mrr),
        0,
    )

    # reorder features
    df = df[
        [
            "month",
            "account_code",
            "company_name",
            "product_code",
            "plan_name",
            "plan_group",
            "charge_description",
            "segment",
            "plan_type",
            "currency",
            "prior_period_mrr",
            "new_business",
            "churn",
            "reactivation",
            "expansion",
            "contraction",
            "mrr",
            "cumulative_sum",
        ]
    ]

    df["check_mrr_movement"] = np.isclose(
        (
            df.prior_period_mrr
            + df.new_business
            + df.expansion
            + df.contraction
            + df.churn
            + df.reactivation
        ),
        df.mrr,
    )

    assert df[df.check_mrr_movement == False].shape[0] == 0

    # cumulative_sum for a specific account
    # making this feature for later Cohort
    df["cumulative_sum_for_account"] = df.groupby(
        [
            "account_code",
        ]
    ).mrr.cumsum()

    # moving the cumsum by account forward in time one month
    df["cumulative_sum_for_account"] = df.groupby(
        "account_code"
    ).cumulative_sum_for_account.shift()

    return df


def create_mrr_features_for_plan_to_plan_movement(df):
    """
    This function creates the MRR features for Trainerize revenue where plan-to-plan movement is captured as Expansion/Contraction
    """

    # this groupby is necessary to aggregate rows to a specific month if there are multiple rows that are the same combination of "month", "account_code", "company_name", "product_code",
    # "plan_name", "plan_group", "charge_description", "segment", "plan_type", "currency",
    df = pd.DataFrame(
        df.groupby(
            [
                "month",
                "account_code",
                "company_name",
                "product_code",
                "plan_name",
                "plan_group",
                "charge_description",
                "segment",
                "plan_type",
                "currency",
            ]
        ).mrr.sum()
    ).reset_index()

    # for each customer shift the mrr values forward one
    # groupby all but the month feature in the base schedule df
    df["prior_period_mrr"] = df.groupby(
        [
            "account_code",
            "company_name",
            "product_code",
            "plan_name",
            "plan_group",
            "charge_description",
            "segment",
            "plan_type",
            "currency",
        ]
    ).mrr.shift()

    # filling null values with 0
    df.mrr.fillna(0, inplace=True)
    df.prior_period_mrr.fillna(0, inplace=True)

    # churn
    df["churn"] = np.where(
        # where current period mrr is 0
        (
            df.groupby(
                [
                    "month",
                    "account_code",
                ]
            ).mrr.transform(sum)
            == 0
        )
        &
        # and prior period mrr is greater than 0
        (
            df.groupby(
                [
                    "month",
                    "account_code",
                ]
            ).prior_period_mrr.transform(sum)
            > 0
        ),
        # give us the difference between the current and prior_period_mrr
        (df.mrr - df.prior_period_mrr),
        0,
    )

    # cumulative_sum for all dimensions other than month
    df["cumulative_sum"] = df.groupby(
        [
            "account_code",
        ]
    ).prior_period_mrr.cumsum()

    # reactivation

    # if the current value for mrr is not 0
    # and if any of the previous values for a specific customer's mrr were not 0
    # and if the prior_period_mrr value was 0
    df["reactivation"] = np.where(
        # where current period mrr greater than 0
        (
            df.groupby(
                [
                    "month",
                    "account_code",
                ]
            ).mrr.transform(sum)
            > 0
        )
        &
        # and prior period mrr is 0
        (
            df.groupby(
                [
                    "month",
                    "account_code",
                ]
            ).prior_period_mrr.transform(sum)
            == 0
        )
        &
        # and this customer has churned at some point in the past
        (
            df.groupby(
                [
                    "month",
                    "account_code",
                ]
            ).cumulative_sum.transform(sum)
            > 0
        ),
        # give us the difference between the current and prior_period_mrr
        (df.mrr - df.prior_period_mrr),
        0,
    )

    # new_business
    df["new_business"] = np.where(
        # where current period mrr greater than 0
        (
            df.groupby(
                [
                    "month",
                    "account_code",
                ]
            ).mrr.transform(sum)
            > 0
        )
        &
        # and prior period mrr is 0
        (
            df.groupby(
                [
                    "month",
                    "account_code",
                ]
            ).prior_period_mrr.transform(sum)
            == 0
        )
        &
        # and this customer has never churned
        (
            df.groupby(
                [
                    "month",
                    "account_code",
                ]
            ).cumulative_sum.transform(sum)
            == 0
        ),
        # give us the difference between the current and prior_period_mrr
        (df.mrr - df.prior_period_mrr),
        0,
    )

    # expansion
    df["expansion"] = np.where(
        # current period mrr is greater than prior period mrr
        (
            df.groupby(
                [
                    "month",
                    "account_code",
                    "company_name",
                    "product_code",
                    "plan_name",
                    "plan_group",
                    "charge_description",
                    "segment",
                    "plan_type",
                    "currency",
                ]
            ).mrr.transform(sum)
            > df.groupby(
                [
                    "month",
                    "account_code",
                    "company_name",
                    "product_code",
                    "plan_name",
                    "plan_group",
                    "charge_description",
                    "segment",
                    "plan_type",
                    "currency",
                ]
            ).prior_period_mrr.transform(sum)
        )
        &
        # where current period mrr greater than 0
        (
            df.groupby(
                [
                    "month",
                    "account_code",
                ]
            ).mrr.transform(sum)
            > 0
        )
        &
        # and prior period mrr is greater than 0
        (
            df.groupby(
                [
                    "month",
                    "account_code",
                ]
            ).prior_period_mrr.transform(sum)
            > 0
        )
        & (
            df.groupby(
                [
                    "month",
                    "account_code",
                ]
            ).new_business.transform(sum)
            == 0
        ),
        # give us the difference between the current and prior_period_mrr
        (df.mrr - df.prior_period_mrr),
        0,
    )

    # contraction
    df["contraction"] = np.where(
        # current period mrr is less than prior period mrr
        (
            df.groupby(
                [
                    "month",
                    "account_code",
                    "company_name",
                    "product_code",
                    "plan_name",
                    "plan_group",
                    "charge_description",
                    "segment",
                    "plan_type",
                    "currency",
                ]
            ).mrr.transform(sum)
            < df.groupby(
                [
                    "month",
                    "account_code",
                    "company_name",
                    "product_code",
                    "plan_name",
                    "plan_group",
                    "charge_description",
                    "segment",
                    "plan_type",
                    "currency",
                ]
            ).prior_period_mrr.transform(sum)
        )
        &
        # where current period mrr greater than 0
        (
            df.groupby(
                [
                    "month",
                    "account_code",
                ]
            ).mrr.transform(sum)
            > 0
        )
        &
        # and prior period mrr is greater than 0
        (
            df.groupby(
                [
                    "month",
                    "account_code",
                ]
            ).prior_period_mrr.transform(sum)
            > 0
        )
        & (
            df.groupby(
                [
                    "month",
                    "account_code",
                ]
            ).churn.transform(sum)
            == 0
        ),
        # give us the difference between the current and prior_period_mrr
        (df.mrr - df.prior_period_mrr),
        0,
    )

    # reorder features
    df = df[
        [
            "month",
            "account_code",
            "company_name",
            "product_code",
            "plan_name",
            "plan_group",
            "charge_description",
            "segment",
            "plan_type",
            "currency",
            "prior_period_mrr",
            "new_business",
            "churn",
            "reactivation",
            "expansion",
            "contraction",
            "mrr",
            "cumulative_sum",
        ]
    ]

    df["check_mrr_movement"] = np.isclose(
        (
            df.prior_period_mrr
            + df.new_business
            + df.expansion
            + df.contraction
            + df.churn
            + df.reactivation
        ),
        df.mrr,
    )

    assert df[df.check_mrr_movement == False].shape[0] == 0

    return df


def create_add_on_flags(df):
    """
    This function creates our boolean flag features for add-ons
    """

    # flag for the discounted $5 nutrition plan
    df["add_on_1"] = np.where(
        (
            (df.plan_name == "Make Believe Add-on Product 1")
            & (df.mrr > 0)
        ),
        1,
        0,
    )

    # flag for the $45 nutrition plan
    df["add_on_1_discounted"] = np.where(
        ((df.plan_name == "Make Believe Add-on Product 1 (Discounted)") & (df.mrr > 0)), 1, 0
    )

    # flag for the tz_pay
    df["add_on_2"] = np.where(
        ((df.segment == "Make Believe Add-on Product 2") & (df.mrr > 0)), 1, 0
    )

    # flag for the video
    df["add_on_3"] = np.where(
        ((df.segment == "Make Believe Add-on Product 3") & (df.mrr > 0)), 1, 0
    )

    return df


def identify_base_plan(df):
    """
    This function identifies the current base plan by customer by month
    """

    # Base Plan by Customer by Month

    # https://chat.openai.com/c/2266e157-4147-45b8-8ebd-1e3818d8fe9e
    # step 4 from ChatGPT (including sorting by mrr as well)
    df = df.sort_values(
        by=["account_code", "mrr", "plan_type", "plan_group", "month"],
        ascending=[1, 0, 0, 0, 0],
    )

    # step 5 from ChatGPT
    # identify the base plan by customer by month
    # df['base_product'] = df.groupby(['customer_id', df['date_column'].dt.to_period("M")])['product'].transform('first')
    df["base_plan_by_month"] = df.groupby(["account_code", "month"])[
        "plan_group"
    ].transform("first")

    # reordering features
    df = df[
        [
            "month",
            "account_code",
            "company_name",
            "product_code",
            "plan_name",
            "plan_group",
            "charge_description",
            "segment",
            "plan_type",
            "base_plan_by_month",
            "currency",
            "mrr",
            "prior_period_mrr",
            "new_business",
            "churn",
            "reactivation",
            "expansion",
            "contraction",
            "cumulative_sum",
            "cumulative_sum_for_account",
            "check_mrr_movement",
            # "active_customer",
            # "customer_lifetime",
            "nutrition_discounted",
            "nutrition",
            "tz_pay",
            "video",
        ]
    ]

    # Base Plan by Customer
    # step 4 from above (including sorting by mrr as well)
    df = df.sort_values(
        by=[
            "account_code",
            "plan_type",
            "mrr",
        ],
        ascending=[1, 0, 0],
    )

    # step 6 from above
    # identify the base plan by customer (excluding month)
    # df["base_plan"] = df.groupby(["account_code"])["plan_group"].transform('first')
    df["base_plan_by_customer"] = df.groupby(["account_code"])["plan_group"].transform(
        "first"
    )

    # reordering features
    df = df[
        [
            "month",
            "account_code",
            "company_name",
            "product_code",
            "plan_name",
            "plan_group",
            "charge_description",
            "segment",
            "plan_type",
            "base_plan_by_month",
            "base_plan_by_customer",
            "currency",
            "mrr",
            "prior_period_mrr",
            "new_business",
            "churn",
            "reactivation",
            "expansion",
            "contraction",
            "cumulative_sum",
            "cumulative_sum_for_account",
            "check_mrr_movement",
            # "active_customer",
            # "customer_lifetime",
            "nutrition_discounted",
            "nutrition",
            "tz_pay",
            "video",
        ]
    ]

    # Replace Base Plan by Month with Base Plan by Customer where Base Plan by Month is an Add-on

    # Exclude add-on products
    mask = (df.base_plan_by_month != df.base_plan_by_customer) & (
        df.base_plan_by_month.isin(
            [
                "Stripe integrated payments",
                "Advanced Nutrition Coaching Add-on",
                "Video Coaching Add-on",
                "Other",
            ]
        )
    )
    # and replace with base plan value
    df.loc[mask, "base_plan_by_month"] = df.loc[mask, "base_plan_by_customer"]

    # checking all my shit
    assert (
        df[
            np.isclose(
                (
                    df.prior_period_mrr
                    + df.new_business
                    + df.expansion
                    + df.contraction
                    + df.churn
                    + df.reactivation
                ),
                df.mrr,
            )
            == False
        ].shape[0]
        == 0
    )

    # sorting the df
    df = df.sort_values(
        by=[
            "account_code",
            "company_name",
            "product_code",
            "plan_name",
            "plan_group",
            "charge_description",
            "segment",
            "plan_type",
            "currency",
            "month",
        ]
    ).reset_index()

    # drop old index
    df.drop(columns=["index", "base_plan_by_customer"], inplace=True)
    df.rename(columns={"base_plan_by_month": "base_plan"}, inplace=True)

    return df


def create_mrr_summary_df(df):
    """
    This function summarizes MRR movement by month
    """

    # creating an aggregated mrr_summary_df
    mrr_summary_df = (
        df.groupby("month")[
            [
                "prior_period_mrr",
                "new_business",
                "churn",
                "reactivation",
                "expansion",
                "contraction",
                "mrr",
            ]
        ]
        .sum()
        .reset_index()
    )

    # shift mrr forward one month to represent opening mrr balance
    mrr_summary_df["opening"] = mrr_summary_df.prior_period_mrr
    # mrr_summary_df.opening.fillna(0, inplace=True)

    # rename the mrr feature as ending balance
    mrr_summary_df["ending"] = mrr_summary_df.mrr

    # reorder features
    mrr_summary_df = mrr_summary_df[
        [
            "month",
            "opening",
            "new_business",
            "expansion",
            "contraction",
            "churn",
            "reactivation",
            "ending",
        ]
    ]

    # creating a check in my df
    mrr_summary_df["check"] = np.isclose(
        (
            mrr_summary_df.opening
            + mrr_summary_df.new_business
            + mrr_summary_df.expansion
            + mrr_summary_df.contraction
            + mrr_summary_df.churn
            + mrr_summary_df.reactivation
        ),
        mrr_summary_df.ending,
    )

    # testing my logic
    # assert mrr_summary_df.check.unique()[0] == True
    # there should be zero rows where the check is False
    assert mrr_summary_df[mrr_summary_df.check == False].shape[0] == 0

    return mrr_summary_df


def create_logo_summary_df(df):
    """
    This function summarizes Logo movement by month
    """

    logo_summary_df = pd.DataFrame(df.month.unique(), columns=["month"])

    prior_period_mrr_logo_summary_df = pd.DataFrame(
        df[df.prior_period_mrr != 0].groupby("month").account_code.nunique()
    ).reset_index()
    prior_period_mrr_logo_summary_df = prior_period_mrr_logo_summary_df.rename(
        columns={"account_code": "prior_period_logos"}
    )
    new_business_logo_summary_df = pd.DataFrame(
        df[df.new_business != 0].groupby("month").account_code.nunique()
    ).reset_index()
    new_business_logo_summary_df = new_business_logo_summary_df.rename(
        columns={"account_code": "new_business"}
    )
    churn_logo_summary_df = pd.DataFrame(
        df[df.churn != 0].groupby("month").account_code.nunique()
    ).reset_index()
    churn_logo_summary_df = churn_logo_summary_df.rename(
        columns={"account_code": "churn"}
    )
    reactivation_logo_summary_df = pd.DataFrame(
        df[df.reactivation != 0].groupby("month").account_code.nunique()
    ).reset_index()
    reactivation_logo_summary_df = reactivation_logo_summary_df.rename(
        columns={"account_code": "reactivation"}
    )
    expansion_logo_summary_df = pd.DataFrame(
        df[df.expansion != 0].groupby("month").account_code.nunique()
    ).reset_index()
    expansion_logo_summary_df = expansion_logo_summary_df.rename(
        columns={"account_code": "expansion"}
    )
    contraction_logo_summary_df = pd.DataFrame(
        df[df.contraction != 0].groupby("month").account_code.nunique()
    ).reset_index()
    contraction_logo_summary_df = contraction_logo_summary_df.rename(
        columns={"account_code": "contraction"}
    )
    mrr_logo_summary_df = pd.DataFrame(
        df[df.mrr != 0].groupby("month").account_code.nunique()
    ).reset_index()
    mrr_logo_summary_df = mrr_logo_summary_df.rename(columns={"account_code": "logos"})

    logo_summary_df = logo_summary_df.merge(
        prior_period_mrr_logo_summary_df, how="left", on="month"
    )
    logo_summary_df = logo_summary_df.merge(
        new_business_logo_summary_df, how="left", on="month"
    )
    logo_summary_df = logo_summary_df.merge(
        churn_logo_summary_df, how="left", on="month"
    )
    logo_summary_df = logo_summary_df.merge(
        reactivation_logo_summary_df, how="left", on="month"
    )
    logo_summary_df = logo_summary_df.merge(
        expansion_logo_summary_df, how="left", on="month"
    )
    logo_summary_df = logo_summary_df.merge(
        contraction_logo_summary_df, how="left", on="month"
    )
    logo_summary_df = logo_summary_df.merge(mrr_logo_summary_df, how="left", on="month")
    logo_summary_df.fillna(0, inplace=True)

    # shift mrr forward one month to represent opening mrr balance
    logo_summary_df["opening"] = logo_summary_df.prior_period_logos
    # logo_summary_df.opening.fillna(0, inplace=True)

    # rename the mrr feature as ending balance
    logo_summary_df["ending"] = logo_summary_df.logos

    # # rename the mrr feature as ending balance
    # logo_summary_df["churn"] = logo_summary_df.churn * -1

    # # rename the mrr feature as ending balance
    # logo_summary_df["contraction"] = logo_summary_df.contraction * -1

    # reorder features
    logo_summary_df = logo_summary_df[
        [
            "month",
            "opening",
            "new_business",
            "expansion",
            "contraction",
            "churn",
            "reactivation",
            "ending",
        ]
    ]

    # creating a check in my df
    logo_summary_df["check"] = np.isclose(
        (
            logo_summary_df.opening
            + logo_summary_df.new_business
            - logo_summary_df.churn
            + logo_summary_df.reactivation
        ),
        logo_summary_df.ending,
    )

    # testing my logic
    # assert logo_summary_df.check.unique()[0] == True
    # there should be zero rows where the check is False
    # assert logo_summary_df[logo_summary_df.check == False].shape[0] == 0

    return logo_summary_df


def generate_add_on_penetration_metrics(df):
    """
    Docstring
    """

    # creating a count of unique active customers by month summary df
    count_of_unique_customers_by_month = (
        df[df.mrr > 0].groupby("month")[["account_code"]].nunique().reset_index()
    )

    # creating a count of add-on customers by month df
    count_of_add_on_customers = (
        df.groupby("month")[
            [
                "nutrition_discounted",
                "nutrition",
                "tz_pay",
                "video",
            ]
        ]
        .sum()
        .reset_index()
    )

    # merging the two customer count dfs above together
    add_on_penetration_df = count_of_unique_customers_by_month.merge(
        count_of_add_on_customers,
        how="left",
        # join on the features of the base schedule
        on=[
            "month",
        ],
    ).sort_values(
        by=[
            "month",
        ]
    )

    # renaming columns
    add_on_penetration_df = add_on_penetration_df.rename(
        columns={
            "account_code": "count_of_unique_active_accounts",
            "nutrition_discounted": "count_of_active_nutrition_discounted_accounts",
            "nutrition": "count_of_active_nutrition_accounts",
            "tz_pay": "count_of_active_tz_pay_accounts",
            "video": "count_of_active_video_accounts",
        }
    )

    # calculating penetration rates
    add_on_penetration_df["nutrition_discounted_penetration_rate"] = (
        add_on_penetration_df.count_of_active_nutrition_discounted_accounts
        / add_on_penetration_df.count_of_unique_active_accounts
    )
    add_on_penetration_df["nutrition_penetration_rate"] = (
        add_on_penetration_df.count_of_active_nutrition_accounts
        / add_on_penetration_df.count_of_unique_active_accounts
    )
    add_on_penetration_df["tz_pay_penetration_rate"] = (
        add_on_penetration_df.count_of_active_tz_pay_accounts
        / add_on_penetration_df.count_of_unique_active_accounts
    )
    add_on_penetration_df["video_penetration_rate"] = (
        add_on_penetration_df.count_of_active_video_accounts
        / add_on_penetration_df.count_of_unique_active_accounts
    )

    return add_on_penetration_df


def generate_add_on_penetration_metrics_by_plan_group(df):
    """
    Docstring
    """

    # creating a count of unique active customers by month summary df
    count_of_unique_customers_by_month_by_base_plan = (
        df[df.mrr > 0]
        .groupby(["month", "base_plan"])[["account_code"]]
        .nunique()
        .sort_values(by=["base_plan", "month"])
        .reset_index()
    )

    # creating a count of add-on customers by month df
    count_of_add_on_customers_by_month_by_base_plan = (
        df.groupby(["month", "base_plan"])[
            [
                "nutrition_discounted",
                "nutrition",
                "tz_pay",
                "video",
            ]
        ]
        .sum()
        .sort_values(by=["base_plan", "month"])
        .reset_index()
    )

    # merging the two customer count dfs above together
    add_on_penetration_by_plan_group_df = count_of_unique_customers_by_month_by_base_plan.merge(
        count_of_add_on_customers_by_month_by_base_plan,
        how="left",
        # join on the features of the base schedule
        on=[
            "month",
            "base_plan",
        ],
    ).sort_values(
        by=[
            "base_plan",
            "month",
        ]
    )

    # renaming columns
    add_on_penetration_by_plan_group_df = add_on_penetration_by_plan_group_df.rename(
        columns={
            "account_code": "count_of_unique_active_accounts",
            "nutrition_discounted": "count_of_active_nutrition_discounted_accounts",
            "nutrition": "count_of_active_nutrition_accounts",
            "tz_pay": "count_of_active_tz_pay_accounts",
            "video": "count_of_active_video_accounts",
        }
    )

    # calculating penetration rate
    add_on_penetration_by_plan_group_df["nutrition_discounted_penetration_rate"] = (
        add_on_penetration_by_plan_group_df.count_of_active_nutrition_discounted_accounts
        / add_on_penetration_by_plan_group_df.count_of_unique_active_accounts
    )
    add_on_penetration_by_plan_group_df["nutrition_penetration_rate"] = (
        add_on_penetration_by_plan_group_df.count_of_active_nutrition_accounts
        / add_on_penetration_by_plan_group_df.count_of_unique_active_accounts
    )
    add_on_penetration_by_plan_group_df["tz_pay_penetration_rate"] = (
        add_on_penetration_by_plan_group_df.count_of_active_tz_pay_accounts
        / add_on_penetration_by_plan_group_df.count_of_unique_active_accounts
    )
    add_on_penetration_by_plan_group_df["video_penetration_rate"] = (
        add_on_penetration_by_plan_group_df.count_of_active_video_accounts
        / add_on_penetration_by_plan_group_df.count_of_unique_active_accounts
    )

    return add_on_penetration_by_plan_group_df


def generate_new_business_by_plan_group_mix(df):
    """
    Docstring
    """

    # creating a count of unique new customers by month summary df
    count_of_unique_new_customers_by_month = (
        df[
            (df.new_business != 0)
            & (
                (df.segment == "Pro")
                | (df.segment == "Studio")
                | (df.segment == "Enterprise")
            )
        ]
        .groupby(["month", "plan_group"])[["account_code"]]
        .nunique()
        .reset_index()
    )

    count_of_unique_new_customers_by_month.rename(
        columns={"account_code": "count_of_unique_new_business_accounts"}, inplace=True
    )

    return count_of_unique_new_customers_by_month


# def generate_add_on_purchase_timing_df(df):
#     """
#     This function does the following:
#         1. Creates an 'active_customer' feature
#         2. Creates a 'customer_lifetime' feature
#         3. Creates an 'add_on_purchase_timing_df' variable that contains the timing of when a customer purchased a specific add-on product in the customer's life cycle
#     """

#     # calculate if a customer is active
#     df["active_customer"] = np.where(
#         # where current period mrr is greater 0
#         (
#             df.groupby(
#                 [
#                     "month",
#                     "account_code",
#                     "plan_type",
#                 ]
#             ).mrr.transform(sum)
#             > 0
#         )
#         & (df.mrr != 0),
#         # count 1
#         1,
#         0,
#     )

#     # if a customer's base plan has > 0 dollars in MRR, then the customer life increases by 1
#     # do a cumsum for the increments
#     df["customer_lifetime"] = df.groupby(
#         [
#             "account_code",
#             "plan_type",
#         ]
#     ).active_customer.cumsum()

#     data = []

#     for i in range(df.account_code.nunique()):
#         try:
#             first_base_plan_purchase = (
#                 df[
#                     (
#                         df.account_code == df.account_code.unique()[i]
#                     )  # filter for customer
#                     & (df.plan_type == "Base Plan")  # filter for base plan
#                     & (df.active_customer == 1)  # filter for active customer
#                 ]
#                 .groupby(["plan_type"])
#                 .month.min()
#                 .dt
#             )  # date of customer first Base Plan purchase

#             add_on_purchase_dates = (
#                 df[
#                     (
#                         df.account_code == df.account_code.unique()[i]
#                     )  # filter for customer
#                     & (df.plan_type == "Add-on")  # filter for add-ons
#                     & (df.active_customer == 1)  # filter for active customer
#                 ]
#                 .groupby(["segment"])
#                 .month.min()
#                 .reset_index()
#             )  # df of add-on segments and dates of customer first add-on purchases

#             for x in range(add_on_purchase_dates.shape[0]):
#                 data.append(
#                     {
#                         "account_code": df.account_code.unique()[i],
#                         "company_name": df[
#                             (df.account_code == df.account_code.unique()[i])
#                         ].company_name.unique()[0],
#                         "segment": add_on_purchase_dates.iloc[x].segment,
#                         "month_add_on_purchased": (
#                             (
#                                 (
#                                     add_on_purchase_dates.iloc[x].month.year
#                                     - first_base_plan_purchase.year.values[0]
#                                 )
#                                 * 12
#                                 + (
#                                     add_on_purchase_dates.iloc[x].month.month
#                                     - first_base_plan_purchase.month.values[0]
#                                 )
#                             )
#                             + 1
#                         ),
#                     }
#                 )
#         except:
#             pass

#     add_on_purchase_timing_df = pd.DataFrame(data)

#     return df, add_on_purchase_timing_df


################################################################################################################
################################################################################################################
##                                                                                                            ##
##                                                                                                            ##
##                                                                                                            ##
##                                                                                                            ##
##                   Multiprocessing attempt for `generate_add_on_purchase_timing_df` below                   ##
##                                                                                                            ##
##                                                                                                            ##
##                                                                                                            ##
##                                                                                                            ##
################################################################################################################
################################################################################################################


# def calculate_customer_lifetime(df):
#     """
#     This function does the following:
#         1. Creates an 'active_customer' feature
#         2. Creates a 'customer_lifetime' feature
#     """

#     # calculate if a customer is active
#     df["active_customer"] = np.where(
#         # where current period mrr is greater 0
#         (
#             df.groupby(
#                 [
#                     "month",
#                     "account_code",
#                     "plan_type",
#                 ]
#             ).mrr.transform(sum)
#             > 0
#         )
#         & (df.mrr != 0),
#         # count 1
#         1,
#         0,
#     )

#     # if a customer's base plan has > 0 dollars in MRR, then the customer life increases by 1
#     # do a cumsum for the increments
#     df["customer_lifetime"] = df.groupby(
#         [
#             "account_code",
#             "plan_type",
#         ]
#     ).active_customer.cumsum()

#     return df


# def calculate_add_on_purchase_timing(df):
#     data = []

#     for i in range(df.account_code.nunique()):
#         try:
#             first_base_plan_purchase = (
#                 df[
#                     (df.account_code == df.account_code.unique()[i])
#                     & (df.plan_type == "Base Plan")  # filter for customer
#                     & (  # filter for base plan
#                         df.active_customer == 1
#                     )  # filter for active customer
#                 ]
#                 .groupby(["plan_type"])
#                 .month.min()
#                 .dt
#             )  # date of customer first Base Plan purchase

#             add_on_purchase_dates = (
#                 df[
#                     (df.account_code == df.account_code.unique()[i])
#                     & (df.plan_type == "Add-on")  # filter for customer
#                     & (  # filter for add-ons
#                         df.active_customer == 1
#                     )  # filter for active customer
#                 ]
#                 .groupby(["segment"])
#                 .month.min()
#                 .reset_index()
#             )  # df of add-on segments and dates of customer first add-on purchases

#             for x in range(add_on_purchase_dates.shape[0]):
#                 data.append(
#                     {
#                         "account_code": df.account_code.unique()[i],
#                         "company_name": df[
#                             (df.account_code == df.account_code.unique()[i])
#                         ].company_name.unique()[0],
#                         "segment": add_on_purchase_dates.iloc[x].segment,
#                         "month_add_on_purchased": (
#                             (
#                                 (
#                                     add_on_purchase_dates.iloc[x].month.year
#                                     - first_base_plan_purchase.year.values[0]
#                                 )
#                                 * 12
#                                 + (
#                                     add_on_purchase_dates.iloc[x].month.month
#                                     - first_base_plan_purchase.month.values[0]
#                                 )
#                             )
#                             + 1
#                         ),
#                     }
#                 )
#         except:
#             pass

#     add_on_purchase_timing_df = pd.DataFrame(data)

#     return add_on_purchase_timing_df


################################################################################################################
################################################################################################################
##                                                                                                            ##
##                                                                                                            ##
##                                                                                                            ##
##                                                                                                            ##
##                  Multiprocessing attempt for `generate_add_on_purchase_timing_df` above^                   ##
##                                                                                                            ##
##                                                                                                            ##
##                                                                                                            ##
##                                                                                                            ##
################################################################################################################
################################################################################################################


################################################################################################################
################################################################################################################
##                                                                                                            ##
##                                                                                                            ##
##                                                                                                            ##
##                                                                                                            ##
##                                               Cohorting                                                    ##
##                                                                                                            ##
##                                                                                                            ##
##                                                                                                            ##
##                                                                                                            ##
################################################################################################################
################################################################################################################


def prep_mrr_df_for_cohort_analysis(df):
    """
    Docstring
    """

    assert np.isclose(
        df[
            ~(
                (
                    df.prior_period_mrr
                    + df.new_business
                    + df.churn
                    + df.reactivation
                    + df.expansion
                    + df.contraction
                    + df.mrr
                )
                == 0
            )
        ].mrr.sum(),
        df.mrr.sum(),
    )

    df = df[
        ~(
            (
                abs(df.prior_period_mrr)
                + abs(df.new_business)
                + abs(df.churn)
                + abs(df.reactivation)
                + abs(df.expansion)
                + abs(df.contraction)
                + abs(df.mrr)
            )
            == 0
        )
    ]

    df = df.sort_values(
        by=[
            "account_code",
            "company_name",
            "product_code",
            "plan_name",
            "plan_group",
            "charge_description",
            "segment",
            "plan_type",
            "currency",
            "month",
        ]
    ).reset_index()

    # drop old index
    df.drop(columns=["index"], inplace=True)
    df.reset_index(inplace=True, drop=True)

    return df


def generate_cohort_analysis(df, assert_test_periods=["2022-05", "2023-01", "2023-10"]):
    """
    This function creates a cohort analysis using the data passed through the `df` variable
    """

    # 1. Create a period column based on the `month`
    df["charge_period"] = df.month.apply(lambda x: x.strftime("%Y-%m"))

    # 2. Determine the user's cohort group (based on their first order)
    df.set_index("account_code", inplace=True)

    df["cohort"] = df.groupby(level=0).month.min().apply(lambda x: x.strftime("%Y-%m"))
    df.reset_index(inplace=True)

    # 3. Rollup data by cohort & charge_period
    grouped = df.groupby(["cohort", "charge_period"])

    # count unique customers and total mrr per cohort and charge_period
    cohorts = grouped.agg({"account_code": pd.Series.nunique, "mrr": np.sum})

    # make the column names more meaningful
    cohorts.rename(
        columns={
            "account_code": "unique_customers",
        },
        inplace=True,
    )

    cohorts.reset_index(inplace=True)

    def cohort_period(df):
        """
        Creates a `CohortPeriod` column, which is the Nth period based on the user's first purchase.

        Example
        -------
        Say you want to get the 3rd month for every user:
            df.sort(['UserId', 'OrderTime', inplace=True)
            df = df.groupby('UserId').apply(cohort_period)
            df[df.CohortPeriod == 3]
        """
        df["relative_period"] = np.arange(len(df)) + 1
        return df

    cohorts = cohorts.groupby(["cohort"]).apply(cohort_period)

    # cohorts.drop(columns="cohort", inplace=True)

    # cohorts.reset_index(inplace=True)

    # cohorts.drop(columns="level_1", inplace=True)

    cohorts.set_index(["cohort", "charge_period"], inplace=True)

    # # testing analysis
    # x = df[
    #     (df.cohort == assert_test_periods[0])
    #     & (df.charge_period == assert_test_periods[0])
    # ]
    # y = cohorts.loc[(assert_test_periods[0], assert_test_periods[0])]

    # assert x.account_code.nunique() == y.unique_customers
    # assert x.mrr.sum().round(2) == y.mrr.round(2)

    # x = df[
    #     (df.cohort == assert_test_periods[1])
    #     & (df.charge_period == assert_test_periods[1])
    # ]
    # y = cohorts.loc[(assert_test_periods[1], assert_test_periods[1])]

    # assert x.account_code.nunique() == y.unique_customers
    # assert x.mrr.sum().round(2) == y.mrr.round(2)

    # x = df[
    #     (df.cohort == assert_test_periods[-1])
    #     & (df.charge_period == assert_test_periods[-1])
    # ]
    # y = cohorts.loc[(assert_test_periods[-1], assert_test_periods[-1])]

    # assert x.account_code.nunique() == y.unique_customers
    # assert x.mrr.sum().round(2) == y.mrr.round(2)

    # creating a datetime64[ns] feature with full date for cohort feature
    df["cohort_month"] = pd.to_datetime((df.cohort + "-01")) + MonthEnd(1)

    return df, cohorts


def prep_cohort_data_for_analysis(df):
    """
    Docstring
    """

    # reset_index
    df = df.reset_index()

    # calc nrr_rate
    df["nrr_rate"] = df.mrr / df.groupby(["cohort"]).mrr.transform("first")

    # # calc prior_period_unique_customers
    # df["prior_period_unique_customers"] = df.groupby(["cohort"]).unique_customers.shift(
    #     -1
    # )
    # df.prior_period_unique_customers.fillna(0, inplace=True)

    # calc nrr_rate
    df["cohort_size"] = (
        df.groupby(["cohort"]).unique_customers.transform("first").astype("int")
    )

    # calc arpu
    df["arpu"] = df.mrr / df.unique_customers

    # reorder features
    df = df[
        [
            "cohort",
            "charge_period",
            "relative_period",
            "unique_customers",
            "cohort_size",
            # "prior_period_unique_customers",
            "mrr",
            "nrr_rate",
            "arpu",
        ]
    ]

    return df


################################################################################################################
################################################################################################################
##                                                                                                            ##
##                                                                                                            ##
##                                                                                                            ##
##                                                                                                            ##
##                                          GymSales MRR Model                                                ##
##                                                                                                            ##
##                                                                                                            ##
##                                                                                                            ##
##                                                                                                            ##
################################################################################################################
################################################################################################################


def raname_workday_revenue_journal_lines(df):
    """
    Docstring
    """

    # renaming columns
    df = df.rename(
        columns={
            "Journal": "journal",
            "Journal Number": "journal_number",
            "Company": "company",
            "Status": "status",
            "Customer Invoice Documents for Customer": "customer_invoice_documents_for_customer",
            "Journal Source": "journal_source",
            "Accounting Date": "accounting_date",
            "Dimension": "dimension",
            "Customer Group": "customer_group",
            "Ledger": "ledger",
            "Currency": "currency",
            "Ledger Account": "ledger_account",
            "Line Memo": "line_memo",
            "Cost Center": "cost_center",
            "Customer": "customer",
            "Spend Category": "spent_cat",
            "Revenue Category": "revenue_category",
            "Transaction Debit Amount": "transaction_debit_amount",
            "Transaction Credit Amount": "transaction_credit_amount",
            "Transaction Net Amount": "transaction_net_amount",
            "Translated Debit Amount": "translated_debit_amount",
            "Translated Credit Amount": "translated_credit_amount",
            "Translated Net Amount": "translated_net_amount",
        }
    )

    return df


def join_journal_lines_with_rev_lookups(df, rev_lookups):
    """
    Docstring
    """

    # joing rev_lookups to df to get product 'revenue_type'
    df = df.merge(
        rev_lookups,
        how="left",
        left_on=[
            "revenue_category",
        ],
        right_on=[
            "revenue_category",
        ],
    )
    # df.drop(columns="revenue_category", inplace=True)

    return df


def prep_datetime_features(df):
    """
    Docstring
    """

    # getting all the necessary pieces to make a datetime feature from the string variable `accounting_date`
    df["year"] = pd.to_datetime(df["accounting_date"]).dt.year
    df["month"] = pd.to_datetime(df["accounting_date"]).dt.month
    df["day"] = 1

    # creating a date feature that is the beginning of the month
    df["accounting_date"] = pd.to_datetime(df.accounting_date)
    df["date"] = pd.to_datetime(
        (df.year * 10000 + df.month * 100 + df.day).apply(str), format="%Y%m%d"
    )

    return df


def gymsales_mrr_misc_data_processing(df):
    """
    Docstring
    """

    # creating a placeholder variables for later appending to Excel model data
    df["group_name"] = "Other"
    df["region"] = "Other"

    # reorder features
    df = df[
        [
            "journal",
            "journal_number",
            "company",
            "status",
            "customer_invoice_documents_for_customer",
            "journal_source",
            "accounting_date",
            "dimension",
            "customer_group",
            "ledger",
            "currency",
            "ledger_account",
            "line_memo",
            "cost_center",
            "customer",
            "spent_cat",
            "revenue_category",
            "transaction_debit_amount",
            "transaction_credit_amount",
            "transaction_net_amount",
            "translated_debit_amount",
            "translated_credit_amount",
            "translated_net_amount",
            "group_name",
            "revenue_type",
            "region",
            "year",
            "month",
            "day",
            "date",
        ]
    ]

    return df


def append_new_revenue_lines_to_master_historic_revenue_lines(
    df, master_historic_revenue_lines
):
    """
    Docstring
    """

    # appending Nov-2023 through Dec-2023 revenue lines to the existing revenue lines that drive the Excel MRR model
    df = pd.concat([master_historic_revenue_lines, df])
    df.reset_index(inplace=True, drop=True)

    return df


def create_gymsales_mrr_features(df, mrr_feature="translated_net_amount"):
    """
    Docstring
    """

    # renaming columns
    df = df.rename(
        columns={
            "date": "month",
            mrr_feature: "mrr",
        }
    )

    df["month"] = df.month + MonthEnd(0)

    # aggregate sum of mrr by month, by customer, by revenue_category, by currency, by revenue_type as there are duplicates in the `df` above
    df = pd.DataFrame(
        df.groupby(
            [
                "month",
                "club_number",
                "customer",
                "revenue_category",
                "currency",
                "revenue_type",
                "product",
                "logo",
                "region",
            ]
        ).mrr.sum()
    ).reset_index()

    # sorting the df
    df = df.sort_values(
        by=[
            "club_number",
            "customer",
            "revenue_category",
            "currency",  # may need to sort on values from the journal lines as priority then feature engineered columns
            "revenue_type",
            "product",
            "logo",
            "region",
            "month",
        ]
    ).reset_index()

    # drop old index
    df.drop(columns=["index"], inplace=True)

    # we need to append/concat a new row of data with the month incremented one (1) from the last `month` value in the df for each group of
    # categorical variables in our df
    for group, value in df.groupby(
        [
            "club_number",
            "customer",
            "revenue_category",
            "currency",
            "revenue_type",
            "product",
            "logo",
            "region",
        ]
    ):  # for each unique grouping of a customer, revenue_category, currency, revenue type

        for i in range(
            len(value)
        ):  # iterate over the length of the rows for the unique group specified above

            try:

                if (
                    round(
                        (value.iloc[i + 1].month - value.iloc[i].month)
                        / np.timedelta64(1, "M")
                    )
                    > 1
                ):  # if the months delta between two sequental rows of data is greater than 1

                    # create some new_data to append to our data frame
                    # we do this in order to give the `prior_period_mrr` calc a row to shift into without losing data

                    new_data = {
                        "month": [
                            f"{value.iloc[i].month + MonthEnd(1)}"
                        ],  # increment `month` by one (1) MonthEnd value
                        "club_number": [f"{group[0]}"],  # club_number
                        "customer": [f"{group[1]}"],  # customer
                        "revenue_category": [
                            f"{group[2]}",
                        ],  # revenue_category
                        "currency": [
                            f"{group[3]}",
                        ],  # currency
                        "revenue_type": [
                            f"{group[4]}",
                        ],  # revenue_type
                        "product": [
                            f"{group[5]}",
                        ],  # product
                        "logo": [
                            f"{group[6]}",
                        ],  # logo
                        "region": [
                            f"{group[7]}",
                        ],  # region
                        "mrr": [
                            f"0",
                        ],  # mrr
                    }

                    df = pd.concat([df, pd.DataFrame(new_data)])
                    df["month"] = pd.to_datetime(df.month)
                    df["mrr"] = df.mrr.astype("float64")

            except IndexError:
                pass

        new_data = {
            "month": [
                f"{value.iloc[-1].month + MonthEnd(1)}"
            ],  # increment `month` by one (1) MonthEnd value
            "club_number": [f"{group[0]}"],  # customer
            "customer": [f"{group[1]}"],  # customer
            "revenue_category": [
                f"{group[2]}",
            ],  # rev_cat
            "currency": [
                f"{group[3]}",
            ],  # currency
            "revenue_type": [
                f"{group[4]}",
            ],  # revenue_type
            "product": [
                f"{group[5]}",
            ],  # product
            "logo": [
                f"{group[6]}",
            ],  # logo
            "region": [
                f"{group[7]}",
            ],  # region
            "mrr": [
                f"0",
            ],  # mrr
        }

        df = pd.concat([df, pd.DataFrame(new_data)])
        df["month"] = pd.to_datetime(df.month)
        df["mrr"] = df.mrr.astype("float64")

    df = df.sort_values(
        by=[
            "club_number",
            "customer",
            "revenue_category",
            "currency",  # may need to sort on values from the journal lines as priority then feature engineered columns
            "revenue_type",
            "product",
            "logo",
            "region",
            "month",
        ]
    ).reset_index()

    # drop old index
    df.drop(columns=["index"], inplace=True)

    # for each customer shift the mrr values forward one
    # groupby all but the month feature in the base schedule df
    df["prior_period_mrr"] = df.groupby(
        [
            "club_number",
            "customer",
            "revenue_category",
            "currency",
            "revenue_type",
            "product",
            "logo",
            "region",
        ]
    ).mrr.shift()

    # filling null values with 0
    df.mrr.fillna(0, inplace=True)
    df.prior_period_mrr.fillna(0, inplace=True)

    # churn
    df["churn"] = np.where(
        # where current period mrr is 0
        (
            df.groupby(
                [
                    "month",
                    "club_number",
                    "customer",
                    "revenue_category",
                    "currency",
                    "revenue_type",
                    "product",
                    "logo",
                    "region",
                ]
            ).mrr.transform(sum)
            == 0
        )
        &
        # and prior period mrr is greater than 0
        (
            df.groupby(
                [
                    "month",
                    "club_number",
                    "customer",
                    "revenue_category",
                    "currency",
                    "revenue_type",
                    "product",
                    "logo",
                    "region",
                ]
            ).prior_period_mrr.transform(sum)
            > 0
        ),
        # give us the difference between the current and prior_period_mrr
        (df.mrr - df.prior_period_mrr),
        0,
    )

    # cumulative_sum for all dimensions other than month
    df["cumulative_sum"] = df.groupby(
        [
            "club_number",
            "customer",
            "revenue_category",
            "currency",
            "revenue_type",
            "product",
            "logo",
            "region",
        ]
    ).prior_period_mrr.cumsum()

    # new_business
    df["new_business"] = np.where(
        # where current period mrr greater than 0
        (
            df.groupby(
                [
                    "month",
                    "club_number",
                    "customer",
                    "revenue_category",
                    "currency",
                    "revenue_type",
                    "product",
                    "logo",
                    "region",
                ]
            ).mrr.transform(sum)
            > 0
        )
        &
        # and prior period mrr is 0
        (
            df.groupby(
                [
                    "month",
                    "club_number",
                    "customer",
                    "revenue_category",
                    "currency",
                    "revenue_type",
                    "product",
                    "logo",
                    "region",
                ]
            ).prior_period_mrr.transform(sum)
            == 0
        )
        &
        # and this customer has never churned
        (
            df.groupby(
                [
                    "month",
                    "club_number",
                    "customer",
                    "revenue_category",
                    "currency",
                    "revenue_type",
                    "product",
                    "logo",
                    "region",
                ]
            ).cumulative_sum.transform(sum)
            == 0
        ),
        # give us the difference between the current and prior_period_mrr
        (df.mrr - df.prior_period_mrr),
        0,
    )

    # reactivation

    # if the current value for mrr is not 0
    # and if any of the previous values for a specific customer's mrr were not 0
    # and if the prior_period_mrr value was 0
    df["reactivation"] = np.where(
        # where current period mrr greater than 0
        (
            df.groupby(
                [
                    "month",
                    "club_number",
                    "customer",
                    "revenue_category",
                    "currency",
                    "revenue_type",
                    "product",
                    "logo",
                    "region",
                ]
            ).mrr.transform(sum)
            > 0
        )
        &
        # and prior period mrr is 0
        (
            df.groupby(
                [
                    "month",
                    "club_number",
                    "customer",
                    "revenue_category",
                    "currency",
                    "revenue_type",
                    "product",
                    "logo",
                    "region",
                ]
            ).prior_period_mrr.transform(sum)
            == 0
        )
        &
        # and this customer has churned at some point in the past
        (
            df.groupby(
                [
                    "month",
                    "club_number",
                    "customer",
                    "revenue_category",
                    "currency",
                    "revenue_type",
                    "product",
                    "logo",
                    "region",
                ]
            ).cumulative_sum.transform(sum)
            > 0
        ),
        # give us the difference between the current and prior_period_mrr
        (df.mrr - df.prior_period_mrr),
        0,
    )

    # expansion
    df["expansion"] = np.where(
        # current period mrr is greater than prior period mrr
        (
            df.groupby(
                [
                    "month",
                    "club_number",
                    "customer",
                    "revenue_category",
                    "currency",
                    "revenue_type",
                    "product",
                    "logo",
                    "region",
                ]
            ).mrr.transform(sum)
            > df.groupby(
                [
                    "month",
                    "club_number",
                    "customer",
                    "revenue_category",
                    "currency",
                    "revenue_type",
                    "product",
                    "logo",
                    "region",
                ]
            ).prior_period_mrr.transform(sum)
        )
        &
        # where new business == 0
        (
            df.groupby(
                [
                    "month",
                    "club_number",
                    "customer",
                    "revenue_category",
                    "currency",
                    "revenue_type",
                    "product",
                    "logo",
                    "region",
                ]
            ).new_business.transform(sum)
            == 0
        )
        &
        # where reactivation == 0
        (
            df.groupby(
                [
                    "month",
                    "club_number",
                    "customer",
                    "revenue_category",
                    "currency",
                    "revenue_type",
                    "product",
                    "logo",
                    "region",
                ]
            ).reactivation.transform(sum)
            == 0
        ),
        # give us the difference between the current and prior_period_mrr
        (df.mrr - df.prior_period_mrr),
        0,
    )

    # contraction
    df["contraction"] = np.where(
        # current period mrr is less than prior period mrr
        (
            df.groupby(
                [
                    "month",
                    "club_number",
                    "customer",
                    "revenue_category",
                    "currency",
                    "revenue_type",
                    "product",
                    "logo",
                    "region",
                ]
            ).mrr.transform(sum)
            < df.groupby(
                [
                    "month",
                    "club_number",
                    "customer",
                    "revenue_category",
                    "currency",
                    "revenue_type",
                    "product",
                    "logo",
                    "region",
                ]
            ).prior_period_mrr.transform(sum)
        )
        &
        # where current period mrr greater than 0
        (
            df.groupby(
                [
                    "month",
                    "club_number",
                    "customer",
                    "revenue_category",
                    "currency",
                    "revenue_type",
                    "product",
                    "logo",
                    "region",
                ]
            ).churn.transform(sum)
            == 0
        ),
        # give us the difference between the current and prior_period_mrr
        (df.mrr - df.prior_period_mrr),
        0,
    )

    df["check_starting_ending"] = np.isclose(
        df.prior_period_mrr.shift(-1).fillna(0),
        df.mrr,
    )

    df["check_mrr_movement"] = np.isclose(
        (
            df.prior_period_mrr
            + df.new_business
            + df.churn
            + df.reactivation
            + df.expansion
            + df.contraction
        ),
        df.mrr,
    )

    # where do we have problem with our check?
    assert df[df.check_mrr_movement == False].shape[0] == 0

    # where do we have problem with our check?
    # the only observations where the check_starting_ending feature should be False is the last row in the dataframe since the df.prior_period_mrr.shift(-1) null (isna() == True)
    assert df[df.check_starting_ending == False].shape[0] == 0

    # reorder features
    df = df[
        [
            "month",
            "club_number",
            "customer",
            "logo",
            "revenue_category",
            "revenue_type",
            "product",
            "region",
            "currency",
            "prior_period_mrr",
            "new_business",
            "churn",
            "reactivation",
            "expansion",
            "contraction",
            "mrr",
            "cumulative_sum",
            "check_starting_ending",
            "check_mrr_movement",
        ]
    ]

    return df
