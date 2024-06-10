# import necessary libraries
import time
import cProfile
import multiprocessing

from datetime import date
from datetime import datetime

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd

from scipy import interpolate

import joypy

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.legend_handler import HandlerBase
from matplotlib.text import Text

# http://stanford.edu/~mwaskom/software/seaborn/
import seaborn as sns

last_month = pd.to_datetime(datetime.today().strftime("%Y-%m-%d")) + MonthEnd(-1)


# source: https://www.analyzecore.com/2015/12/10/cohort-analysis-retention-rate-visualization-r/
# asked ChatGTP to translate the R to python using pandas: https://chat.openai.com/share/9cdee5fc-e756-49b9-aa73-974e21abd62a
def generate_cohort_retention_bubble_chart(
    df_plot,
    df_plot2,
    df_plot3,
    bubble_size_multiple=12,
    chart_title=f"""Net Revenue Retention Rate (as of {last_month.strftime("%b %d, %Y")})""",
    addon_name="",
    customer_segment="",
    consolidated_nrr_text_plot_color="white",
    add_on_text_plot_color="white",
    legend_location="best",
    y_axis_lower_bound=0,
    y_axis_upper_bound=2,
):
    """
    Docstring
    """

    # Custom legend handler for custom text
    class TextHandler(HandlerBase):
        def create_artists(
            self,
            legend,
            orig_handle,
            xdescent,
            ydescent,
            width,
            height,
            fontsize,
            trans,
        ):
            x = width // 2
            y = height // 2
            text = orig_handle
            t = Text(
                x, y, text, fontsize=fontsize, ha="center", va="center", color="black"
            )
            return [t]

    # consolidated
    sns.lineplot(
        data=df_plot.dropna(),
        x="cohort",
        y="nrr_rate",
        # hue="cohort",
        color="black",
        alpha=0.5,
        linewidth=2,
    )

    # cba_cohorts
    sns.lineplot(
        data=df_plot2.dropna(),
        x="cohort",
        y="nrr_rate",
        hue="relative_period",
        color="midnightblue",
        alpha=0.5,
        linewidth=2,
    )

    # non_cba_chorts
    sns.lineplot(
        data=df_plot3.dropna(),
        x="cohort",
        y="nrr_rate",
        hue="relative_period",
        color="grey",
        alpha=0.5,
        linewidth=2,
    )

    # consolidated
    # current cohort size bubble
    ax_1 = sns.scatterplot(
        data=df_plot.tail(1).dropna(),
        x="cohort",
        y="nrr_rate",
        # hue="relative_period",
        color="black",
        size="unique_customers",
        sizes=(
            df_plot.tail(1).unique_customers.values[0] * bubble_size_multiple,
            df_plot.tail(1).unique_customers.values[0] * bubble_size_multiple,
        ),
        alpha=0.8,
    )

    # cba_cohorts
    # current cohort size bubble
    ax_2 = sns.scatterplot(
        data=df_plot2.iloc[:-1].dropna(),
        x="cohort",
        y="nrr_rate",
        # hue="relative_period",
        color="midnightblue",
        size="unique_customers",
        sizes=(
            df_plot2.iloc[:-1].unique_customers.min() * bubble_size_multiple,
            df_plot2.iloc[:-1].unique_customers.max() * bubble_size_multiple,
        ),
        alpha=0.8,
    )

    # non_cba_chorts
    # current cohort size bubble
    ax_3 = sns.scatterplot(
        data=df_plot3.iloc[:-1].dropna(),
        x="cohort",
        y="nrr_rate",
        # hue="relative_period",
        color="grey",
        size="unique_customers",
        sizes=(
            df_plot3.iloc[:-1].unique_customers.min() * bubble_size_multiple,
            df_plot3.iloc[:-1].unique_customers.max() * bubble_size_multiple,
        ),
        alpha=0.8,
    )

    # # consolidated
    # # initial cohort size bubble plot (effectively, this is a halo)
    # ax_1 = sns.scatterplot(
    #     data=df_plot.tail(1).dropna(),
    #     x="cohort",
    #     y="nrr_rate",
    #     # hue="relative_period",
    #     color="black",
    #     size="cohort_size",
    #     sizes=(4000, 4000),
    #     alpha=0.3,
    # )

    # cba_cohorts
    # initial cohort size bubble plot (effectively, this is a halo)
    ax_2 = sns.scatterplot(
        data=df_plot2.iloc[:-1].dropna(),
        x="cohort",
        y="nrr_rate",
        # hue="relative_period",
        color="midnightblue",
        size="cohort_size",
        sizes=(
            df_plot2.iloc[:-1].cohort_size.min() * bubble_size_multiple,
            df_plot2.iloc[:-1].cohort_size.max() * bubble_size_multiple,
        ),
        alpha=0.3,
    )

    # non_cba_chorts
    # initial cohort size bubble plot (effectively, this is a halo)
    ax_3 = sns.scatterplot(
        data=df_plot3.iloc[:-1].dropna(),
        x="cohort",
        y="nrr_rate",
        # hue="relative_period",
        color="grey",
        size="cohort_size",
        sizes=(
            df_plot3.iloc[:-1].cohort_size.min() * bubble_size_multiple,
            df_plot3.iloc[:-1].cohort_size.max() * bubble_size_multiple,
        ),
        alpha=0.3,
    )

    # consolidated
    # this generates a text plot with the current cohort size (unique_customers) divided (/) by the inital cohort size and the NRR rate
    for i, row in df_plot.tail(1).dropna().iterrows():
        plt.text(
            row["cohort"],
            row["nrr_rate"],
            f"{row['unique_customers']:,.0f} / {row['cohort_size']:,.0f}\n{row['nrr_rate']:.0%}",
            color=consolidated_nrr_text_plot_color,
            size=8,
            ha="center",
            va="center",
        )

    # cba_cohorts
    for i, row in df_plot2.iloc[:-1].dropna().iterrows():
        plt.text(
            row["cohort"],
            row["nrr_rate"],
            f"{row['unique_customers']:,.0f} / {row['cohort_size']:,.0f}\n{row['nrr_rate']:.0%}",
            color=add_on_text_plot_color,
            size=8,
            ha="center",
            va="center",
        )

    # non_cba_chorts
    for i, row in df_plot3.iloc[:-1].dropna().iterrows():
        plt.text(
            row["cohort"],
            row["nrr_rate"],
            f"{row['unique_customers']:,.0f} / {row['cohort_size']:,.0f}\n{row['nrr_rate']:.0%}",
            color="black",
            size=8,
            ha="center",
            va="center",
        )

    # chart formatting
    plt.title(
        chart_title,
        fontsize=18,
        fontweight="bold",
        y=1.02,
    )

    plt.xlabel("Cohorts", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45)

    plt.ylabel("Retention Rate", fontsize=14, fontweight="bold")

    # Format y-axis ticks as percentages
    ax_2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.0%}".format(y)))
    plt.ylim(y_axis_lower_bound, y_axis_upper_bound)

    # Custom legend
    custom_handles = [
        "x / y",
        "###%",
        plt.Line2D(
            [0],
            [0],
            color="midnightblue",
            marker="o",
            linestyle="",
            alpha=0.8,
            markersize=8,
        ),
        plt.Line2D(
            [0], [0], color="grey", marker="o", linestyle="", alpha=0.8, markersize=8
        ),
        plt.Line2D(
            [0], [0], color="black", marker="o", linestyle="", alpha=0.8, markersize=8
        ),
        plt.Line2D([0], [0], color="black", alpha=0.5, linewidth=2),
    ]
    custom_labels = [
        "Current Cohort Size / Initial Cohort Size",
        "NRR Rate",
        f"{addon_name} Customers Cohorts",
        f"Non-{addon_name} Customers Cohorts",
        f"""Consolidated {last_month.strftime("%b-%Y")} Cohort""",
        f"{customer_segment} NRR",
    ]
    legend = plt.legend(
        custom_handles,
        custom_labels,
        title="Legend",
        loc=legend_location,
        handler_map={str: TextHandler()},
    )
    plt.gca().add_artist(legend)  # Add legend as an artist to maintain its position

    # Make legend box transparent
    legend.get_frame().set_alpha(0)

    # Access the legend title text object and set its font properties directly
    legend.get_title().set_fontweight("bold")

    plt.show()


def generate_cohort_arpu_bubble_chart(
    df_plot,
    df_plot2,
    df_plot3,
    bubble_size_multiple=12,
    chart_title=f"""ARPU (as of {last_month.strftime("%b %d, %Y")})""",
    addon_name="",
    customer_segment="",
    add_on_text_plot_color="white",
    legend_location="best",
    y_axis_upper_bound=100,
):
    """
    Docstring
    """

    # Custom legend handler for custom text
    class TextHandler(HandlerBase):
        def create_artists(
            self,
            legend,
            orig_handle,
            xdescent,
            ydescent,
            width,
            height,
            fontsize,
            trans,
        ):
            x = width // 2
            y = height // 2
            text = orig_handle
            t = Text(
                x, y, text, fontsize=fontsize, ha="center", va="center", color="black"
            )
            return [t]

    sns.lineplot(
        data=df_plot.dropna(),
        x="cohort",
        y="arpu",
        # hue="relative_period",
        color="black",
        alpha=0.5,
        linewidth=2,
    )

    # cba_cohorts
    sns.lineplot(
        data=df_plot2.dropna(),
        x="cohort",
        y="arpu",
        hue="relative_period",
        color="midnightblue",
        alpha=0.5,
        linewidth=2,
    )

    # non_cba_chorts
    sns.lineplot(
        data=df_plot3.dropna(),
        x="cohort",
        y="arpu",
        hue="relative_period",
        color="grey",
        alpha=0.5,
        linewidth=2,
    )

    # cba_cohorts
    # current cohort size bubble
    ax_2 = sns.scatterplot(
        data=df_plot2.dropna(),
        x="cohort",
        y="arpu",
        # hue="relative_period",
        color="midnightblue",
        size="unique_customers",
        sizes=(
            df_plot2.dropna().unique_customers.min() * bubble_size_multiple,
            df_plot2.dropna().unique_customers.max() * bubble_size_multiple,
        ),
        alpha=0.8,
    )

    # non_cba_chorts
    # current cohort size bubble
    ax_3 = sns.scatterplot(
        data=df_plot3.dropna(),
        x="cohort",
        y="arpu",
        # hue="relative_period",
        color="grey",
        size="unique_customers",
        sizes=(
            df_plot3.dropna().unique_customers.min() * bubble_size_multiple,
            df_plot3.dropna().unique_customers.max() * bubble_size_multiple,
        ),
        alpha=0.8,
    )

    # cba_cohorts
    # initial cohort size bubble plot (effectively, this is a halo)
    ax_2 = sns.scatterplot(
        data=df_plot2.dropna(),
        x="cohort",
        y="arpu",
        # hue="relative_period",
        color="midnightblue",
        size="cohort_size",
        sizes=(
            df_plot2.dropna().cohort_size.min() * bubble_size_multiple,
            df_plot2.dropna().cohort_size.max() * bubble_size_multiple,
        ),
        alpha=0.3,
    )

    # non_cba_chorts
    # initial cohort size bubble plot (effectively, this is a halo)
    ax_3 = sns.scatterplot(
        data=df_plot3.dropna(),
        x="cohort",
        y="arpu",
        # hue="relative_period",
        color="grey",
        size="cohort_size",
        sizes=(
            df_plot3.dropna().cohort_size.min() * bubble_size_multiple,
            df_plot3.dropna().cohort_size.max() * bubble_size_multiple,
        ),
        alpha=0.3,
    )

    # cba_cohorts
    for i, row in df_plot2.dropna().iterrows():
        plt.text(
            row["cohort"],
            row["arpu"],
            f"{row['unique_customers']:,.0f} / {row['cohort_size']:,.0f}\n${row['arpu']:.0f}",
            color=add_on_text_plot_color,
            size=8,
            ha="center",
            va="center",
        )

    # non_cba_chorts
    for i, row in df_plot3.dropna().iterrows():
        plt.text(
            row["cohort"],
            row["arpu"],
            f"{row['unique_customers']:,.0f} / {row['cohort_size']:,.0f}\n${row['arpu']:.0f}",
            color="black",
            size=8,
            ha="center",
            va="center",
        )

    # chart formatting
    plt.title(
        chart_title,
        fontsize=18,
        fontweight="bold",
        y=1.02,
    )

    plt.xlabel("Cohorts", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45)

    plt.ylabel("ARPU", fontsize=14, fontweight="bold")

    # Format y-axis ticks as percentages
    ax_2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: "${:,.0f}".format(y)))
    plt.ylim(0, y_axis_upper_bound)

    # Custom legend
    custom_handles = [
        "x / y",
        "$##",
        plt.Line2D(
            [0],
            [0],
            color="midnightblue",
            marker="o",
            linestyle="",
            alpha=0.8,
            markersize=8,
        ),
        plt.Line2D(
            [0], [0], color="grey", marker="o", linestyle="", alpha=0.8, markersize=8
        ),
        plt.Line2D([0], [0], color="black", alpha=0.5, linewidth=2),
    ]
    custom_labels = [
        "Current Cohort Size / Initial Cohort Size",
        "ARPU",
        f"{addon_name} Customers Cohorts",
        f"Non-{addon_name} Customers Cohorts",
        f"{customer_segment} ARPU",
    ]
    legend = plt.legend(
        custom_handles,
        custom_labels,
        title="Legend",
        loc=legend_location,
        handler_map={str: TextHandler()},
    )
    plt.gca().add_artist(legend)  # Add legend as an artist to maintain its position

    # Make legend box transparent
    legend.get_frame().set_alpha(0)

    # Access the legend title text object and set its font properties directly
    legend.get_title().set_fontweight("bold")

    plt.show()


def plot_add_on_distribution_by_base_plan(df, y="account_code", chart_title=""):
    """
    This function creates a column chart (plt.bar) of customer count by base plan
    """

    base_plans = list(df.base_plan_at_time_of_cba_purchase)
    active_customer_count = list(df[y])

    # creating the bar plot
    fig, ax = plt.subplots()
    bar_container = ax.bar(
        base_plans,
        active_customer_count,
        color="midnightblue",
        width=0.4,
    )
    ax.bar_label(bar_container, fmt="{:,.0f}")

    plt.xlabel("Base Plan", fontsize=14, fontweight="bold")
    plt.ylabel("Customer Count", fontsize=14, fontweight="bold")
    plt.title(
        f"{chart_title}",
        fontsize=18,
        fontweight="bold",
        y=1.02,
    )

    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: "{:,.0f}".format(x))
    )  # Format as integer with comma separator

    plt.show()
