from typing import Literal

import pandas as pd


def calculate_statistic(data: pd.Series, statistic: Literal["mean", "median", "mode"]) -> float:
    if statistic == "mean":
        return data.mean()
    if statistic == "median":
        return data.median()
    if statistic == "mode":
        return data.mode().iloc[0]
    raise ValueError(f"Unknown statistic: {statistic}")
