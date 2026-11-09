"""This file should contain functions for plotting with matplotlib.
It insures the main Jupyter notebook stays clean from plotting stuff."""

import ee
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import figure
from scipy import stats


def plot_histogram_image_collection(
        collection: ee.ImageCollection,
        start: str,
        end: str,
        split_percent: list[int | float] | None = None,
        color_space: str = "coolwarm"
) -> None:
    """Plots the image collection times of recording with color according to Cloudy Pixel Percentage. Bins are quarter yearly, please use start and end dates that align with 3-month cycles.
    :param collection: ee.ImageCollection
    :param start: Date 'yyyy-MM-dd'
    :param end: Date 'yyyy-MM-dd'
    :param split_percent: list of percentiles where colors are split. Numbers must be between 0 and 100.
    :param color_space: matplotlib.pyplot color space name, such as 'coolwarm'.
    :return: None
    """

    if split_percent is None or len(split_percent) == 0:
        split_percent = [10, 20, 30, 50]

    times = collection.aggregate_array('system:time_start').getInfo()
    clouds = collection.aggregate_array('CLOUDY_PIXEL_PERCENTAGE').getInfo()
    dates = pd.to_datetime(times, unit='ms')

    cloud_bins = [0, *split_percent, 100]
    cloud_labels = [
        f"<{split_percent[0]}%",
        *[f"{split_percent[n]}%-{split_percent[n+1]}%" for n in range(len(split_percent)-1)],
        f">{split_percent[-1]}%"
    ]
    cloud_categories = pd.cut(clouds, bins=cloud_bins, labels=cloud_labels, include_lowest=True)

    plt.figure(figsize=(12, 6))

    data = [dates[cloud_categories == category] for category in cloud_labels]
    date_bins = pd.date_range(start=start, end=end, freq='3MS')

    colors = plt.get_cmap(color_space)(np.linspace(0, 1, len(cloud_labels)))

    plt.hist(data, bins=date_bins, stacked=True, label=cloud_labels, color=colors)

    plt.xlabel("Recording time")
    plt.ylabel("Number of images")
    plt.title(f"{collection.get('collection_name').getInfo()} - Histogram of Recording Time, n={collection.size().getInfo()}")
    plt.legend(title="Cloud coverage")
    plt.show()


def plot_gst_series(ts: pd.DataFrame) -> matplotlib.figure.Figure:
    ts = ts.assign(err=np.hypot(ts["gst_sd"], ts["gst_unc"]))

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharex=True)

    # --- gst plot ---
    for zone, d in ts.groupby("zone"):
        ax.errorbar(d["date"], d["gst"], yerr=d["err"], fmt=".-", lw=0.6,
                       elinewidth=0.5, ecolor="gray", capsize=1.5, label=zone)

    ax.axhline(0, color="black", ls="--", linewidth=0.6)
    ax.set(
        ylabel="glacier surface temperature (°C)",
        title="Half-month area-weighted GST"
    )
    ax.legend()

    fig.show()

    return fig