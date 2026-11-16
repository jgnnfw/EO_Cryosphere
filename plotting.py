"""This file should contain functions for plotting with matplotlib.
It insures the main Jupyter notebook stays clean from plotting stuff."""
from typing import Callable
import ee
import calendar
import geemap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import figure
from matplotlib.patches import Patch
from io_param_functions import *
import matplotlib.dates as mdates

#---------------- plot functions returning matplotlig.figure.Figure ------------------------------------------
def plot_histogram_image_collection(
        collection: ee.ImageCollection,
        start: str,
        end: str,
        split_percent: list[int | float] | None = None,
        color_space: str = "coolwarm"
) -> matplotlib.figure.Figure:
    """Plots the image collection times of recording with color according to Cloudy Pixel Percentage.
    Bins are quarter yearly, please use start and end dates that align with 3-month cycles.
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

    fig = plt.figure(figsize=(12, 6))

    data = [dates[cloud_categories == category] for category in cloud_labels]
    date_bins = pd.date_range(start=start, end=end, freq='3MS')

    colors = plt.get_cmap(color_space)(np.linspace(0, 1, len(cloud_labels)))

    plt.hist(data, bins=date_bins, stacked=True, label=cloud_labels, color=colors)

    plt.xlabel("Recording time")
    plt.ylabel("Number of images")
    plt.title(f"{collection.get('collection_name').getInfo()} - Histogram of Recording Time, n={collection.size().getInfo()}")
    plt.legend(title="Cloud coverage")

    return fig

def plot_gst_series(ts: pd.DataFrame, errorbars: bool) -> matplotlib.figure.Figure:
    ts = ts.assign(err=np.hypot(ts["gst_sd"], ts["gst_unc"]))

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharex=True)

    color = plt.rcParams["text.color"]

    # --- gst plot ---
    for zone, d in ts.groupby("zone"):
        if errorbars:
            ax.errorbar(d["date"], d["gst"], yerr=d["err"], fmt=".-", lw=0.6,
                       elinewidth=0.5, ecolor="gray", capsize=1.5, label=zone)
        else:
            ax.plot(d["date"], d["gst"], marker=".", lw=0.6, label=zone)

    ax.axhline(0, color=color, ls="--", linewidth=0.6)
    ax.set(
        ylabel="glacier surface temperature (°C)",
        title="Half-month area-weighted GST"
    )
    ax.legend(loc="upper left")

    if errorbars:
        plt.text(s="error bars: √(spatial_scattering² + mean_surface_temperature_standard_deviation²)",
                x=0.5, y=-0.08, transform=ax.transAxes, ha="center", va="top")

    return fig


def plot_half_months(gst_half_month: pd.DataFrame) -> matplotlib.figure.Figure:
    gst_half_month["month"] = gst_half_month["hm"] // 2 + 1
    monthly = gst_half_month.groupby(["year", "month"]).size().reset_index(name="count")
    monthly["date"] = pd.to_datetime(
        monthly[["year", "month"]].assign(day=1)
    )
    quarterly = gst_half_month.groupby(["year", (gst_half_month["month"] - 1) // 3 + 1]).size().reset_index(
        name="count")

    quarterly["date"] = pd.to_datetime(
        quarterly["year"].astype(str) + "-" + ((quarterly.iloc[:, 1] - 1) * 3 + 1).astype(str) + "-01")

    def seasonal_color(month):
        if month in [12, 1, 2]:
            return "steelblue"  # Winter
        elif month in [3, 4, 5]:
            return "mediumseagreen"  # Spring
        elif month in [6, 7, 8]:
            return "indianred"  # Summer
        else:
            return "darkorange"  # Autumn

    monthly_colors = monthly["month"].apply(seasonal_color)

    # Quarterly colors based on the middle month of each quarter
    quarterly_colors = quarterly.iloc[:, 1].apply(
        lambda q: seasonal_color((q - 1) * 3 + 2)
    )

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.bar(monthly["date"], monthly["count"], width=25, color=monthly_colors)
    ax1.set_title("Monthly observations")
    ax1.set_ylabel("Number of observations")

    ax2.bar(quarterly["date"], quarterly["count"], width=75, color=quarterly_colors)
    ax2.set_title("Quarterly observations")
    ax2.set_ylabel("Number of observations")
    seasons = {
        "Winter": ([12, 1, 2], "steelblue"),
        "Spring": ([3, 4, 5], "mediumseagreen"),
        "Summer": ([6, 7, 8], "indianred"),
        "Autumn": ([9, 10, 11], "darkorange"),
    }

    legend_handles = []

    for season, (months, color) in seasons.items():
        n = monthly.loc[monthly["month"].isin(months), "count"].sum()

        legend_handles.append(
            Patch(facecolor=color, label=f"{season} (n={n:,})")
        )

    ax1.legend(handles=legend_handles, title="Season")
    ax2.legend(handles=legend_handles, title="Season")

    fig.align_ylabels([ax1, ax2])

    ax1.tick_params(axis="x", labelrotation=0)
    ax2.tick_params(axis="x", labelrotation=0)
    ax1.legend(
        handles=legend_handles,
        loc="upper right",
        title="Season"
    )
    plt.tight_layout()
    return fig


def plot_half_months_2(gst_half_month: pd.DataFrame) -> matplotlib.figure.Figure:
    gst_half_month["month"] = gst_half_month["hm"] // 2 + 1

    quarterly = gst_half_month.groupby(["year", (gst_half_month["month"] - 1) // 3 + 1]).size().reset_index(
        name="count")

    quarterly["date"] = pd.to_datetime(
        quarterly["year"].astype(str) + "-" + ((quarterly.iloc[:, 1] - 1) * 3 + 1).astype(str) + "-01")

    def seasonal_color(month):
        if month in [12, 1, 2]:
            return "steelblue"  # Winter
        elif month in [3, 4, 5]:
            return "mediumseagreen"  # Spring
        elif month in [6, 7, 8]:
            return "indianred"  # Summer
        else:
            return "darkorange"  # Autumn

    # Quarterly colors based on the middle month of each quarter
    quarterly_colors = quarterly.iloc[:, 1].apply(
        lambda q: seasonal_color((q - 1) * 3 + 2)
    )

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), width_ratios=[2, 0.8])

    ax1.set_title("Quarterly observations")

    ax1.bar(quarterly["date"], quarterly["count"], width=75, color=quarterly_colors)
    ax1.set_ylabel("Number of observations")
    seasons = {
        "Winter": ([12, 1, 2], "steelblue"),
        "Spring": ([3, 4, 5], "mediumseagreen"),
        "Summer": ([6, 7, 8], "indianred"),
        "Autumn": ([9, 10, 11], "darkorange"),
    }

    counts = [quarterly.loc[quarterly["date"].dt.month.isin(months), "count"].sum() for season, (months, color) in
              seasons.items()]
    ax2.bar(seasons.keys(), counts, color=quarterly_colors)
    ax2.set_title("Total")

    ax1.tick_params(axis="x", labelrotation=0)

    legend_handles = [Patch(facecolor=color, label=season) for season, (month, color) in seasons.items()]
    ax1.legend(handles=legend_handles, loc="upper left", title="Season")

    plt.tight_layout()
    return fig


# ----------------------------------------------------------------------------- plots
def plot_sar_snow_series(df: pd.DataFrame, col: str = "wet") -> matplotlib.figure.Figure:
    """Wet fraction (or other column) per date over the full period, by orbit direction."""
    fig, ax = plt.subplots(figsize=(12, 4))
    for orb, d in df.groupby("orbit"):
        ax.plot(d["date"], d[col], "o", ms=3, label=orb.lower())
    ax.set_ylabel(col)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_sar_snow_season(df: pd.DataFrame, day_ticks: tuple[list[int|float], list[str]], col: str = "wet") -> matplotlib.figure.Figure:
    """Seasonal course (Dec-May) of all years overlaid, one panel per orbit direction."""
    orbs = sorted(df["orbit"].unique())
    sys_ = sorted(df["sy"].unique())
    cmap = plt.get_cmap("viridis")
    fig, axs = plt.subplots(1, len(orbs), figsize=(6 * len(orbs), 4), sharey=True, squeeze=False)
    for ax, orb in zip(axs[0], orbs):
        for sy, d in df[df["orbit"] == orb].groupby("sy"):
            d = d.sort_values("day")
            ax.plot(d["day"], d[col], ".-", color=cmap(sys_.index(sy) / max(len(sys_) - 1, 1)), label=str(sy))
        ax.set_xticks(*day_ticks)
        ax.set_title(orb.lower())
    axs[0, 0].set_ylabel(col)
    axs[0, -1].legend(ncol=2, fontsize=7, title="season year")
    fig.tight_layout()
    return fig


def plot_sar_snow_elev(bdf: pd.DataFrame, sy: int, orbit: str, min_px: int = 20) -> matplotlib.figure.Figure:
    """Wet fraction by elevation band (y) and date (x) for one season year and orbit."""
    d = bdf[(bdf["sy"] == sy) & (bdf["orbit"] == orbit) & (bdf["n_px"] >= min_px)]
    pv = d.pivot_table(index="elev", columns="date", values="wet")
    fig, ax = plt.subplots(figsize=(10, 4))
    mesh = ax.pcolormesh(mdates.date2num(pv.columns), pv.index.values, np.ma.masked_invalid(pv.values),
                         cmap="Blues", vmin=0, vmax=1, shading="nearest")
    ax.xaxis_date()
    fig.autofmt_xdate()
    ax.set_ylabel("elevation (m)")
    ax.set_title(f"{sy} {orbit.lower()}")
    fig.colorbar(mesh, ax=ax, label="wet fraction")
    fig.tight_layout()
    return fig


def plot_sar_snow_diff_hist(roi: ee.Geometry, coll: ee.ImageCollection,
                            months: tuple[int, ...] = (12, 1, 2, 3, 4, 5),  thr: float = -2.0,
                            scale: float = 30, rng: tuple[float, float] = (-15, 10),
                            nbins: int = 50, debug: bool = False) -> matplotlib.figure.Figure:
    """Histogram of backscatter change vs. dry reference per month, with the threshold marked.
    Use coll = wet_coll(roi, sy, orbit, pol, thr) first."""
    fig, ax = plt.subplots(figsize=(7, 4))
    cmap = plt.get_cmap("coolwarm")
    for i, mo in enumerate(months):
        img = coll.filter(ee.Filter.calendarRange(mo, mo, "month")).select("diff").median()
        try:
            h = img.reduceRegion(reducer=ee.Reducer.fixedHistogram(rng[0], rng[1], nbins), geometry=roi,
                                 scale=scale, maxPixels=int(1e9), bestEffort=True,
                                 tileScale=4).get("diff").getInfo()
        except ee.EEException as e:
            myprint(debug, f"month {mo}: {e}")
            continue
        a = np.array(h)
        ax.plot(a[:, 0], a[:, 1] / a[:, 1].sum(), color=cmap(i / (len(months) - 1)), label=calendar.month_abbr[mo])
    ax.axvline(thr, color="k", ls="--", label=f"thr {thr} dB")
    ax.set_xlabel("backscatter change vs. dry reference (dB)")
    ax.set_ylabel("pixel share")
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------- maps -------------------------------------------------------------
def plot_sar_snow_maps(
        roi: ee.Geometry, coll: ee.ImageCollection,
        wet_freqs: dict[int, ee.Image],
        melt_onset: ee.Image,
        sy: int,
        glacier_mask: ee.Geometry | None = None,
) -> geemap.Map:
    """Interactive map: monthly wet-snow frequency layers and melt-onset day."""
    m = geemap.Map()
    m.centerObject(roi, 11)
    vis = {"min": 0, "max": 1, "palette": ["#f0f0f0", "#9ecae1", "#08306b"]}
    for mo, wet_freq in wet_freqs.items():
        m.addLayer(wet_freq.clip(roi), vis, f"wet freq {calendar.month_abbr[mo]} {sy}", mo == 5)
    vis_on = {"min": 0, "max": 182, "palette": ["#440154", "#3b528b", "#21908d", "#5dc863", "#fde725"]}
    m.addLayer(melt_onset.clip(roi), vis_on, f"melt onset (days since Dec 1) {sy}", False)
    m.add_colorbar(vis, label="wet frequency (0 dry, 1 wet)")
    for geom, name, col in ((roi, "roi", "black"), (glacier_mask, "glacier", "red")):
        if geom is not None:
            m.addLayer(ee.FeatureCollection([ee.Feature(geom)]).style(color=col, fillColor="00000000", width=2), {}, name)
    return m


def show_map(dem: ee.Image, roi: ee.Geometry, glacier_mask: ee.Geometry):
    """Optional: quick look in a notebook."""
    m = geemap.Map()
    m.centerObject(roi, 13)
    m.addLayer(dem.clip(roi), {"min": 4500, "max": 6500}, "DEM")
    m.addLayer(glacier_mask.clip(roi), {"palette": "cyan"}, "glacier mask")
    return m

