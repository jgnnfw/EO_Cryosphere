"""This file should contain functions for plotting with matplotlib.
It insures the main Jupyter notebook stays clean from plotting stuff.
return geemap.Map or matplotlib Figure."""
import ee
import calendar
import geemap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure
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


def plot_sar_snow_season(df: pd.DataFrame,
                         day_ticks: tuple[list[int|float], list[str]],
                         col: str = "wet") -> matplotlib.figure.Figure:
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


def map_dem_and_aoi(dem: ee.Image, roi: ee.Geometry, glacier_mask: ee.Geometry):
    """Optional: quick look in a notebook."""
    m = geemap.Map()
    m.centerObject(roi, 13)
    m.addLayer(dem.clip(roi), {"min": 4500, "max": 6500}, "DEM")
    m.addLayer(glacier_mask.clip(roi), {"palette": "cyan"}, "glacier mask")
    return m


# --- glacier_area -----------------------------------------------------------
def map_glacier_area(
    masks: dict[int, ee.Image],
    roi: ee.Geometry,
    glacier_mask: ee.Geometry | None = None,
    zoom: int = 14,
) -> geemap.Map:
    """Yearly snow/ice masks as coloured layers (red = oldest -> blue = newest)."""
    colors = ["#d7191c", "#f57a5a", "#fdbf6f", "#ffffbf", "#c7e9b4",
              "#7fcdbb", "#41b6c4", "#2c7fb8", "#253494"]
    m = geemap.Map(basemap="HYBRID")
    m.centerObject(roi, zoom)
    for i, (year, mask) in enumerate(sorted(masks.items())):
        m.addLayer(mask, {"palette": [colors[i % len(colors)]]}, f"Snow/ice {year}")
    if glacier_mask is not None:
        outline = ee.Image().paint(ee.FeatureCollection([ee.Feature(glacier_mask)]), 1, 2)
        m.addLayer(outline, {"palette": ["orange"]}, "GLIMS outline")
    m.addLayer(ee.Image().paint(ee.FeatureCollection([ee.Feature(roi)]), 1, 2),
               {"palette": ["red"]}, "ROI")
    return m


def plot_glacier_area_seasons(area_df: pd.DataFrame) -> matplotlib.figure.Figure:
    """Snow/ice area per year, one line per season. Needs [season, year, area_km2]."""
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e", "#9467bd", "#8c564b", "#17becf", "#7f7f7f"]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, (season, grp) in enumerate(area_df.groupby("season", sort=False)):
        grp = grp.dropna(subset=["area_km2"])
        if len(grp) > 0:
            ax.plot(grp["year"], grp["area_km2"], marker=markers[i % len(markers)],
                    color=colors[i % len(colors)], linewidth=2, markersize=7, label=season)
    mean = area_df.groupby("year")["area_km2"].mean()
    ax.plot(mean.index, mean.values,
            marker=markers[area_df["season"].nunique() % len(markers)],
            color="black", linewidth=3, markersize=7, label="Mean of seasons")
    ax.set_title(f"Snow and ice cover by season (Median NDSI > {area_df.attrs.get("threshold", 0.4)})",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Area (km²)")
    ax.set_xticks(sorted(area_df["year"].unique()))
    ax.grid(linestyle="--", alpha=0.6)
    ax.legend(title="Period", loc="lower left")
    fig.tight_layout()
    return fig


def plot_glacier_area_months(month_df: pd.DataFrame, year: int) -> matplotlib.figure.Figure:
    """Monthly snow/ice area bars. Needs [month, area_km2]; NaN is shown as 'No data'."""
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    area = month_df.set_index("month")["area_km2"].reindex(range(1, 13))
    ymax = np.nanmax(area.to_numpy()) if area.notna().any() else 1.0
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(names, area.fillna(0), color="#41b6c4", edgecolor="black", width=0.7)
    for i, val in enumerate(area):
        if np.isnan(val):
            ax.text(i, 0.02 * ymax, "No\ndata", ha="center", va="bottom", color="red", fontsize=9)
        else:
            ax.text(i, val + 0.02 * ymax, f"{val:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, ymax * 1.15)
    ax.set_title(f"Monthly snow and ice cover ({year})", fontsize=14, fontweight="bold")
    ax.set_ylabel("Area (km²)")
    ax.set_xlabel("Month")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


# --- gst_line ---------------------------------------------------------------
def map_gst_line_map(
    stack: ee.Image,
    roi: ee.Geometry,
    glacier_mask: ee.Geometry | None = None,
    line: ee.Geometry | None = None,
    zoom: int = 13,
    vmin: float = -25,
    vmax: float = 15,
) -> geemap.Map:
    """Seasonal mean LST layers (bands of `stack`) plus glacier outline and centerline."""
    vis = {"min": vmin, "max": vmax,
           "palette": ["040274", "2c7bb6", "00a6ca", "ffffbf", "fdae61", "d7191c"]}
    m = geemap.Map()
    m.centerObject(roi, zoom)
    for band in stack.bandNames().getInfo():
        m.addLayer(stack.select(band), vis, f"Mean LST - {band}")
    if glacier_mask is not None:
        m.addLayer(ee.FeatureCollection([ee.Feature(glacier_mask)]), {"color": "cyan"}, "GLIMS outline")
    if line is not None:
        m.addLayer(line, {"color": "black"}, "Centerline")
    return m


def plot_gst_line_profile(
    profile_df: pd.DataFrame,
    season_count: dict[str, int],
    glacier_end_m: float | None = None,
) -> matplotlib.figure.Figure:
    """Seasonal LST along the centerline. Needs [distance_m, Winter_DJF, ...]."""
    seasons = [("Winter_DJF", "Winter (DJF)", "#2c7bb6"), ("Spring_MAM", "Spring (MAM)", "#00a6ca"),
               ("Summer_JJA", "Summer (JJA)", "#d7191c"), ("Autumn_SON", "Autumn (SON)", "#fdae61")]
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    for col, label, color in seasons:
        if col in profile_df.columns:
            ax.plot(profile_df["distance_m"], profile_df[col], marker="o", markersize=4, linewidth=2,
                    color=color, label=f"{label} ($n_{{scenes}}={season_count.get(col, 0)}$)")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1, label="Melting point (0 °C)")
    if glacier_end_m is not None:
        ax.axvline(glacier_end_m, color="black", linestyle=":", label="End of glacier")
    ax.set_title("Seasonal surface temperature along glacier centerline")
    ax.set_xlabel("Distance along centerline from top to bottom (m)")
    ax.set_ylabel("Mean surface temperature (°C)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_gst_line_seasons(
    arrays: dict[str, np.ndarray],
    extent: tuple[float, float, float, float],
    season_count: dict[str, int],
    vmin: float = -25,
    vmax: float = 15,
) -> matplotlib.figure.Figure:
    """One LST map per season. `extent` = (lon_min, lon_max, lat_min, lat_max)."""
    labels = {"Winter_DJF": "Winter (DJF)", "Spring_MAM": "Spring (MAM)",
              "Summer_JJA": "Summer (JJA)", "Autumn_SON": "Autumn (SON)"}
    fig, axes = plt.subplots(1, len(arrays), figsize=(14, 5), dpi=300, sharey=True)
    axes = np.atleast_1d(axes)
    im = None
    for ax, (name, arr) in zip(axes, arrays.items()):
        im = ax.imshow(arr, cmap="RdYlBu_r", vmin=vmin, vmax=vmax, extent=extent, aspect="auto")
        ax.set_title(f"{labels.get(name, name)}\n($n={season_count.get(name, 0)}$ scenes)")
        ax.set_xlabel("Longitude (°E)")
    axes[0].set_ylabel("Latitude (°N)")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02).set_label("Mean surface temperature (°C)")
    fig.suptitle("Spatial variability of glacier surface temperature across seasons", y=1.05, fontsize=13)
    return fig


def plot_gst_line_timeseries(
    temp_df: pd.DataFrame,
    curve_df: pd.DataFrame,
    stats: dict[str, float],
) -> matplotlib.figure.Figure:
    """Scene LST scatter with linear trend (`lin_fit`) and seasonal-model curve."""
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    ax.scatter(temp_df["date"], temp_df["LST"], s=18, alpha=0.7, color="#2b5c8f",
               label=f"Scene mean LST ($n = {stats['n_scenes']}$)")
    ax.plot(temp_df["date"], temp_df["lin_fit"], color="#d7191c", linewidth=2,
            label=f"Linear trend ({stats['lin_per_decade']:+.2f} °C/decade, $R^2 = {stats['lin_r2']:.3f}$)")
    ax.plot(curve_df["date"], curve_df["fit"], "--", color="tab:green",
            label=f"Seasonal model ({stats['sea_per_decade']:+.2f} °C/decade, $R^2 = {stats['sea_r2']:.3f}$)")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Glacier surface temperature time series")
    ax.set_xlabel("Date")
    ax.set_ylabel("Mean surface temperature (°C)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


#------- glacier_pixel -----------------------------------
def plot_pixel_maps(pixel_arrays: dict[str, np.ndarray]) -> matplotlib.figure.Figure:
    """Four-panel map: mean ST, ST spread, fraction of scenes >= 0 deg C, number of observations."""
    panels = [("st_mean", "Mean ST (°C)", "RdYlBu_r"), ("st_std", "Std of ST (K)", "viridis"),
              ("frac_nonneg", "Share of scenes ST ≥ 0 °C", "Blues"), ("n_obs", "Observations (n)", "Greys")]
    fig, ax = plt.subplots(1, len(panels), figsize=(16, 6))
    for axis, (band, title, colormap) in zip(ax, panels):
        shown = axis.imshow(pixel_arrays[band], cmap=colormap)
        axis.set_title(title)
        axis.axis("off")
        fig.colorbar(shown, ax=axis, shrink=0.7)
    fig.tight_layout()
    return fig