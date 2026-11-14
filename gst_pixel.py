"""
Winter glacier surface temperature (GST) from Landsat 7/8/9 Collection 2 Level 2 in Earth Engine.

Two products, both with terrain-shadow masking:
  1. Zone series: scene x elevation-zone sums (3 equal-area zones + whole glacier) -> half-month
     series -> winter means/trends.  (download_zone_scenes -> process_zones -> winter_means)
  2. Per-pixel maps and point series.  (pixel_statistics -> image_to_arrays / download_point_series)

Quick start (needs ee.Initialize(project=...) first):

    scenes = download_zone_scenes(roi, glacier_mask, "2014-11-01", "2026-04-01")
    half_monthly = process_zones(scenes, debug=True)
    winters = winter_means(half_monthly)
    trends = winter_trends(winters)

    pixel_image = pixel_statistics(roi, glacier_mask, "2014-11-01", "2026-04-01")
    pixel_arrays = image_to_arrays(pixel_image, roi)
    plot_pixel_maps(pixel_arrays)

    points = download_point_series({"centre": (81.774, 30.220)}, "2014-11-01", "2026-04-01")

Needs: earthengine-api, pandas, numpy, scipy, matplotlib.
"""
from __future__ import annotations
import math
from collections.abc import Sequence
import ee
import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np
import pandas as pd
from scipy import stats
import os
from io_param_functions import *

# ----------------- configuration ------------------------------------------------------------
TARGET_CRS = "EPSG:32644"             # UTM 44N (78-84 deg E); metric grid needed by HillShadow
SCALE_M = 30                          # Landsat L2 products are resampled to 30 m
WINTER_MONTHS = (11, 12, 1, 2, 3)     # dry season; a winter is labelled by its January
N_ZONES = 3                           # equal-area elevation zones
MAX_ST_UNCERTAINTY_K = 5.0            # drop pixels whose ST_QA uncertainty exceeds this
# QA_PIXEL bits that remove a pixel: 1 dilated cloud, 2 cirrus, 3 cloud, 4 cloud shadow.
# Bright snow is often flagged as cirrus/dilated cloud; 0b11000 (cloud + shadow only) keeps more data.
CLOUD_MASK_BITS = 0b11110
MIN_SUN_ELEVATION_DEG = 25.0          # drop scenes with a very low sun (long shadows, drifting L7)
MIN_ILLUMINATION_COS = 0.1            # self-shadow threshold: cos(sun incidence angle) must exceed this
HILLSHADOW_SEARCH_PIXELS = 256        # how far (in pixels) terrain can shade a pixel: 256 * 30 m = 7.7 km
LANDSAT7_LAST_DATE = "2022-04-30"     # after this L7 flies a lower orbit with earlier overpasses
MIN_SCENE_ZONE_COVER = 0.25           # min. share of a zone that must be clear and sunlit in a scene
MIN_YEARS_FOR_OUTLIER_TEST = 6        # leave-one-out outlier test needs this many years
OUTLIER_SIGMA = 3.0
MIN_HALF_MONTHS_PER_WINTER = 5        # of the 10 half-months in Nov-Mar
MIN_PIXEL_OBSERVATIONS = 8            # per-pixel maps: min. clear, sunlit scenes

# Landsat 7 is LE07 and its thermal band is B6; Landsat 8/9 use B10.
_THERMAL_BAND_BY_COLLECTION: dict[str, str] = {
    "LANDSAT/LE07/C02/T1_L2": "ST_B6",
    "LANDSAT/LC08/C02/T1_L2": "ST_B10",
    "LANDSAT/LC09/C02/T1_L2": "ST_B10",
}
_SUM_COLUMNS = ["stsum", "st2sum", "uncsum", "valid", "shaded", "n"]



# ----------------------------------------------------------------------------- Earth Engine side
def _make_band_renamer(thermal_band: str):
    """Return a function that renames the sensor-specific thermal band to ST_RAW."""
    def rename(image: ee.Image) -> ee.Image:
        return image.select([thermal_band, "ST_QA", "QA_PIXEL"], ["ST_RAW", "ST_QA", "QA_PIXEL"])
    return rename


def _month_filter(months: Sequence[int]) -> ee.Filter:
    """Filter that keeps images acquired in any of the given calendar months."""
    return ee.Filter.Or(*[ee.Filter.calendarRange(month, month, "month") for month in months])


def _landsat(roi: ee.Geometry, start: str, end: str,
             months: Sequence[int] | None = WINTER_MONTHS) -> ee.ImageCollection:
    """Merged Landsat 7/8/9 L2 collection with harmonised band names (ST_RAW, ST_QA, QA_PIXEL)."""
    merged: ee.ImageCollection | None = None
    for collection_id, thermal_band in _THERMAL_BAND_BY_COLLECTION.items():
        last_date = min(end, LANDSAT7_LAST_DATE) if "LE07" in collection_id else end
        if last_date <= start:          # e.g. Landsat 7 after its cut-off date
            continue
        collection = (ee.ImageCollection(collection_id)
                      .filterBounds(roi)
                      .filterDate(start, last_date)
                      .filter(ee.Filter.gte("SUN_ELEVATION", MIN_SUN_ELEVATION_DEG)))
        if months:
            collection = collection.filter(_month_filter(months))
        collection = collection.map(_make_band_renamer(thermal_band))
        merged = collection if merged is None else merged.merge(collection)
    if merged is None:
        raise ValueError("No Landsat collection covers the requested period.")
    return merged


def _projected_dem() -> ee.Image:
    """SRTM 30 m on the metric UTM grid (HillShadow only works on projected, parallel-ray grids)."""
    return (ee.Image("USGS/SRTMGL1_003").select("elevation")
            .resample("bilinear").reproject(crs=TARGET_CRS, scale=SCALE_M))


def _illumination_mask(image: ee.Image, dem_projected: ee.Image) -> ee.Image:
    """1 where a pixel is sunlit at overpass: no cast shadow and facing the sun."""
    sun_azimuth_deg = ee.Number(image.get("SUN_AZIMUTH"))
    sun_zenith_deg = ee.Number(90).subtract(ee.Number(image.get("SUN_ELEVATION")))

    # Cast shadow from surrounding terrain: HillShadow returns 1 = illuminated, 0 = shadowed.
    not_cast_shadow = ee.Algorithms.HillShadow(
        image=dem_projected, azimuth=sun_azimuth_deg, zenith=sun_zenith_deg,
        neighborhoodSize=HILLSHADOW_SEARCH_PIXELS, hysteresis=False)

    # Self shadow: cosine of the angle between sun direction and the slope normal.
    degrees_to_radians = math.pi / 180
    zenith_rad = sun_zenith_deg.multiply(degrees_to_radians)
    azimuth_rad = sun_azimuth_deg.multiply(degrees_to_radians)
    slope_rad = ee.Terrain.slope(dem_projected).multiply(degrees_to_radians)
    aspect_rad = ee.Terrain.aspect(dem_projected).multiply(degrees_to_radians)
    cos_incidence = (slope_rad.cos().multiply(zenith_rad.cos())
                     .add(slope_rad.sin().multiply(zenith_rad.sin())
                          .multiply(ee.Image.constant(azimuth_rad).subtract(aspect_rad).cos())))
    facing_sun = cos_incidence.gt(MIN_ILLUMINATION_COS)
    return not_cast_shadow.eq(1).And(facing_sun).rename("lit")


def _prepare_scene(image: ee.Image, dem_projected: ee.Image) -> ee.Image:
    """Bands: ST (deg C) and ST_UNC (K), both masked to clear + sunlit pixels; SHADED (unmasked, 1 = shaded)."""
    clear = image.select("QA_PIXEL").bitwiseAnd(0b11110).eq(0)   # dilated cloud, cirrus, cloud, cloud shadow
    raw = image.select("ST_RAW")
    temperature = raw.multiply(0.00341802).add(149.0).subtract(273.15).rename("ST")
    uncertainty = image.select("ST_QA").multiply(0.01).rename("ST_UNC")
    lit = _illumination_mask(image, dem_projected)
    valid = clear.And(raw.gt(0)).And(uncertainty.lt(MAX_ST_UNCERTAINTY_K)).And(lit)
    shaded = lit.Not().toFloat().rename("SHADED")
    prepared = ee.Image.cat([temperature, uncertainty]).updateMask(valid).addBands(shaded)
    return prepared.copyProperties(image, ["system:time_start", "system:index", "SPACECRAFT_ID",
                                           "SUN_AZIMUTH", "SUN_ELEVATION"])


def _equal_area_zone_image(roi: ee.Geometry, glacier_mask: ee.Image, dem_projected: ee.Image,
                           n_zones: int) -> ee.Image:
    """Image with zone numbers 1 (lowest) ... n_zones (highest), each covering the same glacier area."""
    percentiles = [int(round(100 * index / n_zones)) for index in range(1, n_zones)]
    cut_values = (dem_projected.updateMask(glacier_mask)
                  .reduceRegion(reducer=ee.Reducer.percentile(percentiles), geometry=roi,
                                scale=SCALE_M, crs=TARGET_CRS, maxPixels=1e9)
                  .getInfo())
    thresholds = sorted(value for value in cut_values.values() if value is not None)
    zone_image = ee.Image.constant(1)
    for index, threshold in enumerate(thresholds):
        zone_image = zone_image.where(dem_projected.gte(threshold), index + 2)
    return zone_image.rename("zone").toInt()


def _scene_zone_features(prepared: ee.Image, zone_image: ee.Image, glacier_mask: ee.Image,
                         roi: ee.Geometry) -> ee.FeatureCollection:
    """One feature per zone with sums of ST, ST^2, uncertainty, valid/shaded/total pixel counts."""
    temperature = prepared.select("ST")
    stack = ee.Image.cat([
        temperature.unmask(0).rename("stsum"),
        temperature.unmask(0).pow(2).rename("st2sum"),
        prepared.select("ST_UNC").unmask(0).rename("uncsum"),
        temperature.mask().toFloat().rename("valid"),
        prepared.select("SHADED").rename("shaded"),
        ee.Image.constant(1).rename("n"),
        zone_image,
    ]).updateMask(glacier_mask)
    reducer = ee.Reducer.sum().forEach(_SUM_COLUMNS).group(groupField=len(_SUM_COLUMNS), groupName="zone")
    result = stack.reduceRegion(reducer=reducer, geometry=roi, crs=TARGET_CRS, scale=SCALE_M,
                                maxPixels=1e9, tileScale=4)
    scene_info = ee.Dictionary({"time": prepared.get("system:time_start"),
                                "scene": prepared.get("system:index"),
                                "sensor": prepared.get("SPACECRAFT_ID")})

    def to_feature(group: ee.Dictionary) -> ee.Feature:
        return ee.Feature(None, ee.Dictionary(group).combine(scene_info))

    return ee.FeatureCollection(ee.List(result.get("groups")).map(to_feature))


def download_zone_scenes(roi: ee.Geometry, glacier_mask: ee.Image, start: str, end: str,
                         months: Sequence[int] | None = WINTER_MONTHS,
                         n_zones: int = N_ZONES,
                         debug: bool = False) -> pd.DataFrame:
    """Scene x zone sums for all Landsat 7/8/9 scenes (one Earth Engine request per year)."""
    dem_projected = _projected_dem()
    zone_image = _equal_area_zone_image(roi, glacier_mask, dem_projected, n_zones)
    rows: list[dict] = []
    for year in range(int(start[:4]), int(end[:4]) + 1):
        year_start, year_end = max(f"{year}-01-01", start), min(f"{year + 1}-01-01", end)
        if year_end <= year_start:
            continue
        scenes = _landsat(roi, year_start, year_end, months).map(
            lambda image: _prepare_scene(image, dem_projected))
        features = (ee.FeatureCollection(scenes.map(
            lambda prepared: _scene_zone_features(prepared, zone_image, glacier_mask, roi)))
            .flatten().getInfo()["features"])
        rows += [feature["properties"] for feature in features]
        if debug:
            print(year, "scene x zone rows:", len(features))
    table = pd.DataFrame(rows)
    if table.empty:
        raise RuntimeError("No scenes returned; check dates, months and the region.")
    table["time"] = pd.to_datetime(table["time"], unit="ms")
    return table


def pixel_statistics(roi: ee.Geometry, glacier_mask: ee.Image, start: str, end: str,
                     months: Sequence[int] | None = WINTER_MONTHS,
                     min_observations: int = MIN_PIXEL_OBSERVATIONS) -> ee.Image:
    """
    Per-pixel statistics over all clear, sunlit scenes (lazy ee.Image on the UTM 30 m grid).

    Bands: st_mean, st_std (deg C), frac_nonneg (share of scenes with ST >= 0 deg C), n_obs.
    Only glacier pixels inside roi are kept (outline parts outside the polygon are clipped).
    Pixels with fewer than min_observations scenes are masked, except in n_obs, so the n_obs
    panel shows a larger area on purpose: it shows where data is missing.
    If getInfo times out, export it with ee.batch.Export.image.toDrive instead.
    """
    dem_projected = _projected_dem()
    glacier_in_roi = glacier_mask.clip(roi)
    temperature = (_landsat(roi, start, end, months)
                   .map(lambda image: _prepare_scene(image, dem_projected)).select("ST"))
    observation_count = temperature.count().rename("n_obs")
    enough_observations = observation_count.gte(min_observations)
    statistics = ee.Image.cat([
        temperature.mean().rename("st_mean"),
        temperature.reduce(ee.Reducer.stdDev()).rename("st_std"),
        temperature.map(lambda image: image.gte(0)).mean().rename("frac_nonneg"),
    ]).updateMask(enough_observations)
    result = statistics.addBands(observation_count).updateMask(glacier_in_roi)
    return result.reproject(crs=TARGET_CRS, scale=SCALE_M)


def image_to_arrays(image: ee.Image, roi: ee.Geometry, fill_value: float = -9999.0) -> dict[str, np.ndarray]:
    """Download a small image as {band: 2-D array}; masked pixels become NaN. Row 0 is north."""
    properties = (image.unmask(fill_value)
                  .sampleRectangle(region=roi.bounds(), defaultValue=fill_value)
                  .getInfo()["properties"])
    arrays = {name: np.array(values, dtype=float) for name, values in properties.items()}
    for array in arrays.values():
        array[array == fill_value] = np.nan
    return arrays


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


def download_point_series(points: dict[str, tuple[float, float]], start: str, end: str,
                          months: Sequence[int] | None = WINTER_MONTHS,
                          buffer_m: float = 50.0,
                          debug: bool = False) -> pd.DataFrame:
    """
    Per-scene ST at named points {name: (lon, lat)}: mean of clear, sunlit pixels within buffer_m.

    Columns: point, time, scene, sensor, ST, ST_UNC, SHADED (share of the buffer in shadow).
    Rows where the buffer is fully masked have no ST and are dropped.
    """
    point_collection = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([lon, lat]).buffer(buffer_m), {"point": name})
        for name, (lon, lat) in points.items()])
    dem_projected = _projected_dem()
    all_points_geometry = point_collection.geometry()
    rows: list[dict] = []

    def values_per_scene(prepared: ee.Image) -> ee.FeatureCollection:
        values = prepared.select(["ST", "ST_UNC", "SHADED"]).reduceRegions(
            collection=point_collection, reducer=ee.Reducer.mean(), scale=SCALE_M, crs=TARGET_CRS)
        return values.map(lambda feature: feature.set({
            "time": prepared.get("system:time_start"), "scene": prepared.get("system:index"),
            "sensor": prepared.get("SPACECRAFT_ID")}))

    for year in range(int(start[:4]), int(end[:4]) + 1):
        year_start, year_end = max(f"{year}-01-01", start), min(f"{year + 1}-01-01", end)
        if year_end <= year_start:
            continue
        scenes = _landsat(all_points_geometry, year_start, year_end, months).map(
            lambda image: _prepare_scene(image, dem_projected))
        features = ee.FeatureCollection(scenes.map(values_per_scene)).flatten().getInfo()["features"]
        rows += [feature["properties"] for feature in features]
        if debug:
            print(year, "scene x point rows:", len(features))
    table = pd.DataFrame(rows)
    if table.empty or "ST" not in table:
        raise RuntimeError("No valid point observations returned.")
    table["time"] = pd.to_datetime(table["time"], unit="ms")
    return table.dropna(subset=["ST"]).sort_values(["point", "time"]).reset_index(drop=True)

# ----------------------------------------------------------------------------- diagnostics
def count_available_scenes(roi: ee.Geometry, start: str, end: str,
                           months: Sequence[int] | None = WINTER_MONTHS) -> pd.DataFrame:
    """Scenes per sensor and tier (T1 is used by this module, T2 is not) before any filtering."""
    sizes: dict[str, ee.Number] = {}
    for sensor_code, sensor_name in (("LE07", "L7"), ("LC08", "L8"), ("LC09", "L9")):
        for tier in ("T1", "T2"):
            collection = (ee.ImageCollection(f"LANDSAT/{sensor_code}/C02/{tier}_L2")
                          .filterBounds(roi).filterDate(start, end))
            if months:
                collection = collection.filter(_month_filter(months))
            sizes[f"{sensor_name} {tier}"] = collection.size()
    counts = ee.Dictionary(sizes).getInfo()
    return pd.Series(counts, name="scenes").rename_axis("collection").reset_index()


def diagnose_masks(roi: ee.Geometry, glacier_mask: ee.Image, start: str, end: str,
                   months: Sequence[int] | None = WINTER_MONTHS) -> pd.DataFrame:
    """
    Per scene: share of glacier pixels in roi that pass each filter (1.0 = nothing removed).

    Columns: has_data, no_cloud, no_cloud_shadow, no_dilated_cloud, no_cirrus, uncertainty_ok,
    lit (not shaded by terrain), all_filters (what is actually used). The filter with the lowest
    median per sensor is the one that costs most data: table.groupby("sensor").median(numeric_only=True).
    """
    dem_projected = _projected_dem()
    glacier_in_roi = glacier_mask.clip(roi)

    def pass_shares(image: ee.Image) -> ee.Feature:
        quality = image.select("QA_PIXEL")
        raw = image.select("ST_RAW")
        uncertainty_ok = image.select("ST_QA").multiply(0.01).lt(MAX_ST_UNCERTAINTY_K)
        lit = _illumination_mask(image, dem_projected)
        clear = quality.bitwiseAnd(CLOUD_MASK_BITS).eq(0)
        checks = ee.Image.cat([
            raw.gt(0).rename("has_data"),
            quality.bitwiseAnd(0b1000).eq(0).rename("no_cloud"),
            quality.bitwiseAnd(0b10000).eq(0).rename("no_cloud_shadow"),
            quality.bitwiseAnd(0b10).eq(0).rename("no_dilated_cloud"),
            quality.bitwiseAnd(0b100).eq(0).rename("no_cirrus"),
            uncertainty_ok.rename("uncertainty_ok"),
            lit.rename("lit"),
            clear.And(raw.gt(0)).And(uncertainty_ok).And(lit).rename("all_filters"),
        ]).unmask(0).toFloat().updateMask(glacier_in_roi)
        shares = checks.reduceRegion(reducer=ee.Reducer.mean(), geometry=roi, crs=TARGET_CRS,
                                     scale=SCALE_M, maxPixels=1e9, tileScale=4)
        scene_info = ee.Dictionary({"time": image.get("system:time_start"),
                                    "scene": image.get("system:index"),
                                    "sensor": image.get("SPACECRAFT_ID")})
        return ee.Feature(None, ee.Dictionary(shares).combine(scene_info))

    rows: list[dict] = []
    for year in range(int(start[:4]), int(end[:4]) + 1):
        year_start, year_end = max(f"{year}-01-01", start), min(f"{year + 1}-01-01", end)
        if year_end <= year_start:
            continue
        scenes = _landsat(roi, year_start, year_end, months)
        features = ee.FeatureCollection(scenes.map(pass_shares)).getInfo()["features"]
        rows += [feature["properties"] for feature in features]
    table = pd.DataFrame(rows)
    table["time"] = pd.to_datetime(table["time"], unit="ms")
    return table.sort_values("time").reset_index(drop=True)


# ----------------------------------------------------------------------------- pandas side
def _half_month(time: pd.Series) -> pd.Series:
    """Half-month index 0..23 (two per calendar month, split at day 15)."""
    return (time.dt.month - 1) * 2 + (time.dt.day > 15).astype(int)


def _season_year(time: pd.Series) -> pd.Series:
    """Winter label = year of its January (Nov 2015 - Mar 2016 -> 2016). Assumes winter months only."""
    return time.dt.year + (time.dt.month >= 7).astype(int)


def _zone_label(zone: int, n_zones: int) -> str:
    """Readable zone name; 0 is the whole glacier."""
    if zone == 0:
        return "whole glacier"
    suffix = " (low)" if zone == 1 else " (high)" if zone == n_zones else ""
    return f"zone{zone}{suffix}"


def process_zones(scene_zone_table: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """
    Scene x zone sums (download_zone_scenes) -> half-month GST per zone and for the whole glacier.

    Steps: scene means -> coverage filter -> half-month composite (L7/L8/L9 averaged)
    -> leave-one-out 3-sigma outlier test per (zone, half-month) across years.
    The paper's neighbour-bin spike filter is dropped: three zones are not adjacent narrow bins.
    """
    table = scene_zone_table.copy()
    table["time"] = pd.to_datetime(table["time"])
    table["zone"] = table["zone"].astype(int)
    n_zones = int(table["zone"].max())

    whole_glacier = table.groupby(["scene", "time", "sensor"], as_index=False)[_SUM_COLUMNS].sum()
    whole_glacier["zone"] = 0
    table = pd.concat([table, whole_glacier], ignore_index=True)

    n_rows_before = len(table)
    table = table[table["valid"] > 0].copy()
    table["cover"] = table["valid"] / table["n"]
    table = table[table["cover"] >= MIN_SCENE_ZONE_COVER].copy()
    table["gst"] = table["stsum"] / table["valid"]
    table["gst_spatial_sd"] = np.sqrt((table["st2sum"] / table["valid"] - table["gst"] ** 2).clip(lower=0))
    table["gst_unc"] = table["uncsum"] / table["valid"]
    table["shaded_frac"] = table["shaded"] / table["n"]

    table["year"] = table["time"].dt.year
    table["hm"] = _half_month(table["time"])
    table["season_year"] = _season_year(table["time"])
    composite = (table.groupby(["season_year", "year", "hm", "zone"], as_index=False)
                 .agg(gst=("gst", "mean"), gst_spatial_sd=("gst_spatial_sd", "mean"),
                      gst_unc=("gst_unc", "mean"), cover=("cover", "mean"),
                      shaded_frac=("shaded_frac", "mean"), n_scenes=("scene", "nunique")))

    # leave-one-out 3-sigma test per (zone, half-month) across years
    keys = [composite["zone"], composite["hm"]]
    count = composite["gst"].groupby(keys).transform("count")
    total = composite["gst"].groupby(keys).transform("sum")
    total_squares = (composite["gst"] ** 2).groupby(keys).transform("sum")
    others_mean = (total - composite["gst"]) / (count - 1)
    others_variance = ((total_squares - composite["gst"] ** 2) / (count - 1) - others_mean ** 2).clip(lower=0)
    is_outlier = (count >= MIN_YEARS_FOR_OUTLIER_TEST) & (
        (composite["gst"] - others_mean).abs() > OUTLIER_SIGMA * np.sqrt(others_variance))
    composite = composite[~is_outlier].copy()

    composite["date"] = pd.to_datetime(dict(year=composite["year"], month=composite["hm"] // 2 + 1,
                                            day=np.where(composite["hm"] % 2 == 0, 8, 23)))
    composite["zone_name"] = composite["zone"].map(lambda zone: _zone_label(zone, n_zones))

    if debug:
        print(f"scene x zone rows kept after coverage filter: {len(table)} of {n_rows_before}")
        print(f"half-month values removed as outliers: {int(is_outlier.sum())}")
        print(composite.groupby("zone_name")[["cover", "shaded_frac", "n_scenes"]].mean().round(2))

    columns = ["date", "season_year", "year", "hm", "zone_name", "gst", "gst_spatial_sd", "gst_unc",
               "n_scenes", "cover", "shaded_frac"]
    return (composite.sort_values(["zone_name", "date"])[columns]
            .rename(columns={"zone_name": "zone"}).reset_index(drop=True))


def winter_means(half_monthly: pd.DataFrame,
                 min_half_months: int = MIN_HALF_MONTHS_PER_WINTER) -> pd.DataFrame:
    """
    Winter mean GST per zone. gst_anomaly removes the seasonal cycle first (half-month climatology),
    so a winter that only has November values is not biased warm. Use gst_anomaly for trends.
    """
    table = half_monthly.copy()
    climatology = table.groupby(["zone", "hm"])["gst"].transform("mean")
    table["gst_anomaly"] = table["gst"] - climatology
    winters = (table.groupby(["zone", "season_year"], as_index=False)
               .agg(gst=("gst", "mean"), gst_anomaly=("gst_anomaly", "mean"), gst_unc=("gst_unc", "mean"),
                    shaded_frac=("shaded_frac", "mean"), n_half_months=("gst", "size")))
    return winters[winters["n_half_months"] >= min_half_months].reset_index(drop=True)


def winter_trends(winters: pd.DataFrame, value_column: str = "gst_anomaly") -> pd.DataFrame:
    """OLS and Theil-Sen trend (K per decade) of winter means per zone; needs at least 4 winters."""
    rows = []
    for zone, group in winters.groupby("zone"):
        if len(group) < 4:
            continue
        ols = stats.linregress(group["season_year"], group[value_column])
        theil_sen_slope = stats.theilslopes(group[value_column], group["season_year"])[0]
        rows.append({"zone": zone, "n_winters": len(group),
                     "ols_per_decade": ols.slope * 10, "ols_p_value": ols.pvalue,
                     "theil_sen_per_decade": theil_sen_slope * 10})
    return pd.DataFrame(rows)


def main_gst_pixel(roi, glacier_mask, start, end, debug: bool = False):

    scenes = download_zone_scenes(roi, glacier_mask, start, end, debug=debug)
    half_monthly = process_zones(scenes, debug=debug)

    winters = winter_means(half_monthly)

    if debug:
        print(winter_trends(winters).round(3))

    pixel_arrays = image_to_arrays(pixel_statistics(roi, glacier_mask, start, end), roi)
    fig = plot_pixel_maps(pixel_arrays)
    fig.show()

    point_series_df = download_point_series({"centre": (81.774, 30.220)}, start, end, debug=debug)

    save_any_tables_and_figures(
        {"gst_scene_zone_sums": scenes,
         "gst_halfmonthly_winter": half_monthly,
         "gst_winter_means": winters,
         "gst_points": point_series_df,
         "gst_pixel_maps": fig,
         },
        save_path = "results/gst_pixel/"
    )