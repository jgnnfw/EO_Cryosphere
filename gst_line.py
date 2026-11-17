"""Glacier surface temperature (Landsat 8/9 LST): seasonal maps, centerline profile, time series."""
import ee
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from io_param_functions import lstsq_sin_for_years, myprint

L8 = "LANDSAT/LC08/C02/T1_L2"
L9 = "LANDSAT/LC09/C02/T1_L2"

SEASON_MONTHS: dict[str, tuple[int, int]] = {
    "Winter_DJF": (12, 2),
    "Spring_MAM": (3, 5),
    "Summer_JJA": (6, 8),
    "Autumn_SON": (9, 11),
}


# --- download ---------------------------------------------------------------
def mask_lst(img: ee.Image) -> ee.Image:
    """Band 'LST' in °C from ST_B10, cloud (bit 3) and shadow (bit 4) masked."""
    qa = img.select("QA_PIXEL")
    ok = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
    lst = img.select("ST_B10").multiply(0.00341802).add(149.0).subtract(273.15).rename("LST")
    return lst.updateMask(ok).copyProperties(img, ["system:time_start"])


def download_lst(
    roi: ee.Geometry,
    start: str = "2014-09-01",
    end: str = "2026-12-31",
    max_cloud: float = 60,
    debug: bool = False,
) -> ee.ImageCollection:
    """Landsat 8+9 land surface temperature collection (°C), cloud-masked."""
    col = (
        ee.ImageCollection(L8).filterBounds(roi).filterDate(start, end)
        .merge(ee.ImageCollection(L9).filterBounds(roi).filterDate(start, end))
        .filter(ee.Filter.lt("CLOUD_COVER", max_cloud))
        .map(mask_lst)
    )
    myprint(debug, f"LST scenes {start}..{end}: {col.size().getInfo()}")
    return col


def get_centerline(
    default_coords: list[list[float]],
    user_roi: ee.Geometry | None = None,
    debug: bool = False,
) -> ee.Geometry:
    """Hand-drawn LineString (e.g. `m.user_roi`) if given, else default coordinates."""
    if user_roi is not None and user_roi.type().getInfo() == "LineString":
        myprint(debug, "Using user-drawn centerline.")
        return ee.Geometry.LineString(user_roi.coordinates(), "EPSG:4326", True)
    myprint(debug, "Using default centerline.")
    return ee.Geometry.LineString(default_coords)


# --- process ----------------------------------------------------------------
def season_stack(lst: ee.ImageCollection, roi: ee.Geometry) -> ee.Image:
    """Mean LST per season as one multi-band image (bands = SEASON_MONTHS keys)."""
    bands = [
        lst.filter(ee.Filter.calendarRange(m0, m1, "month")).mean().clip(roi).rename(name)
        for name, (m0, m1) in SEASON_MONTHS.items()
    ]
    return ee.Image.cat(bands)


def season_count(lst: ee.ImageCollection, debug: bool = False) -> dict[str, int]:
    """Number of scenes per season."""
    counts = ee.Dictionary({
        name: lst.filter(ee.Filter.calendarRange(m0, m1, "month")).size()
        for name, (m0, m1) in SEASON_MONTHS.items()
    }).getInfo()
    myprint(debug, f"Scenes per season: {counts}")
    return counts


def study_region(roi: ee.Geometry, glacier_mask: ee.Geometry | None = None) -> ee.Geometry:
    """roi ∩ glacier_mask, or roi if no mask."""
    return roi if glacier_mask is None else glacier_mask.intersection(roi, 1)


def temp_profile(
    stack: ee.Image,
    line: ee.Geometry,
    step_m: float = 150,
    scale: float = 30,
    debug: bool = False,
) -> pd.DataFrame:
    """Seasonal mean LST sampled every `step_m` along line: [distance_m, <season>...]."""
    length = line.length(1).getInfo()
    cuts = ee.List.sequence(step_m, length, step_m)
    segs = line.cutLines(cuts, 1).geometries()

    def make_point(i):
        i = ee.Number(i)
        pt = ee.Geometry.Point(ee.Geometry(segs.get(i)).coordinates().get(0))
        return ee.Feature(pt, {"distance_m": i.multiply(step_m)})

    pts = ee.FeatureCollection(ee.List.sequence(0, segs.length().subtract(1)).map(make_point))
    fc = stack.reduceRegions(collection=pts, reducer=ee.Reducer.mean(), scale=scale)
    rows = [f["properties"] for f in fc.getInfo()["features"]]
    myprint(debug, f"Profile points: {len(rows)}, line length: {length:.0f} m")
    return pd.DataFrame(rows).sort_values("distance_m").reset_index(drop=True)


def season_arrays(stack: ee.Image, roi: ee.Geometry, scale: float = 30) -> dict[str, np.ndarray]:
    """Per-season 2D temperature arrays (NaN = no data) on a regular grid over roi."""
    grid = stack.unmask(-9999).reproject(crs="EPSG:4326", scale=scale)
    props = grid.sampleRectangle(region=roi, defaultValue=-9999).getInfo()["properties"]
    arrays = {}
    for name in SEASON_MONTHS:
        arr = np.array(props[name], dtype=float)
        arr[arr == -9999] = np.nan
        arrays[name] = arr
    return arrays


def roi_extent(roi: ee.Geometry) -> tuple[float, float, float, float]:
    """(lon_min, lon_max, lat_min, lat_max) of roi bounds."""
    ring = roi.bounds().coordinates().getInfo()[0]
    lons, lats = [p[0] for p in ring], [p[1] for p in ring]
    return min(lons), max(lons), min(lats), max(lats)


def scene_temp_table(
    lst: ee.ImageCollection,
    region: ee.Geometry,
    scale: float = 30,
    debug: bool = False,
) -> pd.DataFrame:
    """Per-scene mean LST over region: [date, LST], sorted by date."""
    def extract(img):
        mean = img.reduceRegion(ee.Reducer.mean(), region, scale, maxPixels=1e8)
        return ee.Feature(None, {"date": img.date().format("YYYY-MM-dd"), "LST": mean.get("LST")})

    feats = lst.map(extract).filter(ee.Filter.notNull(["LST"])).getInfo()["features"]
    temp_df = pd.DataFrame([f["properties"] for f in feats])
    temp_df["date"] = pd.to_datetime(temp_df["date"])
    myprint(debug, f"Scenes with LST: {len(temp_df)}")
    return temp_df.sort_values("date").reset_index(drop=True)


def fit_temp_trend(
    temp_df: pd.DataFrame,
    n_curve: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Linear and seasonal (trend + annual sine) fits of the LST series.

    Returns (temp_df with 'lin_fit', smooth seasonal curve [date, fit], stats dict).
    """
    x = mdates.date2num(temp_df["date"])
    y = temp_df["LST"].to_numpy(dtype=float)

    slope, icpt = np.polyfit(x, y, 1)
    lin = slope * x + icpt
    ss_tot = np.sum((y - y.mean()) ** 2)
    lin_r2 = 1 - np.sum((y - lin) ** 2) / ss_tot

    a, b, sin_c, cos_c, sea_r2 = lstsq_sin_for_years(x, y)
    xs = np.linspace(x[0], x[-1], n_curve)
    yrs = (xs - x.min()) / 365.2425
    fit = a + b * yrs + sin_c * np.sin(2 * np.pi * yrs) + cos_c * np.cos(2 * np.pi * yrs)

    fit_df = temp_df.assign(lin_fit=lin)
    curve_df = pd.DataFrame({"date": pd.to_datetime(mdates.num2date(xs)).tz_localize(None), "fit": fit})
    stats = {
        "lin_per_decade": slope * 365.25 * 10,
        "lin_r2": lin_r2,
        "sea_per_decade": b * 10,
        "sea_r2": sea_r2,
        "n_scenes": len(temp_df),
    }
    return fit_df, curve_df, stats
