"""Sentinel-1 wet/dry snow classification (ratio to a dry-snow winter reference, Nagler-style).

Principle: wet snow absorbs C-band radar, so backscatter drops by several dB compared with
dry snow. Per pixel: diff = sigma0_scene[dB] - sigma0_dry_ref[dB]; wet if diff < thr (-2 dB).
The reference is the median of the scenes of the reference months of the same season, same orbit
direction and same relative orbit, so viewing geometry cancels out. A DEM-based layover/shadow/slope
mask removes pixels where the SAR geometry is unreliable.

Months are chosen by the user: `ref_months` (dry reference) and `wet_months` (months classified).
Both are given in season order, the season starts with the first reference month. Season year `sy`:
months >= first reference month belong to sy-1 if the season wraps over New Year, all others to sy.
Default: ref Dec-Feb, evaluated Dec-May, so sy = Dec (sy-1) ... May sy.
"""
from typing import Literal
from plotting import *
import math
import ee
import numpy as np
import pandas as pd
from io_param_functions import myprint

ORBITS = ("ASCENDING", "DESCENDING")
# approx. azimuth (deg from north) pointing from the ground towards the satellite at ~28N
SENSOR_AZ = {"ASCENDING": 261.0, "DESCENDING": 99.0}
MONTHS: tuple[int, ...] = (12, 1, 2, 3, 4, 5)
REF_MONTHS: tuple[int, ...] = (12, 1, 2)
WET_MONTHS: tuple[int, ...] = MONTHS


# --------------- season / months ----------------------------------------------------------
def season_months(ref_months: tuple[int, ...] = REF_MONTHS, wet_months: tuple[int, ...] = WET_MONTHS) -> tuple[int, ...]:
    """All months of the season in chronological order, starting with the first reference month."""
    first = ref_months[0]
    return tuple(sorted(set(ref_months) | set(wet_months), key=lambda m: (m - first) % 12))


def season_window(sy: int, ref_months: tuple[int, ...] = REF_MONTHS,
                  wet_months: tuple[int, ...] = WET_MONTHS) -> tuple[str, str]:
    """(start, end) dates 'YYYY-MM-DD' of season year `sy` (end exclusive)."""
    ms, first = season_months(ref_months, wet_months), ref_months[0]
    wraps = any(m < first for m in ms)

    def _year(m: int) -> int:
        return sy - 1 if (wraps and m >= first) else sy

    start = pd.Timestamp(_year(ms[0]), ms[0], 1)
    end = pd.Timestamp(_year(ms[-1]), ms[-1], 1) + pd.DateOffset(months=1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _month_filter(months: tuple[int, ...]) -> ee.Filter:
    """Filter for images acquired in any of `months`."""
    return ee.Filter.Or(*[ee.Filter.calendarRange(m, m, "month") for m in months])


# --------------- data / masks --------------------------------------------------------------
def get_dem() -> ee.Image:
    """30 m NASADEM elevation."""
    return ee.Image("NASA/NASADEM_HGT/001").select("elevation")


def terrain_mask(orbit: str, inc: float = 39.0, lim: float = 10.0, slope_max: float = 45.0) -> ee.Image:
    """1 where layover, shadow and steep slopes do not spoil the signal (approximate geometry)."""
    dem = get_dem()
    slope = ee.Terrain.slope(dem)
    asp = ee.Terrain.aspect(dem)
    rad = math.pi / 180
    # slope angle in the range plane (>0 = facing the sensor)
    alpha = (slope.multiply(rad).tan()
             .multiply(asp.subtract(SENSOR_AZ[orbit]).multiply(rad).cos())
             .atan().divide(rad))
    loc = ee.Image.constant(inc).subtract(alpha)  # local incidence angle
    return loc.gt(lim).And(loc.lt(90 - lim)).And(slope.lt(slope_max))


def prep(img: ee.Image) -> ee.Image:
    """Mask border noise and reduce speckle with a small median filter (dB)."""
    out = img.updateMask(img.gt(-30)).focal_median(30, "circle", "meters")
    return out.copyProperties(img, ["system:time_start", "relativeOrbitNumber_start"])


def get_s1(roi: ee.Geometry, start: str, end: str, orbit: str, pol: str = "VV") -> ee.ImageCollection:
    """Sentinel-1 IW GRD (dB) of one orbit direction, speckle-filtered."""
    col = (ee.ImageCollection("COPERNICUS/S1_GRD")
           .filterBounds(roi)
           .filterDate(start, end)
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.eq("orbitProperties_pass", orbit))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", pol))
           .select(pol))
    return col.map(prep)


# --------------- classification --------------------------------------------------------------
def wet_coll(roi: ee.Geometry, sy: int, orbit: str, pol: str = "VV", thr: float = -2.0,
             ref_min: float = -20.0, min_ref: int = 3,
             ref_months: tuple[int, ...] = REF_MONTHS,
             wet_months: tuple[int, ...] = WET_MONTHS) -> ee.ImageCollection:
    """Daily classified mosaics of `wet_months` of season `sy`: bands `wet` (1/0) and `diff` (dB vs. dry ref)."""
    start, end = season_window(sy, ref_months, wet_months)
    col = get_s1(roi, start, end, orbit, pol)
    ref_col = col.filter(_month_filter(ref_months))
    eval_col = col.filter(_month_filter(wet_months))
    tmask = terrain_mask(orbit)
    empty = ee.Image.constant(0).rename(pol).updateMask(ee.Image(0))

    def _cls(img: ee.Image) -> ee.Image:
        ro = img.get("relativeOrbitNumber_start")
        rc = ref_col.filter(ee.Filter.eq("relativeOrbitNumber_start", ro))
        ref = ee.Image(ee.Algorithms.If(rc.size().gte(min_ref), rc.median(), empty))
        diff = img.subtract(ref).rename("diff")
        ok = ref.gt(ref_min).And(tmask)  # drop shadow / noise-floor pixels and bad geometry
        out = diff.lt(thr).rename("wet").addBands(diff).updateMask(ok)
        return (out.copyProperties(img,["system:time_start", "relativeOrbitNumber_start"])
                .set("d", img.date().format("YYYY-MM-dd"))
                .set("ref_count", rc.size())
                .set("ref_available", rc.size().gte(min_ref)))

    out = eval_col.map(_cls).filter(ee.Filter.gte("ref_count", min_ref))  # no usable reference -> drop
    days = out.aggregate_array("d").distinct()

    def _mos(d: ee.String) -> ee.Image:
        g = out.filter(ee.Filter.eq("d", d))
        return g.mosaic().copyProperties(
            g.first(), ["system:time_start", "relativeOrbitNumber_start", "d", "ref_count"])

    return ee.ImageCollection.fromImages(days.map(_mos))


def wet_freq(coll: ee.ImageCollection, month: int) -> ee.Image:
    """Fraction of scenes in `month` classified wet (0 = always dry, 1 = always wet)."""
    return coll.filter(ee.Filter.calendarRange(month, month, "month")).select("wet").mean()


def melt_onset(coll: ee.ImageCollection, sy: int, ref_months: tuple[int, ...] = REF_MONTHS,
               wet_months: tuple[int, ...] = WET_MONTHS) -> ee.Image:
    """Days since season start (1st of first month) of the first wet detection per pixel."""
    t0 = ee.Date(season_window(sy, ref_months, wet_months)[0])

    def _d(img: ee.Image) -> ee.Image:
        return ee.Image.constant(img.date().difference(t0, "day")).updateMask(img.select("wet")).rename("onset")

    return coll.map(_d).min()


# --------------- statistics --------------------------------------------------------------
def _annotate(d: pd.DataFrame, sy: int, start: str, ref_months: tuple[int, ...] = REF_MONTHS) -> pd.DataFrame:
    d["date"] = pd.to_datetime(d["date"])
    d["sy"] = sy
    d["month"] = d["date"].dt.month
    d["season"] = np.where(d["month"].isin(ref_months), "ref", "eval")
    d["day"] = (d["date"] - pd.Timestamp(start)).dt.days
    return d


def season_stats(roi: ee.Geometry, sy: int, orbit: str, glacier_mask: ee.Image | None = None,
                 pol: str = "VV", thr: float = -2.0, scale: float = 30,
                 band_h: float = 250.0, debug: bool = False,
                 ref_months: tuple[int, ...] = REF_MONTHS,
                 wet_months: tuple[int, ...] = WET_MONTHS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-date wet fraction (ROI, optional glacier) and per-date/elevation-band wet fraction."""

    myprint(debug, f"sy {sy} {orbit}: ", end="")
    coll = wet_coll(roi, sy, orbit, pol, thr, ref_months=ref_months, wet_months=wet_months)
    start = season_window(sy, ref_months, wet_months)[0]

    eb = get_dem().divide(band_h).floor().multiply(band_h).rename("elev")
    grp = (ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True)
           .group(1, "elev"))

    kw = dict(scale=scale, maxPixels=int(1e9), bestEffort=True, tileScale=4)

    def _st(img: ee.Image) -> ee.Feature:
        wet = img.select("wet")
        info = wet.addBands(img.select("diff")).addBands(wet.mask().unmask(0).rename("valid"))
        # 1. Statistics over the entire ROI
        m = info.reduceRegion(reducer=ee.Reducer.mean(), geometry=roi, **kw)
        props = {"date": img.get("d"), "rel_orbit": img.get("relativeOrbitNumber_start"),
                 "n_ref": img.get("ref_count"),
                 "wet": m.get("wet"), "diff": m.get("diff"), "valid": m.get("valid")}
        # 2. Statistics over glacier pixels only
        if glacier_mask is not None:
            glacier_info = info.updateMask(glacier_mask.eq(1))
            g = glacier_info.reduceRegion(reducer=ee.Reducer.mean(), geometry=roi, **kw)
            props.update({"g_wet": g.get("wet"), "g_diff": g.get("diff"), "g_valid": g.get("valid")})
        # 3. Statistics by elevation band over entire ROI
        props["groups"] = wet.addBands(eb).reduceRegion(reducer=grp, geometry=roi, **kw).get("groups")
        return ee.Feature(None, props)

    feats = ee.FeatureCollection(coll.map(_st)).getInfo()["features"]

    rows, brows = [], []
    bcols = ["date", "orbit", "rel_orbit", "elev", "wet", "n_px"]
    for f in feats:
        p = f["properties"]
        rows.append({k: v for k, v in p.items() if k != "groups"} | {"orbit": orbit})
        for group in p.get("groups") or []:
            if "mean" not in group:  # band without valid pixels
                continue
            brows.append({"date": p["date"], "orbit": orbit, "rel_orbit": p.get("rel_orbit"),
                          "elev": group["elev"], "wet": group["mean"], "n_px": group["count"]})

    myprint(debug, f"{len(rows)} dates, {len(brows)} brows")

    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    return (_annotate(pd.DataFrame(rows), sy, start, ref_months),
            _annotate(pd.DataFrame(brows, columns=bcols), sy, start, ref_months))


def all_stats(roi: ee.Geometry, start: str, end: str,
              glacier_mask: ee.Geometry | None = None,
              orbits: tuple[str, ...] = ORBITS, pol: str = "VV",
              thr: float = -2.0, scale: float = 30,
              band_h: float = 250.0, min_valid: float = 0.3,
              debug: bool = False,
              ref_months: tuple[int, ...] = REF_MONTHS,
              wet_months: tuple[int, ...] = WET_MONTHS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run `season_stats` for all season years touching start..end and both orbit directions."""
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    dfs, bdfs = [], []
    for sy in range(t0.year, t1.year + 2):
        w0, w1 = (pd.Timestamp(x) for x in season_window(sy, ref_months, wet_months))
        if w1 <= t0 or w0 > t1:  # season does not touch start..end
            continue
        for orb in orbits:
            try:
                d, b = season_stats(roi, sy, orb, glacier_mask, pol, thr, scale, band_h, debug,
                                    ref_months=ref_months, wet_months=wet_months)
            except ee.EEException as e:
                myprint(debug, f"sy {sy} {orb} failed: {e}")
                continue
            if len(d):
                dfs.append(d)
                bdfs.append(b)
    if not dfs:
        return pd.DataFrame(), pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    bdf = pd.concat(bdfs, ignore_index=True)
    df = df[df["valid"] >= min_valid].reset_index(drop=True)  # drop scenes barely covering the roi
    bdf = bdf.merge(df[["date", "orbit"]].drop_duplicates(), on=["date", "orbit"])
    return df, bdf


def monthly_mean_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean wet fraction and backscatter change by season year, month and orbit."""
    if df.empty:
        return pd.DataFrame(columns=["sy", "month", "orbit", "wet", "diff", "n"])

    cols = [c for c in ("wet", "diff", "g_wet", "g_diff") if c in df]
    out = df.groupby(["sy", "month", "orbit"])[cols].mean()
    out["n"] = df.groupby(["sy", "month", "orbit"]).size()

    return out.reset_index()


def show_sar_snow_maps(roi: ee.Geometry, sy: int, orbit: Literal["ASCENDING", "DESCENDING"],
                       months: tuple[int, ...] | None = None, glacier_mask: ee.Geometry | None = None,
                       ref_months: tuple[int, ...] = REF_MONTHS,
                       wet_months: tuple[int, ...] = WET_MONTHS) -> geemap.Map:
    """Interactive map; `months` (layers) default to `wet_months`."""
    months = months or wet_months
    coll = wet_coll(roi, sy, orbit, ref_months=ref_months, wet_months=wet_months)
    wet_freqs = {mo:wet_freq(coll, mo) for mo in months}
    m_onset = melt_onset(coll, sy, ref_months, wet_months)
    return plot_sar_snow_maps(roi, coll, wet_freqs, m_onset, sy, glacier_mask=glacier_mask)


def show_sar_snow_diff_history(roi: ee.Geometry, sy: int, orbit: Literal["ASCENDING", "DESCENDING"],
                               ref_months: tuple[int, ...] = REF_MONTHS,
                               wet_months: tuple[int, ...] = WET_MONTHS, **kwargs):
    kwargs.setdefault("months", wet_months)
    coll = wet_coll(roi, sy, orbit, ref_months=ref_months, wet_months=wet_months)
    return plot_sar_snow_diff_hist(roi, coll=coll, **kwargs)


def show_sar_snow_month_images(roi: ee.Geometry, sy: int, orbit: Literal["ASCENDING", "DESCENDING"],
                               months: tuple[int, ...] | None = None, glims: ee.Geometry | None = None,
                               ref_months: tuple[int, ...] = REF_MONTHS,
                               wet_months: tuple[int, ...] = WET_MONTHS, **kwargs):
    """Wet frequency image per month of season year `sy`; `months` default to `wet_months`.
    `glims` = result of get_latest_glims(roi)."""
    coll = wet_coll(roi, sy, orbit, ref_months=ref_months, wet_months=wet_months)
    return plot_sar_snow_month_images(roi, coll, sy, orbit, months or wet_months, glims=glims, **kwargs)