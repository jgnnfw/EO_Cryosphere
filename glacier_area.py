"""Glacier snow/ice area from Sentinel-2 NDSI (Google Earth Engine)."""
from typing import Literal

import ee
import pandas as pd
from io_param_functions import myprint, study_region

S2_TOA = "COPERNICUS/S2_HARMONIZED"
S2_BOA = "COPERNICUS/S2_SR_HARMONIZED" # atmosphere corrected


def month_window(year: int, m0: int, m1: int) -> tuple[str, str]:
    """ISO start (inclusive) and end (exclusive) dates covering months m0..m1."""
    start = pd.Timestamp(year, m0, 1)
    end = pd.Timestamp(year, m1, 1) + pd.offsets.MonthEnd(0)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def area_region(roi: ee.Geometry, glacier_mask: ee.Geometry | None = None) -> ee.Geometry:
    """Region in which area is measured: roi, or roi ∩ glacier_mask if given."""
    return roi if glacier_mask is None else roi.intersection(glacier_mask, 1)


# --- download ---------------------------------------------------------------
def download_s2(roi: ee.Geometry, start: str, end: str,
                max_cloud: float = 40, collection: str | ee.ImageCollection = S2_BOA,
                debug: bool = False) -> ee.Image | None:
    """Sentinel-2 collection for [start, end); None if no scene is left."""
    if isinstance(collection, str):
        img_coll = ee.ImageCollection(collection)
    elif isinstance(collection, ee.ImageCollection):
        img_coll = collection
    else:
        raise(TypeError("Collection not of type str or ee.ImageCollection."))
    col = (img_coll
           .filterBounds(roi)
           .filterDate(start, end)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud)))
    n_scenes = col.size().getInfo()
    myprint(debug, f"{start} .. {end}: {n_scenes} scenes")
    return col if n_scenes > 0 else None


def download_snow_masks(
    roi: ee.Geometry,
    years: list[int],
    months: tuple[int, int] = (8, 9),
    max_cloud: float = 40,
    thresh: float = 0.4,
    collection: str | ee.ImageCollection = S2_BOA,
    debug: bool = False,
) -> dict[int, ee.Image]:
    """Yearly NDSI snow/ice masks (one composite per year over `months`)."""
    masks: dict[int, ee.Image] = {}
    for year in years:
        start, end = month_window(year, *months)
        img_coll = download_s2(roi, start, end, max_cloud, collection, debug)
        if img_coll is None:
            myprint(debug, f"{year}: no data")
            continue
        masks[year] = ndsi_mask(img_coll, roi, thresh)
    return masks


# --- process ----------------------------------------------------------------
def ndsi_mask(img_coll: ee.ImageCollection, roi: ee.Geometry, thr: float = 0.4) -> ee.Image:
    """Binary snow/ice mask (NDSI > thr) clipped to roi, band 'NDSI'."""
    def _ndsi(img: ee.Image) -> ee.Image:
        return img.normalizedDifference(["B3", "B11"]).rename("NDSI")
    ndsi = img_coll.map(_ndsi).median()
    return ndsi.gt(thr).selfMask().clip(roi)


def mask_area_km2(mask: ee.Image, region: ee.Geometry, scale: float = 20) -> float | None:
    """Area (km²) of a masked image within region; None if empty."""
    stats = mask.multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=region, scale=scale, maxPixels=1e9
    )
    area_m2 = stats.get("NDSI").getInfo()
    return None if area_m2 is None else area_m2 / 1e6


def area_window_km2(
    roi: ee.Geometry,
    start: str,
    end: str,
    max_cloud: float = 50,
    thresh: float = 0.4,
    glacier_mask: ee.Geometry | None = None,
    collection: str | ee.ImageCollection = S2_BOA,
    debug: bool = False,
) -> float | None:
    """Snow/ice area (km²) from the median composite of one time window."""
    region = area_region(roi, glacier_mask)
    img_coll = download_s2(region, start, end, max_cloud, collection, debug)
    if img_coll is None:
        return None
    return mask_area_km2(ndsi_mask(img_coll, region, thresh), region)


def area_by_season(
    roi: ee.Geometry,
    years: list[int],
    seasons: dict[str, tuple[int, int]] | None = None,
    max_cloud: float = 50,
    thresh: float = 0.4,
    glacier_mask: ee.Geometry | None = None,
    collection: str | ee.ImageCollection = S2_BOA,
    debug: bool = False,
) -> pd.DataFrame:
    """Table [season, year, area_km2] (NaN where no data)."""

    if seasons is None:
        seasons = {
        "Jan - Mar": (1, 3),
        "Apr - Jun": (4, 6),
        "Jul - Sep": (7, 9),
        "Oct - Dec": (10, 12),
    }

    rows = []
    for season, (m0, m1) in seasons.items():
        for year in years:
            start, end = month_window(year, m0, m1)
            area = area_window_km2(roi, start, end, max_cloud, thresh, glacier_mask, collection, debug)
            myprint(debug, f"{season} {year}: {area}")
            rows.append({"season": season, "year": year, "area_km2": area})
    result = pd.DataFrame(rows).astype({"area_km2": float})
    result.attrs["max_cloud"] = max_cloud
    result.attrs["threshold"] = thresh
    return result


def area_by_month(
    roi: ee.Geometry,
    year: int,
    max_cloud: float = 60,
    thresh: float = 0.4,
    glacier_mask: ee.Geometry | None = None,
    collection: str | ee.ImageCollection = S2_BOA,
    debug: bool = False,
) -> pd.DataFrame:
    """Table [month, area_km2] for one year (NaN where no data)."""
    rows = []
    for month in range(1, 13):
        start, end = month_window(year, month, month)
        area = area_window_km2(roi, start, end, max_cloud, thresh, glacier_mask, collection, debug)
        myprint(debug, f"{year}-{month:02d}: {area}")
        rows.append({"month": month, "area_km2": area})
    return pd.DataFrame(rows).astype({"area_km2": float})


def glims_area_by_year(glims: ee.FeatureCollection, roi: ee.Geometry,
                       year_field: Literal["src_date", "anlys_time"] = "src_date",
                       error_margin: float = 10, debug: bool = False,
                       min_year_inclusive: float | None = None) -> pd.DataFrame:
    """Glacier area [km²] per year: GLIMS outlines intersected with roi."""
    glims = glims.filterBounds(roi)
    feats = glims.toList(glims.size())

    area_by_year: dict[int, float] = {}
    for i in range(glims.size().getInfo()):
        f = ee.Feature(feats.get(i))
        geom = study_region(f.geometry(), roi)
        date = f.get(year_field).getInfo()
        area = geom.area(error_margin).divide(1e6).getInfo()
        if date is None or area is None or area < 1e-6:
            continue
        year = int(str(date)[:4])
        if min_year_inclusive is not None and year >= min_year_inclusive:
            area_by_year[year] = area_by_year.get(year, 0.0) + area

        myprint(debug, f"{year}: {area:.2f}; ", end="")

    myprint(debug, "\nTotal:",  *[f"{year}: {area:.2f};" for year, area in sorted(area_by_year.items())])

    return pd.DataFrame(sorted(area_by_year.items()), columns=["year", "area"])