import os
import ee
import pandas as pd
import matplotlib
import numpy as np

def myprint(debug: bool = False, *arg, **kwarg):
    if debug:
        print(*arg, **kwarg)


def save_any_tables_and_figures(tables_and_figs: dict[str, pd.DataFrame | matplotlib.figure.Figure], save_path: str = ".", debug: bool = False) -> None:
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    myprint(debug, "\nSaving: ", end="")

    for name, item in tables_and_figs.items():
        if isinstance(item, pd.DataFrame):
            item.to_csv(os.path.join(save_path, name + ".csv"), index=False)
            myprint(debug, f"{name}.csv", end=" ")
        elif isinstance(item, matplotlib.figure.Figure):
            item.savefig(os.path.join(save_path, name + ".png"), dpi=300)
            myprint(debug, f"{name}.png", end=" ")
        else:
            myprint(debug, f"\nNot Saved: {name} of type {type(item)}\nSaving: ", end="")
            continue

    myprint(debug, f"\nDone Saving.")


def lstsq_sin_for_years(x, y):
    """
    Least-squares adjustment of:
        y = a + b*x + A*sin(2*pi*x) + B*cos(2*pi*x)

    x contains dates as datetime64 values or Matplotlib date numbers.
    The period is fixed to one year.
    """

    x = np.asarray(x)
    y = np.asarray(y, dtype=float)

    # Convert dates to elapsed days
    if np.issubdtype(x.dtype, np.datetime64):
        x_days = (x - x.min()).astype('timedelta64[D]').astype(float)
    else:
        x_days = x.astype(float) - np.min(x.astype(float))

    # Convert days to years
    x_years = x_days / 365.2425

    s = 2 * np.pi

    M = np.column_stack([
        np.ones_like(x_years),
        x_years,
        np.sin(s * x_years),
        np.cos(s * x_years)
    ])

    params, residuals, rank, singular_values = np.linalg.lstsq(
        M, y, rcond=None
    )

    a, b, A, B = params

    # Fitted values and R﷿﷿
    y_fit = M @ params

    ss_res = np.sum((y - y_fit)**2)
    ss_tot = np.sum((y - np.mean(y))**2)

    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return a, b, A, B, r_squared


def study_region(roi: ee.Geometry, glacier_mask: ee.Geometry | None = None) -> ee.Geometry:
    """roi ∩ glacier_mask, or roi if no mask."""
    return roi if glacier_mask is None else glacier_mask.intersection(roi, 1)
