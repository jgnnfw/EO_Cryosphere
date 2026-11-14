import os
import pandas as pd
import matplotlib

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