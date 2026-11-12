import os
import pandas as pd
import matplotlib

def save_any_tables_and_figures(tables_and_figs: dict[str, pd.DataFrame | matplotlib.figure.Figure], save_path: str = ".", debug: bool = False) -> None:
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if debug:
        print("\nSaving:", end="")

    for name, item in tables_and_figs.items():
        if isinstance(item, pd.DataFrame):
            item.to_csv(os.path.join(save_path, name + ".csv"), index=False)
            print()
        elif isinstance(item, matplotlib.figure.Figure):
            item.savefig(os.path.join(save_path, name + ".png"), dpi=300)
        else:
            continue


def myprint(debug: bool = False, *arg, **kwarg):
    if debug:
        print(*arg, **kwarg)

