"""Save tomography figures, tables, and numerical data."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def save_solver_result(path, data):
    """Save the reconstruction, weights, histories and counters without pickle.

    Large Krylov bases and the sequence of full images are not needed for the
    paper figures/tables. Linear-operator objects are also deliberately omitted.
    """
    excluded = {"V", "U", "Z", "solution_history", "Lopinv", "Lpinv"}
    arrays = {}
    for key, value in data.items():
        if key in excluded or value is None:
            continue
        if isinstance(value, (np.ndarray, np.number, int, float, bool, list, tuple)):
            array = np.asarray(value)
            if array.dtype.kind in "biufc":
                arrays[key] = array
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Writing through a file handle avoids numpy appending another suffix.
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)
    print(f"Saved {path}", flush=True)


def save_raw_results(path, results):
    """Combine already-computed results into one numeric archive."""
    arrays = {}
    excluded = {"V", "U", "Z", "solution_history", "Lopinv", "Lpinv"}
    for weights, method, data in results:
        prefix = weights + "__" + method.replace("-", "_")
        for key, value in data.items():
            if key in excluded or value is None:
                continue
            if isinstance(value, (np.ndarray, np.number, int, float, bool, list, tuple)):
                array = np.asarray(value)
                if array.dtype.kind in "biufc":
                    arrays[prefix + "__" + key] = array
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def save_image_data(fig, path):
    """Export the exact image arrays displayed in a figure, before color mapping.

    Equal-size panels use one row per pixel and one column per panel. Different
    sizes (the phantom and sinogram) use panel/row/column/value rows instead.
    A companion *_panels.csv records panel order, titles and display limits.
    """
    path = Path(path)
    images = [(index, ax, im) for index, ax in enumerate(fig.axes) for im in ax.images]
    panels = []
    for index, ax, im in images:
        array = np.asarray(im.get_array())
        low, high = im.get_clim()
        panels.append({"panel": f"panel_{index:02d}", "title": ax.get_title(),
                       "n_rows": array.shape[0], "n_columns": array.shape[1],
                       "cmap": im.get_cmap().name, "vmin": low, "vmax": high})
    if not images:
        raise ValueError("The figure contains no image panels.")
    arrays = [np.asarray(im.get_array()) for _, _, im in images]
    if len({array.shape for array in arrays}) == 1:
        rows, columns = np.indices(arrays[0].shape)
        table = {"row": rows.ravel(), "column": columns.ravel()}
        for (index, _, _), array in zip(images, arrays):
            table[f"panel_{index:02d}"] = array.ravel()
        pd.DataFrame(table).to_csv(path, index=False)
    else:
        blocks = []
        for (index, _, _), array in zip(images, arrays):
            rows, columns = np.indices(array.shape)
            blocks.append(pd.DataFrame({"panel": f"panel_{index:02d}",
                                        "row": rows.ravel(), "column": columns.ravel(),
                                        "value": array.ravel()}))
        pd.concat(blocks, ignore_index=True).to_csv(path, index=False)
    pd.DataFrame(panels).to_csv(path.with_name(path.stem + "_panels.csv"), index=False)


def save_curve_data(fig, path):
    """Export every plotted line, with its panel title and original iteration index."""
    blocks = []
    for index, ax in enumerate(fig.axes):
        for line in ax.lines:
            blocks.append(pd.DataFrame({"panel": index, "title": ax.get_title(),
                                        "method": line.get_label(),
                                        "iteration": np.asarray(line.get_xdata()),
                                        "value": np.asarray(line.get_ydata())}))
    if not blocks:
        raise ValueError("The figure contains no plotted lines.")
    pd.concat(blocks, ignore_index=True).to_csv(path, index=False)


def save_table_png(frame, path, lambda_digits=1):
    """Render the numerical table using the paper's rounding conventions."""
    rows = []
    for row in frame.itertuples(index=False, name=None):
        formatted = []
        for index, value in enumerate(row):
            if index < 2:
                formatted.append(str(value))
            elif index == 2:
                formatted.append(f"{value:.{lambda_digits - 1}e}")
            elif index in (4, 5, 6):
                formatted.append(f"{value:.3f}")
            else:
                formatted.append(f"{value:.0f}")
        rows.append(formatted)
    labels = ["Weights", "Method", r"$\lambda$", "Iterations", "RRE", "SSIM",
              "Gini index", r"$n_A$", r"$n_L$", r"$n_{L_k^\dagger}$"]
    fig, ax = plt.subplots(figsize=(14, 0.36 * (len(frame) + 2)))
    ax.set_axis_off()
    table = ax.table(cellText=rows, colLabels=labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.65)
    table.auto_set_column_width(range(len(labels)))
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
