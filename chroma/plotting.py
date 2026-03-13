import importlib.resources as resources
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

try:
    with resources.as_file(
        resources.files("chroma") / "paper.mplstyle"
    ) as style_path:
        plt.style.use(str(style_path))
except FileNotFoundError:
    pass


class PlotContactMap:

    def __init__(self, fig=None, ax=None) -> None:
        self.cmap = mcolors.LinearSegmentedColormap.from_list(
            "bright_red", [(1, 1, 1), (1, 0, 0)]
        )

        if not fig and not ax:
            fig, ax = plt.subplots(1, 1)
        elif fig and not ax:
            ax = fig.gca()
        elif ax and not fig:
            fig = ax.gcf()

        self.fig = fig
        self.ax = ax
        self.divider = make_axes_locatable(ax)

    def plot_hic(self, contact_map, scale="log", vmin=1e-4, vmax=1):
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        if scale == "log":
            im = self.ax.imshow(
                contact_map,
                norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
                cmap=self.cmap,
            )
        elif scale == "linear":
            im = self.ax.imshow(
                contact_map,
                vmax=vmax,
                vmin=vmin,
                cmap=self.cmap,
            )
        else:
            raise ValueError("`scale` must be `log` or `linear`")

        return im

    def add_colorbar(self, im, loc="right", size=0.05, pad=0.05):
        cax = self.divider.append_axes(loc, size=size, pad=pad)

        cbar = self.fig.colorbar(
            im, cax=cax, label="Contact probability"
        )

        return cbar

    def add_eigenvector(
        self, eigenvector, loc="bottom", size=0.10, pad=0.05
    ):
        eig_ax = self.divider.append_axes(loc, size=size, pad=pad)

        eig_ax.fill_between(
            range(len(eigenvector)),
            eigenvector,
            0,
            where=eigenvector > 0,
            color="red",
            edgecolor="None",
        )
        eig_ax.fill_between(
            range(len(eigenvector)),
            eigenvector,
            0,
            where=eigenvector < 0,
            color="blue",
            edgecolor="None",
        )
        eig_ax.hlines(
            0, 0, len(eigenvector), linewidth=0.5, color="black"
        )

        eig_ax.set_xlim((0, len(eigenvector)))
        eig_ax.set_xticks([])
        eig_ax.set_yticks([])

        return eig_ax

    def add_track(
        self, track, loc="bottom", size=0.10, pad=0.05, **kwargs
    ):
        track_ax = self.divider.append_axes(loc, size=size, pad=pad)

        track_ax.plot(range(len(track)), track, **kwargs)

        track_ax.set_xlim((0, len(track)))
        track_ax.set_xticks([])
        track_ax.set_yticks([])

        return track_ax

    def add_compartment_annotations(
        self,
        annotations: list[str],
        colors: dict | str = "subcompartment",
        loc: str = "left",
        size=0.05,
        pad=0.05,
    ):
        if colors == "subcompartment":
            colors = {
                "A1": "red",
                "A2": "orange",
                "B1": "blue",
                "B2": "cyan",
                "B3": "green",
                "B4": "lightgreen",
                "NA": "gray",
            }
        elif colors == "compartment":
            colors = {
                "A": "red",
                "B": "blue",
                "NA": "gray",
            }
        elif isinstance(colors, dict):
            pass
        else:
            raise ValueError(
                "Unsupported color scheme for compartment annotations"
            )

        for key, color in colors.items():
            if not mcolors.is_color_like(color):
                raise ValueError(
                    f"Invalid color for compartment annotation: {color}"
                )
            colors[key] = mcolors.to_rgba(color)

        possible_annotations = set(annotations)

        if not possible_annotations.issubset(colors.keys()):
            raise ValueError(
                "Some annotations do not have corresponding colors defined"
            )

        colors_arr = np.array(
            [
                colors.get(str(x), (1, 1, 1))
                for x in annotations
            ],
            dtype=np.float32,
        )
        annot_ax = self.divider.append_axes(loc, size=size, pad=pad)

        if loc in {"top", "bottom"}:
            img = colors_arr[np.newaxis, :, :]
            annot_ax.imshow(
                img,
                aspect="auto",
                interpolation="none",
                origin="upper",
            )
        elif loc in {"left", "right"}:
            img = colors_arr[:, np.newaxis, :]
            annot_ax.imshow(
                img,
                aspect="auto",
                interpolation="none",
                origin="upper",
            )
        else:
            raise ValueError(f"Unsupported AB-bar location: {loc}")

        annot_ax.set_xticks([])
        annot_ax.set_yticks([])

        return annot_ax
