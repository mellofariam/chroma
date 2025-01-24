import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.style.use("chroma.paper")


class PlotContactMap:

    def __init__(self, fig=None, ax=None) -> None:
        self.cmap = mpl.colors.LinearSegmentedColormap.from_list(
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

    def plot_hic(
        self, contact_map, resolution, scale="log", vmin=1e-4, vmax=1
    ):
        self.ax.set_xlim((0, contact_map.shape[0]))
        self.ax.set_ylim((contact_map.shape[1], 0))

        self.ax.set_xticks([])
        self.ax.set_yticks([])

        if scale == "log":
            im = self.ax.imshow(
                contact_map,
                norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax),
                cmap=self.cmap,
            )
        elif scale == "linear":
            im = self.ax.imshow(
                contact_map,
                norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax),
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
