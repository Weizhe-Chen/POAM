from copy import copy

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker as ticker
from matplotlib.patches import Arrow
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams["image.origin"] = "lower"
plt.rcParams["image.cmap"] = "jet"
plt.rcParams["image.interpolation"] = "gaussian"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8


class OOMFormatter(ticker.ScalarFormatter):
    def __init__(self, values, fformat="% 1.1f", offset=True, mathText=True):
        self.fformat = fformat
        self.oom = np.floor(np.log10(values.ptp()))
        super().__init__(
            useOffset=offset,
            useMathText=mathText,
        )

    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom


class Visualizer:
    def __init__(self, env_extent, task_extent, interval):
        self.init_axes()
        self.env_extent = env_extent
        self.task_extent = task_extent
        self.env_rectangle = self._init_rectangle(env_extent, "black")
        self.task_rectangle = self._init_rectangle(task_extent, "gray")
        self.vmins = [None] * 4
        self.vmaxs = [None] * 4
        env_size = np.hypot(
            env_extent[1] - env_extent[0], env_extent[3] - env_extent[2]
        )
        self.arrow = None
        self.interval = interval

    def _init_rectangle(self, extent, color):
        x_min, x_max, y_min, y_max = extent
        return plt.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            fill=False,
            edgecolor=color,
            linewidth=3,
            alpha=0.8,
        )

    def init_axes(self) -> None:
        self.fig, axes = plt.subplots(2, 4, figsize=(16, 6), height_ratios=[2, 1])
        self.fig.subplots_adjust(
            top=0.95,
            bottom=0.1,
            left=0.1,
            right=0.9,
            hspace=0.1,
            wspace=0.2,
        )
        axes = np.asarray(axes).ravel()
        caxes = []
        for ax in axes[:4]:
            ax.axis("off")
            cax = make_axes_locatable(ax).append_axes(
                "right",
                size="5%",
                pad=0.05,
            )
            caxes.append(cax)
        caxes = np.asarray(caxes)
        self.axes = axes
        self.caxes = caxes

    def plot_image(self, index, matrix, title, vmin=None, vmax=None):
        ax, cax = self.axes[index], self.caxes[index]
        im = ax.imshow(
            matrix,
            extent=self.env_extent if index == 0 else self.task_extent,
            vmin=vmin,
            vmax=vmax,
        )
        plt.colorbar(im, cax, format=OOMFormatter(matrix))
        ax.add_patch(copy(self.env_rectangle))
        ax.add_patch(copy(self.task_rectangle))
        ax.set_xlim(self.env_extent[:2])
        ax.set_ylim(self.env_extent[2:])
        ax.set_title(title)

    def plot_prediction(self, mean, std, abs_error):
        self.plot_image(
            index=1,
            matrix=mean,
            title="Mean",
            vmin=self.vmins[1],
            vmax=self.vmaxs[1],
        )
        self.plot_image(
            index=2,
            matrix=std,
            title="Standard Deviation",
            vmin=self.vmins[2],
            vmax=self.vmaxs[2],
        )
        self.plot_image(
            index=3,
            matrix=abs_error,
            title="Absolute Error",
            vmin=self.vmins[3],
            vmax=self.vmaxs[3],
        )

    def plot_lengthscales(self, model, evaluator, min_lengthscale, max_lengthscale):
        self.axes[0].clear()
        self.axes[0].axis("off")
        self.caxes[0].clear()
        lenscale = model.get_ak_lengthscales(evaluator.eval_inputs).reshape(
            *evaluator.eval_grid
        )
        self.plot_image(
            index=0,
            matrix=lenscale,
            title="Lengthscales",
            vmin=min_lengthscale,
            vmax=max_lengthscale,
        )

    def plot_metrics(self, evaluator):
        self.axes[4].plot(evaluator.num_samples, evaluator.smses)
        self.axes[4].set_title("Standardized Mean Squared Error")
        self.axes[4].set_xlabel("Number of Samples")
        self.axes[4].grid("on")
        self.axes[4].yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: "% .2f" % x)
        )
        self.axes[4].set_ylim([0, None])

        self.axes[5].plot(evaluator.num_samples, evaluator.mslls)
        self.axes[5].set_title("Mean Standardized Log Loss")
        self.axes[5].set_xlabel("Number of Samples")
        self.axes[5].set_ylim([None, 0])
        self.axes[5].grid("on")
        self.axes[5].yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: "% .2f" % x)
        )

        self.axes[6].plot(evaluator.num_samples, evaluator.training_times)
        self.axes[6].yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: "% .2f" % x)
        )
        self.axes[6].set_xlabel("Number of Samples")
        self.axes[6].grid("on")
        self.axes[6].set_title("Training Time (Seconds)")

        self.axes[7].plot(evaluator.num_samples, evaluator.prediction_times)
        self.axes[7].yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: "% .2f" % x)
        )
        self.axes[7].set_xlabel("Number of Samples")
        self.axes[7].grid("on")
        self.axes[7].set_title("Prediction Time (Seconds)")

    def plot_data(self, x_train: np.ndarray, alpha=0.5):
        self.axes[2].plot(x_train[:, 0], x_train[:, 1], ".", color="k", alpha=alpha)

    def plot_inducing_inputs(self, x_inducing: np.ndarray):
        self.axes[3].plot(x_inducing[:, 0], x_inducing[:, 1], "+", color="w", alpha=0.9)

    def plot_robot(self, state: np.ndarray, scale: float = 1.0):
        if self.arrow is not None:
            self.arrow.remove()
        self.arrow = Arrow(
            state[0],
            state[1],
            scale * np.cos(state[2]),
            scale * np.sin(state[2]),
            width=scale,
            color="black",
            alpha=0.8,
        )
        self.axes[2].add_patch(self.arrow)

    def plot_goal(self, goal: np.ndarray):
        self.axes[2].plot(
            goal[0, 0],
            goal[0, 1],
            "*",
            color="white",
            markersize=20,
            alpha=0.8,
        )

    def plot_title(self, decision_epoch, time_elapsed):
        self.fig.suptitle(
            f"Decision Epoch: {decision_epoch} | "
            + f"Time Elapsed: {time_elapsed:.2f} Seconds."
        )

    def clear(self):
        for i in range(1, 4):
            self.axes[i].clear()
            self.axes[i].axis("off")
            self.caxes[i].clear()
        for i in range(4, 8):
            self.axes[i].clear()

    def pause(self, interval=1e-3):
        plt.gcf().canvas.mpl_connect(
            "key_release_event",
            lambda event: [exit(0) if event.key == "escape" else None],
        )
        plt.pause(interval)

    def flash(self):
        self.pause()
        self.clear()

    def show(self):
        plt.show()
