import os
import numpy as np
import pickle
from scipy import interpolate
from matplotlib import pyplot as plt
from matplotlib import ticker as ticker
import scienceplots as _
import pandas as pd

plt.style.use(["science", "nature"])
plt.rcParams["image.origin"] = "lower"
plt.rcParams["image.cmap"] = "jet"
plt.rcParams["image.interpolation"] = "gaussian"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["text.usetex"] = True
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8

init_num_samples = 507
max_num_samples = 5000
image_path = os.path.expanduser("~/Projects/RSS2024/paper/figures/images")


def set_size(width=516, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27
    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2
    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio
    fig_dim = (fig_width_in, fig_height_in)
    return fig_dim


def get_metrics(run_id, xs):
    with open(f"./outputs/{run_id}/evaluator.pkl", "rb") as f:
        evaluator = pickle.load(f)
    num_samples = evaluator.num_samples
    smse_fn = interpolate.interp1d(num_samples, evaluator.smses)
    smse = smse_fn(xs)
    msll_fn = interpolate.interp1d(num_samples, evaluator.mslls)
    msll = msll_fn(xs)
    train_time_fn = interpolate.interp1d(num_samples, evaluator.training_times)
    train_time = train_time_fn(xs)
    pred_time_fn = interpolate.interp1d(num_samples, evaluator.prediction_times)
    pred_time = pred_time_fn(xs)
    return smse, msll, train_time, pred_time


def get_legend(methods):
    fig, ax = plt.subplots()
    legend = plt.figure(figsize=set_size(fraction=1.0))
    lines = ax.plot(*[range(2) for _ in range(len(methods) * 2)])
    legend.legend(
        lines,
        methods,
        loc="center",
        bbox_to_anchor=(0.5, 0.5),
        shadow=True,
        ncol=len(methods),
    )
    plt.close(fig)
    legend.savefig(
        f"{image_path}/ablation_legend.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.close(legend)


def plot_metrics(xs, df, coordinate, models, metric_name, gap=250):
    fig, ax = plt.subplots(figsize=set_size(fraction=0.25))
    for model in models:
        results = df.loc[coordinate, model, 0, metric_name]
        # mean = results.mean(axis=0)
        # std = results.std(axis=0)
        ax.plot(xs, results, alpha=0.8)
        # ax.fill_between(xs, mean - std, mean + std, alpha=0.2, color=plt.gca().lines[-1].get_color())
        # ax.errorbar(
        #     xs[::gap],
        #     mean[::gap],
        #     yerr=std[::gap],
        #     alpha=0.8,
        #     color=plt.gca().lines[-1].get_color(),
        # )
    ax.grid("on")
    ax.autoscale(tight=True)
    ax.set_xticks([xs[0], xs[len(xs) // 2], xs[-1]])
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: "%4.2f" % x))
    fig.tight_layout()
    fig.savefig(
        f"{image_path}/ablation_{coordinate}_{metric_name}.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.close(fig)
    plt.show()


def join_mean_std(value, error, error_points=1):
    is_negative = value < 0
    if is_negative:
        value *= -1
    val = f"{value:.20e}".split("e")
    err = f"{error:.20e}".split("e")
    first_uncertain = int(val[1]) - int(err[1]) + error_points
    my_val = f"{np.round(float(val[0]), first_uncertain-1):.10f}"
    my_err = f"{np.round(float(err[0]), error_points-1):.10f}".replace(".", "")
    # Avoid 1. and write 1 instead
    if first_uncertain > 1:
        first_uncertain = first_uncertain + 1
    result = f"{my_val[:first_uncertain]}({my_err[:error_points]})e{val[1]}"
    if is_negative:
        return "-" + result
    else:
        return result

def main():
    # xs = np.arange(init_num_samples, max_num_samples + 1)
    xs = np.arange(1000, max_num_samples + 1)
    multiindex = []
    table = []
    coords = ["n35w107"]
    models = ["poam", "poam-z-opt","poam-z-rand", "poam-var-opt", "poam-var-ssgp", "poam-online-elbo"]
    seeds = [0]
    metric_names = ["smse", "msll", "train_time"]
    for coord in coords:
        for model in models:
            for seed in seeds:
                metrics = get_metrics(f"{coord}_{model}_{seed}", xs)
                for index, metric_name in enumerate(metric_names):
                    multiindex.append([coord, model, seed, metric_name])
                    table.append(metrics[index])
    index = pd.MultiIndex.from_tuples(
        multiindex, names=["Coordinates", "Models", "Seeds", "Metrics"]
    )
    df = pd.DataFrame(table, index=index, columns=xs)
    # print(df)

    # for coord in coords:
    #     print("==================================")
    #     print(coord)
    #     print("==================================")
    #     for model in models:
    #         print("-------")
    #         print(model)
    #         print("-------")
    #         for metric_name in metric_names:
    #             results = df.loc[coord, model, :, metric_name]
    #             mean = results.mean(axis=1).mean()
    #             std = results.mean(axis=1).std()
    #             print(f"{metric_name:10}  {mean: 4.2e} \t {std: .4f}")
    # exit()

    get_legend([model.upper() for model in models])
    for coord in coords:
        for metric_name in metric_names:
            plot_metrics(xs, df, coord, models, metric_name)


if __name__ == "__main__":
    main()
