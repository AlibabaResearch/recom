import argparse
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_axes(fig, models, workers, data_list, nrows, ncols, fontsize):
    labels = ["TF-CPU", "RECom"]
    colors = ["#B4C7E7", "#F8CBAD"]
    hatches = ["\\\\", "--"]

    axes = fig.subplots(nrows, ncols, sharex='col', sharey='row')

    for r in range(nrows):
        for c in range(ncols):
            if nrows == 1 or ncols == 1:
                ax = axes[r * ncols + c]
            else:
                ax = axes[r, c]
            # tf-cpu, recom
            data = data_list[r * ncols + c]

            width = 1.0 / (len(data) + 1)
            location = np.arange(len(workers))

            for i, (bar, label, color, hatch) in enumerate(zip(
                data, labels, colors, hatches
            )):
                ax.bar(location + width * i, bar, width=width, label=label,
                       color=color, hatch=hatch, edgecolor="k", alpha=1)
            ax.bar(location + width * len(labels) / 2,
                   np.zeros_like(bar), tick_label=workers)

            ax.set_ylim(0, 145000)
            ax.set_xlim(-width, len(workers) - width)
            for tick in ax.get_xticklabels():
                tick.set_fontsize(fontsize)
                x, y = tick.get_position()
                tick.set_position((x, y))
            for tick in ax.get_yticklabels():
                tick.set_fontsize(fontsize)
            ax.tick_params(bottom=False, top=False, right=False)
            ax.set_title(f"Model {models[r * ncols + c]}", fontsize=fontsize, y=0.75)


def read_data(models, log_dir, workers):
    data_list = []
    targets = ["tf_cpu", "recom"]
    for model in models:
        bars = [[], []]
        for bar, target in zip(bars, targets):
            for w in workers:
                log = f"{log_dir}/t_{target}_{model}_{w}.log"
                lat = float(os.popen(
                    f"grep 'max_throughput' {log} | awk '{{print $10}}'").read())
                bar.append(lat)
        data_list.append(bars)
    return data_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--output", type=str, default="./throughput.pdf")
    args = parser.parse_args()

    nrows = 1
    ncols = 2
    models = ["E", "F"]
    workers = [2, 4, 8]

    plt.rcParams["pdf.fonttype"] = 42
    fontsize = 18

    fig = plt.figure(figsize=(2 * ncols + 2, 1.5 * nrows + 1.5))

    data_list = read_data(models, args.log_dir, workers)
    plot_axes(fig, models, workers, data_list, nrows, ncols, fontsize)

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor="none", top=False,
                    bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel("# Worker Threads", fontsize=fontsize, labelpad=10)
    plt.ylabel("Throughput (inference/s)", fontsize=fontsize, labelpad=40)
    bars, labels = fig.axes[0].get_legend_handles_labels()
    plt.legend(bars, labels, bbox_to_anchor=(0.5, 1.0), loc="lower center", ncol=2,
               handletextpad=1, fontsize=fontsize, columnspacing=3.0, frameon=False)
    fig.tight_layout()
    # plt.show()
    fig.savefig(args.output, bbox_inches="tight")
