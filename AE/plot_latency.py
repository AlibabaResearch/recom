import argparse
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_axes(fig, models, batch_sizes, data_list, nrows, ncols, fontsize):
    labels = ["TF-CPU", "TF-GPU", "TF-CPU-GPU", "RECom"]
    colors = ["#B4C7E7", "#C5E0B4", "#FFE699", "#F8CBAD"]
    hatches = ["\\\\", "//", "||", "--"]

    axes = fig.subplots(nrows, ncols, sharex="col", sharey="row")

    for r in range(nrows):
        for c in range(ncols):
            if nrows == 1 or ncols == 1:
                ax = axes[r * ncols + c]
            else:
                ax = axes[r, c]
            # tf-cpu, tf-gpu, tf-cpu-gpu, recom
            data = data_list[r * ncols + c]

            width = 1.0 / (len(data) + 1)
            location = np.arange(len(batch_sizes))

            for i, (bar, label, color, hatch) in enumerate(zip(
                data, labels, colors, hatches
            )):
                ax.bar(location + width * i, bar, width=width, label=label,
                       color=color, hatch=hatch, edgecolor="k", alpha=1)
            ax.bar(location + width * len(labels) / 2,
                   np.zeros_like(bar), tick_label=batch_sizes)

            ax.set_ylim(2, 4000)
            ax.set_xlim(-width, len(batch_sizes) - width)
            for tick in ax.get_xticklabels():
                tick.set_fontsize(fontsize)
                tick.set_rotation(30)
                x, y = tick.get_position()
                tick.set_position((x, y))
            for tick in ax.get_yticklabels():
                tick.set_fontsize(fontsize)
            ax.tick_params(bottom=False, top=False, right=False)
            ax.set_yscale('log', subs=[2, 3, 4, 5])
            ax.set_title(
                f"Model {models[r * ncols + c]}", fontsize=fontsize, y=0.75)


def read_data(models, log_dir, batch_sizes):
    data_list = []
    targets = ["tf_cpu", "tf_gpu", "tf_cpu_gpu", "recom"]
    for model in models:
        bars = [[], [], [], []]
        for bar, target in zip(bars, targets):
            for bs in batch_sizes:
                log = f"{log_dir}/l_{target}_{model}_{bs}.log"
                try:
                    lat = float(os.popen(
                        f"grep 'average latency' {log} | awk '{{print $14}}'").read())
                except ValueError as e:
                    print(f"Error: {e}")
                    print(f"Cannot read data correctly. Please check log {log}")
                    print(f"Assign zero to the bar")
                    lat = 0
                bar.append(lat)
        data_list.append(bars)
    return data_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--output", type=str, default="./latency.pdf")
    args = parser.parse_args()

    nrows = 1
    ncols = 2
    models = ["E", "F"]
    batch_sizes = [32, 64, 128, 256, 512, 1024, 2048]

    plt.rcParams["pdf.fonttype"] = 42
    fontsize = 18

    fig = plt.figure(figsize=(4 * ncols + 3, 1.5 * nrows + 2))

    data_list = read_data(models, args.log_dir, batch_sizes)
    plot_axes(fig, models, batch_sizes, data_list, nrows, ncols, fontsize)

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor="none", top=False,
                    bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel("Batch Size", fontsize=fontsize, labelpad=30)
    plt.ylabel("Latency (ms)", fontsize=fontsize, labelpad=20)
    bars, labels = fig.axes[0].get_legend_handles_labels()
    plt.legend(bars, labels, bbox_to_anchor=(0.5, 1.0), loc="lower center", ncol=4,
               handletextpad=1, fontsize=fontsize, columnspacing=2.0, frameon=False)
    fig.tight_layout()
    # plt.show()
    fig.savefig(args.output, bbox_inches="tight")

