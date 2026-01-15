import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to .pt from compare_smoothquant_dump.py")
    parser.add_argument("--output_dir", type=str, default="viz_out")
    parser.add_argument("--bins", type=int, default=120)
    parser.add_argument("--max_points", type=int, default=500000)
    return parser.parse_args()


def to_numpy(tensor, max_points):
    data = tensor.detach().float().cpu().numpy().ravel()
    if data.size > max_points:
        idx = np.random.choice(data.size, size=max_points, replace=False)
        data = data[idx]
    return data


def stats(tensor):
    data = tensor.detach().float().cpu()
    return {
        "mean": data.mean().item(),
        "std": data.std().item(),
        "min": data.min().item(),
        "max": data.max().item(),
    }


def plot_hist(ax, data, title, bins):
    ax.hist(data, bins=bins, density=True, alpha=0.7)
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.input, map_location="cpu")["max_stats"]
    stats_out = {}

    def stack_samples(sample_dict):
        tensors = []
        for _, sample_map in sample_dict.items():
            if isinstance(sample_map, dict):
                for i in range(len(sample_map)):
                    tensors.append(sample_map[i])
            else:
                tensors.append(sample_map)
        return torch.tensor(tensors)

    # for name in payload["fp16_weight"].keys():
    fp16_in = stack_samples(payload["fp16_input"])
    quant_in = stack_samples(payload["quant_input_fp"])
    quant_in_q = stack_samples(payload["quant_input_q"])

    smooth_w = stack_samples(payload["smooth_weight"])
    fp16_w = stack_samples(payload["fp16_weight"])
    quant_w = stack_samples(payload["quant_weight"])
    name = "self_attn.q_proj"
    stats_out[name] = {
        "fp16_input": stats(fp16_in),
        "quant_input_fp": stats(quant_in),
        "quant_input_q": stats(quant_in_q),
        "fp16_weight": stats(fp16_w),
        "smooth_weight": stats(smooth_w),
        "quant_weight": stats(quant_w),
    }
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    plot_hist(axes[0, 0], to_numpy(fp16_in, args.max_points), f"{name} fp16 input", args.bins)
    plot_hist(axes[0, 1], to_numpy(quant_in, args.max_points), f"{name} quant input fp", args.bins)
    plot_hist(axes[0, 2], to_numpy(quant_in_q, args.max_points), f"{name} quant input q", args.bins)
    plot_hist(axes[1, 0], to_numpy(fp16_w, args.max_points), f"{name} fp16 weight", args.bins)
    plot_hist(axes[1, 1], to_numpy(smooth_w, args.max_points), f"{name} smooth weight", args.bins)
    plot_hist(axes[1, 2], to_numpy(quant_w, args.max_points), f"{name} quant weight", args.bins)
    fig.tight_layout()
    fig.savefig(output_dir / f"{name.replace('.', '_')}.png", dpi=160)
    plt.close(fig)

    # max_stats = payload.get("max_stats", None)
    # if max_stats:
    #     summary = {}
    #     global_stats = {
    #         "activation": {"fp16_input": [], "quant_input_fp": [], "quant_input_q": []},
    #         "weight": {"fp16_weight": [], "smooth_weight": [], "quant_weight": []},
    #     }
    #     for name in payload["fp16_weights"].keys():
    #         summary[name] = {}
    #         for key in ("fp16_input", "quant_input_fp", "quant_input_q"):
    #             values = list(max_stats[key][name].values())
    #             if values:
    #                 summary[name][key] = {
    #                     "mean": float(np.mean(values)),
    #                     "std": float(np.std(values)),
    #                     "min": float(np.min(values)),
    #                     "max": float(np.max(values)),
    #                 }
    #                 global_stats["activation"][key].extend(values)
    #         for key in ("fp16_weight", "smooth_weight", "quant_weight"):
    #             summary[name][key] = max_stats[key][name]
    #             if max_stats[key][name] is not None:
    #                 global_stats["weight"][key].append(max_stats[key][name])

    #     with open(output_dir / "max_stats_summary.json", "w") as f:
    #         json.dump(summary, f, indent=2)

    #     for category, series in global_stats.items():
    #         fig, ax = plt.subplots(figsize=(10, 6))
    #         for key, values in series.items():
    #             if values:
    #                 ax.hist(values, bins=args.bins, alpha=0.5, density=True, label=key)
    #         ax.set_title(f"{category} max distribution (all layers)")
    #         ax.legend()
    #         ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    #         fig.tight_layout()
    #         fig.savefig(output_dir / f"{category}_max_all_layers.png", dpi=160)
    #         plt.close(fig)

    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats_out, f, indent=2)

    print(json.dumps({"output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
