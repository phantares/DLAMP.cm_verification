import argparse
from pathlib import Path

import h5py as h5
import numpy as np
import pandas as pd
import yaml
from calculators import DetectionCalculator, calculate_bin, calculate_water_path
from constants import STANDARD_LEVEL, WPS, q_bins, wp_bins
from dotenv import dotenv_values
from plotters import BarPlotter, GridPlotter, PDPlotter, plot_heat_map, plot_profile


def main(exp: str, mask_threshold: float = 0.5) -> None:
    env = dotenv_values(".env")

    fig_dir = Path(env.get("FIG_DIR"), exp)
    fig_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(env.get("DATA_DIR"), exp)
    files = sorted(data_dir.glob("*.h5"))

    with open(data_dir / "config.yaml", "r", encoding="utf-8") as f:
        configs = yaml.safe_load(f)
    use_mask = configs["model"]["system"]["use_mask"]
    data_configs = configs["dataset"]["var"]

    vars = data_configs["target"]
    sample_var = len(vars)
    annots = [var[1].capitalize() for var in vars] + ["A"]
    vars_all = vars + ["all"]

    thresholds = data_configs["threshold"]

    z_tar = data_configs["z_target"]
    sample_z = len(z_tar)
    standard_indices = [z_tar.index(z) for z in STANDARD_LEVEL]

    sample_t = 0
    sample_r = {var: np.zeros(sample_z) for var in vars}

    hits = {var: np.zeros(sample_z) for var in vars}
    misses = {var: np.zeros(sample_z) for var in vars}
    fas = {var: np.zeros(sample_z) for var in vars}
    crs = {var: np.zeros(sample_z) for var in vars}

    sources = ["prediction", "target"]
    mean = {source: {var: np.zeros(sample_z) for var in vars} for source in sources}
    square = {source: {var: np.zeros(sample_z) for var in vars} for source in sources}
    std = {source: {var: np.zeros(sample_z) for var in vars} for source in sources}
    q_counts = {
        source: {var: np.zeros((len(q_bins) - 1, sample_z)) for var in vars}
        for source in sources
    }
    wp_counts = {
        source: {wp: np.zeros(len(wp_bins) - 1) for wp in WPS} for source in sources
    }

    mae = {var: np.zeros(sample_z) for var in vars}
    sum_abs_diff = {var: np.zeros(sample_z) for var in vars}
    r_mae = {var: np.zeros(sample_z) for var in vars}
    bias = {var: np.zeros(sample_z) for var in vars}
    r_bias = {var: np.zeros(sample_z) for var in vars}
    sum_diff = {var: np.zeros(sample_z) for var in vars}

    stats_metrics = ["Mean", "Std"]
    comparison_metrics = ["MAE", "RelativeMAE", "Bias", "RelativeBias"]
    stats = {
        source: {var: {metric: 0.0 for metric in stats_metrics} for var in vars_all}
        for source in sources
    }
    comparisons = {
        var: {metric: 0.0 for metric in comparison_metrics} for var in vars_all
    }

    print("load data")
    for file in files:
        month = file.stem[-2:]
        print(month)

        dadd = {source: [] for source in sources}
        wps = {source: {wp: 0 for wp in WPS} for source in sources}

        with h5.File(file, "r") as f:
            for v, var in enumerate(vars):
                print(var)

                if var in ["qi", "qs", "qg"]:
                    wp = "iwp"
                elif var in ["qc", "qr"]:
                    wp = "lwp"

                datas = {}
                for source in sources:
                    data = f[f"{source}s"][var][:]
                    data[data < thresholds[var]] = 0

                    if source == "prediction":
                        if use_mask:
                            mask = f["predictions"][f"{var}_mask"][:]
                            data[mask <= mask_threshold] = 0

                    dadd[source].append(data)
                    datas[source] = data
                    wps[source][wp] += data

                    axes_to_sum = tuple(
                        i for i in range(data.ndim) if i != data.ndim - 3
                    )
                    mean[source][var] += np.sum(data, axis=axes_to_sum)
                    square[source][var] += np.sum(data**2, axis=axes_to_sum)

                for k in range(sample_z):
                    pred = datas["prediction"][..., k, :, :]
                    tar = datas["target"][..., k, :, :]

                    detection = DetectionCalculator(pred, tar, [0])

                    hits[var][k] += detection.hits[0]
                    misses[var][k] += detection.misses[0]
                    fas[var][k] += detection.false_alarms[0]
                    crs[var][k] += detection.correct_rejections[0]

                    q_counts["prediction"][var][:, k] += calculate_bin(q_bins, pred)
                    q_counts["target"][var][:, k] += calculate_bin(q_bins, tar)

                    diff = pred - tar
                    valid_r = tar > 0
                    sample_r[var][k] += np.sum(valid_r.astype(int))
                    r_diff = diff[valid_r]
                    r_tar = tar[valid_r]

                    mae[var][k] += np.sum(np.abs(diff))
                    sum_abs_diff[var][k] += np.sum(np.abs(r_diff) / r_tar)
                    bias[var][k] += np.sum(diff)
                    sum_diff[var][k] += np.sum(r_diff / r_tar)

            sample_t += np.size(data, 0)

            for wp in WPS:
                for source in sources:
                    wp_counts[source][wp] += calculate_bin(
                        wp_bins, calculate_water_path(wps[source][wp], z_tar)
                    )

    print("calculate statistics")
    sample_h = np.size(data, axis=-1) * np.size(data, axis=-2)
    sample_th = sample_t * sample_h

    for var in vars:
        for source in sources:
            mean[source][var] = mean[source][var] / sample_th
            stats[source][var]["Mean"] = np.sum(mean[source][var]) / sample_z

            std[source][var] = np.sqrt(
                square[source][var] / sample_th - mean[source][var] ** 2
            )
            stats[source][var]["Std"] = np.sqrt(
                np.sum(square[source][var]) / (sample_th * sample_z)
                - stats[source][var]["Mean"] ** 2
            )

        valid_r = sample_r[var] > 0

        mae[var] = mae[var] / sample_th
        comparisons[var]["MAE"] = np.sum(mae[var]) / sample_z

        r_mae[var][valid_r] = sum_abs_diff[var][valid_r] / sample_r[var][valid_r]
        r_mae[var][~valid_r] = np.nan
        comparisons[var]["RelativeMAE"] = (
            np.sum(sum_abs_diff[var][valid_r]) / np.sum(sample_r[var][valid_r]) * 100
        )

        bias[var] = bias[var] / sample_th
        comparisons[var]["Bias"] = np.sum(bias[var]) / sample_z

        r_bias[var][valid_r] = sum_diff[var][valid_r] / sample_r[var][valid_r]
        r_bias[var][~valid_r] = np.nan
        comparisons[var]["RelativeBias"] = (
            np.sum(sum_diff[var][valid_r]) / np.sum(sample_r[var][valid_r]) * 100
        )

    for source in sources:
        stats[source]["all"]["Mean"] = np.sum(list(mean[source].values())) / (
            sample_z * sample_var
        )
        stats[source]["all"]["Std"] = np.sqrt(
            np.sum(list(square[source].values())) / (sample_th * sample_z * sample_var)
            - stats[source]["all"]["Mean"] ** 2
        )
    comparisons["all"]["MAE"] = np.sum(list(mae.values())) / (sample_z * sample_var)
    comparisons["all"]["RelativeMAE"] = (
        np.sum(list(sum_abs_diff.values())) / np.sum(list(sample_r.values())) * 100
    )

    comparisons["all"]["Bias"] = np.sum(list(bias.values())) / (sample_z * sample_var)
    comparisons["all"]["RelativeBias"] = (
        np.sum(list(sum_diff.values())) / np.sum(list(sample_r.values())) * 100
    )

    rows = {}

    for source in sources:
        rows[f"{source}_Mean"] = {var: stats[source][var]["Mean"] for var in vars_all}
        rows[f"{source}_Std"] = {var: stats[source][var]["Std"] for var in vars_all}

    for metric in comparison_metrics:
        rows[metric] = {var: comparisons[var][metric] for var in vars_all}

    df = pd.DataFrame(rows).T
    df = df[vars_all]

    row_order = [
        f"{s}_{m}" for s in sources for m in ["Mean", "Std"]
    ] + comparison_metrics
    df = df.loc[row_order]

    df.to_csv(fig_dir / "metrics_summary.csv", float_format="%.4e")

    print("plot")
    cmap_scalar = "plasma"
    cmap_value = "Blues"
    cmap_diff = "RdBu"
    nan_color = "lightgray"

    heat_configs = {
        "xticklabels": vars,
        "yticks": standard_indices,
        "yticklabels": STANDARD_LEVEL,
    }

    cfad_value_configs = {
        "vmin": np.log10(1e-10),
        "vmax": np.log10(1e-2),
        "extend": "both",
        "cbar_log_scale": True,
    }

    label_configs = {
        "yticks": STANDARD_LEVEL[:-4] + STANDARD_LEVEL[-3:-2],
        "xlim": [q_bins[1], np.amax(q_bins)],
        "plot_legend": False,
    }

    bin_configs = {"align": "edge", "alpha": 0.6, "width": np.diff(wp_bins)}

    # Performance Diagram
    sr = []
    pod = []
    for var in vars:
        hit = np.sum(hits[var])
        miss = np.sum(misses[var])
        fa = np.sum(fas[var])

        sr.append(detection.calculate_SR(hit, fa))
        pod.append(detection.calculate_POD(hit, miss))

    hit = np.sum(list(hits.values()))
    miss = np.sum(list(misses.values()))
    fa = np.sum(list(fas.values()))

    sr.append(detection.calculate_SR(hit, fa))
    pod.append(detection.calculate_POD(hit, miss))

    plotter = PDPlotter()
    plotter.plot([sr], [pod], ["C6"], annots)
    plotter.save(fig_dir / "pd.png")

    # Heat map for Detection
    hit = np.array(list(hits.values()))
    miss = np.array(list(misses.values()))
    fa = np.array(list(fas.values()))
    cr = np.array(list(crs.values()))

    csi = detection.calculate_CSI(hit, miss, fa)
    hr = detection.calculate_POD(hit, miss)
    sr = detection.calculate_SR(hit, fa)
    far = detection.calculate_FAR(fa, cr)

    plot_heat_map(
        fig_dir / "csi.png",
        csi.transpose(),
        cmap=cmap_scalar,
        title="CSI",
        **heat_configs,
    )

    plot_heat_map(
        fig_dir / "hr.png",
        hr.transpose(),
        cmap=cmap_scalar,
        title="Hit Rate",
        **heat_configs,
    )

    plot_heat_map(
        fig_dir / "sr.png",
        sr.transpose(),
        cmap=cmap_scalar,
        title="Success Ratio",
        **heat_configs,
    )

    plot_heat_map(
        fig_dir / "far.png",
        far.transpose(),
        cmap=cmap_scalar,
        title="False Alarm Rate",
        **heat_configs,
    )

    # CFAD
    pressure_edges = np.concatenate([[1050], np.diff(z_tar) / 2 + z_tar[:-1], [0]])

    for var in vars:
        var_title = rf"${var[0].capitalize()}_{var[1]}$"

        for source in sources:
            prob = q_counts[source][var] / sample_th

            plotter = GridPlotter()
            plotter.plot_pcolor(
                q_bins,
                pressure_edges,
                prob.transpose(),
                cmap=cmap_value,
                **cfad_value_configs,
            )
            plotter.plot_label(
                title=f"{var_title}: {source.capitalize()}", **label_configs
            )
            plotter.save(fig_dir / f"cfad_{var}_{source}.png")

        prediction_prob, target_prob = [
            q_counts[source][var] / sample_th for source in sources
        ]
        plotter = GridPlotter()
        plotter.plot_pcolor(
            q_bins,
            pressure_edges,
            (prediction_prob - target_prob).transpose(),
            cmap=cmap_diff,
            extend="both",
            bounds=[
                -1e0,
                -1e-2,
                -1e-4,
                -1e-6,
                -1e-8,
                1e-8,
                1e-6,
                1e-4,
                1e-2,
                1e0,
            ],
            assign_ctick=True,
            ctick_format="{:.0e}",
        )
        plotter.plot_label(title=f"{var_title}: Prediction - Target", **label_configs)
        plotter.save(fig_dir / f"cfad_{var}_diff.png")

        plotter = GridPlotter()
        plotter.plot_pcolor(
            q_bins,
            pressure_edges,
            ((prediction_prob - target_prob) / target_prob).transpose() * 100,
            cmap=cmap_diff,
            extend="both",
            bounds=[-1e2, -1e1, -1e0, -1e-1, 1e-1, 1e0, 1e1, 1e2],
            nan_color=nan_color,
            cbar_label="%",
            assign_ctick=True,
            ctick_format="{:.0e}",
        )
        plotter.plot_label(title=f"{var_title}: (Pred - Tar) / Tar", **label_configs)
        plotter.save(fig_dir / f"cfad_{var}_rdiff.png")

    # Histogram
    for wp in WPS:
        plotter = BarPlotter()
        for source in sources:
            plotter.plot_bar(
                wp_bins[:-1],
                wp_counts[source][wp] / sample_th,
                label=source.capitalize(),
                **bin_configs,
            )
        plotter.plot_label(title=wp.upper())
        plotter.save(fig_dir / f"{wp}.png")

    # Profile
    plot_profile(
        fig_dir / "profile.png",
        z_tar,
        mean["prediction"],
        mean["target"],
        xlim=[1e-10, 1e-4],
        yticks=label_configs["yticks"],
    )

    # Heat map for Statistics
    plot_heat_map(
        fig_dir / "mae.png",
        np.array(list(mae.values())).transpose(),
        cmap=cmap_scalar,
        bounds=[0, 1e-10, 1e-8, 1e-6, 1e-4],
        extend="max",
        assign_ctick=True,
        ctick_format="{:.0e}",
        title="MAE",
        **heat_configs,
    )

    plot_heat_map(
        fig_dir / "mae_r.png",
        np.array(list(r_mae.values())).transpose() * 100,
        cmap=cmap_scalar,
        bounds=[0, 1e-1, 1e0, 1e1, 1e2],
        extend="max",
        nan_color=nan_color,
        cbar_label="%",
        assign_ctick=True,
        ctick_format="{:.0e}",
        title="Relative MAE",
        **heat_configs,
    )

    plot_heat_map(
        fig_dir / "bias.png",
        np.array(list(bias.values())).transpose(),
        cmap=cmap_diff,
        bounds=[-1e-4, -1e-6, -1e-8, -1e-10, 1e-10, 1e-8, 1e-6, 1e-4],
        extend="both",
        assign_ctick=True,
        ctick_format="{:.0e}",
        title="BIAS",
        **heat_configs,
    )

    plot_heat_map(
        fig_dir / "bias_r.png",
        np.array(list(r_bias.values())).transpose() * 100,
        cmap=cmap_diff,
        cbar_label="%",
        bounds=[-1e2, -1e1, -1e0, -1e-1, 1e-1, 1e0, 1e1, 1e2],
        extend="both",
        nan_color=nan_color,
        assign_ctick=True,
        ctick_format="{:.0e}",
        title="Relative BIAS",
        **heat_configs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "exp",
        type=str,
        help="Enter experiment name.",
    )
    parser.add_argument(
        "--mask",
        type=float,
        default=0.5,
        help="Enter mask threshold for cloud.",
    )
    args = parser.parse_args()

    main(args.exp, mask_threshold=args.mask)
