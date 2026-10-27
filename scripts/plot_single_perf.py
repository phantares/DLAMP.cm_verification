import argparse
from datetime import datetime, timezone
from pathlib import Path

import h5py as h5
import numpy as np
import pandas as pd
import yaml
from calculators import (
    DetectionCalculator,
    calculate_bin,
    calculate_isotropic_spectrum_sum,
    calculate_vertical_wasserstein,
    calculate_water_path,
    get_spectrum_wavelength,
    smooth_spectrum,
)
from constants import STANDARD_LEVEL, WPS, q_bins, wp_bins
from dotenv import dotenv_values
from plotters import BarPlotter, GridPlotter, PDPlotter, plot_heat_map, plot_profile
from scipy.stats import wasserstein_distance
from utils import find_data_files


def main(
    exp: str, data_source: str, initial_time: datetime, mask_threshold: float = 0.5
) -> None:
    env = dotenv_values(".env")

    fig_dir = Path(env.get("FIG_DIR"), exp, "statistics")
    fig_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(env.get("DATA_DIR"), exp)
    files = find_data_files(data_dir / data_source, initial_time=initial_time)

    sources = ["prediction", "target"]
    source_to_title = {"prediction": exp, "target": "RWRF"}

    with open(data_dir / "config.yaml", "r", encoding="utf-8") as f:
        configs = yaml.safe_load(f)
    grid_spacing = configs["dataset"]["res"]["resolution_target"]
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
    sample_r = {var: 0 for var in vars}
    sample_c = {source: {var: np.zeros(sample_z) for var in vars} for source in sources}
    sample_w = {var: 0 for var in vars_all}

    hits = {var: np.zeros(sample_z) for var in vars}
    misses = {var: np.zeros(sample_z) for var in vars}
    fas = {var: np.zeros(sample_z) for var in vars}
    crs = {var: np.zeros(sample_z) for var in vars}
    std_r = {var: np.zeros(sample_z) for var in vars}

    q_counts = {
        source: {var: np.zeros((len(q_bins) - 1, sample_z)) for var in vars}
        for source in sources
    }
    wp_counts = {
        source: {wp: np.zeros(len(wp_bins) - 1) for wp in WPS} for source in sources
    }
    spectrum_counts = {source: {wp: 0 for wp in WPS} for source in sources}
    spectrum_energy = {source: {wp: 0 for wp in WPS} for source in sources}

    mean = {source: {var: np.zeros(sample_z) for var in vars} for source in sources}
    square = {source: {var: np.zeros(sample_z) for var in vars} for source in sources}
    std = {source: {var: np.zeros(sample_z) for var in vars} for source in sources}
    c_mean = {source: {var: np.zeros(sample_z) for var in vars} for source in sources}
    c_std = {source: {var: np.zeros(sample_z) for var in vars} for source in sources}

    mae = {var: np.zeros(sample_z) for var in vars}
    r_mae = {var: np.zeros(sample_z) for var in vars}
    bias = {var: np.zeros(sample_z) for var in vars}
    r_bias = {var: np.zeros(sample_z) for var in vars}

    stats_metrics = ["Mean", "Std"]
    comparison_metrics = ["MAE", "RelativeMAE", "Bias", "RelativeBias", "VerticalWD"]
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

        wps = {source: {wp: 0 for wp in WPS} for source in sources}

        with h5.File(file, "r") as f:
            cloud = {source: 0 for source in sources}

            for var in vars:
                print(var)

                if var in ["qi", "qs", "qg"]:
                    wp = "iwp"
                elif var in ["qc", "qr"]:
                    wp = "lwp"

                datas = {}
                for source in sources:
                    data = f[f"{source}s"][var][:]
                    data[data < thresholds[var]] = 0

                    if source == "prediction" and use_mask:
                        mask = f["predictions"][f"{var}_mask"][:]
                        data[mask <= mask_threshold] = 0

                    datas[source] = data
                    wps[source][wp] += data
                    cloud[source] += data

                    axes_to_sum = tuple(
                        i for i in range(data.ndim) if i != data.ndim - 3
                    )
                    mean[source][var] += np.sum(data, axis=axes_to_sum)
                    square[source][var] += np.sum(data**2, axis=axes_to_sum)
                wd_sum, wd_sample = calculate_vertical_wasserstein(
                    datas["prediction"], datas["target"], z_tar
                )
                comparisons[var]["VerticalWD"] += wd_sum
                sample_w[var] += wd_sample

                for k in range(sample_z):
                    pred = datas["prediction"][..., k, :, :]
                    tar = datas["target"][..., k, :, :]

                    sample_c["prediction"][var][k] += np.sum(pred > 0)
                    sample_c["target"][var][k] += np.sum(tar > 0)

                    detection = DetectionCalculator(pred, tar, [0])

                    hits[var][k] += detection.hits[0]
                    misses[var][k] += detection.misses[0]
                    fas[var][k] += detection.false_alarms[0]
                    crs[var][k] += detection.correct_rejections[0]

                    q_counts["prediction"][var][:, k] += calculate_bin(
                        q_bins, pred[pred > 0]
                    )
                    q_counts["target"][var][:, k] += calculate_bin(q_bins, tar[tar > 0])

                    diff = pred - tar
                    mae[var][k] += np.sum(np.abs(diff))
                    bias[var][k] += np.sum(diff)

            sample_t += np.size(data, 0)

            for wp in WPS:
                for source in sources:
                    wp_value = calculate_water_path(wps[source][wp], z_tar)

                    s_energy, s_count = calculate_isotropic_spectrum_sum(
                        wp_value, grid_spacing
                    )
                    spectrum_energy[source][wp] += s_energy
                    spectrum_counts[source][wp] += s_count

                    wp_counts[source][wp] += calculate_bin(
                        wp_bins, wp_value[wp_value > 0]
                    )

            wd_sum, wd_sample = calculate_vertical_wasserstein(
                cloud["prediction"], cloud["target"], z_tar
            )
            comparisons["all"]["VerticalWD"] += wd_sum
            sample_w["all"] += wd_sample

    print("calculate statistics")
    sample_h = np.size(data, axis=-1) * np.size(data, axis=-2)
    sample_th = sample_t * sample_h

    for var in vars:
        for source in sources:
            c_mean[source][var] = np.where(
                sample_c[source][var] > 0, mean[source][var] / sample_c[source][var], 0
            )
            c_std[source][var] = np.where(
                sample_c[source][var] > 0,
                square[source][var] / sample_c[source][var] - c_mean[source][var] ** 2,
                0,
            )

            mean[source][var] = mean[source][var] / sample_th
            stats[source][var]["Mean"] = np.sum(mean[source][var]) / sample_z

            std[source][var] = np.sqrt(
                square[source][var] / sample_th - mean[source][var] ** 2
            )
            stats[source][var]["Std"] = np.sqrt(
                np.sum(square[source][var]) / (sample_th * sample_z)
                - stats[source][var]["Mean"] ** 2
            )

        valid_r = mean["target"][var] > 0
        sample_r[var] = np.sum(valid_r)

        mae[var] = mae[var] / sample_th
        comparisons[var]["MAE"] = np.sum(mae[var]) / sample_z

        r_mae[var][valid_r] = mae[var][valid_r] / mean["target"][var][valid_r]
        comparisons[var]["RelativeMAE"] = np.sum(r_mae[var]) / sample_r[var]

        bias[var] = bias[var] / sample_th
        comparisons[var]["Bias"] = np.sum(bias[var]) / sample_z

        r_bias[var][valid_r] = bias[var][valid_r] / mean["target"][var][valid_r]
        comparisons[var]["RelativeBias"] = np.sum(r_bias[var]) / sample_r[var]

    for source in sources:
        stats[source]["all"]["Mean"] = np.sum(list(mean[source].values())) / (
            sample_z * sample_var
        )
        stats[source]["all"]["Std"] = np.sqrt(
            np.sum(list(square[source].values())) / (sample_th * sample_z * sample_var)
            - stats[source]["all"]["Mean"] ** 2
        )
    comparisons["all"]["MAE"] = np.sum(list(mae.values())) / (sample_z * sample_var)
    comparisons["all"]["RelativeMAE"] = np.sum(list(r_mae.values())) / np.sum(
        list(sample_r.values())
    )

    comparisons["all"]["Bias"] = np.sum(list(bias.values())) / (sample_z * sample_var)
    comparisons["all"]["RelativeBias"] = np.sum(list(r_bias.values())) / np.sum(
        list(sample_r.values())
    )

    for var in vars_all:
        comparisons[var]["VerticalWD"] = comparisons[var]["VerticalWD"] / sample_w[var]

    rows = {}

    for source in sources:
        rows[f"{source}_Mean"] = {var: stats[source][var]["Mean"] for var in vars_all}
        rows[f"{source}_Std"] = {var: stats[source][var]["Std"] for var in vars_all}

    for metric in comparison_metrics:
        rows[metric] = {var: comparisons[var][metric] for var in vars_all}

    df = pd.DataFrame(rows).T
    df = df[vars_all]

    row_order = [
        f"{s}_{m}" for m in ["Mean", "Std"] for s in sources
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

    plotter = PDPlotter([{"label": f"Performance Diagram: {exp}", "loc": "left"}])
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
        title_configs=[{"label": f"CSI: {exp}", "loc": "left"}],
        **heat_configs,
    )

    plot_heat_map(
        fig_dir / "hr.png",
        hr.transpose(),
        cmap=cmap_scalar,
        title_configs=[{"label": f"Hit Rate: {exp}", "loc": "left"}],
        **heat_configs,
    )

    plot_heat_map(
        fig_dir / "sr.png",
        sr.transpose(),
        cmap=cmap_scalar,
        title_configs=[{"label": f"Success Ratio: {exp}", "loc": "left"}],
        **heat_configs,
    )

    plot_heat_map(
        fig_dir / "far.png",
        far.transpose(),
        cmap=cmap_scalar,
        title_configs=[{"label": f"False Alarm Rate: {exp}", "loc": "left"}],
        **heat_configs,
    )

    # CFAD
    pressure_edges = np.concatenate([[1050], np.diff(z_tar) / 2 + z_tar[:-1], [0]])

    for var in vars:
        for source in sources:
            prob = q_counts[source][var] / np.sum(q_counts[source][var], axis=0)

            plotter = GridPlotter()
            plotter.plot_pcolor(
                q_bins,
                pressure_edges,
                prob.transpose(),
                cmap=cmap_value,
                **cfad_value_configs,
            )
            plotter.plot_line(c_mean[source][var], z_tar, color="k", linewidth=2)
            plotter.plot_label(
                title_configs=[
                    {
                        "label": rf"${var[0].capitalize()}_{var[1]}$: {source_to_title[source]}",
                        "loc": "left",
                    }
                ],
                **label_configs,
            )
            plotter.save(fig_dir / f"cfad_{var}_{source}.png")

        prediction_prob, target_prob = [
            q_counts[source][var] / np.sum(q_counts[source][var], axis=0)
            for source in sources
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
        plotter.plot_line(c_mean["prediction"][var], z_tar, color="k", linewidth=2)
        plotter.plot_line(
            c_mean["target"][var], z_tar, color="k", linestyle=":", linewidth=2
        )
        plotter.plot_label(
            title_configs=[
                {
                    "label": rf"${var[0].capitalize()}_{var[1]}$ Diff: {source_to_title[source]}",
                    "loc": "left",
                }
            ],
            **label_configs,
        )
        plotter.save(fig_dir / f"cfad_{var}_diff.png")

        plotter = GridPlotter()
        plotter.plot_pcolor(
            q_bins,
            pressure_edges,
            ((prediction_prob - target_prob) / target_prob).transpose() * 100,
            cmap=cmap_diff,
            vmin=-100,
            vmax=100,
            extend="both",
            nan_color=nan_color,
            cbar_label="%",
        )
        plotter.plot_line(c_mean["prediction"][var], z_tar, color="k", linewidth=2)
        plotter.plot_line(
            c_mean["target"][var], z_tar, color="k", linestyle=":", linewidth=2
        )
        plotter.plot_label(
            title_configs=[
                {
                    "label": rf"${var[0].capitalize()}_{var[1]}$ Relative Diff: {source_to_title[source]}",
                    "loc": "left",
                }
            ],
            **label_configs,
        )
        plotter.save(fig_dir / f"cfad_{var}_rdiff.png")

    # WP Histogram
    centers = np.sqrt(wp_bins[1:-1] * wp_bins[2:])

    for wp in WPS:
        plotter = BarPlotter()
        for source in sources:
            plotter.plot_bar(
                wp_bins[:-1],
                wp_counts[source][wp] / np.sum(wp_counts[source][wp]),
                label=source_to_title[source],
                **bin_configs,
            )
        wd = wasserstein_distance(
            centers,
            centers,
            wp_counts["prediction"][wp][1:],
            wp_counts["target"][wp][1:],
        )

        plotter.plot_label(
            xlabel="kg m⁻²",
            title_configs=[
                {"label": f"{wp.upper()}: {exp}", "loc": "left"},
                {"label": f"Wasserstein Distance: {wd:.2f}", "loc": "right"},
            ],
        )
        plotter.save(fig_dir / f"distribution_{wp}.png")

    # WP Spectrum
    for wp in WPS:
        plotter = GridPlotter()

        for source in sources:
            energy = spectrum_energy[source][wp]
            counts = spectrum_counts[source][wp]
            spectrum = np.divide(
                energy, counts, out=np.full_like(energy, np.nan), where=counts > 0
            )
            wavelength_km = get_spectrum_wavelength(
                shape=data.shape[-2:], grid_spacing_km=grid_spacing
            )

            valid_bin = (counts > 0) & np.isfinite(spectrum)
            wavelength_km = wavelength_km[valid_bin]
            spectrum = smooth_spectrum(spectrum[valid_bin])

            plotter.plot_line(
                wavelength_km, spectrum, linewidth=2, label=source_to_title[source]
            )
        plotter.plot_label(
            xlim=[1, 1000],
            invert_x=True,
            xlabel="km",
            ylim=[1e-4, 1e4],
            invert_y=False,
            y_log_scale=True,
            ylabel="Power",
            title_configs=[{"label": f"{wp.upper()}: {exp}", "loc": "left"}],
        )
        plotter.save(fig_dir / f"spectrum_{wp}.png")

    # Profile
    plot_profile(
        fig_dir / "profile.png",
        z_tar,
        c_mean["prediction"],
        c_mean["target"],
        xlim=[1e-8, 1e-2],
        yticks=label_configs["yticks"],
    )

    # Heat map for Statistics
    for var in vars:
        std_r[var] = c_std["prediction"][var] / c_std["target"][var]
    plot_heat_map(
        fig_dir / "std_ratio.png",
        np.array(list(std_r.values())).transpose(),
        cmap=cmap_diff,
        bounds=[0.1, 0.2, 0.5, 0.8, 1.25, 2, 5, 10],
        extend="both",
        nan_color=nan_color,
        assign_ctick=True,
        title_configs=[{"label": f"Std Ratio: {exp}", "loc": "left"}],
        **heat_configs,
    )

    plot_heat_map(
        fig_dir / "mae.png",
        np.array(list(mae.values())).transpose(),
        cmap=cmap_scalar,
        bounds=[0, 1e-10, 1e-8, 1e-6, 1e-4],
        extend="max",
        assign_ctick=True,
        ctick_format="{:.0e}",
        title_configs=[{"label": f"MAE: {exp}", "loc": "left"}],
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
        title_configs=[{"label": f"Relative MAE: {exp}", "loc": "left"}],
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
        title_configs=[{"label": f"Bias: {exp}", "loc": "left"}],
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
        title_configs=[{"label": f"Relative Bias: {exp}", "loc": "left"}],
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
        "--source",
        "-s",
        type=str,
        default="testing",
        help="Enter data source name.",
    )
    parser.add_argument(
        "--initial_time",
        "-i",
        type=str,
        help="Enter initial time of prediction in format YYYYmmddHH.",
    )
    parser.add_argument(
        "--mask",
        type=float,
        default=0.5,
        help="Enter mask threshold for cloud.",
    )
    args = parser.parse_args()

    initial_time = (
        datetime.strptime(args.initial_time, "%Y%m%d%H").replace(tzinfo=timezone.utc)
        if args.initial_time
        else None
    )

    main(args.exp, args.source, initial_time, mask_threshold=args.mask)
