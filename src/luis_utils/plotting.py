"""Plotting and visualization utilities."""

from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogFormatterMathtext


@dataclass
class Colors:
    """
    UM Colors: https://brand.umich.edu/design-resources/colors/
    """

    # PointState
    FREE: Tuple[float, float, float] = (255 / 255, 255 / 255, 255 / 255)  # White
    OCCUPIED: Tuple[float, float, float] = (47 / 255, 101 / 255, 167 / 255)  # (UM) Arboretum Blue
    LOW_FRICTION: Tuple[float, float, float] = (255 / 255, 203 / 255, 5 / 255)
    UNKNOWN: Tuple[float, float, float] = (100 / 255, 100 / 255, 100 / 255)  # Gray

    # Other
    ROBOT: Tuple[float, float, float] = (0 / 255, 0 / 255, 0 / 255)
    GOAL: Tuple[float, float, float] = (198 / 255, 70 / 255, 227 / 255)
    SENSOR_BOUNDARY = (0, 0, 255, 150)  # Blue, semi-transparent (150 out of 255 alpha)


# Color/style defaults
DEFAULT_COLORS = ["#4169E1", "#B22222", "#2E8B57", "#FFA500", "#8A2BE2"]
DEFAULT_LINESTYLES = ["solid", "dash", "dot", "dashdot"]


def visualize_confidence_rerun(  # pylint: disable=too-many-positional-arguments
    boundary_mesh,  # pyvista PolyData with .points and .faces
    sampled_points_MC,
    boundary_pts,
    mask_algebra,
    convex_hull,
    mask,  # (N,) boolean array: True=inside, False=outside
    save_path: Optional[str] = None,
    marker_size: float = 0.01,
):
    """Visualize confidence regions using rerun."""
    rr.init("confidence_rerun")
    # rr.notebook_show()  # comment out if you don’t need the inline viewer

    # 1) log the 3D mesh
    verts = boundary_mesh.points.tolist()  # (V,3)
    faces_raw = boundary_mesh.faces
    tris = []
    i = 0
    while i < len(faces_raw):
        if faces_raw[i] == 3:  # triangle
            tris.append([faces_raw[i + 1], faces_raw[i + 2], faces_raw[i + 3]])
            i += 4
        else:
            i += faces_raw[i] + 1  # skip non-triangular faces

    rr.log(
        "confidence/region",
        rr.Mesh3D(
            vertex_positions=verts,  # required
            triangle_indices=tris,  # required
        ),
    )

    # split your N points into three categories:
    pts = sampled_points_MC.tolist()
    both_mask = mask_algebra & mask
    drop_mask = mask_algebra & ~mask
    out_mask = ~mask_algebra

    rr.log(
        "confidence/both",
        rr.Points3D(
            positions=[pts[i] for i in range(len(pts)) if both_mask[i]],
            radii=[marker_size] * both_mask.sum(),
            colors=[0, 255, 0],  # green = “in algebra *and* in state”
        ),
    )

    rr.log(
        "confidence/dropped",
        rr.Points3D(
            positions=[pts[i] for i in range(len(pts)) if drop_mask[i]],
            radii=[marker_size] * drop_mask.sum(),
            colors=[255, 165, 0],  # orange = “in algebra *but not* in state”
        ),
    )

    rr.log(
        "confidence/outside",
        rr.Points3D(
            positions=[pts[i] for i in range(len(pts)) if out_mask[i]],
            radii=[marker_size] * out_mask.sum(),
            colors=[255, 0, 0],  # red = “outside algebra”
        ),
    )

    # boundary points as before…
    rr.log("confidence/boundary_pts", rr.Points3D(positions=boundary_pts.tolist(), radii=[marker_size] * len(boundary_pts), colors=[0, 0, 255]))

    # Fix convex_hull mesh as well
    convex_verts = convex_hull.points.tolist()
    convex_faces_raw = convex_hull.faces
    convex_tris = []
    i = 0
    while i < len(convex_faces_raw):
        if convex_faces_raw[i] == 3:  # triangle
            convex_tris.append([convex_faces_raw[i + 1], convex_faces_raw[i + 2], convex_faces_raw[i + 3]])
            i += 4
        else:
            i += convex_faces_raw[i] + 1  # skip non-triangular faces

    rr.log(
        "confidence/convex_hull",
        rr.Mesh3D(
            vertex_positions=convex_verts,
            triangle_indices=convex_tris,
        ),
    )

    if save_path:
        rr.save(f"{save_path}.rrd")


def plot_accuracy_and_runtime(  # pylint: disable=too-many-positional-arguments
    lie_metrics: dict,
    state_space_metrics: dict,
    figsize=(12, 3.5),
    ncols: int = 2,
    annotate_lie: bool = False,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Plot accuracy and runtime metrics for different integrators."""

    # figure setup
    if ncols == 2:
        fig, (ax1, _) = plt.subplots(1, 2, figsize=figsize, constrained_layout=False)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(figsize[0] / 2, figsize[1]), constrained_layout=False)

    plt.subplots_adjust(top=0.85, right=0.98)

    # union of all integrator names
    integrators = sorted(set(state_space_metrics) | set(lie_metrics))
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {name: default_colors[i % len(default_colors)] for i, name in enumerate(integrators)}

    # --- (1) error vs dt ---

    # annotate lie if requested
    # custom_offsets = {
    #     "SS fe": (0.0, 0.01),
    #     "SS se": (0.0, 0.01),
    #     "SS heun": (0.0, 0.005),
    #     "SS rk2": (0.0, 0.01),
    #     "SS rk4": (-0.1e-3, 3e-14),
    #     "Lie cf4": (0.0, -3e-15),
    #     "Lie rk2": (0.0, 10),
    #     "Lie se": (0.0, 10),
    #     "Lie heun": (0.0, 10),
    #     "Lie rk4": (0.0, 10),
    #     "Lie fe": (0.0, 0.001),
    #     "Lie dla01": (0.0, -2e-8),
    # }
    for name in integrators:
        c = colors[name]

        # state‐space
        if name in state_space_metrics:
            a = state_space_metrics[name]
            dts_a = np.array(sorted(a["q_errors"]))
            errs_a = np.array([a["q_errors"][dt] for dt in dts_a])
            if dts_a.size >= 2:
                _ = np.polyfit(np.log(dts_a), np.log(errs_a), 1)[0]  # type: ignore[no-untyped-call]
            ax1.loglog(dts_a, errs_a, "-", marker="s", color=c, alpha=0.5)

        # lie‐group
        if name in lie_metrics:
            l = lie_metrics[name]
            dts_l = np.array(sorted(l["q_errors"]))
            errs_l = np.array([l["q_errors"][dt] for dt in dts_l])
            if dts_l.size >= 2:
                _ = np.polyfit(np.log(dts_l), np.log(errs_l), 1)[0]  # type: ignore[no-untyped-call]
            ax1.loglog(dts_l, errs_l, "--", marker="o", color=c, alpha=0.5)

    ax1.set_xlabel(r"Timestep $\Delta t$ (sec)")
    ax1.set_ylabel(r"RMSE of $e_1$ Relative to Reference Trajectory")
    ax1.grid(True, which="major", ls="--", alpha=0.5)
    ax1.minorticks_off()
    ax1.set_xticks(sorted({dt for m in state_space_metrics.values() for dt in m["q_errors"]}))
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.1e}"))
    ax1.set_yticks([10**i for i in range(-15, 0)])
    ax1.yaxis.set_major_formatter(LogFormatterMathtext())
    ax1.tick_params(axis="both", which="major", direction="in")

    # annotate lie if requested
    if annotate_lie:
        dt_max = max((max(m["q_errors"]) for m in lie_metrics.values()), default=None)
        if dt_max:
            x_edge = dt_max * 1.15
            ax1.set_xlim(ax1.get_xlim()[0], x_edge * 1.02)
            dy = 8
            offsets = np.linspace(-dy * (len(integrators) - 1) / 2, +dy * (len(integrators) - 1) / 2, len(integrators)) + 10
            for i, name in enumerate(integrators):
                if name in lie_metrics:
                    errs_l = np.array([lie_metrics[name]["q_errors"][dt] for dt in sorted(lie_metrics[name]["q_errors"])])
                    ax1.annotate(
                        name,
                        xy=(dt_max, errs_l[-1]),
                        xytext=(5, offsets[i]),
                        textcoords="offset points",
                        ha="left",
                        va="center",
                        color=colors[name],
                        fontsize=8,
                        clip_on=False,
                    )

    # --- (2) error vs runtime ---
    # if ncols == 2:
    #     for name in integrators:
    #         c = colors[name]
    #         if name in state_space_metrics:
    #             a = state_space_metrics[name]
    #             ts = np.array([a["total_time"][dt] for dt in sorted(a["total_time"])])
    #             es = np.array([a["q_errors"][dt] for dt in sorted(a["q_errors"])])
    #             ax2.loglog(ts, es, '-', marker='s', color=c, alpha=0.8)
    #         if name in lie_metrics:
    #             l = lie_metrics[name]
    #             ts = np.array([l["total_time"][dt] for dt in sorted(l["total_time"])])
    #             es = np.array([l["q_errors"][dt] for dt in sorted(l["q_errors"])])
    #             ax2.loglog(ts, es, '--', marker='o', color=c, alpha=0.8)

    #     ax2.set_xlabel("Wall-clock time (ms)")
    #     ax2.set_ylabel(r"$\log(\tilde g^{-1}g)$ Error Relative to Reference (RMSE over trajectory)")
    #     ax2.grid(True, which='both', ls='--', alpha=0.5)

    # build legends
    # type_handles = [
    #     Line2D([0], [0], linestyle="-", marker="s", color="black", label="State Space"),
    #     Line2D([0], [0], linestyle="--", marker="o", color="black", label="Lie Group"),
    # ]
    integrator_handles = [Line2D([0], [0], color=colors[name], linestyle="-", label=name) for name in integrators]
    # fig.legend(handles=type_handles,   title="Dynamics Representation", loc="upper center",
    #    ncol=2, frameon=False, bbox_to_anchor=(0.5,1.20))
    fig.legend(
        handles=integrator_handles,
        title="Numerical Integrator",
        loc="upper center",
        ncol=len(integrators),
        frameon=False,
        bbox_to_anchor=(0.5, 0.99),
        fontsize=8,
        title_fontsize=9,
    )

    if save_path:
        fig.savefig(save_path, dpi=300)
    if show:
        plt.show()
