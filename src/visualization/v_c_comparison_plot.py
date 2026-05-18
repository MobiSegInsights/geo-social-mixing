#!/usr/bin/env python
"""
Plot residential diversity, transit catchment diversity, and binned bootstrap effects.

For each city, this figure shows:
1. Residential diversity map
2. Transit/catchment diversity map
3. Error-bar plot:
   x = residential diversity bins
   y = median(visitor diversity - catchment diversity)

Creates:
# fig_birth_residential_catchment_bootstrap_errorbar.pdf
# fig_income_residential_catchment_bootstrap_errorbar.pdf
# fig_birth_residential_catchment_colorbar.pdf
# fig_income_residential_catchment_colorbar.pdf

Usage:
    python -m src.visualization.v_c_comparison_plot
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import contextily as ctx
from shapely.geometry import box

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs/figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ROUTING_DIR = PROJECT_ROOT / "dbs/routing"
US_FILE = ROUTING_DIR / "us_poi_diversity_metrics.parquet"
SWEDEN_FILE = ROUTING_DIR / "sweden_poi_diversity_metrics.parquet"

CITY_COL = "city"
LON_COL = "lon"
LAT_COL = "lat"

BIRTH_VISITOR_COL = "visitor_entropy_birth_norm"
BIRTH_CATCHMENT_COL = "catchment_entropy_birth_norm"
BIRTH_RESIDENTIAL_COL = "residential_entropy_birth_norm"

INCOME_VISITOR_COL = "visitor_entropy_income_norm"
INCOME_CATCHMENT_COL = "catchment_entropy_income_norm"
INCOME_RESIDENTIAL_COL = "residential_entropy_income_norm"

CITY_ORDER = [
    "Stockholm",
    "Göteborg",
    "Malmö",
    "new_york",
    "washington_dc",
    "atlanta",
]

CITY_BOUNDS = {
    "Stockholm": [59.20, 59.45, 17.70, 18.30],
    "Göteborg": [57.60, 57.80, 11.85, 12.10],
    "Malmö": [55.50, 55.65, 12.90, 13.10],
    "new_york": [40.50, 40.95, -74.30, -73.65],
    "washington_dc": [38.0, 39.75, -78.75, -76.25],
    "atlanta": [32.75, 34.75, -85.25, -83],
}

CITY_CENTERS = {
    "Stockholm": [59.3293, 18.0686],
    "Göteborg": [57.7089, 11.9746],
    "Malmö": [55.6050, 13.0038],
    "new_york": [40.7128, -74.0060],
    "washington_dc": [38.9072, -77.0369],
    "atlanta": [33.7490, -84.3880],
}

CITY_DISPLAY = {
    "new_york": "New York",
    "washington_dc": "Washington DC",
    "atlanta": "Atlanta",
    "Stockholm": "Stockholm",
    "Göteborg": "Gothenburg",
    "Malmö": "Malmö",
}


# =============================================================================
# MATPLOTLIB SETUP
# =============================================================================

def setup_mpl():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "lines.linewidth": 0.8,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.pad": 1.5,
        "ytick.major.pad": 1.5,
        "legend.frameon": False,
    })


setup_mpl()


# =============================================================================
# HELPERS
# =============================================================================

def mm_to_inch(mm):
    return mm / 25.4


def get_dimension_columns(dimension):
    if dimension == "birth":
        return {
            "visitor": BIRTH_VISITOR_COL,
            "catchment": BIRTH_CATCHMENT_COL,
            "residential": BIRTH_RESIDENTIAL_COL,
        }

    if dimension == "income":
        return {
            "visitor": INCOME_VISITOR_COL,
            "catchment": INCOME_CATCHMENT_COL,
            "residential": INCOME_RESIDENTIAL_COL,
        }

    raise ValueError("dimension must be either 'birth' or 'income'")


def city_bounds_to_3857(city_name, make_square=True):
    lat_min, lat_max, lon_min, lon_max = CITY_BOUNDS[city_name]

    bbox = gpd.GeoSeries(
        [box(lon_min, lat_min, lon_max, lat_max)],
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    xmin, ymin, xmax, ymax = bbox.total_bounds

    if make_square:
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        side = max(xmax - xmin, ymax - ymin)

        xmin = x_center - side / 2
        xmax = x_center + side / 2
        ymin = y_center - side / 2
        ymax = y_center + side / 2

    return xmin, xmax, ymin, ymax


def city_center_to_3857(city_name):
    lat, lon = CITY_CENTERS[city_name]

    gdf = gpd.GeoDataFrame(
        {"city": [city_name]},
        geometry=gpd.points_from_xy([lon], [lat]),
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    geom = gdf.geometry.iloc[0]
    return geom.x, geom.y


def base_clean_coordinates(df):
    df = df.copy()
    df = df.dropna(subset=[LON_COL, LAT_COL])
    df = df[
        np.isfinite(df[LON_COL])
        & np.isfinite(df[LAT_COL])
    ].copy()
    return df


def clean_available_value(df, value_col):
    df = base_clean_coordinates(df)

    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[value_col])

    return df


def clean_residential_for_map(df, city, visitor_col, catchment_col, residential_col):
    """
    Residential map filtering rule:
    - New York: require visitor, catchment, and residential values.
    - Other cities: require residential value only.
    """
    df = base_clean_coordinates(df)

    if city == "new_york":
        required = [visitor_col, catchment_col, residential_col]
    else:
        required = [residential_col]

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required)

    return df


def clean_catchment_for_map(df, visitor_col, catchment_col, residential_col):
    """
    Transit/catchment map filtering rule:
    - All cities: require visitor, catchment, and residential values.
    """
    df = base_clean_coordinates(df)

    required = [visitor_col, catchment_col, residential_col]

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required)

    return df


def clean_scatter_data(df, visitor_col, catchment_col, residential_col):
    """
    Scatter filtering rule:
    - Need residential, visitor, and catchment values.
    """
    df = base_clean_coordinates(df)

    required = [visitor_col, catchment_col, residential_col]

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required)

    df["visitor_minus_catchment"] = df[visitor_col] - df[catchment_col]

    return df


def add_city_center_marker(ax, city):
    x, y = city_center_to_3857(city)

    ax.scatter(
        [x],
        [y],
        marker="*",
        s=28,
        color="black",
        edgecolor="white",
        linewidth=0.35,
        zorder=11,
    )


def nice_scale_length(width_m):
    raw = width_m / 5
    candidates = np.array([
        500, 1000, 2000, 5000,
        10000, 20000, 50000,
        100000, 200000,
    ])
    return candidates[np.argmin(np.abs(candidates - raw))]


def add_scale_bar(
    ax,
    length_m=None,
    location=(0.94, 0.06),
    linewidth=1.0,
    fontsize=5.5,
):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    width_m = xlim[1] - xlim[0]

    if length_m is None:
        length_m = nice_scale_length(width_m)

    x0 = xlim[0] + location[0] * width_m - length_m
    x1 = x0 + length_m
    y0 = ylim[0] + location[1] * (ylim[1] - ylim[0])

    ax.plot(
        [x0, x1],
        [y0, y0],
        color="black",
        linewidth=linewidth,
        solid_capstyle="butt",
        zorder=10,
    )

    tick_h = 0.012 * (ylim[1] - ylim[0])

    ax.plot([x0, x0], [y0 - tick_h, y0 + tick_h], color="black", linewidth=linewidth, zorder=10)
    ax.plot([x1, x1], [y0 - tick_h, y0 + tick_h], color="black", linewidth=linewidth, zorder=10)

    label = f"{int(length_m / 1000)} km" if length_m >= 1000 else f"{int(length_m)} m"

    ax.text(
        (x0 + x1) / 2,
        y0 + 0.018 * (ylim[1] - ylim[0]),
        label,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        color="black",
        zorder=10,
    )


# =============================================================================
# BOOTSTRAP BINNING
# =============================================================================

def bootstrap_median_ci(values, n_boot=500, ci=95, random_state=42):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(random_state)

    med = np.median(values)

    if len(values) == 1:
        return med, med, med

    boot = np.empty(n_boot)

    for i in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boot[i] = np.median(sample)

    lo = np.percentile(boot, (100 - ci) / 2)
    hi = np.percentile(boot, 100 - (100 - ci) / 2)

    return med, lo, hi


def make_equal_count_bins(
    df,
    x_col,
    y_col,
    n_bins=15,
    n_boot=500,
    ci=95,
    min_n=10,
    random_state=42,
):
    """
    Bin x into equal-count quantile groups.
    For each bin, bootstrap median y.
    """
    tmp = df[[x_col, y_col]].copy()
    tmp = tmp.replace([np.inf, -np.inf], np.nan)
    tmp = tmp.dropna()

    if tmp.empty:
        return pd.DataFrame(columns=["x", "y50", "yerr_low", "yerr_high", "n"])

    tmp = tmp.sort_values(x_col).reset_index(drop=True)

    try:
        tmp["bin"] = pd.qcut(
            tmp[x_col],
            q=n_bins,
            labels=False,
            duplicates="drop",
        )
    except ValueError:
        return pd.DataFrame(columns=["x", "y50", "yerr_low", "yerr_high", "n"])

    records = []

    for b, sub in tmp.groupby("bin"):
        if len(sub) < min_n:
            continue

        med, lo, hi = bootstrap_median_ci(
            sub[y_col].values,
            n_boot=n_boot,
            ci=ci,
            random_state=random_state + int(b),
        )

        x_mid = float(sub[x_col].median())

        records.append({
            "bin": int(b),
            "x": x_mid,
            "y50": med,
            "yerr_low": med - lo,
            "yerr_high": hi - med,
            "n": len(sub),
        })

    return pd.DataFrame(records)


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_dimension_data(all_pois, dimension, cities=CITY_ORDER):
    cols = get_dimension_columns(dimension)

    city_data = {}

    for city in cities:
        city_df = all_pois[all_pois[CITY_COL] == city].copy()

        residential_df = clean_residential_for_map(
            city_df,
            city=city,
            visitor_col=cols["visitor"],
            catchment_col=cols["catchment"],
            residential_col=cols["residential"],
        )

        catchment_df = clean_catchment_for_map(
            city_df,
            visitor_col=cols["visitor"],
            catchment_col=cols["catchment"],
            residential_col=cols["residential"],
        )

        scatter_df = clean_scatter_data(
            city_df,
            visitor_col=cols["visitor"],
            catchment_col=cols["catchment"],
            residential_col=cols["residential"],
        )

        city_data[city] = {
            "residential": residential_df,
            "catchment": catchment_df,
            "scatter": scatter_df,
        }

    return city_data, cols


def compute_norms(city_data, cols):
    residential_values = []
    catchment_values = []
    diff_values = []

    for parts in city_data.values():
        if not parts["residential"].empty:
            residential_values.append(parts["residential"][cols["residential"]])

        if not parts["catchment"].empty:
            catchment_values.append(parts["catchment"][cols["catchment"]])

        if not parts["scatter"].empty:
            diff_values.append(parts["scatter"]["visitor_minus_catchment"])

    residential_values = pd.concat(residential_values)
    catchment_values = pd.concat(catchment_values)
    diff_values = pd.concat(diff_values)

    div_values = pd.concat([residential_values, catchment_values])

    div_vmin = float(div_values.quantile(0.01))
    div_vmax = float(div_values.quantile(0.99))

    if np.isclose(div_vmin, div_vmax):
        div_vmin = float(div_values.min())
        div_vmax = float(div_values.max())

    if np.isclose(div_vmin, div_vmax):
        div_vmax = div_vmin + 1e-6

    y_abs = float(np.nanquantile(np.abs(diff_values), 0.99))
    if np.isclose(y_abs, 0):
        y_abs = 1e-6

    diversity_norm = mcolors.Normalize(vmin=div_vmin, vmax=div_vmax)

    return diversity_norm, (-y_abs, y_abs)


# =============================================================================
# PLOTTING
# =============================================================================

def plot_map_panel(
    ax,
    city_df,
    city,
    value_col,
    norm,
    cmap,
    point_size=0.55,
    alpha=0.70,
):
    if city_df.empty:
        ax.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=7,
        )
        ax.axis("off")
        return

    gdf = gpd.GeoDataFrame(
        city_df,
        geometry=gpd.points_from_xy(city_df[LON_COL], city_df[LAT_COL]),
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    extent = city_bounds_to_3857(city, make_square=True)

    gdf.plot(
        ax=ax,
        column=value_col,
        cmap=cmap,
        norm=norm,
        markersize=point_size,
        alpha=alpha,
        linewidth=0,
        rasterized=True,
    )

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)

    ctx.add_basemap(
        ax,
        source=ctx.providers.CartoDB.PositronNoLabels,
        attribution=False,
    )

    add_city_center_marker(ax, city)
    add_scale_bar(ax)

    ax.axis("off")

def plot_binned_errorbar_panel(
    ax,
    city_df,
    residential_col,
    visitor_col,
    diff_col="visitor_minus_catchment",
    xlim=None,
    ylim_diff=None,
    ylim_visitor=None,
    n_bins=15,
    n_boot=500,
    diff_color="#0fbcf9",
    visitor_color="#05c46b",
):
    if city_df.empty:
        ax.text(
            0.5, 0.5, "No data",
            ha="center", va="center",
            transform=ax.transAxes,
            fontsize=6,
        )
        ax.set_xlabel("R")
        ax.text(0.02, 1.04, "V-T", transform=ax.transAxes, ha="left", va="bottom", fontsize=6)
        ax_r = ax.twinx()
        ax_r.text(0.98, 1.04, "V", transform=ax_r.transAxes, ha="right", va="bottom", fontsize=6)
        return

    # --------------------------------------------------
    # Left axis: V-T
    # --------------------------------------------------
    binned_diff = make_equal_count_bins(
        city_df,
        x_col=residential_col,
        y_col=diff_col,
        n_bins=n_bins,
        n_boot=n_boot,
        ci=95,
        min_n=10,
    )

    # --------------------------------------------------
    # Right axis: V
    # --------------------------------------------------
    binned_visitor = make_equal_count_bins(
        city_df,
        x_col=residential_col,
        y_col=visitor_col,
        n_bins=n_bins,
        n_boot=n_boot,
        ci=95,
        min_n=10,
    )

    if binned_diff.empty and binned_visitor.empty:
        ax.text(
            0.5, 0.5, "Too few data",
            ha="center", va="center",
            transform=ax.transAxes,
            fontsize=6,
        )
        ax.set_xlabel("R")
        ax.text(0.02, 1.04, "V-T", transform=ax.transAxes, ha="left", va="bottom", fontsize=6)
        ax_r = ax.twinx()
        ax_r.text(0.98, 1.04, "V", transform=ax_r.transAxes, ha="right", va="bottom", fontsize=6)
        return

    ax_r = ax.twinx()

    def draw_errorbar_and_fit(axis, binned, color, y_is_positive=False):
        if binned.empty:
            return

        x = binned["x"].to_numpy(dtype=float)
        y50 = binned["y50"].to_numpy(dtype=float)
        yerr_low = binned["yerr_low"].to_numpy(dtype=float)
        yerr_high = binned["yerr_high"].to_numpy(dtype=float)

        axis.errorbar(
            x,
            y50,
            yerr=[yerr_low, yerr_high],
            fmt="o",
            capsize=1.5,
            color=color,
            markersize=2.0,
            linewidth=0.50,
            elinewidth=0.50,
            capthick=0.50,
            alpha=0.95,
        )

        mask = np.isfinite(x) & np.isfinite(y50)

        if mask.sum() >= 2:
            x_fit = x[mask]
            y_fit = y50[mask]

            m, b = np.polyfit(x_fit, y_fit, deg=1)
            xs = np.linspace(x_fit.min(), x_fit.max(), 200)

            axis.plot(
                xs,
                m * xs + b,
                color=color,
                lw=0.55,
                alpha=0.70,
            )

    draw_errorbar_and_fit(ax, binned_diff, diff_color)
    draw_errorbar_and_fit(ax_r, binned_visitor, visitor_color)

    ax.axhline(0, color="black", linewidth=0.5, alpha=0.70)

    if xlim is not None:
        ax.set_xlim(xlim)
        ax_r.set_xlim(xlim)

    if ylim_diff is not None:
        ax.set_ylim(ylim_diff)

    if ylim_visitor is not None:
        ax_r.set_ylim(ylim_visitor)

    ax.grid(True, linewidth=0.25, alpha=0.22)

    for spine in ["top"]:
        ax.spines[spine].set_visible(False)
        ax_r.spines[spine].set_visible(False)

    ax.spines["right"].set_visible(False)
    ax_r.spines["left"].set_visible(False)

    ax.set_xlabel("R", fontsize=6)
    ax.set_ylabel("")
    ax_r.set_ylabel("")

    ax.text(
        0.02,
        1.04,
        "V-T",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6,
        color=diff_color,
    )

    ax_r.text(
        0.98,
        1.04,
        "V",
        transform=ax_r.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
        color=visitor_color,
    )

    ax.tick_params(
        axis="both",
        labelsize=5,
        length=2,
        width=0.5,
        colors="black",
    )

    ax_r.tick_params(
        axis="y",
        labelsize=5,
        length=2,
        width=0.5,
        colors=visitor_color,
    )

    ax.tick_params(axis="y", colors=diff_color)

def plot_dimension_figure(
    all_pois,
    dimension,
    diversity_norm,
    cities_sweden=("Stockholm", "Göteborg", "Malmö"),
    cities_us=("new_york", "washington_dc", "atlanta"),
    diversity_cmap_name="viridis",
    diff_color="#0fbcf9",
    visitor_color="#05c46b",
):
    city_order = list(cities_sweden) + list(cities_us)

    city_data, cols = prepare_dimension_data(
        all_pois,
        dimension,
        cities=city_order,
    )

    diversity_cmap = plt.get_cmap(diversity_cmap_name)

    all_x = pd.concat([
        parts["scatter"][cols["residential"]]
        for parts in city_data.values()
        if not parts["scatter"].empty
    ])

    all_y_diff = pd.concat([
        parts["scatter"]["visitor_minus_catchment"]
        for parts in city_data.values()
        if not parts["scatter"].empty
    ])

    all_y_visitor = pd.concat([
        parts["scatter"][cols["visitor"]]
        for parts in city_data.values()
        if not parts["scatter"].empty
    ])

    xlim = (
        float(all_x.quantile(0.01)),
        float(all_x.quantile(0.99)),
    )

    y_abs = float(np.nanquantile(np.abs(all_y_diff), 0.99))
    if np.isclose(y_abs, 0):
        y_abs = 1e-6

    ylim_diff = (-y_abs, y_abs)

    ylim_visitor = (
        float(all_y_visitor.quantile(0.01)),
        float(all_y_visitor.quantile(0.99)),
    )

    fig, axes = plt.subplots(
        nrows=3,
        ncols=9,
        figsize=(mm_to_inch(150), mm_to_inch(82)),
        gridspec_kw={
            "width_ratios": [1, 1, 0.34, 0.68, 0.30, 1, 1, 0.34, 0.68],
        },
        constrained_layout=False,
    )

    for r in range(3):
        axes[r, 2].axis("off")
        axes[r, 4].axis("off")
        axes[r, 7].axis("off")

    def draw_city_row(row_idx, city, col_res, col_cat, col_err):
        parts = city_data[city]

        ax_res = axes[row_idx, col_res]
        ax_cat = axes[row_idx, col_cat]
        ax_err = axes[row_idx, col_err]

        plot_map_panel(
            ax=ax_res,
            city_df=parts["residential"],
            city=city,
            value_col=cols["residential"],
            norm=diversity_norm,
            cmap=diversity_cmap,
            point_size=0.45,
            alpha=0.70,
        )

        plot_map_panel(
            ax=ax_cat,
            city_df=parts["catchment"],
            city=city,
            value_col=cols["catchment"],
            norm=diversity_norm,
            cmap=diversity_cmap,
            point_size=0.45,
            alpha=0.70,
        )

        plot_binned_errorbar_panel(
            ax=ax_err,
            city_df=parts["scatter"],
            residential_col=cols["residential"],
            visitor_col=cols["visitor"],
            diff_col="visitor_minus_catchment",
            xlim=xlim,
            ylim_diff=ylim_diff,
            ylim_visitor=ylim_visitor,
            n_bins=15,
            n_boot=500,
            diff_color=diff_color,
            visitor_color=visitor_color,
        )

        ax_err.set_box_aspect(1)

        ax_res.text(
            -0.07,
            0.5,
            CITY_DISPLAY.get(city, city),
            transform=ax_res.transAxes,
            ha="right",
            va="center",
            rotation=90,
            fontsize=7,
        )

        n_res = len(parts["residential"])
        n_cat = len(parts["catchment"])
        share = 100 * n_cat / n_res if n_res > 0 else np.nan

        count_label = f"n={n_cat:,} ({share:.0f}%)" if np.isfinite(share) else f"n={n_cat:,}"

        ax_cat.text(
            0.5,
            -0.08,
            count_label,
            transform=ax_cat.transAxes,
            ha="center",
            va="top",
            fontsize=5.5,
        )

    for row_idx, city in enumerate(cities_sweden):
        draw_city_row(
            row_idx=row_idx,
            city=city,
            col_res=0,
            col_cat=1,
            col_err=3,
        )

    for row_idx, city in enumerate(cities_us):
        draw_city_row(
            row_idx=row_idx,
            city=city,
            col_res=5,
            col_cat=6,
            col_err=8,
        )

    # Aligned column labels using fig.text instead of axes titles
    fig.text(0.135, 0.955, "Residential", ha="center", va="bottom", fontsize=7)
    fig.text(0.268, 0.955, "Transit", ha="center", va="bottom", fontsize=7)
    fig.text(0.620, 0.955, "Residential", ha="center", va="bottom", fontsize=7)
    fig.text(0.753, 0.955, "Transit", ha="center", va="bottom", fontsize=7)

    plt.subplots_adjust(
        left=0.035,
        right=0.995,
        bottom=0.065,
        top=0.925,
        wspace=0.08,
        hspace=0.22,
    )

    return fig

def save_diversity_colorbar(
    norm,
    cmap,
    out_file,
    label="Diversity",
):
    fig = plt.figure(figsize=(mm_to_inch(70), mm_to_inch(8)))

    cax = fig.add_axes([0.05, 0.35, 0.90, 0.28])

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        cax=cax,
        orientation="horizontal",
    )

    cbar.set_label(label, fontsize=7)
    cbar.ax.tick_params(labelsize=6, length=2, width=0.4)

    cbar.outline.set_visible(False)

    for spine in cbar.ax.spines.values():
        spine.set_visible(False)

    fig.savefig(
        out_file,
        dpi=600,
        facecolor="white",
        bbox_inches="tight",
    )

    plt.close(fig)
# =============================================================================
# DATA LOADING
# =============================================================================

def load_poi_data():
    dfs = []

    if US_FILE.exists():
        us = pd.read_parquet(US_FILE)
        us["country"] = "US"
        dfs.append(us)
        print(f"Loaded US file: {US_FILE}")

    if SWEDEN_FILE.exists():
        sweden = pd.read_parquet(SWEDEN_FILE)
        sweden["country"] = "Sweden"
        dfs.append(sweden)
        print(f"Loaded Sweden file: {SWEDEN_FILE}")

    if not dfs:
        raise FileNotFoundError(
            f"No input files found:\n{US_FILE}\n{SWEDEN_FILE}"
        )

    all_pois = pd.concat(dfs, ignore_index=True)

    required_cols = [
        CITY_COL,
        LON_COL,
        LAT_COL,
        BIRTH_VISITOR_COL,
        BIRTH_CATCHMENT_COL,
        BIRTH_RESIDENTIAL_COL,
        INCOME_VISITOR_COL,
        INCOME_CATCHMENT_COL,
        INCOME_RESIDENTIAL_COL,
    ]

    missing = [col for col in required_cols if col not in all_pois.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return all_pois


# =============================================================================
# MAIN
# =============================================================================
def main():
    all_pois = load_poi_data()

    for dimension in ["birth", "income"]:
        city_data, cols = prepare_dimension_data(
            all_pois,
            dimension,
            cities=CITY_ORDER,
        )

        diversity_norm, _ = compute_norms(city_data, cols)
        diversity_cmap = plt.get_cmap("viridis")

        fig = plot_dimension_figure(
            all_pois=all_pois,
            dimension=dimension,
            diversity_norm=diversity_norm,
            cities_sweden=("Stockholm", "Göteborg", "Malmö"),
            cities_us=("new_york", "washington_dc", "atlanta"),
            diversity_cmap_name="viridis",
            diff_color="#0fbcf9",
            visitor_color="#05c46b",
        )
        out_file = OUTPUT_DIR / f"fig_{dimension}_residential_catchment_bootstrap_errorbar.pdf"

        fig.savefig(
            out_file,
            dpi=600,
            facecolor="white",
            bbox_inches="tight",
        )

        plt.close(fig)
        print(f"Saved figure to: {out_file}")

        legend_file = OUTPUT_DIR / f"fig_{dimension}_residential_catchment_colorbar.pdf"

        save_diversity_colorbar(
            norm=diversity_norm,
            cmap=diversity_cmap,
            out_file=legend_file,
            label="Diversity",
        )

        print(f"Saved legend to: {legend_file}")

if __name__ == "__main__":
    main()