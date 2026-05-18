#!/usr/bin/env python
"""
Plot GWR local transit catchment coefficient maps.

Creates publication-style coefficient maps for the same six cities used in
visiting_plot.py:
1. fig_gwr_coef_birth.pdf
2. fig_gwr_coef_birth.png
3. fig_gwr_coef_birth_legend.pdf

Usage:
    python src/visualization/gwr_coef_plot.py
    python -m src.visualization.gwr_coef_plot
"""

import argparse
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import box

warnings.filterwarnings("ignore")

try:
    import contextily as ctx
except ImportError:
    ctx = None

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs/figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COEF_DIR = PROJECT_ROOT / "outputs/phase3/gwr_local_coefficients"

LON_COL = "lon"
LAT_COL = "lat"
COEF_COL = "beta_catchment"

SWEDEN_CITIES = ("Stockholm", "Gothenburg", "Malmo")
US_CITIES = ("new_york", "washington_dc", "atlanta")
ALL_CITIES = SWEDEN_CITIES + US_CITIES

CITY_TO_SLUG = {
    "Stockholm": "sweden_stockholm",
    "Gothenburg": "sweden_g\u00f6teborg",
    "Malmo": "sweden_malm\u00f6",
    "new_york": "us_new_york",
    "washington_dc": "us_washington_dc",
    "atlanta": "us_atlanta",
}

CITY_DISPLAY = {
    "Stockholm": "Stockholm",
    "Gothenburg": "Gothenburg",
    "Malmo": "Malm\u00f6",
    "new_york": "New York",
    "washington_dc": "Washington DC",
    "atlanta": "Atlanta",
}

CITY_BOUNDS = {
    "Stockholm": [59.20, 59.45, 17.70, 18.30],
    "Gothenburg": [57.60, 57.80, 11.85, 12.10],
    "Malmo": [55.50, 55.65, 12.90, 13.10],
    "new_york": [40.50, 40.95, -74.30, -73.65],
    "washington_dc": [38.0, 39.75, -78.75, -76.25],
    "atlanta": [32.75, 34.75, -85.25, -83.0],
}

CITY_CENTERS = {
    "Stockholm": [59.3293, 18.0686],
    "Gothenburg": [57.7089, 11.9746],
    "Malmo": [55.6050, 13.0038],
    "new_york": [40.7128, -74.0060],
    "washington_dc": [38.9072, -77.0369],
    "atlanta": [33.7490, -84.3880],
}


# =============================================================================
# HELPERS
# =============================================================================

def setup_mpl():
    """Set matplotlib defaults matching the publication figure scripts."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "lines.linewidth": 0.8,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.pad": 2,
        "ytick.major.pad": 2,
        "xtick.top": False,
        "ytick.right": False,
        "axes.labelpad": 2,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


setup_mpl()


def mm_to_inch(mm):
    return mm / 25.4


def red_gray_blue():
    """Diverging map used by the Figure 4 notebook cell."""
    return mcolors.LinearSegmentedColormap.from_list(
        "RedGrayBlue",
        [
            (0.0, "#b2182b"),
            (0.5, "#bdbdbd"),
            (1.0, "#2166ac"),
        ],
    )


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
    linewidth=1.2,
    fontsize=6,
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


def clean_coef_data(df):
    df = df.copy()
    required_cols = [LON_COL, LAT_COL, COEF_COL]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=[LON_COL, LAT_COL])
    df = df[np.isfinite(df[LON_COL]) & np.isfinite(df[LAT_COL])].copy()
    df[COEF_COL] = pd.to_numeric(df[COEF_COL], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[COEF_COL])
    return df


# =============================================================================
# DATA LOADING
# =============================================================================

def load_gwr_birth_data(city, coef_dir=COEF_DIR):
    """Load pooled birth-background GWR coefficients for one city."""
    city_slug = CITY_TO_SLUG[city]
    all_file = coef_dir / f"{city_slug}_birth_all.parquet"

    if all_file.exists():
        df = pd.read_parquet(all_file)
    else:
        pattern = f"{city_slug}_birth_*.parquet"
        matching_files = [
            f for f in coef_dir.glob(pattern)
            if not f.name.endswith("_birth_all.parquet")
        ]
        if not matching_files:
            return gpd.GeoDataFrame()

        df = pd.concat([pd.read_parquet(f) for f in matching_files], ignore_index=True)

    df = clean_coef_data(df)

    lat_min, lat_max, lon_min, lon_max = CITY_BOUNDS[city]
    df = df[
        (df[LAT_COL] >= lat_min)
        & (df[LAT_COL] <= lat_max)
        & (df[LON_COL] >= lon_min)
        & (df[LON_COL] <= lon_max)
    ].copy()

    if df.empty:
        return gpd.GeoDataFrame()

    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[LON_COL], df[LAT_COL]),
        crs="EPSG:4326",
    ).to_crs(epsg=3857)


# =============================================================================
# PLOTTING
# =============================================================================

def plot_single_gwr_map(
    ax,
    city,
    coef_dir,
    norm,
    cmap,
    point_size=0.7,
    alpha=0.75,
    add_basemap=True,
    add_scalebar=True,
    add_center=True,
):
    gdf = load_gwr_birth_data(city, coef_dir=coef_dir)

    if gdf.empty:
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

    plot_values = gdf[COEF_COL].clip(norm.vmin, norm.vmax)

    ax.scatter(
        gdf.geometry.x,
        gdf.geometry.y,
        c=plot_values,
        cmap=cmap,
        norm=norm,
        s=point_size,
        alpha=alpha,
        linewidths=0,
        rasterized=True,
    )

    extent = city_bounds_to_3857(city, make_square=True)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)

    if add_basemap:
        if ctx is None:
            print(
                f"Warning: contextily is not installed; "
                f"skipping basemap for {CITY_DISPLAY[city]}."
            )
        else:
            try:
                ctx.add_basemap(
                    ax,
                    source=ctx.providers.CartoDB.PositronNoLabels,
                    attribution=False,
                )
            except Exception as exc:
                print(f"Warning: could not add basemap for {CITY_DISPLAY[city]}: {exc}")

    if add_center:
        add_city_center_marker(ax, city)

    if add_scalebar:
        add_scale_bar(ax)

    ax.axis("off")


def plot_gwr_coef_grid(
    coef_dir=COEF_DIR,
    vmin=-0.5,
    vmax=0.5,
    point_size=0.7,
    alpha=0.75,
    add_basemap=True,
):
    cmap = red_gray_blue()
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    rows = [SWEDEN_CITIES, US_CITIES]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(mm_to_inch(138), mm_to_inch(92)),
        constrained_layout=False,
    )
    axes = np.asarray(axes)

    for row_idx, cities in enumerate(rows):
        for col_idx, city in enumerate(cities):
            ax = axes[row_idx, col_idx]

            plot_single_gwr_map(
                ax=ax,
                city=city,
                coef_dir=coef_dir,
                norm=norm,
                cmap=cmap,
                point_size=point_size,
                alpha=alpha,
                add_basemap=add_basemap,
                add_scalebar=True,
                add_center=True,
            )

            ax.text(
                0.5,
                1.035,
                CITY_DISPLAY[city],
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.subplots_adjust(
        left=0.01,
        right=0.995,
        bottom=0.035,
        top=0.955,
        wspace=-0.16,
        hspace=0.24,
    )

    return fig, cmap, norm


def save_colorbar_legend(
    norm,
    cmap,
    out_file,
    label="Local transit catchment coefficient",
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
        transparent=False,
    )

    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot GWR birth-background local catchment coefficients."
    )
    parser.add_argument(
        "--coef-dir",
        type=Path,
        default=COEF_DIR,
        help="Directory containing GWR local coefficient parquet files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for figure outputs.",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=-0.5,
        help="Lower bound for coefficient color scale.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=0.5,
        help="Upper bound for coefficient color scale.",
    )
    parser.add_argument(
        "--no-basemap",
        action="store_true",
        help="Skip contextily basemap tiles.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.coef_dir.exists():
        raise FileNotFoundError(f"GWR coefficient directory not found: {args.coef_dir}")

    fig, cmap, norm = plot_gwr_coef_grid(
        coef_dir=args.coef_dir,
        vmin=args.vmin,
        vmax=args.vmax,
        add_basemap=not args.no_basemap,
    )

    pdf_file = args.output_dir / "fig_gwr_coef_birth.pdf"
    png_file = args.output_dir / "fig_gwr_coef_birth.png"

    fig.savefig(
        pdf_file,
        dpi=600,
        facecolor="white",
        bbox_inches="tight",
    )
    fig.savefig(
        png_file,
        dpi=300,
        facecolor="white",
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved GWR coefficient PDF to: {pdf_file}")
    print(f"Saved GWR coefficient PNG to: {png_file}")

    legend_file = args.output_dir / "fig_gwr_coef_birth_legend.pdf"
    save_colorbar_legend(
        norm=norm,
        cmap=cmap,
        out_file=legend_file,
    )
    print(f"Saved GWR coefficient legend to: {legend_file}")


if __name__ == "__main__":
    main()
