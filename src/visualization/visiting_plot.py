#!/usr/bin/env python
"""
Plot POI visitor diversity maps separately for birth and income.

Creates three PDFs:
1. fig_visitor_diversity_birth.pdf
2. fig_visitor_diversity_income.pdf
3. fig_visitor_diversity_shared_legend.pdf

Usage:
    python -m src.visualization.plot_visitor_diversity_maps
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
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
# HELPERS
# =============================================================================
import matplotlib as mpl
def setup_mpl():
    """Setup matplotlib for publication-quality figures."""
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


def clean_value_data(df, required_cols):
    df = df.copy()

    df = df.dropna(subset=[LON_COL, LAT_COL])
    df = df[
        np.isfinite(df[LON_COL])
        & np.isfinite(df[LAT_COL])
    ].copy()

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required_cols)

    return df


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

def compute_shared_norm(
    all_pois,
    dimensions=("birth", "income"),
    cities=None,
    q_low=0.01,
    q_high=0.99,
):
    """
    Compute one shared normalization across birth and income visitor diversity.

    Filtering rule:
    - New York: keep only POIs with visitor, catchment, and residential values.
    - Other cities: keep POIs with valid visitor value.
    """
    all_values = []

    for dimension in dimensions:
        cols = get_dimension_columns(dimension)

        value_col = cols["visitor"]
        required_cols = [
            cols["visitor"],
            cols["catchment"],
            cols["residential"],
        ]

        tmp = all_pois.copy()

        if cities is not None:
            tmp = tmp[tmp[CITY_COL].isin(cities)].copy()

        city_values = []

        for city, city_df in tmp.groupby(CITY_COL):
            city_df = city_df.copy()

            if city == "new_york":
                city_df = clean_value_data(city_df, required_cols)
            else:
                city_df = city_df.dropna(subset=[LON_COL, LAT_COL])
                city_df = city_df[
                    np.isfinite(city_df[LON_COL])
                    & np.isfinite(city_df[LAT_COL])
                ].copy()

                city_df[value_col] = pd.to_numeric(city_df[value_col], errors="coerce")
                city_df = city_df.replace([np.inf, -np.inf], np.nan)
                city_df = city_df.dropna(subset=[value_col])

            vals = pd.to_numeric(city_df[value_col], errors="coerce")
            vals = vals.replace([np.inf, -np.inf], np.nan).dropna()

            if not vals.empty:
                city_values.append(vals)

        if city_values:
            all_values.append(pd.concat(city_values))

    all_values = pd.concat(all_values)

    vmin = float(all_values.quantile(q_low))
    vmax = float(all_values.quantile(q_high))

    if np.isclose(vmin, vmax):
        vmin = float(all_values.min())
        vmax = float(all_values.max())

    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    return mcolors.Normalize(vmin=vmin, vmax=vmax)

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

    if length_m >= 1000:
        label = f"{int(length_m / 1000)} km"
    else:
        label = f"{int(length_m)} m"

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


# =============================================================================
# PLOTTING
# =============================================================================
def plot_single_visitor_map(
    ax,
    city_df,
    city,
    value_col,
    required_cols,
    norm,
    cmap,
    point_size=0.7,
    alpha=0.65,
    add_basemap=True,
    add_scalebar=True,
    add_center=True,
):
    city_df = city_df.copy()

    # Always require valid lon/lat
    city_df = city_df.dropna(subset=[LON_COL, LAT_COL])
    city_df = city_df[
        np.isfinite(city_df[LON_COL])
        & np.isfinite(city_df[LAT_COL])
    ].copy()

    # New York: require all three values
    if city == "new_york":
        city_df = clean_value_data(city_df, required_cols)

    # Other cities: only require visitor value
    else:
        city_df[value_col] = pd.to_numeric(city_df[value_col], errors="coerce")
        city_df = city_df.replace([np.inf, -np.inf], np.nan)
        city_df = city_df.dropna(subset=[value_col])

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

    if add_basemap:
        ctx.add_basemap(
            ax,
            source=ctx.providers.CartoDB.PositronNoLabels,
            attribution=False,
        )

    if add_center:
        add_city_center_marker(ax, city)

    if add_scalebar:
        add_scale_bar(ax)

    ax.axis("off")


def plot_visitor_diversity_single_dimension(
    all_pois,
    dimension,
    norm,
    sweden_cities=("Stockholm", "Göteborg", "Malmö"),
    us_cities=("new_york", "washington_dc", "atlanta"),
    point_size=0.7,
    alpha=0.65,
    cmap_name="viridis",
):
    cols = get_dimension_columns(dimension)

    value_col = cols["visitor"]

    required_cols = [
        cols["visitor"],
        cols["catchment"],
        cols["residential"],
    ]

    cmap = plt.get_cmap(cmap_name)

    rows = [
        sweden_cities,
        us_cities,
    ]

    ncols = max(len(sweden_cities), len(us_cities))

    fig, axes = plt.subplots(
        nrows=2,
        ncols=ncols,
        figsize=(mm_to_inch(138), mm_to_inch(92)),
        constrained_layout=False,
    )

    axes = np.asarray(axes)

    for row_idx, cities in enumerate(rows):
        for col_idx in range(ncols):
            ax = axes[row_idx, col_idx]

            if col_idx >= len(cities):
                ax.axis("off")
                continue

            city = cities[col_idx]

            city_df = all_pois[
                all_pois[CITY_COL] == city
            ].copy()

            plot_single_visitor_map(
                ax=ax,
                city_df=city_df,
                city=city,
                value_col=value_col,
                required_cols=required_cols,
                norm=norm,
                cmap=cmap,
                point_size=point_size,
                alpha=alpha,
                add_basemap=True,
                add_scalebar=True,
                add_center=True,
            )

            ax.text(
                0.5,
                1.035,
                CITY_DISPLAY.get(city, city),
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

    return fig, cmap


def save_colorbar_legend(
    norm,
    cmap,
    out_file,
    label="Normalized visitor diversity",
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

    cmap_name = "viridis"

    all_cities = [
        "Stockholm",
        "Göteborg",
        "Malmö",
        "new_york",
        "washington_dc",
        "atlanta",
    ]

    shared_norm = compute_shared_norm(
        all_pois=all_pois,
        dimensions=("birth", "income"),
        cities=all_cities,
    )

    fig_birth, cmap_birth = plot_visitor_diversity_single_dimension(
        all_pois=all_pois,
        dimension="birth",
        norm=shared_norm,
        sweden_cities=("Stockholm", "Göteborg", "Malmö"),
        us_cities=("new_york", "washington_dc", "atlanta"),
        point_size=0.7,
        alpha=0.65,
        cmap_name=cmap_name,
    )

    birth_file = OUTPUT_DIR / "fig_visitor_diversity_birth.pdf"

    fig_birth.savefig(
        birth_file,
        dpi=600,
        facecolor="white",
        bbox_inches="tight",
    )

    plt.close(fig_birth)
    print(f"Saved birth figure to: {birth_file}")

    fig_income, cmap_income = plot_visitor_diversity_single_dimension(
        all_pois=all_pois,
        dimension="income",
        norm=shared_norm,
        sweden_cities=("Stockholm", "Göteborg", "Malmö"),
        us_cities=("new_york", "washington_dc", "atlanta"),
        point_size=0.7,
        alpha=0.65,
        cmap_name=cmap_name,
    )

    income_file = OUTPUT_DIR / "fig_visitor_diversity_income.pdf"

    fig_income.savefig(
        income_file,
        dpi=600,
        facecolor="white",
        bbox_inches="tight",
    )

    plt.close(fig_income)
    print(f"Saved income figure to: {income_file}")

    save_colorbar_legend(
        norm=shared_norm,
        cmap=cmap_birth,
        out_file=OUTPUT_DIR / "fig_visitor_diversity_shared_legend.pdf",
        label="Normalized visitor diversity",
    )

    print("Saved shared legend.")


if __name__ == "__main__":
    main()
