#!/usr/bin/env python
"""
Compute spatial control variables for POIs.

Adds to diversity metrics:
- dist_to_center_km: Distance to city center (haversine)
- poi_density_500m: Count of POIs within 500m radius
- dist_to_transit_m: Distance to nearest transit stop

These variables are used for:
1. Stepwise OLS robustness checks (R2.5)
2. Hotspot characterization (notebook 16, Section 3b)

Usage:
    python -m src.features.compute_spatial_controls
"""

import argparse
import zipfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

def get_project_root():
    """Get project root directory."""
    script_dir = Path(__file__).resolve().parent
    if script_dir.name == 'features' and script_dir.parent.name == 'src':
        return script_dir.parent.parent
    return Path.cwd()

PROJECT_ROOT = get_project_root()

ROUTING_DIR = PROJECT_ROOT / 'dbs/routing'
GTFS_DIR = PROJECT_ROOT / 'dbs/gtfs'

# City center coordinates (lat, lon)
# Swedish cities: central station or main square
# US cities: downtown core
CITY_CENTERS = {
    # Sweden
    'Stockholm': (59.3293, 18.0686),      # Central Station
    'Göteborg': (57.7089, 11.9746),       # Central Station
    'Malmö': (55.6059, 13.0007),          # Central Station
    'Uppsala': (59.8588, 17.6389),        # Central Station
    'Västerås': (59.6099, 16.5448),       # Central Station
    'Örebro': (59.2753, 15.2134),         # Central Station
    'Linköping': (58.4169, 15.6253),      # Central Station
    'Helsingborg': (56.0465, 12.6945),    # Central Station
    'Lund': (55.7047, 13.1910),           # Central Station
    # US
    'new_york': (40.7128, -74.0060),      # Lower Manhattan
    'washington_dc': (38.8951, -77.0364), # National Mall
    'atlanta': (33.7490, -84.3880),       # Five Points
}

# GTFS paths
US_CITY_GTFS = {
    'new_york': GTFS_DIR / 'new_york',
    'washington_dc': GTFS_DIR / 'washington_dc',
    'atlanta': GTFS_DIR / 'atlanta',
}

# Swedish city to county code mapping
# Each city uses its county's filtered GTFS data
SWEDEN_CITY_COUNTY = {
    'Stockholm': '01',
    'Södertälje': '01',
    'Uppsala': '03',
    'Linköping': '05',
    'Norrköping': '05',
    'Jönköping': '06',
    'Växjö': '07',
    'Helsingborg': '12',
    'Lund': '12',
    'Malmö': '12',
    'Halmstad': '13',
    'Göteborg': '14',
    'Borås': '14',
    'Trollhättan': '14',
    'Karlstad': '17',
    'Örebro': '18',
    'Västerås': '19',
    'Eskilstuna': '04',
}

def get_sweden_gtfs_path(city):
    """Get GTFS path for a Swedish city based on its county."""
    county = SWEDEN_CITY_COUNTY.get(city)
    if county:
        return GTFS_DIR / f'sweden_south/c_{county}'
    return None


# =============================================================================
# DISTANCE FUNCTIONS
# =============================================================================

def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate haversine distance in kilometers."""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def compute_dist_to_center(df, city_col='city', lat_col='lat', lon_col='lon'):
    """
    Compute distance to city center for each POI.

    Returns array of distances in km.
    """
    distances = np.full(len(df), np.nan)

    for city, center in CITY_CENTERS.items():
        # Match city names (handle both "Stockholm" and "Sweden - Stockholm" formats)
        mask = df[city_col].str.contains(city, case=False, na=False)
        if mask.sum() == 0:
            continue

        city_lats = df.loc[mask, lat_col].values
        city_lons = df.loc[mask, lon_col].values

        valid = ~(np.isnan(city_lats) | np.isnan(city_lons))
        if valid.sum() > 0:
            dists = haversine_km(
                city_lats[valid], city_lons[valid],
                center[0], center[1]
            )
            distances[np.where(mask)[0][valid]] = dists

    return distances


def compute_poi_density(df, lat_col='lat', lon_col='lon', radius_m=500):
    """
    Compute POI density within radius for each POI.

    Uses approximate meter conversion for efficiency.
    Returns array of counts (excluding self).
    """
    valid_mask = df[lat_col].notna() & df[lon_col].notna()
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return np.full(len(df), np.nan)

    # Approximate conversion to meters (at mid-latitude)
    mean_lat = df.loc[valid_mask, lat_col].mean()
    lat_to_m = 111320
    lon_to_m = 111320 * np.cos(np.radians(mean_lat))

    coords_m = np.column_stack([
        df.loc[valid_mask, lat_col].values * lat_to_m,
        df.loc[valid_mask, lon_col].values * lon_to_m
    ])

    tree = cKDTree(coords_m)
    counts = tree.query_ball_point(coords_m, r=radius_m, return_length=True)

    # Subtract 1 to exclude self
    density = np.full(len(df), np.nan)
    density[valid_indices] = counts - 1

    return density


# =============================================================================
# TRANSIT STOP FUNCTIONS
# =============================================================================

def load_gtfs_stops(gtfs_path):
    """Load transit stops from GTFS directory."""
    if not gtfs_path.exists():
        return None

    # Try direct stops.txt file
    stops_file = gtfs_path / 'stops.txt'
    if stops_file.exists():
        stops_df = pd.read_csv(stops_file)
        if 'stop_lat' in stops_df.columns and 'stop_lon' in stops_df.columns:
            return stops_df[['stop_id', 'stop_lat', 'stop_lon']].dropna()

    # Try zip files in directory
    for zf in gtfs_path.glob('*.zip'):
        try:
            with zipfile.ZipFile(zf, 'r') as z:
                if 'stops.txt' in z.namelist():
                    stops_df = pd.read_csv(z.open('stops.txt'))
                    if 'stop_lat' in stops_df.columns and 'stop_lon' in stops_df.columns:
                        return stops_df[['stop_id', 'stop_lat', 'stop_lon']].dropna()
        except Exception:
            continue

    return None


def compute_nearest_stop_distance(poi_lats, poi_lons, stops_df, batch_size=1000):
    """
    Compute distance to nearest transit stop for each POI.

    Returns array of distances in meters.
    """
    if stops_df is None or len(stops_df) == 0:
        return np.full(len(poi_lats), np.nan)

    R = 6371000  # Earth radius in meters

    # Convert to radians
    poi_lat_rad = np.radians(poi_lats)
    poi_lon_rad = np.radians(poi_lons)
    stop_lat_rad = np.radians(stops_df['stop_lat'].values)
    stop_lon_rad = np.radians(stops_df['stop_lon'].values)

    n_pois = len(poi_lats)
    min_distances = np.full(n_pois, np.nan)

    # Process in batches to avoid memory issues
    for i in range(0, n_pois, batch_size):
        end_i = min(i + batch_size, n_pois)
        batch_lat = poi_lat_rad[i:end_i, np.newaxis]
        batch_lon = poi_lon_rad[i:end_i, np.newaxis]

        # Haversine formula (vectorized)
        dlat = stop_lat_rad - batch_lat
        dlon = stop_lon_rad - batch_lon
        a = np.sin(dlat/2)**2 + np.cos(batch_lat) * np.cos(stop_lat_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        distances = R * c

        min_distances[i:end_i] = np.min(distances, axis=1)

    return min_distances


def compute_dist_to_transit(df, city_col='city', lat_col='lat', lon_col='lon', country=None):
    """
    Compute distance to nearest transit stop for each POI.

    Args:
        country: 'sweden' or 'us' to force processing with specific GTFS.
                 If None, auto-detect based on city names.

    Returns array of distances in meters.
    """
    distances = np.full(len(df), np.nan)

    # Auto-detect country if not specified
    if country is None:
        us_cities_present = df[city_col].str.contains('new_york|washington|atlanta', case=False, na=False).any()
        country = 'us' if us_cities_present else 'sweden'

    if country == 'sweden':
        # Process each Swedish city with its county's GTFS
        for city in SWEDEN_CITY_COUNTY.keys():
            city_mask = df[city_col].str.contains(city, case=False, na=False)
            if city_mask.sum() == 0:
                continue

            gtfs_path = get_sweden_gtfs_path(city)
            if gtfs_path is None or not gtfs_path.exists():
                print(f"  {city}: no GTFS found at {gtfs_path}")
                continue

            stops_df = load_gtfs_stops(gtfs_path)
            if stops_df is None:
                print(f"  {city}: could not load stops")
                continue

            print(f"  {city} (c_{SWEDEN_CITY_COUNTY[city]}): {len(stops_df):,} stops")

            valid_mask = city_mask & df[lat_col].notna() & df[lon_col].notna()
            if valid_mask.sum() > 0:
                dists = compute_nearest_stop_distance(
                    df.loc[valid_mask, lat_col].values,
                    df.loc[valid_mask, lon_col].values,
                    stops_df
                )
                distances[np.where(valid_mask)[0]] = dists
                print(f"    {valid_mask.sum():,} POIs, median dist: {np.nanmedian(dists):.0f}m")

        return distances

    # US processing - process each city separately
    for city, gtfs_path in US_CITY_GTFS.items():
        city_mask = df[city_col].str.contains(city, case=False, na=False)
        if city_mask.sum() == 0:
            continue

        stops_df = load_gtfs_stops(gtfs_path)
        if stops_df is None:
            print(f"  {city}: no GTFS found")
            continue

        print(f"  {city}: loaded {len(stops_df):,} stops")

        valid_mask = city_mask & df[lat_col].notna() & df[lon_col].notna()
        if valid_mask.sum() > 0:
            dists = compute_nearest_stop_distance(
                df.loc[valid_mask, lat_col].values,
                df.loc[valid_mask, lon_col].values,
                stops_df
            )
            distances[np.where(valid_mask)[0]] = dists
            print(f"  {city}: {valid_mask.sum():,} POIs processed")

    return distances


# =============================================================================
# MAIN
# =============================================================================

def process_file(filepath, overwrite=False):
    """Process a single diversity metrics file."""
    print(f"\nProcessing: {filepath.name}")

    df = pd.read_parquet(filepath)
    print(f"  Loaded {len(df):,} POIs")

    # Detect country from filename
    country = 'sweden' if 'sweden' in filepath.name.lower() else 'us'
    print(f"  Detected country: {country.upper()}")

    # Check if already computed
    existing_cols = set(df.columns)
    new_cols = {'dist_to_center_km', 'poi_density_500m', 'dist_to_transit_m'}

    if new_cols.issubset(existing_cols) and not overwrite:
        print("  Spatial controls already computed. Use --overwrite to recompute.")
        return df

    # Compute distance to city center
    print("  Computing distance to city center...")
    df['dist_to_center_km'] = compute_dist_to_center(df)
    valid = df['dist_to_center_km'].notna().sum()
    print(f"    {valid:,} valid values ({100*valid/len(df):.1f}%)")

    # Compute POI density
    print("  Computing POI density (500m radius)...")
    df['poi_density_500m'] = compute_poi_density(df)
    valid = df['poi_density_500m'].notna().sum()
    print(f"    {valid:,} valid values, median = {df['poi_density_500m'].median():.0f}")

    # Compute distance to transit
    print("  Computing distance to transit stops...")
    df['dist_to_transit_m'] = compute_dist_to_transit(df, country=country)
    valid = df['dist_to_transit_m'].notna().sum()
    print(f"    {valid:,} valid values, median = {df['dist_to_transit_m'].median():.0f}m")

    # Save
    df.to_parquet(filepath, index=False)
    print(f"  Saved: {filepath}")

    return df


def main():
    parser = argparse.ArgumentParser(description='Compute spatial control variables')
    parser.add_argument('--overwrite', action='store_true',
                        help='Recompute even if columns exist')
    parser.add_argument('--file', type=str, default=None,
                        help='Process specific file only')
    args = parser.parse_args()

    print("=" * 70)
    print("COMPUTING SPATIAL CONTROL VARIABLES")
    print("=" * 70)
    print(f"Output columns: dist_to_center_km, poi_density_500m, dist_to_transit_m")

    if args.file:
        filepath = Path(args.file)
        if not filepath.exists():
            print(f"File not found: {filepath}")
            return
        process_file(filepath, overwrite=args.overwrite)
    else:
        # Process both US and Sweden files
        files = [
            ROUTING_DIR / 'us_poi_diversity_metrics.parquet',
            ROUTING_DIR / 'sweden_poi_diversity_metrics.parquet'
        ]

        for filepath in files:
            if filepath.exists():
                process_file(filepath, overwrite=args.overwrite)
            else:
                print(f"\nFile not found: {filepath}")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
