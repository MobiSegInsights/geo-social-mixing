#!/usr/bin/env python
"""
Compute geographic catchment diversity (simple Euclidean buffer).

For each POI, computes population-weighted diversity within X km radius,
without transit routing. This provides a comparison baseline for the
transit-based catchment diversity.

By default, calibrates the radius to match the median distance of the
45-min transit catchment for each city (--calibrate flag).

Output: Adds geo_catchment_entropy_birth_norm and geo_catchment_entropy_income_norm
        to the diversity metrics parquet files.

Usage:
    python -m src.features.compute_geographic_catchment              # calibrated radius
    python -m src.features.compute_geographic_catchment --radius 5   # fixed 5km radius
"""

import argparse
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
US_CENSUS_DIR = PROJECT_ROOT / 'dbs/us_census'
DESO_DIR = PROJECT_ROOT / 'dbs/deso'

DEFAULT_RADIUS_KM = 5.0

# City to routing folder mapping
US_CITY_ROUTING = {
    'new_york': ROUTING_DIR / 'new_york',
    'washington_dc': ROUTING_DIR / 'washington_dc',
    'atlanta': ROUTING_DIR / 'atlanta',
}

# Sweden county codes to cities (approximate mapping)
SWEDEN_COUNTY_CITIES = {
    'c01': ['Stockholm'],
    'c03': ['Uppsala'],
    'c04': ['Södermanland'],
    'c05': ['Östergötland', 'Linköping'],
    'c06': ['Jönköping'],
    'c07': ['Kronoberg'],
    'c08': ['Kalmar'],
    'c09': ['Gotland'],
    'c10': ['Blekinge'],
    'c12': ['Skåne', 'Malmö', 'Lund', 'Helsingborg'],
    'c13': ['Halland'],
    'c14': ['Västra Götaland', 'Göteborg'],
    'c17': ['Värmland'],
    'c18': ['Örebro'],
    'c19': ['Västmanland', 'Västerås'],
}


# =============================================================================
# HAVERSINE DISTANCE
# =============================================================================

def haversine_km(lat1, lon1, lat2, lon2):
    """Compute haversine distance in kilometers (vectorized)."""
    R = 6371.0  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


# =============================================================================
# TRANSIT CATCHMENT RADIUS CALIBRATION
# =============================================================================

def compute_us_transit_catchment_radius(city):
    """
    Compute median MAX distance of 45-min transit catchment for a US city.

    Only considers POIs that have valid transit catchment diversity values.
    For each POI, finds the furthest reachable tract (max distance).
    Then takes the median of these max distances across POIs.

    Returns median of max distances in km.
    """
    routing_dir = US_CITY_ROUTING.get(city)
    if routing_dir is None or not routing_dir.exists():
        return None

    # Load diversity metrics to get POIs with valid transit catchment diversity
    diversity_file = ROUTING_DIR / 'us_poi_diversity_metrics.parquet'
    if not diversity_file.exists():
        return None

    diversity = pd.read_parquet(diversity_file)
    diversity['poi_id'] = diversity['poi_id'].astype(str)

    # Filter to city and POIs with valid transit catchment diversity
    diversity = diversity[diversity['city'] == city]
    diversity = diversity[diversity['catchment_entropy_birth_norm'].notna()]

    if len(diversity) == 0:
        print(f"    {city}: no POIs with valid transit catchment diversity")
        return None

    valid_poi_ids = set(diversity['poi_id'])
    print(f"    {city}: {len(valid_poi_ids):,} POIs with transit catchment diversity")

    # Load catchment summary
    summary_file = routing_dir / 'catchment_summary_walk_transit_45min.parquet'
    if not summary_file.exists():
        return None

    summary = pd.read_parquet(summary_file)
    summary['poi_id'] = summary['poi_id'].astype(str)

    # Filter to POIs with valid transit diversity
    summary = summary[summary['poi_id'].isin(valid_poi_ids)]

    if 'reachable_tract_ids' not in summary.columns:
        print(f"    WARNING: {city} - no reachable_tract_ids column")
        return None

    # Load tract centroids
    centroids_file = ROUTING_DIR / 'tract_centroids.csv'
    if not centroids_file.exists():
        return None

    centroids = pd.read_csv(centroids_file)
    centroids['id'] = centroids['id'].astype(str)

    # Build tract coordinate lookup (all tracts, not just city - catchment can cross boundaries)
    tract_coords = {
        row['id']: (row['lat'], row['lon'])
        for _, row in centroids.iterrows()
    }

    # Filter to valid data
    summary = summary.dropna(subset=['lat', 'lon', 'reachable_tract_ids'])

    # Sample POIs for efficiency
    sample_size = min(1000, len(summary))
    summary_sample = summary.sample(n=sample_size, random_state=42)

    max_distances = []

    for _, row in summary_sample.iterrows():
        poi_lat, poi_lon = row['lat'], row['lon']
        tract_ids_str = row['reachable_tract_ids']

        if pd.isna(poi_lat) or pd.isna(poi_lon) or pd.isna(tract_ids_str):
            continue

        # Parse tract IDs (pipe-separated)
        tract_ids = str(tract_ids_str).split('|')

        poi_distances = []
        for tract_id in tract_ids:
            tract_id = tract_id.strip()
            if tract_id in tract_coords:
                tract_lat, tract_lon = tract_coords[tract_id]
                dist = haversine_km(poi_lat, poi_lon, tract_lat, tract_lon)
                poi_distances.append(dist)

        if poi_distances:
            max_distances.append(max(poi_distances))

    if len(max_distances) == 0:
        return None

    median_max_dist = np.median(max_distances)
    print(f"    {city}: median of max distances from {len(max_distances)} POIs = {median_max_dist:.2f} km")
    return median_max_dist


def compute_sweden_transit_catchment_radius(city):
    """
    Compute median MAX distance of 45-min transit catchment for a Swedish city.

    Only considers POIs that have valid transit catchment diversity values.
    For each POI, finds the furthest reachable DeSO (max distance).
    Then takes the median of these max distances across POIs.

    Returns median of max distances in km.
    """
    # Find which county this city is in
    county_code = None
    for code, cities in SWEDEN_COUNTY_CITIES.items():
        if any(c.lower() in city.lower() or city.lower() in c.lower() for c in cities):
            county_code = code
            break

    if county_code is None:
        return None

    routing_dir = ROUTING_DIR / f'sweden_{county_code}'
    if not routing_dir.exists():
        return None

    # Load diversity metrics to get POIs with valid transit catchment diversity
    diversity_file = ROUTING_DIR / 'sweden_poi_diversity_metrics.parquet'
    if not diversity_file.exists():
        return None

    diversity = pd.read_parquet(diversity_file)
    diversity['poi_id'] = diversity['poi_id'].astype(str)

    # Filter to city and POIs with valid transit catchment diversity
    diversity = diversity[diversity['city'].str.contains(city[:4], case=False, na=False)]
    diversity = diversity[diversity['catchment_entropy_birth_norm'].notna()]

    if len(diversity) == 0:
        print(f"    {city}: no POIs with valid transit catchment diversity")
        return None

    valid_poi_ids = set(diversity['poi_id'])
    print(f"    {city}: {len(valid_poi_ids):,} POIs with transit catchment diversity")

    # Load catchment summary
    summary_file = routing_dir / 'catchment_summary_walk_transit_45min.parquet'
    if not summary_file.exists():
        return None

    summary = pd.read_parquet(summary_file)
    summary['poi_id'] = summary['poi_id'].astype(str)

    # Filter to POIs with valid transit diversity
    summary = summary[summary['poi_id'].isin(valid_poi_ids)]

    if 'reachable_tract_ids' not in summary.columns:
        print(f"    WARNING: {city} - no reachable_tract_ids column")
        return None

    # Load DeSO centroids from all county files (catchment can cross boundaries)
    all_centroids = []
    for county_dir in ROUTING_DIR.glob('sweden_c*'):
        centroid_file = county_dir / 'deso_centroids.csv'
        if centroid_file.exists():
            all_centroids.append(pd.read_csv(centroid_file))

    if not all_centroids:
        return None

    centroids = pd.concat(all_centroids, ignore_index=True).drop_duplicates(subset='id')
    centroids['id'] = centroids['id'].astype(str)

    # Build DeSO coordinate lookup
    deso_coords = {
        row['id']: (row['lat'], row['lon'])
        for _, row in centroids.iterrows()
    }

    # Filter to valid data
    summary = summary.dropna(subset=['lat', 'lon', 'reachable_tract_ids'])

    # Sample POIs for efficiency
    sample_size = min(1000, len(summary))
    if len(summary) == 0:
        return None

    summary_sample = summary.sample(n=sample_size, random_state=42)

    max_distances = []

    for _, row in summary_sample.iterrows():
        poi_lat, poi_lon = row['lat'], row['lon']
        tract_ids_str = row['reachable_tract_ids']

        if pd.isna(poi_lat) or pd.isna(poi_lon) or pd.isna(tract_ids_str):
            continue

        # Parse DeSO IDs (pipe-separated)
        deso_ids = str(tract_ids_str).split('|')

        poi_distances = []
        for deso_id in deso_ids:
            deso_id = deso_id.strip()
            if deso_id in deso_coords:
                deso_lat, deso_lon = deso_coords[deso_id]
                dist = haversine_km(poi_lat, poi_lon, deso_lat, deso_lon)
                poi_distances.append(dist)

        if poi_distances:
            max_distances.append(max(poi_distances))

    if len(max_distances) == 0:
        return None

    median_max_dist = np.median(max_distances)
    print(f"    {city}: median of max distances from {len(max_distances)} POIs = {median_max_dist:.2f} km")
    return median_max_dist


def get_calibrated_radii():
    """
    Compute calibrated radius for each city based on median transit catchment distance.

    Returns dict mapping city name to radius in km.
    """
    print("  Calibrating radii based on median transit catchment distance...")

    radii = {}

    # US cities
    for city in ['new_york', 'washington_dc', 'atlanta']:
        radius = compute_us_transit_catchment_radius(city)
        if radius is not None:
            radii[city] = radius
            print(f"    {city}: {radius:.2f} km")
        else:
            radii[city] = DEFAULT_RADIUS_KM
            print(f"    {city}: {DEFAULT_RADIUS_KM} km (default, calibration failed)")

    # Swedish cities
    sweden_cities = ['Stockholm', 'Göteborg', 'Malmö', 'Uppsala', 'Västerås',
                     'Örebro', 'Linköping', 'Helsingborg', 'Lund']
    for city in sweden_cities:
        radius = compute_sweden_transit_catchment_radius(city)
        if radius is not None:
            radii[city] = radius
            print(f"    {city}: {radius:.2f} km")
        else:
            radii[city] = DEFAULT_RADIUS_KM
            print(f"    {city}: {DEFAULT_RADIUS_KM} km (default, calibration failed)")

    return radii


# =============================================================================
# ENTROPY COMPUTATION
# =============================================================================

def compute_entropy(counts):
    """Compute Shannon entropy from counts array."""
    counts = np.array(counts, dtype=float)
    total = counts.sum()
    if total == 0:
        return np.nan
    probs = counts / total
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs))


def normalize_entropy(entropy, n_categories):
    """Normalize entropy to [0, 1] range."""
    if np.isnan(entropy) or n_categories <= 1:
        return np.nan
    max_entropy = np.log(n_categories)
    return entropy / max_entropy if max_entropy > 0 else 0


# =============================================================================
# US DATA
# =============================================================================

def load_us_tract_data():
    """Load US tract centroids and population data."""
    print("  Loading US tract data...")

    # Tract centroids
    centroids = pd.read_csv(ROUTING_DIR / 'tract_centroids.csv')
    centroids['GEOID'] = centroids['id'].astype(str)

    # Birth background
    census = pd.read_parquet(US_CENSUS_DIR / 'acs2024_tracts_study_cities.parquet')
    census['GEOID'] = census['GEOID'].astype(str)

    # Income distribution
    income = pd.read_csv(US_CENSUS_DIR / 'acs2024_tracts_income_distribution.csv')
    income['GEOID'] = income['GEOID'].astype(str)

    # Merge
    tracts = centroids.merge(
        census[['GEOID', 'native_born', 'foreign_born', 'total_population', 'city']],
        on='GEOID', how='left', suffixes=('', '_census')
    )
    tracts = tracts.merge(
        income[['GEOID', 'q1_pct', 'q2_pct', 'q3_pct', 'q4_pct', 'total_households']],
        on='GEOID', how='left'
    )

    # Use city from centroids if available
    if 'city_census' in tracts.columns:
        tracts['city'] = tracts['city'].fillna(tracts['city_census'])
        tracts = tracts.drop(columns=['city_census'])

    print(f"    Loaded {len(tracts):,} tracts")
    return tracts


def compute_us_geo_catchment(poi_df, tract_df, radius_km=DEFAULT_RADIUS_KM):
    """
    Compute geographic catchment diversity for US POIs.

    For each POI, aggregates population from all tracts within radius_km,
    then computes entropy.
    """
    print(f"  Computing US geographic catchment (radius={radius_km}km)...")

    # Filter to POIs and tracts with valid coordinates
    poi_valid = poi_df[poi_df['lat'].notna() & poi_df['lon'].notna()].copy()
    tract_valid = tract_df[tract_df['lat'].notna() & tract_df['lon'].notna()].copy()

    if len(poi_valid) == 0 or len(tract_valid) == 0:
        return np.full(len(poi_df), np.nan), np.full(len(poi_df), np.nan)

    # Build KD-tree for tracts (approximate meter conversion)
    mean_lat = tract_valid['lat'].mean()
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(mean_lat))

    tract_coords = np.column_stack([
        tract_valid['lat'].values * km_per_deg_lat,
        tract_valid['lon'].values * km_per_deg_lon
    ])
    tree = cKDTree(tract_coords)

    # POI coordinates in same units
    poi_coords = np.column_stack([
        poi_valid['lat'].values * km_per_deg_lat,
        poi_valid['lon'].values * km_per_deg_lon
    ])

    # Tract population arrays
    birth_native = tract_valid['native_born'].fillna(0).values
    birth_foreign = tract_valid['foreign_born'].fillna(0).values

    # Income: convert percentages to counts (approximate)
    total_hh = tract_valid['total_households'].fillna(0).values
    income_q1 = tract_valid['q1_pct'].fillna(0).values * total_hh / 100
    income_q2 = tract_valid['q2_pct'].fillna(0).values * total_hh / 100
    income_q3 = tract_valid['q3_pct'].fillna(0).values * total_hh / 100
    income_q4 = tract_valid['q4_pct'].fillna(0).values * total_hh / 100

    # Results arrays
    n_pois = len(poi_valid)
    birth_entropy = np.full(n_pois, np.nan)
    income_entropy = np.full(n_pois, np.nan)
    n_tracts_arr = np.zeros(n_pois, dtype=int)

    # Process in batches
    batch_size = 5000
    for start in range(0, n_pois, batch_size):
        end = min(start + batch_size, n_pois)

        for i in range(start, end):
            # Find tracts within radius
            indices = tree.query_ball_point(poi_coords[i], radius_km)

            if len(indices) == 0:
                continue

            n_tracts_arr[i] = len(indices)

            # Aggregate birth background
            total_native = birth_native[indices].sum()
            total_foreign = birth_foreign[indices].sum()

            if total_native + total_foreign > 0:
                ent = compute_entropy([total_native, total_foreign])
                birth_entropy[i] = normalize_entropy(ent, 2)

            # Aggregate income
            total_q1 = income_q1[indices].sum()
            total_q2 = income_q2[indices].sum()
            total_q3 = income_q3[indices].sum()
            total_q4 = income_q4[indices].sum()

            if total_q1 + total_q2 + total_q3 + total_q4 > 0:
                ent = compute_entropy([total_q1, total_q2, total_q3, total_q4])
                income_entropy[i] = normalize_entropy(ent, 4)

        if (end - start) == batch_size:
            print(f"    Processed {end:,}/{n_pois:,} POIs...")

    # Map back to original POI order
    birth_result = np.full(len(poi_df), np.nan)
    income_result = np.full(len(poi_df), np.nan)

    valid_indices = poi_df.index[poi_df['lat'].notna() & poi_df['lon'].notna()]
    birth_result[valid_indices] = birth_entropy
    income_result[valid_indices] = income_entropy

    valid_count = np.sum(~np.isnan(birth_result))
    print(f"    Computed for {valid_count:,} POIs")

    return birth_result, income_result


# =============================================================================
# SWEDEN DATA
# =============================================================================

def load_sweden_deso_data():
    """Load Swedish DeSO centroids and population data."""
    print("  Loading Swedish DeSO data...")

    # Load all DeSO centroids
    centroid_files = list(ROUTING_DIR.glob('sweden_c*/deso_centroids.csv'))
    centroids_list = []
    for f in centroid_files:
        df = pd.read_csv(f)
        centroids_list.append(df)

    if not centroids_list:
        print("    No DeSO centroid files found")
        return None

    centroids = pd.concat(centroids_list, ignore_index=True)
    centroids = centroids.drop_duplicates(subset='id')
    centroids = centroids.rename(columns={'id': 'deso_code'})

    # Load demographics
    demo = pd.read_parquet(DESO_DIR / 'deso_harmonized_2024.parquet')

    # Merge
    deso = centroids.merge(demo, on='deso_code', how='left')

    print(f"    Loaded {len(deso):,} DeSO zones")
    return deso


def compute_sweden_geo_catchment(poi_df, deso_df, radius_km=DEFAULT_RADIUS_KM):
    """
    Compute geographic catchment diversity for Swedish POIs.
    """
    print(f"  Computing Swedish geographic catchment (radius={radius_km}km)...")

    if deso_df is None:
        return np.full(len(poi_df), np.nan), np.full(len(poi_df), np.nan)

    # Filter to valid coordinates
    poi_valid = poi_df[poi_df['lat'].notna() & poi_df['lon'].notna()].copy()
    deso_valid = deso_df[deso_df['lat'].notna() & deso_df['lon'].notna()].copy()

    if len(poi_valid) == 0 or len(deso_valid) == 0:
        return np.full(len(poi_df), np.nan), np.full(len(poi_df), np.nan)

    # Build KD-tree
    mean_lat = deso_valid['lat'].mean()
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(mean_lat))

    deso_coords = np.column_stack([
        deso_valid['lat'].values * km_per_deg_lat,
        deso_valid['lon'].values * km_per_deg_lon
    ])
    tree = cKDTree(deso_coords)

    poi_coords = np.column_stack([
        poi_valid['lat'].values * km_per_deg_lat,
        poi_valid['lon'].values * km_per_deg_lon
    ])

    # Population arrays - ensure numeric types
    birth_sweden = pd.to_numeric(deso_valid['birth_sweden'], errors='coerce').fillna(0).values
    birth_europe = pd.to_numeric(deso_valid['birth_europe'], errors='coerce').fillna(0).values
    birth_other = pd.to_numeric(deso_valid['birth_other'], errors='coerce').fillna(0).values

    # Income quintiles (stored as percentages, use with total population)
    pop_total = pd.to_numeric(deso_valid['pop_total'], errors='coerce').fillna(0).values
    income_q1_pct = pd.to_numeric(deso_valid['income_q1_pct'], errors='coerce').fillna(0).values
    income_q2_pct = pd.to_numeric(deso_valid['income_q2_pct'], errors='coerce').fillna(0).values
    income_q3_pct = pd.to_numeric(deso_valid['income_q3_pct'], errors='coerce').fillna(0).values
    income_q4_pct = pd.to_numeric(deso_valid['income_q4_pct'], errors='coerce').fillna(0).values

    income_q1 = income_q1_pct * pop_total / 100
    income_q2 = income_q2_pct * pop_total / 100
    income_q3 = income_q3_pct * pop_total / 100
    income_q4 = income_q4_pct * pop_total / 100

    # Results
    n_pois = len(poi_valid)
    birth_entropy = np.full(n_pois, np.nan)
    income_entropy = np.full(n_pois, np.nan)

    batch_size = 5000
    for start in range(0, n_pois, batch_size):
        end = min(start + batch_size, n_pois)

        for i in range(start, end):
            indices = tree.query_ball_point(poi_coords[i], radius_km)

            if len(indices) == 0:
                continue

            # Birth background (3 categories)
            total_sweden = birth_sweden[indices].sum()
            total_europe = birth_europe[indices].sum()
            total_other = birth_other[indices].sum()

            if total_sweden + total_europe + total_other > 0:
                ent = compute_entropy([total_sweden, total_europe, total_other])
                birth_entropy[i] = normalize_entropy(ent, 3)

            # Income (4 categories)
            total_q1 = income_q1[indices].sum()
            total_q2 = income_q2[indices].sum()
            total_q3 = income_q3[indices].sum()
            total_q4 = income_q4[indices].sum()

            if total_q1 + total_q2 + total_q3 + total_q4 > 0:
                ent = compute_entropy([total_q1, total_q2, total_q3, total_q4])
                income_entropy[i] = normalize_entropy(ent, 4)

        if (end - start) == batch_size:
            print(f"    Processed {end:,}/{n_pois:,} POIs...")

    # Map back
    birth_result = np.full(len(poi_df), np.nan)
    income_result = np.full(len(poi_df), np.nan)

    valid_indices = poi_df.index[poi_df['lat'].notna() & poi_df['lon'].notna()]
    birth_result[valid_indices] = birth_entropy
    income_result[valid_indices] = income_entropy

    valid_count = np.sum(~np.isnan(birth_result))
    print(f"    Computed for {valid_count:,} POIs")

    return birth_result, income_result


# =============================================================================
# MAIN
# =============================================================================

def compute_us_geo_catchment_by_city(poi_df, tract_df, city_radii):
    """
    Compute geographic catchment for US POIs with per-city radius.
    """
    birth_result = np.full(len(poi_df), np.nan)
    income_result = np.full(len(poi_df), np.nan)
    radius_result = np.full(len(poi_df), np.nan)

    for city, radius in city_radii.items():
        city_mask = poi_df['city'] == city
        if city_mask.sum() == 0:
            continue

        print(f"    {city}: {city_mask.sum():,} POIs, radius={radius:.2f}km")

        city_pois = poi_df[city_mask].copy()
        city_pois = city_pois.reset_index(drop=True)

        birth, income = compute_us_geo_catchment(city_pois, tract_df, radius)

        birth_result[city_mask] = birth
        income_result[city_mask] = income
        radius_result[city_mask] = radius

    return birth_result, income_result, radius_result


def compute_sweden_geo_catchment_by_city(poi_df, deso_df, city_radii):
    """
    Compute geographic catchment for Swedish POIs with per-city radius.
    """
    birth_result = np.full(len(poi_df), np.nan)
    income_result = np.full(len(poi_df), np.nan)
    radius_result = np.full(len(poi_df), np.nan)

    for city, radius in city_radii.items():
        # Match city names (handle encoding)
        city_mask = poi_df['city'].str.contains(city[:4], case=False, na=False)
        if city_mask.sum() == 0:
            continue

        print(f"    {city}: {city_mask.sum():,} POIs, radius={radius:.2f}km")

        city_pois = poi_df[city_mask].copy()
        city_pois = city_pois.reset_index(drop=True)

        birth, income = compute_sweden_geo_catchment(city_pois, deso_df, radius)

        birth_result[city_mask] = birth
        income_result[city_mask] = income
        radius_result[city_mask] = radius

    return birth_result, income_result, radius_result


def main():
    parser = argparse.ArgumentParser(description='Compute geographic catchment diversity')
    parser.add_argument('--radius', type=float, default=None,
                        help=f'Fixed catchment radius in km. If not set, calibrates to match transit catchment.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing columns')
    args = parser.parse_args()

    print("=" * 70)
    print("GEOGRAPHIC CATCHMENT DIVERSITY")
    print("=" * 70)

    # Determine radii
    if args.radius is not None:
        print(f"Using fixed radius: {args.radius} km")
        us_radii = {city: args.radius for city in ['new_york', 'washington_dc', 'atlanta']}
        se_radii = {city: args.radius for city in ['Stockholm', 'Göteborg', 'Malmö', 'Uppsala',
                                                     'Västerås', 'Örebro', 'Linköping', 'Helsingborg', 'Lund']}
    else:
        print("Calibrating radius to match median transit catchment distance...")
        all_radii = get_calibrated_radii()
        us_radii = {k: v for k, v in all_radii.items() if k in ['new_york', 'washington_dc', 'atlanta']}
        se_radii = {k: v for k, v in all_radii.items() if k not in ['new_york', 'washington_dc', 'atlanta']}

    print("=" * 70)

    # Process US
    print("\n[1/2] Processing US...")
    us_file = ROUTING_DIR / 'us_poi_diversity_metrics.parquet'

    if us_file.exists():
        us_df = pd.read_parquet(us_file)
        print(f"  Loaded {len(us_df):,} US POIs")

        col_birth = 'geo_catchment_entropy_birth_norm'
        col_income = 'geo_catchment_entropy_income_norm'

        if col_birth in us_df.columns and not args.overwrite:
            print("  Geographic catchment already computed. Use --overwrite to recompute.")
        else:
            tract_df = load_us_tract_data()
            birth, income, radii = compute_us_geo_catchment_by_city(us_df, tract_df, us_radii)

            us_df[col_birth] = birth
            us_df[col_income] = income
            us_df['geo_catchment_radius_km'] = radii

            us_df.to_parquet(us_file, index=False)
            print(f"  Saved: {us_file}")

    # Process Sweden
    print("\n[2/2] Processing Sweden...")
    se_file = ROUTING_DIR / 'sweden_poi_diversity_metrics.parquet'

    if se_file.exists():
        se_df = pd.read_parquet(se_file)
        print(f"  Loaded {len(se_df):,} Swedish POIs")

        col_birth = 'geo_catchment_entropy_birth_norm'
        col_income = 'geo_catchment_entropy_income_norm'

        if col_birth in se_df.columns and not args.overwrite:
            print("  Geographic catchment already computed. Use --overwrite to recompute.")
        else:
            deso_df = load_sweden_deso_data()
            birth, income, radii = compute_sweden_geo_catchment_by_city(se_df, deso_df, se_radii)

            se_df[col_birth] = birth
            se_df[col_income] = income
            se_df['geo_catchment_radius_km'] = radii

            se_df.to_parquet(se_file, index=False)
            print(f"  Saved: {se_file}")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
