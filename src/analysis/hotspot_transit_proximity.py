#!/usr/bin/env python
"""
Analyze spatial correlation between GWR hotspots and transit stop locations.

This validates whether identified hotspots are actually near transit infrastructure.

Usage:
    python -m src.analysis.hotspot_transit_proximity
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Check for scipy availability
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, statistical tests will be skipped")

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
COEF_DIR = PROJECT_ROOT / 'outputs/phase3/gwr_local_coefficients'
GTFS_DIR = PROJECT_ROOT / 'dbs/gtfs'
OUTPUT_DIR = PROJECT_ROOT / 'outputs/phase3'

# City to GTFS mapping
# US: separate GTFS per city
# Sweden: national GTFS (same file for all cities, load once from any county folder)
US_CITY_GTFS = {
    'us_new_york': GTFS_DIR / 'new_york',
    'us_washington_dc': GTFS_DIR / 'washington_dc',
    'us_atlanta': GTFS_DIR / 'atlanta',
}
SWEDEN_GTFS_PATH = GTFS_DIR / 'sweden_south/c_01'  # National GTFS, same in all county folders


def load_gtfs_stops(gtfs_path):
    """Load transit stops from GTFS directory."""
    import zipfile

    # First try direct stops.txt file
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
        except Exception as e:
            print(f"    Error reading {zf.name}: {e}")
            continue

    return None


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate haversine distance in meters."""
    R = 6371000  # Earth radius in meters

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def compute_nearest_stop_distance(poi_df, stops_df, batch_size=1000):
    """Compute distance to nearest transit stop for each POI using vectorized haversine."""
    if stops_df is None or len(stops_df) == 0:
        return np.full(len(poi_df), np.nan)

    # Earth radius in meters
    R = 6371000

    # Convert to radians
    poi_lat = np.radians(poi_df['lat'].values)
    poi_lon = np.radians(poi_df['lon'].values)
    stop_lat = np.radians(stops_df['stop_lat'].values)
    stop_lon = np.radians(stops_df['stop_lon'].values)

    n_pois = len(poi_df)
    min_distances = np.zeros(n_pois)

    # Process in batches to avoid memory issues
    for i in range(0, n_pois, batch_size):
        end_i = min(i + batch_size, n_pois)
        batch_lat = poi_lat[i:end_i, np.newaxis]
        batch_lon = poi_lon[i:end_i, np.newaxis]

        # Haversine formula (vectorized)
        dlat = stop_lat - batch_lat
        dlon = stop_lon - batch_lon
        a = np.sin(dlat/2)**2 + np.cos(batch_lat) * np.cos(stop_lat) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        distances = R * c

        # Get minimum distance for each POI in batch
        min_distances[i:end_i] = np.min(distances, axis=1)

    return min_distances


def main():
    print("=" * 70)
    print("HOTSPOT-TRANSIT PROXIMITY ANALYSIS")
    print("=" * 70)

    # Load all coefficient files
    all_results = []

    for coef_file in COEF_DIR.glob('*.parquet'):
        df = pd.read_parquet(coef_file)

        # Extract city from filename
        city_slug = '_'.join(coef_file.stem.split('_')[:2])  # e.g., "us_new_york"

        all_results.append(df)

    # Combine all POIs
    all_pois = pd.concat(all_results, ignore_index=True)
    print(f"\nTotal POIs with GWR results: {len(all_pois):,}")
    print(f"Total hotspots: {all_pois['is_hotspot'].sum():,}")
    print(f"Total coldspots: {all_pois['is_coldspot'].sum():,}")

    # Process by city
    print("\n" + "=" * 70)
    print("COMPUTING DISTANCE TO NEAREST TRANSIT STOP")
    print("=" * 70)

    # Create city slug and country from city column
    all_pois['city_slug'] = all_pois['city'].str.lower().str.replace(' - ', '_').str.replace(' ', '_')
    all_pois['country'] = all_pois['city'].apply(lambda x: 'US' if 'US' in x else 'Sweden')

    # Load Swedish GTFS once (national coverage)
    print("\nLoading Swedish national GTFS...")
    sweden_stops = None
    if SWEDEN_GTFS_PATH.exists():
        sweden_stops = load_gtfs_stops(SWEDEN_GTFS_PATH)
        if sweden_stops is not None:
            print(f"  Loaded {len(sweden_stops):,} Swedish transit stops")
        else:
            print("  Could not load Swedish stops.txt")
    else:
        print(f"  Swedish GTFS path not found: {SWEDEN_GTFS_PATH}")

    # Process US cities (separate GTFS each)
    print("\nProcessing US cities...")
    for city_slug, gtfs_path in US_CITY_GTFS.items():
        print(f"\n  {city_slug}:")

        if not gtfs_path.exists():
            print(f"    GTFS not found, skipping")
            continue

        stops_df = load_gtfs_stops(gtfs_path)
        if stops_df is None:
            print(f"    Could not load stops.txt")
            continue

        print(f"    Loaded {len(stops_df):,} transit stops")

        # Get POIs for this city
        city_mask = all_pois['city_slug'] == city_slug
        city_pois = all_pois[city_mask].copy()

        if len(city_pois) == 0:
            print(f"    No POIs found")
            continue

        # Compute distances
        distances = compute_nearest_stop_distance(city_pois, stops_df)
        all_pois.loc[city_mask, 'dist_to_transit_m'] = distances
        print(f"    Computed distances for {len(city_pois):,} POIs")

    # Process Swedish cities (all use same national GTFS)
    print("\nProcessing Swedish cities...")
    if sweden_stops is not None:
        sweden_mask = all_pois['country'] == 'Sweden'
        sweden_pois = all_pois[sweden_mask].copy()

        if len(sweden_pois) > 0:
            print(f"  Computing distances for {len(sweden_pois):,} Swedish POIs...")
            distances = compute_nearest_stop_distance(sweden_pois, sweden_stops)
            all_pois.loc[sweden_mask, 'dist_to_transit_m'] = distances
            print(f"  Done.")

    # Analysis
    print("\n" + "=" * 70)
    print("RESULTS: ARE HOTSPOTS CLOSER TO TRANSIT?")
    print("=" * 70)

    # Filter to POIs with distance data
    with_dist = all_pois[all_pois['dist_to_transit_m'].notna()].copy()
    print(f"\nPOIs with transit distance data: {len(with_dist):,}")

    # Overall comparison
    hotspots = with_dist[with_dist['is_hotspot'] == True]
    coldspots = with_dist[with_dist['is_coldspot'] == True]
    non_sig = with_dist[(with_dist['is_hotspot'] == False) & (with_dist['is_coldspot'] == False)]

    print(f"\n{'Category':<20} {'N':>10} {'Mean Dist (m)':>15} {'Median Dist (m)':>15}")
    print("-" * 65)
    print(f"{'Hotspots':<20} {len(hotspots):>10,} {hotspots['dist_to_transit_m'].mean():>15.0f} {hotspots['dist_to_transit_m'].median():>15.0f}")
    print(f"{'Coldspots':<20} {len(coldspots):>10,} {coldspots['dist_to_transit_m'].mean():>15.0f} {coldspots['dist_to_transit_m'].median():>15.0f}")
    print(f"{'Non-significant':<20} {len(non_sig):>10,} {non_sig['dist_to_transit_m'].mean():>15.0f} {non_sig['dist_to_transit_m'].median():>15.0f}")

    # Statistical test (Mann-Whitney U)
    print("\n" + "-" * 65)
    print("STATISTICAL TESTS (Mann-Whitney U)")
    print("-" * 65)

    if not SCIPY_AVAILABLE:
        print("scipy not available - skipping statistical tests")
    elif len(hotspots) > 0 and len(non_sig) > 0:
        stat, p_value = stats.mannwhitneyu(
            hotspots['dist_to_transit_m'].dropna(),
            non_sig['dist_to_transit_m'].dropna(),
            alternative='less'  # Test if hotspots are CLOSER
        )
        print(f"Hotspots vs Non-significant: U={stat:,.0f}, p={p_value:.2e}")
        print(f"  Interpretation: Hotspots are {'SIGNIFICANTLY CLOSER' if p_value < 0.05 else 'NOT significantly closer'} to transit")

    # By country
    print("\n" + "=" * 70)
    print("RESULTS BY COUNTRY")
    print("=" * 70)

    for country in ['US', 'Sweden']:
        country_df = with_dist[with_dist['country'] == country]
        hotspots_c = country_df[country_df['is_hotspot'] == True]
        non_sig_c = country_df[(country_df['is_hotspot'] == False) & (country_df['is_coldspot'] == False)]

        print(f"\n{country}:")
        print(f"  Hotspots mean distance: {hotspots_c['dist_to_transit_m'].mean():.0f} m (median: {hotspots_c['dist_to_transit_m'].median():.0f} m)")
        print(f"  Non-sig mean distance:  {non_sig_c['dist_to_transit_m'].mean():.0f} m (median: {non_sig_c['dist_to_transit_m'].median():.0f} m)")

        if len(hotspots_c) > 0 and len(non_sig_c) > 0:
            diff_pct = 100 * (non_sig_c['dist_to_transit_m'].mean() - hotspots_c['dist_to_transit_m'].mean()) / non_sig_c['dist_to_transit_m'].mean()
            if SCIPY_AVAILABLE:
                stat, p_value = stats.mannwhitneyu(
                    hotspots_c['dist_to_transit_m'].dropna(),
                    non_sig_c['dist_to_transit_m'].dropna(),
                    alternative='less'
                )
                print(f"  Hotspots are {diff_pct:.1f}% closer to transit (p={p_value:.2e})")
            else:
                print(f"  Hotspots are {diff_pct:.1f}% closer to transit")

    # By entropy type
    print("\n" + "=" * 70)
    print("RESULTS BY ENTROPY TYPE")
    print("=" * 70)

    for etype in ['birth', 'income']:
        etype_df = with_dist[with_dist['entropy_type'] == etype]
        hotspots_e = etype_df[etype_df['is_hotspot'] == True]
        non_sig_e = etype_df[(etype_df['is_hotspot'] == False) & (etype_df['is_coldspot'] == False)]

        print(f"\n{etype.upper()} entropy:")
        print(f"  Hotspots: N={len(hotspots_e):,}, mean dist={hotspots_e['dist_to_transit_m'].mean():.0f} m")
        print(f"  Non-sig:  N={len(non_sig_e):,}, mean dist={non_sig_e['dist_to_transit_m'].mean():.0f} m")

        if len(hotspots_e) > 0 and len(non_sig_e) > 0 and SCIPY_AVAILABLE:
            stat, p_value = stats.mannwhitneyu(
                hotspots_e['dist_to_transit_m'].dropna(),
                non_sig_e['dist_to_transit_m'].dropna(),
                alternative='less'
            )
            print(f"  Mann-Whitney U: p={p_value:.2e}")

    # Percentage within transit walking distance
    print("\n" + "=" * 70)
    print("PERCENTAGE WITHIN WALKING DISTANCE OF TRANSIT")
    print("=" * 70)

    for threshold in [200, 400, 800]:
        hotspot_pct = 100 * (hotspots['dist_to_transit_m'] <= threshold).mean()
        nonsig_pct = 100 * (non_sig['dist_to_transit_m'] <= threshold).mean()
        print(f"\nWithin {threshold}m of transit:")
        print(f"  Hotspots:       {hotspot_pct:.1f}%")
        print(f"  Non-significant: {nonsig_pct:.1f}%")
        print(f"  Difference:      {hotspot_pct - nonsig_pct:+.1f} percentage points")

    # Save results
    output_file = OUTPUT_DIR / 'hotspot_transit_proximity.csv'
    summary_data = []

    for city in with_dist['city'].unique():
        for etype in with_dist['entropy_type'].unique():
            mask = (with_dist['city'] == city) & (with_dist['entropy_type'] == etype)
            subset = with_dist[mask]

            if len(subset) == 0:
                continue

            hotspots_sub = subset[subset['is_hotspot'] == True]
            nonsig_sub = subset[(subset['is_hotspot'] == False) & (subset['is_coldspot'] == False)]

            summary_data.append({
                'city': city,
                'entropy_type': etype,
                'n_total': len(subset),
                'n_hotspots': len(hotspots_sub),
                'n_nonsig': len(nonsig_sub),
                'hotspot_mean_dist_m': hotspots_sub['dist_to_transit_m'].mean(),
                'hotspot_median_dist_m': hotspots_sub['dist_to_transit_m'].median(),
                'nonsig_mean_dist_m': nonsig_sub['dist_to_transit_m'].mean(),
                'nonsig_median_dist_m': nonsig_sub['dist_to_transit_m'].median(),
                'hotspot_pct_within_400m': 100 * (hotspots_sub['dist_to_transit_m'] <= 400).mean() if len(hotspots_sub) > 0 else np.nan,
                'nonsig_pct_within_400m': 100 * (nonsig_sub['dist_to_transit_m'] <= 400).mean() if len(nonsig_sub) > 0 else np.nan,
            })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_file, index=False)
    print(f"\nSaved detailed results to: {output_file}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)


if __name__ == '__main__':
    main()
