#!/usr/bin/env python
"""
Compute LISA (Local Indicators of Spatial Association) clusters.

Identifies High-High (diversity hotspots) and Low-Low (diversity coldspots) clusters.

Usage:
    python -m src.analysis.compute_lisa_clusters
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

try:
    from esda.moran import Moran, Moran_Local
    from libpysal.weights import KNN
    PYSAL_AVAILABLE = True
except ImportError:
    PYSAL_AVAILABLE = False
    print("Warning: PySAL not available")

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ROUTING_DIR = PROJECT_ROOT / 'dbs/routing'
OUTPUT_DIR = PROJECT_ROOT / 'outputs/phase2'

US_FILE = ROUTING_DIR / 'us_poi_diversity_metrics.parquet'
SWEDEN_FILE = ROUTING_DIR / 'sweden_poi_diversity_metrics.parquet'

# Parameters
K_NEIGHBORS = 10
ALPHA = 0.05
MAX_SAMPLE = None  # No sampling - use all POIs
RANDOM_SEED = 42


def compute_lisa(gdf, var_col, k=10, alpha=0.05):
    """
    Compute LISA clusters for a variable.

    Returns DataFrame with cluster labels and statistics.
    """
    # Build spatial weights
    coords = np.column_stack([gdf.geometry.x, gdf.geometry.y])
    W = KNN.from_array(coords, k=k)
    W.transform = 'R'

    # Compute Local Moran's I
    y = gdf[var_col].values
    lisa = Moran_Local(y, W, seed=RANDOM_SEED, permutations=100)

    # Compute Global Moran's I (proper statistic, not mean of local values)
    global_moran = Moran(y, W, permutations=100)
    lisa.global_I = global_moran.I  # Attach global Moran's I to lisa object
    lisa.global_p = global_moran.p_sim  # Attach p-value

    # Classify clusters
    # q: 1=HH, 2=LH, 3=LL, 4=HL
    sig = lisa.p_sim < alpha

    cluster_labels = ['Not Significant'] * len(gdf)
    for i in range(len(gdf)):
        if sig[i]:
            if lisa.q[i] == 1:
                cluster_labels[i] = 'High-High'
            elif lisa.q[i] == 2:
                cluster_labels[i] = 'Low-High'
            elif lisa.q[i] == 3:
                cluster_labels[i] = 'Low-Low'
            elif lisa.q[i] == 4:
                cluster_labels[i] = 'High-Low'

    result = pd.DataFrame({
        'poi_id': gdf['poi_id'].values,
        'lon': gdf.geometry.x.values,
        'lat': gdf.geometry.y.values,
        'value': y,
        'local_i': lisa.Is,
        'p_value': lisa.p_sim,
        'quadrant': lisa.q,
        'cluster': cluster_labels
    })

    return result, lisa


def main():
    if not PYSAL_AVAILABLE:
        print("PySAL required for LISA analysis")
        return

    print("=" * 70)
    print("LISA CLUSTER ANALYSIS")
    print("=" * 70)

    import geopandas as gpd
    from shapely.geometry import Point

    # Load data
    print("\nLoading data...")
    us_df = pd.read_parquet(US_FILE)
    sweden_df = pd.read_parquet(SWEDEN_FILE)

    # Create GeoDataFrames
    us_gdf = gpd.GeoDataFrame(
        us_df,
        geometry=[Point(x, y) for x, y in zip(us_df['lon'], us_df['lat'])],
        crs='EPSG:4326'
    )
    sweden_gdf = gpd.GeoDataFrame(
        sweden_df,
        geometry=[Point(x, y) for x, y in zip(sweden_df['lon'], sweden_df['lat'])],
        crs='EPSG:4326'
    )

    # Combine and add country
    us_gdf['country'] = 'US'
    sweden_gdf['country'] = 'Sweden'
    all_gdf = pd.concat([us_gdf, sweden_gdf], ignore_index=True)

    print(f"Total POIs: {len(all_gdf):,}")

    # Cities to analyze
    cities = all_gdf['city'].unique()

    # Variables to analyze
    variables = [
        ('visitor_entropy_birth_norm', 'birth'),
        ('visitor_entropy_income_norm', 'income')
    ]

    all_results = []
    summary_data = []

    for city in cities:
        city_gdf = all_gdf[all_gdf['city'] == city].copy()

        if len(city_gdf) < 100:
            print(f"\n{city}: Too few POIs ({len(city_gdf)}), skipping")
            continue

        # Project to local UTM for accurate distances
        city_center_lon = city_gdf.geometry.x.mean()
        utm_zone = int((city_center_lon + 180) / 6) + 1
        utm_crs = f'EPSG:326{utm_zone:02d}' if city_gdf.geometry.y.mean() >= 0 else f'EPSG:327{utm_zone:02d}'
        city_proj = city_gdf.to_crs(utm_crs)

        # Filter to analysis-ready POIs (valid visitor + catchment entropy)
        analysis_mask = (
            city_proj['visitor_entropy_birth_norm'].notna() &
            city_proj['catchment_entropy_birth_norm'].notna()
        )
        city_proj = city_proj[analysis_mask].copy()

        # Sample if too large (only if MAX_SAMPLE is set)
        if MAX_SAMPLE is not None and len(city_proj) > MAX_SAMPLE:
            print(f"\n{city}: Sampling {MAX_SAMPLE:,} from {len(city_proj):,} POIs")
            city_proj = city_proj.sample(n=MAX_SAMPLE, random_state=RANDOM_SEED)
        else:
            print(f"\n{city}: {len(city_proj):,} POIs")

        for var_col, var_name in variables:
            if var_col not in city_proj.columns:
                continue

            # Filter valid values
            valid_mask = city_proj[var_col].notna()
            valid_gdf = city_proj[valid_mask].copy()

            if len(valid_gdf) < 100:
                print(f"  {var_name}: Too few valid values, skipping")
                continue

            print(f"  {var_name}: Computing LISA for {len(valid_gdf):,} POIs...")

            try:
                lisa_result, lisa_obj = compute_lisa(valid_gdf, var_col, k=K_NEIGHBORS, alpha=ALPHA)
                lisa_result['city'] = city
                lisa_result['entropy_type'] = var_name
                lisa_result['country'] = city_gdf['country'].iloc[0]

                all_results.append(lisa_result)

                # Summary statistics
                cluster_counts = lisa_result['cluster'].value_counts()
                summary_data.append({
                    'city': city,
                    'country': city_gdf['country'].iloc[0],
                    'entropy_type': var_name,
                    'n_total': len(lisa_result),
                    'n_high_high': cluster_counts.get('High-High', 0),
                    'n_low_low': cluster_counts.get('Low-Low', 0),
                    'n_high_low': cluster_counts.get('High-Low', 0),
                    'n_low_high': cluster_counts.get('Low-High', 0),
                    'n_not_sig': cluster_counts.get('Not Significant', 0),
                    'pct_high_high': 100 * cluster_counts.get('High-High', 0) / len(lisa_result),
                    'pct_low_low': 100 * cluster_counts.get('Low-Low', 0) / len(lisa_result),
                    'global_moran_i': lisa_obj.global_I,
                    'global_moran_p': lisa_obj.global_p
                })

                print(f"    HH: {cluster_counts.get('High-High', 0):,}, LL: {cluster_counts.get('Low-Low', 0):,}")

            except Exception as e:
                print(f"    Error: {e}")
                continue

    # Save results
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        output_file = OUTPUT_DIR / 'lisa_clusters.parquet'
        combined.to_parquet(output_file, index=False)
        print(f"\nSaved LISA results to: {output_file}")
        print(f"Total rows: {len(combined):,}")

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_file = OUTPUT_DIR / 'lisa_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"Saved summary to: {summary_file}")

        # Print summary table
        print("\n" + "=" * 70)
        print("LISA CLUSTER SUMMARY")
        print("=" * 70)
        print(f"\n{'City':<25} {'Type':<8} {'N':>8} {'HH%':>8} {'LL%':>8}")
        print("-" * 60)
        for _, row in summary_df.iterrows():
            print(f"{row['city']:<25} {row['entropy_type']:<8} {row['n_total']:>8,} {row['pct_high_high']:>7.1f}% {row['pct_low_low']:>7.1f}%")


if __name__ == '__main__':
    main()
