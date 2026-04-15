#!/usr/bin/env python
"""
Phase 2: Spatial Spillover Analysis - Production Script

Analyzes transit vs non-transit spillover effects on visitor diversity.
Compares birth/origin entropy and income entropy across US and Swedish cities.

Spatial weights specification:
- W1: Distance-decay weights with cutoff (default: 800m, alpha=1.0)
      w_ij = 1/d_ij^alpha if d_ij <= cutoff, else 0
      Row-standardized so weights sum to 1
- W2: Transit stop clustering (POIs sharing same nearest stop)
- W1_ex: W1 excluding transit-connected pairs

Uses GM_Lag (GMM estimation) for spatial lag models. Results where |rho| >= 1
are flagged as non-stationary (the spatial multiplier doesn't converge).
Non-stationary results should be interpreted with caution or excluded.

Output: CSV files with regression results including stationarity flags.

Usage:
    # Run all cities:
    python -m src.models.run_spatial_spillover_analysis

    # Run single city:
    python -m src.models.run_spatial_spillover_analysis --city Stockholm
    python -m src.models.run_spatial_spillover_analysis --city new_york

    # With options:
    python -m src.models.run_spatial_spillover_analysis --city Atlanta --max-n 50000
"""

import argparse
import os
import sys
import zipfile
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import sparse
from scipy.spatial import cKDTree
from libpysal import weights
from spreg import OLS, GM_Lag  # GM_Lag is faster than ML_Lag for large samples

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

def get_project_root():
    """Get project root directory (works from any location)."""
    # If running as script, go up from src/models/
    script_dir = Path(__file__).resolve().parent
    if script_dir.name == 'models' and script_dir.parent.name == 'src':
        return script_dir.parent.parent
    # Fallback: current working directory
    return Path.cwd()

PROJECT_ROOT = get_project_root()

# Data paths (relative to project root)
ROUTING_DIR = PROJECT_ROOT / 'dbs/routing'
US_TRAFFIC_DIR = PROJECT_ROOT / 'dbs/us_foot_traffic/cities'
SE_TRAFFIC_DIR = PROJECT_ROOT / 'dbs/sweden_weekly_patterns'
GTFS_DIR = PROJECT_ROOT / 'dbs/gtfs'
OUTPUT_DIR = PROJECT_ROOT / 'outputs/phase2'

# US cities to analyze
US_CITIES = ['new_york', 'washington_dc', 'atlanta']

# Top 9 Swedish cities
SWEDEN_CITIES = [
    'Stockholm', 'Göteborg', 'Malmö', 'Uppsala', 'Västerås',
    'Örebro', 'Linköping', 'Helsingborg', 'Lund'
]

# Spatial weights parameters
DISTANCE_CUTOFF_M = 800  # Distance decay cutoff in meters
DISTANCE_ALPHA = 1.0     # Decay exponent: w_ij = 1/d_ij^alpha
TRANSIT_STOP_RADIUS_M = 400


# =============================================================================
# GTFS STOPS EXTRACTION (with caching)
# =============================================================================

_gtfs_cache = {}

def extract_stops_from_gtfs(gtfs_dir):
    """Extract transit stops from GTFS zip files (cached)."""
    gtfs_dir = str(gtfs_dir)
    if gtfs_dir in _gtfs_cache:
        return _gtfs_cache[gtfs_dir]

    gtfs_path = Path(gtfs_dir)
    if not gtfs_path.exists():
        return None

    all_stops = []
    for zf in gtfs_path.glob('*.zip'):
        try:
            with zipfile.ZipFile(zf, 'r') as z:
                if 'stops.txt' in z.namelist():
                    with z.open('stops.txt') as f:
                        stops = pd.read_csv(f)
                        all_stops.append(stops)
        except Exception:
            continue

    if all_stops:
        combined = pd.concat(all_stops, ignore_index=True)
        if 'location_type' in combined.columns:
            combined = combined[(combined['location_type'].isna()) | (combined['location_type'] == 0)]
        result = combined[['stop_id', 'stop_lat', 'stop_lon']].drop_duplicates()
        _gtfs_cache[gtfs_dir] = result
        return result

    _gtfs_cache[gtfs_dir] = None
    return None


# =============================================================================
# SPATIAL WEIGHTS CONSTRUCTION
# =============================================================================

def get_utm_epsg(gdf):
    """Get appropriate UTM EPSG code for projection."""
    centroid = gdf.geometry.unary_union.centroid
    utm_zone = int((centroid.x + 180) / 6) + 1
    hemisphere = 'north' if centroid.y >= 0 else 'south'
    return 32600 + utm_zone if hemisphere == 'north' else 32700 + utm_zone


def build_W1_distance_decay(gdf, cutoff_m=DISTANCE_CUTOFF_M, alpha=DISTANCE_ALPHA):
    """
    W1: Distance-decay weights with cutoff.

    w_ij = 1/d_ij^alpha if d_ij <= cutoff, else 0
    Row-standardized so weights sum to 1 for each POI.

    This addresses KNN's limitation where neighbors at different distances
    get equal weight.
    """
    epsg = get_utm_epsg(gdf)
    gdf_proj = gdf.to_crs(epsg=epsg)

    # Get coordinates in meters
    coords = np.column_stack([gdf_proj.geometry.x, gdf_proj.geometry.y])
    n = len(coords)

    # Build KD-tree for efficient neighbor search
    tree = cKDTree(coords)

    # Build sparse matrix directly (more robust than dict-based W)
    rows, cols, data = [], [], []

    for i in range(n):
        # Query all points within cutoff distance
        indices = tree.query_ball_point(coords[i], cutoff_m)

        neighbors_i = []
        weights_i = []

        for j in indices:
            if i != j:  # Exclude self
                d = np.sqrt(np.sum((coords[i] - coords[j])**2))
                if d > 0 and d <= cutoff_m:
                    neighbors_i.append(j)
                    # Distance decay: 1/d^alpha
                    weights_i.append(1.0 / (d ** alpha))

        if neighbors_i:
            # Row-standardize
            total = sum(weights_i)
            for j, w in zip(neighbors_i, weights_i):
                rows.append(i)
                cols.append(j)
                data.append(w / total)
        else:
            # Isolated point - connect to nearest neighbor as fallback
            dist, idx = tree.query(coords[i], k=2)  # k=2 because first is self
            if len(idx) > 1 and idx[1] != i:
                rows.append(i)
                cols.append(idx[1])
                data.append(1.0)

    # Build W from sparse matrix
    adj_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    W = weights.WSP(adj_matrix).to_W(silence_warnings=True)

    return W


def build_W2_nearest_stop(gdf, gtfs_dir, max_dist_m=TRANSIT_STOP_RADIUS_M):
    """
    W2: Nearest transit stop clustering.
    POIs connected if they share the SAME nearest transit stop.

    Returns: (W2, poi_nearest_stop dict)
    """
    stops = extract_stops_from_gtfs(gtfs_dir)
    if stops is None or len(stops) == 0:
        return None, {}

    epsg = get_utm_epsg(gdf)
    gdf_proj = gdf.to_crs(epsg=epsg)

    stops_gdf = gpd.GeoDataFrame(
        stops,
        geometry=gpd.points_from_xy(stops['stop_lon'], stops['stop_lat']),
        crs='EPSG:4326'
    ).to_crs(epsg=epsg)

    poi_coords = np.array([(g.x, g.y) for g in gdf_proj.geometry])
    stop_coords = np.array([(g.x, g.y) for g in stops_gdf.geometry])

    stop_tree = cKDTree(stop_coords)
    distances, indices = stop_tree.query(poi_coords, k=1)

    # Map POI to nearest stop (if within threshold)
    poi_nearest_stop = {}
    for i, (dist, stop_idx) in enumerate(zip(distances, indices)):
        if dist <= max_dist_m:
            poi_nearest_stop[i] = stop_idx

    if len(poi_nearest_stop) < 100:
        return None, poi_nearest_stop

    # Group POIs by nearest stop
    stop_to_pois = {}
    for poi_idx, stop_idx in poi_nearest_stop.items():
        if stop_idx not in stop_to_pois:
            stop_to_pois[stop_idx] = []
        stop_to_pois[stop_idx].append(poi_idx)

    # Build adjacency matrix
    n = len(gdf)
    rows, cols, data = [], [], []

    for stop_idx, poi_list in stop_to_pois.items():
        if len(poi_list) < 2:
            continue
        for i in range(len(poi_list)):
            for j in range(i + 1, len(poi_list)):
                pi, pj = poi_list[i], poi_list[j]
                rows.extend([pi, pj])
                cols.extend([pj, pi])
                data.extend([1, 1])

    if len(rows) == 0:
        return None, poi_nearest_stop

    adj_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    W = weights.WSP(adj_matrix).to_W()
    W.transform = 'r'

    return W, poi_nearest_stop


def build_W1_exclusive(gdf, poi_nearest_stop, cutoff_m=DISTANCE_CUTOFF_M, alpha=DISTANCE_ALPHA):
    """
    W1_exclusive: Distance-decay weights excluding transit-connected pairs.

    Same as W1 but removes edges between POIs that share a transit stop.
    """
    epsg = get_utm_epsg(gdf)
    gdf_proj = gdf.to_crs(epsg=epsg)
    coords = np.column_stack([gdf_proj.geometry.x, gdf_proj.geometry.y])
    n = len(coords)

    # Get transit pairs to exclude
    stop_to_pois = {}
    for poi_idx, stop_idx in poi_nearest_stop.items():
        if stop_idx not in stop_to_pois:
            stop_to_pois[stop_idx] = []
        stop_to_pois[stop_idx].append(poi_idx)

    transit_pairs = set()
    for stop_idx, poi_list in stop_to_pois.items():
        for i in range(len(poi_list)):
            for j in range(i + 1, len(poi_list)):
                pi, pj = poi_list[i], poi_list[j]
                transit_pairs.add((min(pi, pj), max(pi, pj)))

    # Build distance-decay weights excluding transit pairs using sparse matrix
    tree = cKDTree(coords)
    rows, cols, data = [], [], []

    for i in range(n):
        indices = tree.query_ball_point(coords[i], cutoff_m)

        neighbors_i = []
        weights_i = []

        for j in indices:
            if i != j:
                pair = (min(i, j), max(i, j))
                if pair in transit_pairs:
                    continue  # Skip transit-connected pairs

                d = np.sqrt(np.sum((coords[i] - coords[j])**2))
                if d > 0 and d <= cutoff_m:
                    neighbors_i.append(j)
                    weights_i.append(1.0 / (d ** alpha))

        if neighbors_i:
            total = sum(weights_i)
            for j, w in zip(neighbors_i, weights_i):
                rows.append(i)
                cols.append(j)
                data.append(w / total)

    if len(rows) == 0:
        return None

    adj_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    W = weights.WSP(adj_matrix).to_W(silence_warnings=True)

    return W


# =============================================================================
# DATA LOADING
# =============================================================================

def load_us_categories():
    """Load venue categories for US POIs."""
    dfs = []
    for city in US_CITIES:
        city_dir = US_TRAFFIC_DIR / city
        if city_dir.exists():
            city_dfs = [
                pd.read_parquet(f, columns=['ID_STORE', 'unified_category'])
                for f in city_dir.glob('*.parquet')
            ]
            if city_dfs:
                dfs.append(pd.concat(city_dfs, ignore_index=True))

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset='ID_STORE')
    combined = combined.rename(columns={'ID_STORE': 'poi_id', 'unified_category': 'category'})
    combined['poi_id'] = combined['poi_id'].astype(str)
    return combined


def load_sweden_categories():
    """Load venue categories for Sweden POIs."""
    df = pd.read_parquet(
        SE_TRAFFIC_DIR / 'sweden_weekly_patterns_2024.parquet',
        columns=['PLACEKEY', 'unified_category']
    )
    df = df.drop_duplicates(subset='PLACEKEY')
    df = df.rename(columns={'PLACEKEY': 'poi_id', 'unified_category': 'category'})
    return df


def prepare_analysis_data(df, city, entropy_type='birth', max_n=None):
    """
    Prepare analysis data for a city.

    Args:
        df: Full dataframe with diversity metrics
        city: City name
        entropy_type: 'birth' or 'income'
        max_n: Maximum sample size (None = use all)

    Returns:
        GeoDataFrame ready for analysis
    """
    y_col = f'visitor_entropy_{entropy_type}_norm'
    catch_col = f'catchment_entropy_{entropy_type}_norm'
    res_col = f'residential_entropy_{entropy_type}_norm'

    mask = (
        (df['city'] == city) &
        df[y_col].notna() &
        df[catch_col].notna() &
        df[res_col].notna() &
        df['category'].notna()
    )

    sample = df[mask].copy()

    if max_n and len(sample) > max_n:
        sample = sample.sample(n=max_n, random_state=42)

    sample = sample.reset_index(drop=True)

    gdf = gpd.GeoDataFrame(
        sample,
        geometry=gpd.points_from_xy(sample['lon'], sample['lat']),
        crs='EPSG:4326'
    )

    return gdf


# =============================================================================
# REGRESSION ANALYSIS
# =============================================================================

def extract_gm_lag_stats(model, prefix):
    """
    Extract comprehensive statistics from GM_Lag model.

    Returns dict with:
    - rho: spatial lag coefficient
    - rho_se: standard error
    - rho_z: z-statistic
    - rho_p: p-value
    - rho_ci_lower, rho_ci_upper: 95% CI
    - rho_stationary: True if |rho| < 1 (model is stationary/convergent)
    - pr2: pseudo R-squared
    - sig2: sigma squared (error variance)
    - catchment_coef, catchment_se, catchment_p: catchment variable stats
    """
    results = {}

    # Rho is the last coefficient in GM_Lag
    rho_idx = -1

    # Basic rho stats
    rho = model.betas[rho_idx][0]
    results[f'{prefix}_rho'] = rho
    results[f'{prefix}_rho_se'] = model.std_err[rho_idx]

    # Z-stat and p-value (z_stat is array of [z, p] for each coef)
    if hasattr(model, 'z_stat') and model.z_stat is not None:
        results[f'{prefix}_rho_z'] = model.z_stat[rho_idx][0]
        results[f'{prefix}_rho_p'] = model.z_stat[rho_idx][1]

    # 95% Confidence Interval: rho ± 1.96 * SE
    se = results[f'{prefix}_rho_se']
    results[f'{prefix}_rho_ci_lower'] = rho - 1.96 * se
    results[f'{prefix}_rho_ci_upper'] = rho + 1.96 * se

    # Stationarity check: |rho| < 1 required for valid spatial multiplier
    # If |rho| >= 1, the model is non-stationary and results should be
    # interpreted with caution or excluded from analysis
    results[f'{prefix}_rho_stationary'] = abs(rho) < 1.0

    # Model fit
    results[f'{prefix}_pr2'] = model.pr2
    if hasattr(model, 'sig2'):
        results[f'{prefix}_sig2'] = model.sig2

    # Catchment coefficient stats (index 1, after constant)
    results[f'{prefix}_catchment_coef'] = model.betas[1][0]
    results[f'{prefix}_catchment_se'] = model.std_err[1]
    if hasattr(model, 'z_stat') and model.z_stat is not None:
        results[f'{prefix}_catchment_z'] = model.z_stat[1][0]
        results[f'{prefix}_catchment_p'] = model.z_stat[1][1]

    return results


def run_city_analysis(gdf, gtfs_dir, city_name, entropy_type='birth'):
    """
    Run full spatial regression analysis for one city.

    Uses GM_Lag (GMM estimation) which is faster than ML_Lag for large samples.

    Outputs include:
    - Coefficients with standard errors, z-stats, p-values
    - 95% confidence intervals for rho
    - Model diagnostics (pseudo-R², sigma²)

    Returns: dict with results
    """
    y_col = f'visitor_entropy_{entropy_type}_norm'
    catch_col = f'catchment_entropy_{entropy_type}_norm'
    res_col = f'residential_entropy_{entropy_type}_norm'

    results = {
        'city': city_name,
        'entropy_type': entropy_type,
        'n': len(gdf)
    }

    # Prepare variables
    y = gdf[y_col].values.reshape(-1, 1)

    # Category dummies (top 5 + other)
    top_cats = gdf['category'].value_counts().head(5).index.tolist()
    gdf = gdf.copy()
    gdf['cat_group'] = gdf['category'].apply(lambda x: x if x in top_cats else 'other')
    cat_dummies = pd.get_dummies(gdf['cat_group'], prefix='cat', drop_first=True)

    X_base = gdf[[catch_col, res_col]].values
    X_with_cats = np.hstack([X_base, cat_dummies.values])
    X_names = ['catchment', 'residential'] + cat_dummies.columns.tolist()

    results['n_covariates'] = X_with_cats.shape[1]

    # OLS baseline
    try:
        ols = OLS(y, X_with_cats, name_y='visitor_div', name_x=X_names)
        results['ols_r2'] = ols.r2
        results['ols_adj_r2'] = ols.ar2
        results['ols_catchment_coef'] = ols.betas[1][0]
        results['ols_catchment_se'] = ols.std_err[1]
        results['ols_catchment_t'] = ols.t_stat[1][0]
        results['ols_catchment_p'] = ols.t_stat[1][1]
        results['ols_aic'] = ols.aic
        results['ols_schwarz'] = ols.schwarz  # BIC

        # LM diagnostics for spatial dependence
        if hasattr(ols, 'lm_lag'):
            results['ols_lm_lag_stat'] = ols.lm_lag[0]
            results['ols_lm_lag_p'] = ols.lm_lag[1]
        if hasattr(ols, 'lm_error'):
            results['ols_lm_error_stat'] = ols.lm_error[0]
            results['ols_lm_error_p'] = ols.lm_error[1]
    except Exception as e:
        print(f"  OLS failed: {e}")

    # Build spatial weights using distance-decay with cutoff
    W1 = build_W1_distance_decay(gdf, cutoff_m=DISTANCE_CUTOFF_M, alpha=DISTANCE_ALPHA)
    results['w1_mean_neighbors'] = W1.mean_neighbors
    results['w1_n_nonzero'] = W1.n * W1.mean_neighbors
    results['w1_cutoff_m'] = DISTANCE_CUTOFF_M
    results['w1_decay_alpha'] = DISTANCE_ALPHA

    W2, poi_nearest_stop = build_W2_nearest_stop(gdf, gtfs_dir)
    if W2:
        results['w2_mean_neighbors'] = W2.mean_neighbors
        results['w2_n_nonzero'] = W2.n * W2.mean_neighbors
        results['pois_near_transit'] = len(poi_nearest_stop)
        results['pois_near_transit_pct'] = 100 * len(poi_nearest_stop) / len(gdf)

    W1_ex = None
    if poi_nearest_stop:
        W1_ex = build_W1_exclusive(gdf, poi_nearest_stop, cutoff_m=DISTANCE_CUTOFF_M, alpha=DISTANCE_ALPHA)
        if W1_ex:
            results['w1ex_mean_neighbors'] = W1_ex.mean_neighbors
            results['w1ex_n_nonzero'] = W1_ex.n * W1_ex.mean_neighbors

    # Spatial Lag models using GM_Lag (GMM estimation)
    # Results flagged if |rho| >= 1 (non-stationary)

    # W1: Distance-decay weights
    try:
        gm_w1 = GM_Lag(y, X_with_cats, w=W1, name_y='visitor_div', name_x=X_names)
        w1_stats = extract_gm_lag_stats(gm_w1, 'w1')
        results.update(w1_stats)
        if not w1_stats.get('w1_rho_stationary', True):
            print(f"      W1: rho={w1_stats['w1_rho']:.4f} [NON-STATIONARY]")
    except Exception as e:
        print(f"      W1: GM_Lag failed: {e}")

    # W2: Transit clustering (shared nearest stop)
    if W2:
        try:
            gm_w2 = GM_Lag(y, X_with_cats, w=W2, name_y='visitor_div', name_x=X_names)
            w2_stats = extract_gm_lag_stats(gm_w2, 'w2')
            results.update(w2_stats)
            if not w2_stats.get('w2_rho_stationary', True):
                print(f"      W2: rho={w2_stats['w2_rho']:.4f} [NON-STATIONARY]")
        except Exception as e:
            print(f"      W2: GM_Lag failed: {e}")

    # W1_exclusive: Distance-decay excluding transit-connected pairs
    # W1_ex is n×n matrix with transit edges zeroed out - use full y and X
    if W1_ex:
        try:
            gm_w1ex = GM_Lag(y, X_with_cats, w=W1_ex, name_y='visitor_div', name_x=X_names)
            w1ex_stats = extract_gm_lag_stats(gm_w1ex, 'w1ex')
            results.update(w1ex_stats)
            if not w1ex_stats.get('w1ex_rho_stationary', True):
                print(f"      W1_ex: rho={w1ex_stats['w1ex_rho']:.4f} [NON-STATIONARY]")
        except Exception as e:
            print(f"      W1_ex: GM_Lag failed: {e}")

    return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Spatial Spillover Analysis')
    parser.add_argument('--city', type=str, default=None,
                        help='Single city to analyze (e.g., Stockholm, new_york). Default: all cities')
    parser.add_argument('--max-n', type=int, default=None,
                        help='Maximum POIs per city (default: use all)')
    parser.add_argument('--output-dir', type=str, default='outputs/phase2',
                        help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which cities to run
    if args.city:
        city_lower = args.city.lower()
        if city_lower in [c.lower() for c in US_CITIES]:
            us_cities = [c for c in US_CITIES if c.lower() == city_lower]
            se_cities = []
        elif any(args.city.lower().startswith(c.lower()[:4]) for c in SWEDEN_CITIES):
            us_cities = []
            se_cities = [c for c in SWEDEN_CITIES if c.lower().startswith(args.city.lower()[:4])]
        else:
            print(f"City '{args.city}' not found. Available: {US_CITIES + SWEDEN_CITIES}")
            return
    else:
        us_cities = US_CITIES
        se_cities = SWEDEN_CITIES

    print("=" * 70)
    print("SPATIAL SPILLOVER ANALYSIS - Distance-Decay Weights")
    print(f"Cities: {us_cities + se_cities}")
    print(f"Max N per city: {args.max_n or 'ALL'}")
    print(f"W1: Distance-decay (cutoff={DISTANCE_CUTOFF_M}m, alpha={DISTANCE_ALPHA})")
    print(f"Output dir: {output_dir}")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading diversity metrics...")
    us_df = pd.read_parquet(ROUTING_DIR / 'us_poi_diversity_metrics.parquet')
    se_df = pd.read_parquet(ROUTING_DIR / 'sweden_poi_diversity_metrics.parquet')
    us_df['poi_id'] = us_df['poi_id'].astype(str)
    se_df['poi_id'] = se_df['poi_id'].astype(str)
    print(f"  US POIs: {len(us_df):,}")
    print(f"  Sweden POIs: {len(se_df):,}")

    print("\n[2/4] Loading venue categories...")
    us_cats = load_us_categories()
    se_cats = load_sweden_categories()
    us_df = us_df.merge(us_cats, on='poi_id', how='left')
    se_df = se_df.merge(se_cats, on='poi_id', how='left')
    print(f"  US with categories: {us_df['category'].notna().sum():,}")
    print(f"  Sweden with categories: {se_df['category'].notna().sum():,}")

    # Run analysis for each entropy type
    all_results = []

    for entropy_type in ['birth', 'income']:
        print(f"\n[3/4] Running {entropy_type.upper()} entropy analysis...")

        # US cities
        for city in us_cities:
            print(f"\n  Processing US - {city}...")
            gtfs_path = GTFS_DIR / city

            gdf = prepare_analysis_data(us_df, city, entropy_type, args.max_n)
            if len(gdf) < 100:
                print(f"    Skipped: only {len(gdf)} valid POIs")
                continue

            print(f"    N = {len(gdf):,}")

            if gtfs_path.exists():
                result = run_city_analysis(gdf, str(gtfs_path), f"US - {city}", entropy_type)
                result['country'] = 'US'
                all_results.append(result)
                # Print rho with stationarity check
                w1_rho = result.get('w1_rho', float('nan'))
                w1_stat = "" if result.get('w1_rho_stationary', True) else " [NON-STATIONARY]"
                w1_sig = "***" if result.get('w1_rho_p', 1) < 0.001 else "**" if result.get('w1_rho_p', 1) < 0.01 else "*" if result.get('w1_rho_p', 1) < 0.05 else ""
                print(f"    ρ(W1)={w1_rho:.4f}{w1_sig}{w1_stat}")
            else:
                print(f"    Skipped: no GTFS data")

        # Sweden cities (use national GTFS from c_01)
        gtfs_se_path = GTFS_DIR / 'sweden_south' / 'c_01'

        for city in se_cities:
            print(f"\n  Processing Sweden - {city}...")

            # Handle encoding variations in city names
            city_mask = se_df['city'].str.contains(city[:4], case=False, na=False)
            if city_mask.sum() == 0:
                print(f"    Skipped: city not found")
                continue

            actual_city = se_df[city_mask]['city'].iloc[0]
            gdf = prepare_analysis_data(se_df, actual_city, entropy_type, args.max_n)

            if len(gdf) < 100:
                print(f"    Skipped: only {len(gdf)} valid POIs")
                continue

            print(f"    N = {len(gdf):,}")

            if gtfs_se_path.exists():
                result = run_city_analysis(gdf, str(gtfs_se_path), f"Sweden - {city}", entropy_type)
                result['country'] = 'Sweden'
                all_results.append(result)
                # Print rho with stationarity check
                w1_rho = result.get('w1_rho', float('nan'))
                w1_stat = "" if result.get('w1_rho_stationary', True) else " [NON-STATIONARY]"
                w1_sig = "***" if result.get('w1_rho_p', 1) < 0.001 else "**" if result.get('w1_rho_p', 1) < 0.01 else "*" if result.get('w1_rho_p', 1) < 0.05 else ""
                print(f"    ρ(W1)={w1_rho:.4f}{w1_sig}{w1_stat}")
            else:
                print(f"    Skipped: no GTFS data at {gtfs_se_path}")

    # Save results
    print("\n[4/4] Saving results...")
    results_df = pd.DataFrame(all_results)

    # Save combined results
    output_file = output_dir / 'spatial_spillover_results_full.csv'
    results_df.to_csv(output_file, index=False)
    print(f"  Saved: {output_file}")

    # Save separate files by entropy type
    for etype in ['birth', 'income']:
        subset = results_df[results_df['entropy_type'] == etype]
        if len(subset) > 0:
            etype_file = output_dir / f'spatial_spillover_results_{etype}.csv'
            subset.to_csv(etype_file, index=False)
            print(f"  Saved: {etype_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for etype in ['birth', 'income']:
        subset = results_df[results_df['entropy_type'] == etype]
        if len(subset) > 0:
            print(f"\n{'='*70}")
            print(f"{etype.upper()} ENTROPY RESULTS")
            print(f"{'='*70}")

            # Rho coefficients with CIs
            print("\n1. SPATIAL LAG COEFFICIENTS (ρ) with 95% CI:")
            print("-" * 70)
            for _, row in subset.iterrows():
                city = row['city']
                n = row['n']
                print(f"\n  {city} (n={n:,}):")

                # W1
                if 'w1_rho' in row and pd.notna(row['w1_rho']):
                    rho = row['w1_rho']
                    ci_l = row.get('w1_rho_ci_lower', float('nan'))
                    ci_u = row.get('w1_rho_ci_upper', float('nan'))
                    p = row.get('w1_rho_p', float('nan'))
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                    print(f"    W1 (distance):    ρ = {rho:.4f} [{ci_l:.4f}, {ci_u:.4f}] p={p:.4f} {sig}")

                # W2
                if 'w2_rho' in row and pd.notna(row['w2_rho']):
                    rho = row['w2_rho']
                    ci_l = row.get('w2_rho_ci_lower', float('nan'))
                    ci_u = row.get('w2_rho_ci_upper', float('nan'))
                    p = row.get('w2_rho_p', float('nan'))
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                    print(f"    W2 (transit):     ρ = {rho:.4f} [{ci_l:.4f}, {ci_u:.4f}] p={p:.4f} {sig}")

                # W1_ex
                if 'w1ex_rho' in row and pd.notna(row['w1ex_rho']):
                    rho = row['w1ex_rho']
                    ci_l = row.get('w1ex_rho_ci_lower', float('nan'))
                    ci_u = row.get('w1ex_rho_ci_upper', float('nan'))
                    p = row.get('w1ex_rho_p', float('nan'))
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                    print(f"    W1_ex (non-tran): ρ = {rho:.4f} [{ci_l:.4f}, {ci_u:.4f}] p={p:.4f} {sig}")

            # Model fit comparison
            print("\n2. MODEL FIT (Pseudo-R²):")
            print("-" * 70)
            fit_cols = ['city', 'ols_r2', 'w1_pr2', 'w2_pr2', 'w1ex_pr2']
            available = [c for c in fit_cols if c in subset.columns]
            if available:
                print(subset[available].round(4).to_string(index=False))

    print("\n" + "=" * 70)
    print("SIGNIFICANCE: *** p<0.001, ** p<0.01, * p<0.05")
    print("=" * 70)
    print(f"\nCompleted at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
