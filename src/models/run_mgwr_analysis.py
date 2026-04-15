#!/usr/bin/env python
"""
Phase 3: GWR (Geographically Weighted Regression) Analysis

Analyzes local heterogeneity in the transit-diversity relationship.
Identifies transit mixing hotspots where catchment diversity has strongest local effect.

Uses standard GWR (single adaptive bandwidth) with da Silva-Fotheringham (2016)
multiple testing correction for proper local significance inference.

Usage:
    # Run for a single city (modular approach)
    python -m src.models.run_mgwr_analysis --city "US - new_york" --entropy birth

    # Run for all cities (parallel)
    python -m src.models.run_mgwr_analysis --all --entropy birth

    # Run for all cities (sequential, for debugging)
    python -m src.models.run_mgwr_analysis --all --entropy birth --no-parallel

    # List available cities
    python -m src.models.run_mgwr_analysis --list-cities
"""

import argparse
import warnings
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count
import traceback

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats
from scipy.linalg import LinAlgWarning

# GWR imports
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW

# Suppress numerical warnings (handled via jittering and min bandwidth)
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=LinAlgWarning)


# =============================================================================
# CONFIGURATION
# =============================================================================

def get_project_root():
    """Get project root directory."""
    script_dir = Path(__file__).resolve().parent
    if script_dir.name == 'models' and script_dir.parent.name == 'src':
        return script_dir.parent.parent
    return Path.cwd()

PROJECT_ROOT = get_project_root()

# Data paths
ROUTING_DIR = PROJECT_ROOT / 'dbs/routing'
US_TRAFFIC_DIR = PROJECT_ROOT / 'dbs/us_foot_traffic/cities'
SE_TRAFFIC_DIR = PROJECT_ROOT / 'dbs/sweden_weekly_patterns'
OUTPUT_DIR = PROJECT_ROOT / 'outputs/phase3'

# Analysis parameters
MAX_N_PER_CATEGORY = None  # No cap - use all available POIs
RANDOM_SEED = 42
ALPHA = 0.05  # Nominal significance level (before correction)

# Venue categories to analyze
VENUE_CATEGORIES = [
    'all',  # All POIs combined (no category filter)
    'food_dining',
    'entertainment_recreation',
    'accommodation_travel',
    'education_higher',
    'civic_community',
    'other'  # All others combined
]

# Cities
US_CITIES = ['US - new_york', 'US - washington_dc', 'US - atlanta']
SWEDEN_CITIES = [
    'Sweden - Stockholm', 'Sweden - Göteborg', 'Sweden - Malmö',
    'Sweden - Uppsala', 'Sweden - Västerås', 'Sweden - Örebro',
    'Sweden - Linköping', 'Sweden - Helsingborg', 'Sweden - Lund'
]
ALL_CITIES = US_CITIES + SWEDEN_CITIES


# =============================================================================
# DATA LOADING
# =============================================================================

def load_diversity_data():
    """Load and merge diversity metrics with venue categories."""
    print("Loading diversity metrics...")

    # Load diversity data
    us_df = pd.read_parquet(ROUTING_DIR / 'us_poi_diversity_metrics.parquet')
    se_df = pd.read_parquet(ROUTING_DIR / 'sweden_poi_diversity_metrics.parquet')

    us_df['poi_id'] = us_df['poi_id'].astype(str)
    se_df['poi_id'] = se_df['poi_id'].astype(str)

    # Load US categories
    us_cats_list = []
    for city in ['new_york', 'washington_dc', 'atlanta']:
        city_dir = US_TRAFFIC_DIR / city
        if city_dir.exists():
            for f in city_dir.glob('*.parquet'):
                df = pd.read_parquet(f, columns=['ID_STORE', 'unified_category'])
                us_cats_list.append(df)

    if us_cats_list:
        us_cats = pd.concat(us_cats_list, ignore_index=True)
        us_cats = us_cats.drop_duplicates(subset='ID_STORE')
        us_cats = us_cats.rename(columns={'ID_STORE': 'poi_id', 'unified_category': 'category'})
        us_cats['poi_id'] = us_cats['poi_id'].astype(str)
        us_df = us_df.merge(us_cats, on='poi_id', how='left')

    # Load Sweden categories
    se_cats = pd.read_parquet(
        SE_TRAFFIC_DIR / 'sweden_weekly_patterns_2024.parquet',
        columns=['PLACEKEY', 'unified_category']
    )
    se_cats = se_cats.drop_duplicates(subset='PLACEKEY')
    se_cats = se_cats.rename(columns={'PLACEKEY': 'poi_id', 'unified_category': 'category'})
    se_df = se_df.merge(se_cats, on='poi_id', how='left')

    # Add country prefix to city names
    us_df['city'] = 'US - ' + us_df['city']
    se_df['city'] = 'Sweden - ' + se_df['city']

    # Combine
    df = pd.concat([us_df, se_df], ignore_index=True)

    print(f"  Total POIs: {len(df):,}")
    print(f"  With categories: {df['category'].notna().sum():,}")

    return df


def prepare_city_venue_data(df, city, venue_category, entropy_type, max_n=MAX_N_PER_CATEGORY):
    """
    Prepare data for a specific city and venue category.

    Returns GeoDataFrame with coordinates and variables, or None if insufficient data.
    """
    y_col = f'visitor_entropy_{entropy_type}_norm'
    catch_col = f'catchment_entropy_{entropy_type}_norm'
    res_col = f'residential_entropy_{entropy_type}_norm'

    # Filter by city
    city_mask = df['city'].str.contains(city.split(' - ')[1][:4], case=False, na=False)
    city_df = df[city_mask].copy()

    if len(city_df) == 0:
        return None

    # Filter by venue category
    if venue_category == 'all':
        # No category filter - use all POIs
        venue_df = city_df.copy()
    elif venue_category == 'other':
        # All categories not in the main list
        main_cats = [c for c in VENUE_CATEGORIES if c not in ('other', 'all')]
        venue_mask = ~city_df['category'].isin(main_cats)
        venue_df = city_df[venue_mask].copy()
    else:
        venue_mask = city_df['category'] == venue_category
        venue_df = city_df[venue_mask].copy()

    # Filter for complete data
    complete_mask = (
        venue_df[y_col].notna() &
        venue_df[catch_col].notna() &
        venue_df[res_col].notna()
    )
    venue_df = venue_df[complete_mask].copy()

    if len(venue_df) < 100:
        return None

    # Sample if needed (only if max_n is set)
    if max_n is not None and len(venue_df) > max_n:
        venue_df = venue_df.sample(n=max_n, random_state=RANDOM_SEED)

    # Reset index
    venue_df = venue_df.reset_index(drop=True)

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        venue_df,
        geometry=gpd.points_from_xy(venue_df['lon'], venue_df['lat']),
        crs='EPSG:4326'
    )

    return gdf


def get_utm_epsg(gdf):
    """Get appropriate UTM EPSG code for projection."""
    centroid = gdf.geometry.unary_union.centroid
    utm_zone = int((centroid.x + 180) / 6) + 1
    hemisphere = 'north' if centroid.y >= 0 else 'south'
    return 32600 + utm_zone if hemisphere == 'north' else 32700 + utm_zone


# =============================================================================
# GWR ANALYSIS
# =============================================================================

def run_gwr(gdf, entropy_type):
    """
    Run GWR analysis with da Silva-Fotheringham (2016) multiple testing correction.

    Uses standard GWR (single adaptive bandwidth) for robust inference.
    Significance is determined using the corrected alpha threshold:
        alpha_corrected = alpha * (tr(S) / n)
    where tr(S) is the effective number of parameters.

    Model: visitor_diversity ~ intercept + catchment_diversity + residential_diversity

    Returns dict with results including local coefficients and corrected significance.
    """
    y_col = f'visitor_entropy_{entropy_type}_norm'
    catch_col = f'catchment_entropy_{entropy_type}_norm'
    res_col = f'residential_entropy_{entropy_type}_norm'

    # Project to UTM for proper distance calculations
    epsg = get_utm_epsg(gdf)
    gdf_proj = gdf.to_crs(epsg=epsg)

    # Extract coordinates (in meters)
    coords = np.array([(geom.x, geom.y) for geom in gdf_proj.geometry])

    # Add small jitter to handle duplicate coordinates (1 meter random noise)
    np.random.seed(RANDOM_SEED)
    jitter = np.random.normal(0, 1, coords.shape)  # 1 meter std dev
    coords = coords + jitter

    # Prepare variables
    y = gdf[y_col].values.reshape(-1, 1)
    X = gdf[[catch_col, res_col]].values

    # Check for low variance (can cause singular matrices)
    x_var = np.var(X, axis=0)
    if np.any(x_var < 1e-10):
        return {
            'n': len(y),
            'success': False,
            'error': f'Near-zero variance in predictors: catchment_var={x_var[0]:.2e}, residential_var={x_var[1]:.2e}'
        }

    # Standardize X for better convergence
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1  # Avoid division by zero
    X_scaled = (X - X_mean) / X_std

    n = len(y)

    results = {
        'n': n,
        'coords': coords,
        'y': y.flatten(),
        'X_raw': X,
        'X_scaled': X_scaled,
        'X_mean': X_mean,
        'X_std': X_std
    }

    try:
        # =================================================================
        # Standard GWR with single adaptive bandwidth (full inference)
        # =================================================================
        print("    Selecting GWR bandwidth (adaptive bisquare)...")

        # Set minimum bandwidth to avoid singular matrices in small neighborhoods
        # Minimum = max(50, 1% of sample size)
        min_bw = max(50, int(0.01 * n))

        gwr_selector = Sel_BW(coords, y, X_scaled, multi=False, constant=True)
        bw = gwr_selector.search(criterion='AICc', bw_min=min_bw)

        results['bandwidth'] = bw
        print(f"    GWR bandwidth: {bw:.0f} neighbors (min={min_bw})")

        # Fit GWR
        print("    Fitting GWR model...")
        gwr = GWR(coords, y, X_scaled, bw, constant=True)
        gwr_results = gwr.fit()

        # Extract local coefficients (transform back to original scale)
        # For standardized X: beta_original = beta_scaled / X_std
        local_betas_scaled = gwr_results.params
        local_betas = np.zeros_like(local_betas_scaled)
        local_betas[:, 0] = local_betas_scaled[:, 0]  # Intercept
        local_betas[:, 1] = local_betas_scaled[:, 1] / X_std[0]  # Catchment
        local_betas[:, 2] = local_betas_scaled[:, 2] / X_std[1]  # Residential

        results['local_beta_intercept'] = local_betas[:, 0]
        results['local_beta_catchment'] = local_betas[:, 1]
        results['local_beta_residential'] = local_betas[:, 2]

        # Standard errors (also transform)
        local_se_scaled = gwr_results.bse
        local_se = np.zeros_like(local_se_scaled)
        local_se[:, 0] = local_se_scaled[:, 0]
        local_se[:, 1] = local_se_scaled[:, 1] / X_std[0]
        local_se[:, 2] = local_se_scaled[:, 2] / X_std[1]

        results['local_se_intercept'] = local_se[:, 0]
        results['local_se_catchment'] = local_se[:, 1]
        results['local_se_residential'] = local_se[:, 2]

        # Local t-statistics
        results['local_t_intercept'] = local_betas[:, 0] / local_se[:, 0]
        results['local_t_catchment'] = local_betas[:, 1] / local_se[:, 1]
        results['local_t_residential'] = local_betas[:, 2] / local_se[:, 2]

        # Local p-values (two-tailed, raw)
        df_residual = n - np.sum(gwr_results.tr_S)  # Effective degrees of freedom
        results['df_residual'] = df_residual
        results['local_p_intercept'] = 2 * (1 - stats.t.cdf(np.abs(results['local_t_intercept']), df_residual))
        results['local_p_catchment'] = 2 * (1 - stats.t.cdf(np.abs(results['local_t_catchment']), df_residual))
        results['local_p_residential'] = 2 * (1 - stats.t.cdf(np.abs(results['local_t_residential']), df_residual))

        # Local R²
        results['local_r2'] = gwr_results.localR2.flatten()

        # Model diagnostics
        results['aic'] = gwr_results.aicc
        results['r2'] = gwr_results.R2
        results['adj_r2'] = gwr_results.adj_R2
        results['effective_df'] = np.sum(gwr_results.tr_S)
        results['residual_ss'] = gwr_results.RSS

        # =================================================================
        # da Silva-Fotheringham (2016) multiple testing correction
        # =================================================================
        # The effective number of independent tests is n / tr(S)
        # Corrected significance threshold: alpha_corrected = alpha * tr(S) / n
        tr_S = np.sum(gwr_results.tr_S)
        n_eff = n / tr_S  # Effective number of independent tests
        alpha_corrected = ALPHA * tr_S / n  # Corrected significance level

        results['tr_S'] = tr_S
        results['n_eff'] = n_eff
        results['alpha_corrected'] = alpha_corrected

        print(f"    da Silva-Fotheringham correction: n_eff={n_eff:.1f}, α_corrected={alpha_corrected:.6f}")

        # =================================================================
        # Hotspot identification (using corrected significance)
        # =================================================================
        # Hotspot = local_p_catchment < alpha_corrected AND local_beta_catchment > 0
        is_significant = results['local_p_catchment'] < alpha_corrected
        is_positive = results['local_beta_catchment'] > 0
        results['is_hotspot'] = is_significant & is_positive
        results['n_hotspots'] = np.sum(results['is_hotspot'])
        results['hotspot_pct'] = 100 * results['n_hotspots'] / n

        # Coldspot (significant negative)
        is_negative = results['local_beta_catchment'] < 0
        results['is_coldspot'] = is_significant & is_negative
        results['n_coldspots'] = np.sum(results['is_coldspot'])
        results['coldspot_pct'] = 100 * results['n_coldspots'] / n

        # Non-significant
        results['n_nonsig'] = np.sum(~is_significant)
        results['nonsig_pct'] = 100 * results['n_nonsig'] / n

        # Also store the significance mask for output
        results['is_significant'] = is_significant

        # =================================================================
        # Coefficient heterogeneity statistics
        # =================================================================
        beta_catch = results['local_beta_catchment']
        results['beta_catchment_mean'] = np.mean(beta_catch)
        results['beta_catchment_std'] = np.std(beta_catch)
        results['beta_catchment_cv'] = np.std(beta_catch) / np.abs(np.mean(beta_catch)) if np.mean(beta_catch) != 0 else np.nan
        results['beta_catchment_min'] = np.min(beta_catch)
        results['beta_catchment_max'] = np.max(beta_catch)
        results['beta_catchment_q25'] = np.percentile(beta_catch, 25)
        results['beta_catchment_median'] = np.percentile(beta_catch, 50)
        results['beta_catchment_q75'] = np.percentile(beta_catch, 75)
        results['beta_catchment_iqr'] = results['beta_catchment_q75'] - results['beta_catchment_q25']

        # Residential coefficient stats
        beta_res = results['local_beta_residential']
        results['beta_residential_mean'] = np.mean(beta_res)
        results['beta_residential_std'] = np.std(beta_res)
        results['beta_residential_cv'] = np.std(beta_res) / np.abs(np.mean(beta_res)) if np.mean(beta_res) != 0 else np.nan

        results['success'] = True
        print(f"    GWR fit successful: R²={results['r2']:.4f}, adj.R²={results['adj_r2']:.4f}")
        print(f"    Hotspots: {results['n_hotspots']} ({results['hotspot_pct']:.1f}%), Coldspots: {results['n_coldspots']} ({results['coldspot_pct']:.1f}%)")

    except Exception as e:
        print(f"    GWR failed: {e}")
        results['success'] = False
        results['error'] = str(e)

    return results


# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================

def save_local_coefficients(gdf, gwr_results, city, venue_category, entropy_type, output_dir):
    """Save local coefficients to parquet for visualization."""
    if not gwr_results['success']:
        return None

    # Create output dataframe
    out_df = pd.DataFrame({
        'poi_id': gdf['poi_id'].values,
        'lon': gdf['lon'].values,
        'lat': gdf['lat'].values,
        'city': city,
        'venue_category': venue_category,
        'entropy_type': entropy_type,
        'visitor_diversity': gwr_results['y'],
        'catchment_diversity': gwr_results['X_raw'][:, 0],
        'residential_diversity': gwr_results['X_raw'][:, 1],
        'beta_intercept': gwr_results['local_beta_intercept'],
        'beta_catchment': gwr_results['local_beta_catchment'],
        'beta_residential': gwr_results['local_beta_residential'],
        'se_intercept': gwr_results['local_se_intercept'],
        'se_catchment': gwr_results['local_se_catchment'],
        'se_residential': gwr_results['local_se_residential'],
        't_intercept': gwr_results['local_t_intercept'],
        't_catchment': gwr_results['local_t_catchment'],
        't_residential': gwr_results['local_t_residential'],
        'p_catchment': gwr_results['local_p_catchment'],  # Raw p-value
        'local_r2': gwr_results['local_r2'],
        'is_significant': gwr_results['is_significant'],  # da Silva-Fotheringham corrected
        'is_hotspot': gwr_results['is_hotspot'],
        'is_coldspot': gwr_results['is_coldspot'],
        'alpha_corrected': gwr_results['alpha_corrected'],  # Store for reference
    })

    # Create filename
    city_slug = city.lower().replace(' - ', '_').replace(' ', '_')
    venue_slug = venue_category.lower().replace(' ', '_')
    filename = f"{city_slug}_{entropy_type}_{venue_slug}.parquet"

    # Save
    coef_dir = output_dir / 'gwr_local_coefficients'
    coef_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(coef_dir / filename, index=False)

    return coef_dir / filename


def create_summary_row(city, venue_category, entropy_type, gwr_results):
    """Create a summary row for the results CSV."""
    row = {
        'city': city,
        'venue_category': venue_category,
        'entropy_type': entropy_type,
        'n': gwr_results['n'],
        'success': gwr_results['success']
    }

    if gwr_results['success']:
        row.update({
            'r2': gwr_results['r2'],
            'adj_r2': gwr_results['adj_r2'],
            'aic': gwr_results['aic'],
            'effective_df': gwr_results['effective_df'],
            'df_residual': gwr_results['df_residual'],
            'bandwidth': gwr_results['bandwidth'],
            # da Silva-Fotheringham correction parameters
            'tr_S': gwr_results['tr_S'],
            'n_eff': gwr_results['n_eff'],
            'alpha_corrected': gwr_results['alpha_corrected'],
            # Catchment coefficient stats
            'beta_catchment_mean': gwr_results['beta_catchment_mean'],
            'beta_catchment_std': gwr_results['beta_catchment_std'],
            'beta_catchment_cv': gwr_results['beta_catchment_cv'],
            'beta_catchment_min': gwr_results['beta_catchment_min'],
            'beta_catchment_q25': gwr_results['beta_catchment_q25'],
            'beta_catchment_median': gwr_results['beta_catchment_median'],
            'beta_catchment_q75': gwr_results['beta_catchment_q75'],
            'beta_catchment_max': gwr_results['beta_catchment_max'],
            'beta_catchment_iqr': gwr_results['beta_catchment_iqr'],
            # Residential coefficient stats
            'beta_residential_mean': gwr_results['beta_residential_mean'],
            'beta_residential_std': gwr_results['beta_residential_std'],
            'beta_residential_cv': gwr_results['beta_residential_cv'],
            # Hotspot/coldspot counts (using corrected significance)
            'n_hotspots': gwr_results['n_hotspots'],
            'hotspot_pct': gwr_results['hotspot_pct'],
            'n_coldspots': gwr_results['n_coldspots'],
            'coldspot_pct': gwr_results['coldspot_pct'],
            'n_nonsig': gwr_results['n_nonsig'],
            'nonsig_pct': gwr_results['nonsig_pct']
        })
    else:
        row['error'] = gwr_results.get('error', 'Unknown error')

    return row


# =============================================================================
# PARALLEL EXECUTION
# =============================================================================

def run_single_task(task):
    """
    Worker function for parallel execution.

    Args:
        task: tuple of (df, city, venue_cat, entropy_type, output_dir)

    Returns:
        Summary row dict
    """
    df, city, venue_cat, entropy_type, output_dir = task

    try:
        print(f"\n  [{city} / {venue_cat} / {entropy_type}]")

        # Prepare data
        gdf = prepare_city_venue_data(df, city, venue_cat, entropy_type)

        if gdf is None or len(gdf) < 100:
            print(f"    Skipped: insufficient data")
            return {
                'city': city,
                'venue_category': venue_cat,
                'entropy_type': entropy_type,
                'n': 0 if gdf is None else len(gdf),
                'success': False,
                'error': 'Insufficient data'
            }

        print(f"    N = {len(gdf):,} POIs")

        # Run GWR
        gwr_results = run_gwr(gdf, entropy_type)

        # Save local coefficients
        if gwr_results['success']:
            save_local_coefficients(gdf, gwr_results, city, venue_cat, entropy_type, Path(output_dir))

        # Create summary row
        return create_summary_row(city, venue_cat, entropy_type, gwr_results)

    except Exception as e:
        print(f"    Error in worker: {e}")
        traceback.print_exc()
        return {
            'city': city,
            'venue_category': venue_cat,
            'entropy_type': entropy_type,
            'n': 0,
            'success': False,
            'error': str(e)
        }


def run_city_analysis(df, city, entropy_type, output_dir, venue_categories=None):
    """Run GWR analysis for specified venue categories in one city (sequential)."""
    if venue_categories is None:
        venue_categories = VENUE_CATEGORIES

    print(f"\n{'='*70}")
    print(f"GWR ANALYSIS: {city} - {entropy_type.upper()} entropy")
    print(f"{'='*70}")

    results = []

    for venue_cat in venue_categories:
        print(f"\n  [{venue_cat}]")

        # Prepare data
        gdf = prepare_city_venue_data(df, city, venue_cat, entropy_type)

        if gdf is None or len(gdf) < 100:
            print(f"    Skipped: insufficient data")
            results.append({
                'city': city,
                'venue_category': venue_cat,
                'entropy_type': entropy_type,
                'n': 0 if gdf is None else len(gdf),
                'success': False,
                'error': 'Insufficient data'
            })
            continue

        print(f"    N = {len(gdf):,} POIs")

        # Run GWR
        gwr_results = run_gwr(gdf, entropy_type)

        # Save local coefficients
        if gwr_results['success']:
            save_local_coefficients(gdf, gwr_results, city, venue_cat, entropy_type, output_dir)

        # Create summary row
        results.append(create_summary_row(city, venue_cat, entropy_type, gwr_results))

    return results


def main():
    parser = argparse.ArgumentParser(description='GWR Analysis for Transit-Diversity Relationship')
    parser.add_argument('--city', type=str, help='City to analyze (e.g., "US - new_york")')
    parser.add_argument('--entropy', type=str, choices=['birth', 'income', 'both'], default='both',
                        help='Entropy type to analyze')
    parser.add_argument('--all', action='store_true', help='Run for all cities')
    parser.add_argument('--list-cities', action='store_true', help='List available cities')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel processing')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count - 1)')
    parser.add_argument('--append', action='store_true',
                        help='Append to existing results instead of overwriting')
    parser.add_argument('--category', type=str, default=None,
                        help='Run only specific category (e.g., "all", "food_dining")')

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # List cities
    if args.list_cities:
        print("Available cities:")
        for city in ALL_CITIES:
            print(f"  {city}")
        return

    # Determine which cities to run
    if args.all:
        cities = ALL_CITIES
    elif args.city:
        cities = [args.city]
    else:
        print("Error: Specify --city or --all")
        parser.print_help()
        return

    # Determine entropy types
    if args.entropy == 'both':
        entropy_types = ['birth', 'income']
    else:
        entropy_types = [args.entropy]

    # Determine venue categories
    if args.category:
        if args.category not in VENUE_CATEGORIES:
            print(f"Error: Unknown category '{args.category}'")
            print(f"Available categories: {VENUE_CATEGORIES}")
            return
        venue_categories = [args.category]
    else:
        venue_categories = VENUE_CATEGORIES

    # Determine number of workers
    n_workers = args.workers if args.workers else max(1, cpu_count() - 1)
    use_parallel = not args.no_parallel and len(cities) * len(venue_categories) * len(entropy_types) > 1

    print("=" * 70)
    print("GWR ANALYSIS - PHASE 3")
    print(f"Cities: {len(cities)}")
    print(f"Entropy types: {entropy_types}")
    print(f"Venue categories: {venue_categories}")
    print(f"Output dir: {output_dir}")
    print(f"Sample cap: {'None (all POIs)' if MAX_N_PER_CATEGORY is None else MAX_N_PER_CATEGORY}")
    print(f"Significance: da Silva-Fotheringham (2016) corrected, α={ALPHA}")
    print(f"Parallel: {use_parallel} (workers={n_workers})")
    print(f"Append mode: {args.append}")
    print("=" * 70)

    # Load data once
    df = load_diversity_data()

    # Load existing results if append mode
    all_results = []
    existing_count = 0
    results_file = output_dir / 'gwr_summary_results.csv'
    if args.append and results_file.exists():
        existing_df = pd.read_csv(results_file)
        all_results = existing_df.to_dict('records')
        existing_count = len(all_results)
        print(f"\nLoaded {existing_count} existing results from {results_file}")
        print(f"Existing entropy types: {existing_df['entropy_type'].unique().tolist()}")

    if use_parallel:
        # Build task list
        tasks = []
        for entropy_type in entropy_types:
            for city in cities:
                for venue_cat in venue_categories:
                    tasks.append((df, city, venue_cat, entropy_type, str(output_dir)))

        print(f"\nRunning {len(tasks)} tasks with {n_workers} workers...")

        # Run in parallel
        with Pool(processes=n_workers) as pool:
            all_results = pool.map(run_single_task, tasks)

        # Save results
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_dir / 'gwr_summary_results.csv', index=False)

    else:
        # Sequential execution
        for entropy_type in entropy_types:
            for city in cities:
                city_results = run_city_analysis(df, city, entropy_type, output_dir, venue_categories)
                all_results.extend(city_results)

                # Save intermediate results
                results_df = pd.DataFrame(all_results)
                results_df.to_csv(output_dir / 'gwr_summary_results.csv', index=False)

    # Final summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    results_df = pd.DataFrame(all_results)
    successful = results_df[results_df['success'] == True]

    n_new = len(results_df) - existing_count
    print(f"\nTotal results: {len(results_df)} ({existing_count} existing + {n_new} new)")
    print(f"Successful runs: {len(successful)} / {len(results_df)}")

    if len(successful) > 0:
        print(f"\nModel Performance Summary:")
        perf_summary = successful.groupby(['entropy_type', 'city']).agg({
            'r2': 'mean',
            'adj_r2': 'mean',
            'bandwidth': 'mean',
            'alpha_corrected': 'mean'
        }).round(6).reset_index()
        print(perf_summary.to_string(index=False))

        print(f"\nHotspot Summary (da Silva-Fotheringham corrected significance):")
        hotspot_summary = successful.groupby(['entropy_type', 'city']).agg({
            'n_hotspots': 'sum',
            'n_coldspots': 'sum',
            'n': 'sum'
        }).reset_index()
        hotspot_summary['hotspot_pct'] = (100 * hotspot_summary['n_hotspots'] / hotspot_summary['n']).round(1)
        hotspot_summary['coldspot_pct'] = (100 * hotspot_summary['n_coldspots'] / hotspot_summary['n']).round(1)
        print(hotspot_summary.to_string(index=False))

    print(f"\nResults saved to: {output_dir}")
    print(f"  - gwr_summary_results.csv")
    print(f"  - gwr_local_coefficients/*.parquet")
    print(f"\nCompleted at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
