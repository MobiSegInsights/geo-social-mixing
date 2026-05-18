#!/usr/bin/env python
"""
Stepwise OLS Robustness Checks (R2.5)

Tests stability of transit catchment coefficient when adding controls:
1. Base: catchment + residential diversity + category FE
2. + geographic catchment diversity (1.5km Euclidean buffer)
3. + distance to city center (centrality proxy)
4. + POI density within 500m (agglomeration proxy)

Output: outputs/robustness/stepwise_ols_results.csv

Usage:
    python -m src.models.run_stepwise_robustness
    python -m src.models.run_stepwise_robustness --city Stockholm
"""

import argparse
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from spreg import OLS

warnings.filterwarnings('ignore')


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

ROUTING_DIR = PROJECT_ROOT / 'dbs/routing'
US_TRAFFIC_DIR = PROJECT_ROOT / 'dbs/us_foot_traffic/cities'
SE_TRAFFIC_DIR = PROJECT_ROOT / 'dbs/sweden_weekly_patterns'
OUTPUT_DIR = PROJECT_ROOT / 'outputs/robustness'

# Cities to analyze
US_CITIES = ['new_york', 'washington_dc', 'atlanta']
SWEDEN_CITIES = [
    'Stockholm', 'Göteborg', 'Malmö', 'Uppsala', 'Västerås',
    'Örebro', 'Linköping', 'Helsingborg', 'Lund'
]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_categories():
    """Load venue categories for all POIs."""
    # US categories
    us_cats_list = []
    for city in US_CITIES:
        city_dir = US_TRAFFIC_DIR / city
        if city_dir.exists():
            for f in city_dir.glob('*.parquet'):
                df = pd.read_parquet(f, columns=['ID_STORE', 'unified_category'])
                us_cats_list.append(df)

    us_cats = pd.DataFrame()
    if us_cats_list:
        us_cats = pd.concat(us_cats_list, ignore_index=True)
        us_cats = us_cats.drop_duplicates(subset='ID_STORE')
        us_cats = us_cats.rename(columns={'ID_STORE': 'poi_id', 'unified_category': 'category'})
        us_cats['poi_id'] = us_cats['poi_id'].astype(str)

    # Sweden categories
    se_cats = pd.read_parquet(
        SE_TRAFFIC_DIR / 'sweden_weekly_patterns_2024.parquet',
        columns=['PLACEKEY', 'unified_category']
    )
    se_cats = se_cats.drop_duplicates(subset='PLACEKEY')
    se_cats = se_cats.rename(columns={'PLACEKEY': 'poi_id', 'unified_category': 'category'})

    return us_cats, se_cats


def prepare_data(df, city, entropy_type='birth'):
    """Prepare analysis data for a city."""
    y_col = f'visitor_entropy_{entropy_type}_norm'
    catch_col = f'catchment_entropy_{entropy_type}_norm'
    geo_col = f'geo_catchment_entropy_{entropy_type}_norm'
    res_col = f'residential_entropy_{entropy_type}_norm'

    # Filter by city
    city_mask = df['city'].str.contains(city[:4], case=False, na=False)
    city_df = df[city_mask].copy()

    # Require all variables
    required = [y_col, catch_col, geo_col, res_col, 'category',
                'dist_to_center_km', 'poi_density_500m']
    valid_mask = city_df[required].notna().all(axis=1)
    city_df = city_df[valid_mask].reset_index(drop=True)

    return city_df


# =============================================================================
# STEPWISE OLS
# =============================================================================

def run_stepwise_ols(df, entropy_type='birth'):
    """
    Run stepwise OLS adding controls sequentially.

    Models:
    1. Base: transit_catchment + residential + category FE
    2. + Geo: + geo_catchment (1.5km Euclidean buffer)
    3. + Centrality: + dist_to_center_km
    4. + Density: + poi_density_500m (full model)

    Returns dict with coefficients and model stats for each specification.
    """
    y_col = f'visitor_entropy_{entropy_type}_norm'
    catch_col = f'catchment_entropy_{entropy_type}_norm'
    geo_col = f'geo_catchment_entropy_{entropy_type}_norm'
    res_col = f'residential_entropy_{entropy_type}_norm'

    y = df[y_col].values.reshape(-1, 1)

    # Category dummies (top 5 + other)
    top_cats = df['category'].value_counts().head(5).index.tolist()
    df = df.copy()
    df['cat_group'] = df['category'].apply(lambda x: x if x in top_cats else 'other')
    cat_dummies = pd.get_dummies(df['cat_group'], prefix='cat', drop_first=True)

    results = {}

    # Model 1: Base (transit_catchment + residential + category FE)
    X1 = np.hstack([
        df[[catch_col, res_col]].values,
        cat_dummies.values
    ])
    X1_names = ['transit_catch', 'residential'] + cat_dummies.columns.tolist()

    try:
        ols1 = OLS(y, X1, name_y='visitor_div', name_x=X1_names)
        results['m1_base'] = {
            'spec': 'Base',
            'n': len(y),
            'r2': ols1.r2,
            'adj_r2': ols1.ar2,
            'aic': ols1.aic,
            'catchment_coef': ols1.betas[1][0],
            'catchment_se': ols1.std_err[1],
            'catchment_t': ols1.t_stat[1][0],
            'catchment_p': ols1.t_stat[1][1],
            'residential_coef': ols1.betas[2][0],
            'residential_se': ols1.std_err[2],
        }
    except Exception as e:
        results['m1_base'] = {'spec': 'Base', 'error': str(e)}

    # Model 2: + Geo catchment (1.5km Euclidean buffer diversity)
    X2 = np.hstack([
        df[[catch_col, res_col, geo_col]].values,
        cat_dummies.values
    ])
    X2_names = ['transit_catch', 'residential', 'geo_catch'] + cat_dummies.columns.tolist()

    try:
        ols2 = OLS(y, X2, name_y='visitor_div', name_x=X2_names)
        results['m2_geo'] = {
            'spec': '+ Geo Catchment',
            'n': len(y),
            'r2': ols2.r2,
            'adj_r2': ols2.ar2,
            'aic': ols2.aic,
            'catchment_coef': ols2.betas[1][0],
            'catchment_se': ols2.std_err[1],
            'catchment_t': ols2.t_stat[1][0],
            'catchment_p': ols2.t_stat[1][1],
            'residential_coef': ols2.betas[2][0],
            'residential_se': ols2.std_err[2],
            'geo_catch_coef': ols2.betas[3][0],
            'geo_catch_se': ols2.std_err[3],
        }
    except Exception as e:
        results['m2_geo'] = {'spec': '+ Geo Catchment', 'error': str(e)}

    # Model 3: + Centrality (dist_to_center_km)
    X3 = np.hstack([
        df[[catch_col, res_col, geo_col, 'dist_to_center_km']].values,
        cat_dummies.values
    ])
    X3_names = ['transit_catch', 'residential', 'geo_catch', 'dist_to_center'] + cat_dummies.columns.tolist()

    try:
        ols3 = OLS(y, X3, name_y='visitor_div', name_x=X3_names)
        results['m3_centrality'] = {
            'spec': '+ Geo + Centrality',
            'n': len(y),
            'r2': ols3.r2,
            'adj_r2': ols3.ar2,
            'aic': ols3.aic,
            'catchment_coef': ols3.betas[1][0],
            'catchment_se': ols3.std_err[1],
            'catchment_t': ols3.t_stat[1][0],
            'catchment_p': ols3.t_stat[1][1],
            'residential_coef': ols3.betas[2][0],
            'residential_se': ols3.std_err[2],
            'geo_catch_coef': ols3.betas[3][0],
            'geo_catch_se': ols3.std_err[3],
            'centrality_coef': ols3.betas[4][0],
            'centrality_se': ols3.std_err[4],
        }
    except Exception as e:
        results['m3_centrality'] = {'spec': '+ Geo + Centrality', 'error': str(e)}

    # Model 4: + Density (poi_density_500m) - Full model
    X4 = np.hstack([
        df[[catch_col, res_col, geo_col, 'dist_to_center_km', 'poi_density_500m']].values,
        cat_dummies.values
    ])
    X4_names = ['transit_catch', 'residential', 'geo_catch', 'dist_to_center', 'poi_density'] + cat_dummies.columns.tolist()

    try:
        ols4 = OLS(y, X4, name_y='visitor_div', name_x=X4_names)
        results['m4_full'] = {
            'spec': 'Full',
            'n': len(y),
            'r2': ols4.r2,
            'adj_r2': ols4.ar2,
            'aic': ols4.aic,
            'catchment_coef': ols4.betas[1][0],
            'catchment_se': ols4.std_err[1],
            'catchment_t': ols4.t_stat[1][0],
            'catchment_p': ols4.t_stat[1][1],
            'residential_coef': ols4.betas[2][0],
            'residential_se': ols4.std_err[2],
            'geo_catch_coef': ols4.betas[3][0],
            'geo_catch_se': ols4.std_err[3],
            'centrality_coef': ols4.betas[4][0],
            'centrality_se': ols4.std_err[4],
            'density_coef': ols4.betas[5][0],
            'density_se': ols4.std_err[5],
        }
    except Exception as e:
        results['m4_full'] = {'spec': 'Full', 'error': str(e)}

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Stepwise OLS Robustness Checks')
    parser.add_argument('--city', type=str, default=None,
                        help='Single city to analyze (default: all)')
    parser.add_argument('--entropy', type=str, choices=['birth', 'income', 'both'],
                        default='both', help='Entropy type')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STEPWISE OLS ROBUSTNESS CHECKS")
    print("=" * 70)
    print("Tests whether transit catchment coefficient is robust to controls:")
    print("  - Geographic catchment (1.5km Euclidean buffer diversity)")
    print("  - Centrality (distance to center)")
    print("  - Agglomeration (POI density)")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    us_df = pd.read_parquet(ROUTING_DIR / 'us_poi_diversity_metrics.parquet')
    se_df = pd.read_parquet(ROUTING_DIR / 'sweden_poi_diversity_metrics.parquet')
    us_df['poi_id'] = us_df['poi_id'].astype(str)
    se_df['poi_id'] = se_df['poi_id'].astype(str)

    # Check for required columns
    required_cols = ['dist_to_center_km', 'poi_density_500m', 'geo_catchment_entropy_birth_norm']
    for col in required_cols:
        if col not in us_df.columns or col not in se_df.columns:
            print(f"\nERROR: Column '{col}' not found.")
            print("Run: python -m src.features.compute_spatial_controls")
            print("     python -m src.features.compute_geographic_catchment")
            return

    print(f"  US POIs: {len(us_df):,}")
    print(f"  Sweden POIs: {len(se_df):,}")

    # Load categories
    print("Loading categories...")
    us_cats, se_cats = load_categories()
    us_df = us_df.merge(us_cats, on='poi_id', how='left')
    se_df = se_df.merge(se_cats, on='poi_id', how='left')

    # Determine cities
    if args.city:
        if args.city.lower() in [c.lower() for c in US_CITIES]:
            us_cities = [c for c in US_CITIES if c.lower() == args.city.lower()]
            se_cities = []
        else:
            us_cities = []
            se_cities = [c for c in SWEDEN_CITIES if c.lower().startswith(args.city.lower()[:4])]
    else:
        us_cities = US_CITIES
        se_cities = SWEDEN_CITIES

    # Determine entropy types
    entropy_types = ['birth', 'income'] if args.entropy == 'both' else [args.entropy]

    all_results = []

    for entropy_type in entropy_types:
        print(f"\n{'='*70}")
        print(f"{entropy_type.upper()} ENTROPY")
        print("=" * 70)

        # US cities
        for city in us_cities:
            print(f"\n  US - {city}")
            city_df = prepare_data(us_df, city, entropy_type)

            if len(city_df) < 100:
                print(f"    Skipped: only {len(city_df)} valid POIs")
                continue

            print(f"    N = {len(city_df):,}")

            step_results = run_stepwise_ols(city_df, entropy_type)

            for model_key, model_res in step_results.items():
                model_res['city'] = f"US - {city}"
                model_res['country'] = 'US'
                model_res['entropy_type'] = entropy_type
                all_results.append(model_res)

                if 'catchment_coef' in model_res:
                    sig = "***" if model_res.get('catchment_p', 1) < 0.001 else \
                          "**" if model_res.get('catchment_p', 1) < 0.01 else \
                          "*" if model_res.get('catchment_p', 1) < 0.05 else ""
                    print(f"    {model_res['spec']:<25}: β = {model_res['catchment_coef']:.4f} "
                          f"(SE = {model_res['catchment_se']:.4f}){sig}")

        # Sweden cities
        for city in se_cities:
            print(f"\n  Sweden - {city}")
            city_df = prepare_data(se_df, city, entropy_type)

            if len(city_df) < 100:
                print(f"    Skipped: only {len(city_df)} valid POIs")
                continue

            print(f"    N = {len(city_df):,}")

            step_results = run_stepwise_ols(city_df, entropy_type)

            for model_key, model_res in step_results.items():
                model_res['city'] = f"Sweden - {city}"
                model_res['country'] = 'Sweden'
                model_res['entropy_type'] = entropy_type
                all_results.append(model_res)

                if 'catchment_coef' in model_res:
                    sig = "***" if model_res.get('catchment_p', 1) < 0.001 else \
                          "**" if model_res.get('catchment_p', 1) < 0.01 else \
                          "*" if model_res.get('catchment_p', 1) < 0.05 else ""
                    print(f"    {model_res['spec']:<25}: β = {model_res['catchment_coef']:.4f} "
                          f"(SE = {model_res['catchment_se']:.4f}){sig}")

    # Save results
    results_df = pd.DataFrame(all_results)
    output_file = OUTPUT_DIR / 'stepwise_ols_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")

    # Summary: coefficient stability
    print("\n" + "=" * 70)
    print("COEFFICIENT STABILITY SUMMARY")
    print("=" * 70)

    for entropy_type in entropy_types:
        subset = results_df[results_df['entropy_type'] == entropy_type]
        if len(subset) == 0:
            continue

        print(f"\n{entropy_type.upper()} ENTROPY:")
        print("-" * 70)

        # Group by city, compare base vs full
        cities = subset['city'].unique()
        for city in cities:
            city_sub = subset[subset['city'] == city]
            base = city_sub[city_sub['spec'] == 'Base']
            full = city_sub[city_sub['spec'] == 'Full']

            if len(base) > 0 and len(full) > 0:
                base_coef = base.iloc[0].get('catchment_coef', np.nan)
                full_coef = full.iloc[0].get('catchment_coef', np.nan)

                if pd.notna(base_coef) and pd.notna(full_coef) and base_coef != 0:
                    change_pct = 100 * (full_coef - base_coef) / abs(base_coef)
                    print(f"  {city:<25}: {base_coef:.4f} → {full_coef:.4f} ({change_pct:+.1f}%)")

    print("\n" + "=" * 70)
    print(f"Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
