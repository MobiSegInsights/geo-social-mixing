#!/usr/bin/env python
"""
Compute pairwise correlations between diversity measures.

Addresses R2.13: Report correlations instead of Wilcoxon p-values.

Output: outputs/tables/diversity_correlations.csv

Usage:
    python -m src.analysis.compute_diversity_correlations
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

def get_project_root():
    """Get project root directory."""
    script_dir = Path(__file__).resolve().parent
    if script_dir.name == 'analysis' and script_dir.parent.name == 'src':
        return script_dir.parent.parent
    return Path.cwd()

PROJECT_ROOT = get_project_root()

ROUTING_DIR = PROJECT_ROOT / 'dbs/routing'
OUTPUT_DIR = PROJECT_ROOT / 'outputs/tables'

# Diversity columns to correlate
DIVERSITY_COLS = {
    'birth': {
        'residential': 'residential_entropy_birth_norm',
        'visitor': 'visitor_entropy_birth_norm',
        'catchment': 'catchment_entropy_birth_norm',
    },
    'income': {
        'residential': 'residential_entropy_income_norm',
        'visitor': 'visitor_entropy_income_norm',
        'catchment': 'catchment_entropy_income_norm',
    }
}


def compute_correlations(df, cols, method='pearson'):
    """
    Compute pairwise correlations between columns.

    Returns dict with correlation values and p-values.
    """
    results = {}
    col_names = list(cols.keys())

    for i, name1 in enumerate(col_names):
        for name2 in col_names[i+1:]:
            col1 = cols[name1]
            col2 = cols[name2]

            # Filter valid pairs
            valid_mask = df[col1].notna() & df[col2].notna()
            x = df.loc[valid_mask, col1].values
            y = df.loc[valid_mask, col2].values

            if len(x) < 10:
                continue

            if method == 'pearson':
                r, p = stats.pearsonr(x, y)
            else:
                r, p = stats.spearmanr(x, y)

            pair_key = f"{name1}_vs_{name2}"
            results[pair_key] = {
                'r': r,
                'p': p,
                'n': len(x)
            }

    return results


def main():
    print("=" * 70)
    print("DIVERSITY MEASURE CORRELATIONS")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    us_df = pd.read_parquet(ROUTING_DIR / 'us_poi_diversity_metrics.parquet')
    se_df = pd.read_parquet(ROUTING_DIR / 'sweden_poi_diversity_metrics.parquet')

    # Add country
    us_df['country'] = 'US'
    se_df['country'] = 'Sweden'

    print(f"  US POIs: {len(us_df):,}")
    print(f"  Sweden POIs: {len(se_df):,}")

    # Compute correlations by city and entropy type
    all_results = []

    for df, country in [(us_df, 'US'), (se_df, 'Sweden')]:
        cities = df['city'].unique()

        for city in cities:
            city_df = df[df['city'] == city]

            for entropy_type, cols in DIVERSITY_COLS.items():
                # Check if columns exist
                if not all(c in city_df.columns for c in cols.values()):
                    continue

                # Compute correlations
                corrs = compute_correlations(city_df, cols, method='pearson')

                for pair, vals in corrs.items():
                    all_results.append({
                        'country': country,
                        'city': city,
                        'entropy_type': entropy_type,
                        'pair': pair,
                        'pearson_r': vals['r'],
                        'p_value': vals['p'],
                        'n': vals['n']
                    })

    # Create results DataFrame
    results_df = pd.DataFrame(all_results)

    # Also compute overall correlations by country
    print("\n" + "=" * 70)
    print("CORRELATIONS BY COUNTRY")
    print("=" * 70)

    for df, country in [(us_df, 'US'), (se_df, 'Sweden')]:
        print(f"\n{country}:")
        for entropy_type, cols in DIVERSITY_COLS.items():
            if not all(c in df.columns for c in cols.values()):
                continue

            print(f"\n  {entropy_type.upper()} entropy:")
            corrs = compute_correlations(df, cols, method='pearson')

            for pair, vals in corrs.items():
                print(f"    {pair}: r = {vals['r']:.3f} (n = {vals['n']:,})")

    # Save results
    output_file = OUTPUT_DIR / 'diversity_correlations.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")

    # Create summary table (mean correlation by country)
    print("\n" + "=" * 70)
    print("SUMMARY: MEAN CORRELATIONS BY COUNTRY")
    print("=" * 70)

    summary = results_df.groupby(['country', 'entropy_type', 'pair']).agg({
        'pearson_r': 'mean',
        'n': 'sum'
    }).round(3).reset_index()

    print(summary.to_string(index=False))

    # Save summary
    summary_file = OUTPUT_DIR / 'diversity_correlations_summary.csv'
    summary.to_csv(summary_file, index=False)
    print(f"\nSaved: {summary_file}")

    # Effect sizes (mean differences)
    print("\n" + "=" * 70)
    print("EFFECT SIZES: MEAN DIFFERENCES BETWEEN DIVERSITY MEASURES")
    print("=" * 70)

    effect_results = []
    for df, country in [(us_df, 'US'), (se_df, 'Sweden')]:
        for entropy_type, cols in DIVERSITY_COLS.items():
            if not all(c in df.columns for c in cols.values()):
                continue

            # Complete cases only
            valid_mask = df[cols['residential']].notna() & \
                         df[cols['visitor']].notna() & \
                         df[cols['catchment']].notna()
            valid_df = df[valid_mask]

            res = valid_df[cols['residential']].values
            vis = valid_df[cols['visitor']].values
            cat = valid_df[cols['catchment']].values

            effect_results.append({
                'country': country,
                'entropy_type': entropy_type,
                'n': len(valid_df),
                'mean_residential': np.mean(res),
                'mean_visitor': np.mean(vis),
                'mean_catchment': np.mean(cat),
                'diff_visitor_minus_residential': np.mean(vis - res),
                'diff_catchment_minus_residential': np.mean(cat - res),
                'diff_catchment_minus_visitor': np.mean(cat - vis),
            })

    effect_df = pd.DataFrame(effect_results).round(4)
    print(effect_df.to_string(index=False))

    effect_file = OUTPUT_DIR / 'diversity_effect_sizes.csv'
    effect_df.to_csv(effect_file, index=False)
    print(f"\nSaved: {effect_file}")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
