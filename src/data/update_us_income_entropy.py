#!/usr/bin/env python
"""
Update US POI Diversity Metrics with Real Income Entropy

Uses the tract-level income distributions from ACS B19001 to compute
proper residential income entropy (instead of the previous 0 values).

Usage:
    python -m src.data.update_us_income_entropy
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ROUTING_DIR = PROJECT_ROOT / 'dbs/routing'
CENSUS_DIR = PROJECT_ROOT / 'dbs/us_census'

INPUT_FILE = ROUTING_DIR / 'us_poi_diversity_metrics.parquet'
INCOME_DIST_FILE = CENSUS_DIR / 'acs2024_tracts_income_distribution.csv'
OUTPUT_FILE = ROUTING_DIR / 'us_poi_diversity_metrics.parquet'
BACKUP_FILE = ROUTING_DIR / 'us_poi_diversity_metrics_backup.parquet'


def compute_entropy(q1, q2, q3, q4, normalize=True) -> float:
    """Compute Shannon entropy from quartile percentages."""
    if pd.isna(q1) or pd.isna(q2) or pd.isna(q3) or pd.isna(q4):
        return np.nan

    props = np.array([q1, q2, q3, q4]) / 100  # Convert to proportions
    props = props[props > 0]  # Filter zeros

    if len(props) == 0:
        return np.nan

    entropy = -np.sum(props * np.log(props))

    if normalize:
        max_entropy = np.log(4)  # Max entropy for 4 categories
        entropy = entropy / max_entropy

    return entropy


def main():
    print("=" * 70)
    print("UPDATING US POI DIVERSITY METRICS WITH REAL INCOME ENTROPY")
    print("=" * 70)

    # Load POI diversity data
    print("\nLoading POI diversity metrics...")
    poi_df = pd.read_parquet(INPUT_FILE)
    print(f"  Total POIs: {len(poi_df):,}")

    # Backup original
    print(f"\nBacking up to: {BACKUP_FILE}")
    poi_df.to_parquet(BACKUP_FILE, index=False)

    # Load income distribution data
    print("\nLoading tract income distributions...")
    income_df = pd.read_csv(INCOME_DIST_FILE)
    print(f"  Total tracts with income data: {len(income_df):,}")

    # Create GEOID lookup
    income_df['GEOID'] = income_df['GEOID'].astype(str).str.zfill(11)
    income_lookup = income_df.set_index('GEOID')[['q1_pct', 'q2_pct', 'q3_pct', 'q4_pct', 'income_entropy_norm']].to_dict('index')

    # Check current residential_tract format
    print("\nSample residential_tract values:")
    print(poi_df['residential_tract'].dropna().head())

    # Standardize tract IDs
    poi_df['residential_tract_str'] = poi_df['residential_tract'].astype(str).str.zfill(11)

    # Count matches before update
    matched_before = poi_df['residential_tract_str'].isin(income_lookup.keys()).sum()
    print(f"\nTracts matched to income data: {matched_before:,} / {len(poi_df):,} ({100*matched_before/len(poi_df):.1f}%)")

    # Update residential income entropy
    print("\nUpdating residential_entropy_income_norm...")

    def get_income_entropy(tract_id):
        if pd.isna(tract_id):
            return np.nan
        tract_str = str(tract_id).zfill(11)
        if tract_str in income_lookup:
            return income_lookup[tract_str]['income_entropy_norm']
        return np.nan

    def get_income_entropy_raw(tract_id):
        if pd.isna(tract_id):
            return np.nan
        tract_str = str(tract_id).zfill(11)
        if tract_str in income_lookup:
            data = income_lookup[tract_str]
            return compute_entropy(data['q1_pct'], data['q2_pct'], data['q3_pct'], data['q4_pct'], normalize=False)
        return np.nan

    # Update the columns
    poi_df['residential_entropy_income_norm'] = poi_df['residential_tract'].apply(get_income_entropy)
    poi_df['residential_entropy_income'] = poi_df['residential_tract'].apply(get_income_entropy_raw)

    # Also update ICE income (Q4 - Q1) / (Q4 + Q1)
    def get_ice_income(tract_id):
        if pd.isna(tract_id):
            return np.nan
        tract_str = str(tract_id).zfill(11)
        if tract_str in income_lookup:
            data = income_lookup[tract_str]
            q1 = data['q1_pct']
            q4 = data['q4_pct']
            if pd.isna(q1) or pd.isna(q4) or (q1 + q4) == 0:
                return np.nan
            return (q4 - q1) / (q4 + q1)
        return np.nan

    poi_df['residential_ice_income'] = poi_df['residential_tract'].apply(get_ice_income)

    # Drop temporary column
    poi_df = poi_df.drop(columns=['residential_tract_str'])

    # Verify update
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    for city in poi_df['city'].unique():
        city_df = poi_df[poi_df['city'] == city]
        res_income = city_df['residential_entropy_income_norm']

        print(f"\n{city.upper()}")
        print(f"  N POIs: {len(city_df):,}")
        print(f"  residential_entropy_income_norm:")
        print(f"    non-null: {res_income.notna().sum():,}")
        print(f"    unique values: {res_income.nunique():,}")
        print(f"    min: {res_income.min():.4f}")
        print(f"    max: {res_income.max():.4f}")
        print(f"    mean: {res_income.mean():.4f}")
        print(f"    std: {res_income.std():.4f}")

    # Save updated data
    print(f"\nSaving updated data to: {OUTPUT_FILE}")
    poi_df.to_parquet(OUTPUT_FILE, index=False)

    print("\n" + "=" * 70)
    print("UPDATE COMPLETE")
    print("=" * 70)
    print(f"Backup saved to: {BACKUP_FILE}")
    print(f"Updated file: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
