"""
Step 1.1.1e: Assign DeSO Zones to Home Locations with IPW

Spatial join between device home locations and DeSO zones,
with inverse probability weighting to correct sampling bias.

IMPORTANT: Only processes devices with confirmed residential building matches
(has_home=True AND has_building=True) to ensure location quality.

Usage:
    python assign_home_deso_ipw.py

Input:
    - dbs/home_work/home_work_buildings.parquet
    - dbs/deso/deso_harmonized_2024.gpkg

Output:
    - dbs/home_work/device_homes_deso_ipw.parquet
      Adds: home_deso_code, DeSO characteristics, ipw_weight (trimmed)
      Note: Only devices with building matches get DeSO assignment and IPW weights

IPW Methodology:
    1. Count devices per DeSO zone (only those with building matches)
    2. Calculate raw weight: w_raw = population / n_devices
    3. Van de Kerckhove trimming: w0 = sqrt(CV² + 1) × 3.5 × median(w_raw)
    4. Trim: w = min(w_raw, w0)

Reference:
    - Liao, Yuan, et al. "The effect of limited mobility on the experienced segregation of foreign-born minorities." 
    npj Sustainable Mobility and Transport 2.1 (2025): 29.
"""

from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point

# Paths
ROOT_DIR = Path(__file__).parent.parent.parent
HW_DIR = ROOT_DIR / "dbs" / "home_work"
DESO_DIR = ROOT_DIR / "dbs" / "deso"

print("="*60)
print("Assign DeSO Zones with Inverse Probability Weighting")
print("="*60)

# ============================================================================
# 1. Load Home/Work with Building Linkage
# ============================================================================

print("\n1. Loading home/work with building linkage...")
hw_file = HW_DIR / "home_work_buildings.parquet"
df_hw = pd.read_parquet(hw_file)

print(f"   Loaded {len(df_hw):,} devices")

# Drop any existing DeSO/IPW columns from previous runs
cols_to_drop = [c for c in df_hw.columns if c.startswith('home_deso_') or c == 'ipw_weight']
if cols_to_drop:
    print(f"   Dropping existing columns from previous run: {cols_to_drop}")
    df_hw = df_hw.drop(columns=cols_to_drop)

print(f"   Columns: {df_hw.columns.tolist()}")
print(f"   Devices with home: {df_hw['has_home'].sum():,} ({df_hw['has_home'].mean()*100:.1f}%)")
print(f"   Devices with building match: {df_hw['has_building'].sum():,} ({df_hw['has_building'].mean()*100:.1f}%)")

# ============================================================================
# 2. Load DeSO Zones
# ============================================================================

print("\n2. Loading DeSO zones...")
deso_file = DESO_DIR / "deso_harmonized_2024.gpkg"
gdf_deso = gpd.read_file(deso_file)

print(f"   Loaded {len(gdf_deso):,} DeSO zones")
print(f"   CRS: {gdf_deso.crs}")
print(f"   Total population: {gdf_deso['pop_total'].sum():,.0f}")

# ============================================================================
# 3. Create GeoDataFrame from Home Locations
# ============================================================================

print("\n3. Converting home locations to GeoDataFrame...")

# Filter to devices with home AND building match
df_homes = df_hw[df_hw['has_home'] & df_hw['has_building']].copy()
print(f"   Devices with home: {df_hw['has_home'].sum():,}")
print(f"   Devices with building: {df_hw['has_building'].sum():,}")
print(f"   Devices with home AND building: {len(df_homes):,}")

# Create Point geometries from home lat/lon
geometry = [Point(lon, lat) for lon, lat in zip(df_homes['home_lon'], df_homes['home_lat'])]
gdf_homes = gpd.GeoDataFrame(df_homes, geometry=geometry, crs='EPSG:4326')

# Transform to match DeSO CRS (EPSG:3006 - SWEREF99 TM)
if gdf_homes.crs != gdf_deso.crs:
    print(f"   Transforming from {gdf_homes.crs} to {gdf_deso.crs}...")
    gdf_homes = gdf_homes.to_crs(gdf_deso.crs)

# ============================================================================
# 4. Spatial Join
# ============================================================================

print("\n4. Performing spatial join (home → DeSO)...")
print("   This may take a few minutes...")

# Spatial join: assign DeSO zone to each home location
gdf_joined = gpd.sjoin(
    gdf_homes,
    gdf_deso[['deso_code', 'geometry', 'pop_total', 'income_median_tkr',
              'income_mean_tkr', 'pct_foreign_born', 'cars_per_capita']],
    how='left',
    predicate='within'
)

# Drop duplicate index column from join
if 'index_right' in gdf_joined.columns:
    gdf_joined = gdf_joined.drop(columns='index_right')

# Rename deso_code to home_deso_code
gdf_joined = gdf_joined.rename(columns={'deso_code': 'home_deso_code'})

# Check join success rate
n_matched = gdf_joined['home_deso_code'].notna().sum()
n_total = len(gdf_joined)
match_rate = n_matched / n_total * 100

print(f"\n   Join results:")
print(f"     Total homes: {n_total:,}")
print(f"     Matched to DeSO: {n_matched:,} ({match_rate:.1f}%)")
print(f"     Unmatched: {n_total - n_matched:,} ({100-match_rate:.1f}%)")

# ============================================================================
# 5. Calculate Inverse Probability Weights (IPW)
# ============================================================================

print("\n5. Calculating inverse probability weights (IPW)...")

# Count devices per DeSO zone
deso_device_counts = gdf_joined[gdf_joined['home_deso_code'].notna()].groupby('home_deso_code').size().reset_index(name='n_devices')

# Merge with DeSO population
deso_weights = pd.merge(
    gdf_deso[['deso_code', 'pop_total']],
    deso_device_counts,
    left_on='deso_code',
    right_on='home_deso_code',
    how='inner'
)

# Calculate initial weight: population / device_count
deso_weights['ipw_weight_raw'] = deso_weights['pop_total'] / deso_weights['n_devices']

print(f"\n   Weight statistics (before trimming):")
print(f"     Mean: {deso_weights['ipw_weight_raw'].mean():.2f}")
print(f"     Median: {deso_weights['ipw_weight_raw'].median():.2f}")
print(f"     Std: {deso_weights['ipw_weight_raw'].std():.2f}")
print(f"     Min: {deso_weights['ipw_weight_raw'].min():.2f}")
print(f"     Max: {deso_weights['ipw_weight_raw'].max():.2f}")
print(f"     CV: {deso_weights['ipw_weight_raw'].std() / deso_weights['ipw_weight_raw'].mean():.3f}")

# ============================================================================
# 6. Weight Trimming (Van de Kerckhove et al. 2014)
# ============================================================================

print("\n6. Applying weight trimming...")

# Calculate trimming threshold using Van de Kerckhove et al. method:
# w0 = sqrt(CV^2 + 1) * 3.5 * median(weight)
cv = deso_weights['ipw_weight_raw'].std() / deso_weights['ipw_weight_raw'].mean()
median_weight = deso_weights['ipw_weight_raw'].median()
w0 = np.sqrt(cv**2 + 1) * 3.5 * median_weight

print(f"\n   Trimming calculation:")
print(f"     Coefficient of variation (CV): {cv:.3f}")
print(f"     Median weight: {median_weight:.2f}")
print(f"     Trimming threshold (w0): {w0:.2f}")

# Apply trimming
deso_weights['ipw_weight'] = deso_weights['ipw_weight_raw'].clip(upper=w0)

# Count trimmed weights
n_trimmed = (deso_weights['ipw_weight_raw'] > w0).sum()
print(f"     DeSO zones with trimmed weights: {n_trimmed} ({n_trimmed/len(deso_weights)*100:.1f}%)")

print(f"\n   Weight statistics (after trimming):")
print(f"     Mean: {deso_weights['ipw_weight'].mean():.2f}")
print(f"     Median: {deso_weights['ipw_weight'].median():.2f}")
print(f"     Std: {deso_weights['ipw_weight'].std():.2f}")
print(f"     Min: {deso_weights['ipw_weight'].min():.2f}")
print(f"     Max: {deso_weights['ipw_weight'].max():.2f}")
print(f"     CV: {deso_weights['ipw_weight'].std() / deso_weights['ipw_weight'].mean():.3f}")

# ============================================================================
# 7. Assign Weights to Devices
# ============================================================================

print("\n7. Assigning weights to devices...")

# Merge weights back to joined data
weight_map = deso_weights.set_index('home_deso_code')['ipw_weight'].to_dict()
gdf_joined['ipw_weight'] = gdf_joined['home_deso_code'].map(weight_map)

# Devices without DeSO match get weight = NaN (will be excluded from weighted analyses)
print(f"   Devices with IPW weight: {gdf_joined['ipw_weight'].notna().sum():,}")

# ============================================================================
# 8. Add Back Devices Without Building Match
# ============================================================================

print("\n8. Adding back devices without building match...")

# Get devices without home OR without building
# These devices won't have DeSO assignment or IPW weights
df_excluded = df_hw[~(df_hw['has_home'] & df_hw['has_building'])].copy()

# Add missing columns with None/NaN
deso_cols = ['home_deso_code', 'home_deso_pop', 'home_deso_income_median',
             'home_deso_income_mean', 'home_deso_pct_foreign', 'home_deso_cars_per_capita']
for col in deso_cols:
    if col not in df_excluded.columns:
        df_excluded[col] = None

df_excluded['ipw_weight'] = None

print(f"   Devices without home: {(~df_hw['has_home']).sum():,}")
print(f"   Devices with home but no building: {(df_hw['has_home'] & ~df_hw['has_building']).sum():,}")
print(f"   Total excluded from IPW: {len(df_excluded):,}")

# Convert joined GeoDataFrame back to regular DataFrame and rename columns
df_joined = pd.DataFrame(gdf_joined.drop(columns='geometry'))

# Rename DeSO characteristics in df_joined to match df_excluded
rename_map = {
    'pop_total': 'home_deso_pop',
    'income_median_tkr': 'home_deso_income_median',
    'income_mean_tkr': 'home_deso_income_mean',
    'pct_foreign_born': 'home_deso_pct_foreign',
    'cars_per_capita': 'home_deso_cars_per_capita'
}
df_joined = df_joined.rename(columns=rename_map)

# Combine (both dataframes now have consistent column names)
result = pd.concat([df_joined, df_excluded], ignore_index=True)

print(f"   Total devices in output: {len(result):,}")

# ============================================================================
# 9. Finalize Column Order
# ============================================================================

print("\n9. Finalizing dataset...")

# Reorder columns
cols_order = [
    'device_aid',
    # Home location
    'home_loc', 'home_lat', 'home_lon',
    # Building linkage
    'building_id', 'building_class', 'building_source', 'has_building',
    # DeSO assignment
    'home_deso_code', 'home_deso_pop', 'home_deso_income_median', 'home_deso_income_mean',
    'home_deso_pct_foreign', 'home_deso_cars_per_capita',
    # IPW
    'ipw_weight',
    # Flags
    'has_home',
    # Work
    'work_loc', 'work_lat', 'work_lon', 'has_work'
]

# Filter to existing columns
cols_order = [c for c in cols_order if c in result.columns]
result = result[cols_order]

# ============================================================================
# 10. Save Output
# ============================================================================

print("\n10. Saving output...")

output_file = HW_DIR / "device_homes_deso_ipw.parquet"
result.to_parquet(output_file, index=False)

print(f"   ✓ Saved: {output_file}")

# ============================================================================
# 11. Summary Statistics
# ============================================================================

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

print(f"\nDataset Overview:")
print(f"  Total devices: {len(result):,}")
print(f"  Devices with home: {result['has_home'].sum():,} ({result['has_home'].mean()*100:.1f}%)")
print(f"  Devices with building match: {result['has_building'].sum():,} ({result['has_building'].mean()*100:.1f}%)")
print(f"  Devices with home AND building: {(result['has_home'] & result['has_building']).sum():,} ({(result['has_home'] & result['has_building']).mean()*100:.1f}%)")
print(f"  Devices with DeSO assigned: {result['home_deso_code'].notna().sum():,} ({result['home_deso_code'].notna().mean()*100:.1f}%)")
print(f"  Devices with IPW weight: {result['ipw_weight'].notna().sum():,} ({result['ipw_weight'].notna().mean()*100:.1f}%)")
print(f"  Devices with work: {result['has_work'].sum():,} ({result['has_work'].mean()*100:.1f}%)")

if result['home_deso_code'].notna().any():
    print(f"\n  Unique DeSO zones (homes): {result['home_deso_code'].nunique():,}")

weighted_devices = result[result['ipw_weight'].notna()]
if len(weighted_devices) > 0:
    print(f"\nIPW Weight Statistics (for devices with weights):")
    print(f"  Mean: {weighted_devices['ipw_weight'].mean():.2f}")
    print(f"  Median: {weighted_devices['ipw_weight'].median():.2f}")
    print(f"  Std: {weighted_devices['ipw_weight'].std():.2f}")
    print(f"  Min: {weighted_devices['ipw_weight'].min():.2f}")
    print(f"  Max: {weighted_devices['ipw_weight'].max():.2f}")

    print(f"\nWeighted Population Representation:")
    print(f"  Raw device count: {len(weighted_devices):,}")
    print(f"  Weighted population: {weighted_devices['ipw_weight'].sum():,.0f}")
    print(f"  Actual DeSO population (matched zones): {weighted_devices['home_deso_pop'].sum():,.0f}")

matched = result[result['home_deso_code'].notna()]
if len(matched) > 0:
    print(f"\nHome DeSO Characteristics (for matched devices):")
    if 'home_deso_pop' in matched.columns:
        print(f"  Population (mean): {matched['home_deso_pop'].mean():.0f}")
    if 'home_deso_income_median' in matched.columns:
        print(f"  Income median (mean): {matched['home_deso_income_median'].mean():.1f} tkr")
    if 'home_deso_pct_foreign' in matched.columns:
        print(f"  Foreign-born % (mean): {matched['home_deso_pct_foreign'].mean():.1f}%")

print(f"\nSample Data (devices with weights):")
print(result[result['ipw_weight'].notna()].head(3))

print("\n" + "="*60)
print("✅ DeSO assignment with IPW complete!")
print("="*60)
print("\nIPW Usage:")
print("  When aggregating device visits to tract-to-POI flows:")
print("    weighted_visits = sum(device_visits * ipw_weight)")
print("  This ensures results are representative of actual population")
print("\nNext steps:")
print("  1. POI matching (match stops to points of interest)")
print("  2. Aggregate to tract-to-POI flows with IPW weighting")
