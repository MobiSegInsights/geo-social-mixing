"""
Step 1.1.1d: Harmonize DeSO Zone Data for 2024

Combines geometry, income, birth background, and car ownership data
into a single analysis-ready GeoDataFrame.

Usage:
    python harmonize_deso.py

Output:
    dbs/deso/deso_harmonized_2024.gpkg
    dbs/deso/deso_harmonized_2024.parquet
"""

from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np

# Paths
ROOT_DIR = Path(__file__).parent.parent.parent
DESO_DIR = ROOT_DIR / "dbs" / "deso"

print("="*60)
print("DeSO Data Harmonization - 2024")
print("="*60)

# ============================================================================
# 1. Load Geometry
# ============================================================================

print("\n1. Loading DeSO geometry...")
geometry_file = DESO_DIR / "DeSO_2025.gpkg"
gdf = gpd.read_file(geometry_file)

print(f"   Loaded {len(gdf):,} DeSO zones")
print(f"   CRS: {gdf.crs}")
print(f"   Columns: {gdf.columns.tolist()}")

# Standardize deso_code column name
# The geometry file uses 'desokod' as the DeSO code column
if 'desokod' in gdf.columns:
    gdf = gdf.rename(columns={'desokod': 'deso_code'})
elif 'deso' in gdf.columns:
    gdf = gdf.rename(columns={'deso': 'deso_code'})
elif 'DeSO' in gdf.columns:
    gdf = gdf.rename(columns={'DeSO': 'deso_code'})

# Keep only deso_code and geometry
gdf = gdf[['deso_code', 'geometry']]

# ============================================================================
# 2. Load Birth Background Data (use "totalt" gender rows)
# ============================================================================

print("\n2. Processing birth background data...")
birth_file = DESO_DIR / "deso_birth_background_gender.csv"
df_birth = pd.read_csv(birth_file, skiprows=1, encoding='latin1')

print(f"   Raw shape: {df_birth.shape}")
print(f"   Columns: {df_birth.columns.tolist()}")

# Rename columns
df_birth.columns = ['region', 'birth_region', 'gender', 'count']

# Filter to "totalt" gender rows only
df_birth = df_birth[df_birth['gender'] == 'totalt'].copy()

# Filter out "totalt" birth region (keep Sverige, Europa, Övriga separate)
df_birth = df_birth[df_birth['birth_region'] != 'totalt'].copy()

print(f"   After filtering to totalt gender: {len(df_birth)} rows")
print(f"   Unique birth regions: {df_birth['birth_region'].unique()}")

# Pivot to wide format
birth_wide = df_birth.pivot(index='region', columns='birth_region', values='count').reset_index()
birth_wide.columns.name = None

# Rename columns to English
birth_rename = {}
for col in birth_wide.columns:
    if col == 'region':
        continue
    elif col == 'Sverige':
        birth_rename[col] = 'birth_sweden'
    elif 'Europa' in col:
        birth_rename[col] = 'birth_europe'
    elif 'vriga' in col or 'vriga' in str(col):  # Övriga världen
        birth_rename[col] = 'birth_other'
    else:
        # Clean other column names
        clean_name = col.lower().replace(' ', '_').replace('å', 'a').replace('ä', 'a').replace('ö', 'o')
        birth_rename[col] = f'birth_{clean_name}'

birth_wide = birth_wide.rename(columns=birth_rename)
birth_wide = birth_wide.rename(columns={'region': 'deso_code'})

# Calculate total population and foreign-born percentage
birth_cols = [c for c in birth_wide.columns if c.startswith('birth_')]
birth_wide['pop_total'] = birth_wide[birth_cols].sum(axis=1)

# Foreign born = europe + other
foreign_cols = [c for c in birth_cols if c not in ['birth_sweden']]
if foreign_cols:
    birth_wide['pop_foreign_born'] = birth_wide[foreign_cols].sum(axis=1)
    birth_wide['pct_foreign_born'] = (birth_wide['pop_foreign_born'] / birth_wide['pop_total'] * 100).round(1)
else:
    birth_wide['pop_foreign_born'] = 0
    birth_wide['pct_foreign_born'] = 0.0

print(f"   Birth columns created: {birth_cols}")
print(f"   Total DeSO zones: {len(birth_wide):,}")

# ============================================================================
# 3. Load Income Data (use "totalt" gender rows)
# ============================================================================

print("\n3. Processing income data...")
income_file = DESO_DIR / "deso_income_gender.csv"
df_income = pd.read_csv(income_file, skiprows=1, encoding='latin1')

print(f"   Raw shape: {df_income.shape}")
print(f"   Columns: {df_income.columns.tolist()}")

# Standardize column names
df_income.columns = ['region', 'income_type', 'gender', 'q1_pct', 'q2_pct',
                     'q3_pct', 'q4_pct', 'median_tkr', 'mean_tkr', 'n_persons']

# Filter to "sammanräknad förvärvsinkomst" and "totalt" gender
df_income = df_income[
    (df_income['income_type'].str.contains('sammanr', na=False)) &
    (df_income['gender'] == 'totalt')
].copy()

print(f"   After filtering: {len(df_income)} rows")

# Select and rename relevant columns
df_income = df_income[[
    'region', 'q1_pct', 'q2_pct', 'q3_pct', 'q4_pct',
    'median_tkr', 'mean_tkr', 'n_persons'
]].rename(columns={
    'region': 'deso_code',
    'q1_pct': 'income_q1_pct',
    'q2_pct': 'income_q2_pct',
    'q3_pct': 'income_q3_pct',
    'q4_pct': 'income_q4_pct',
    'median_tkr': 'income_median_tkr',
    'mean_tkr': 'income_mean_tkr',
    'n_persons': 'income_n_persons'
})

print(f"   Total DeSO zones: {len(df_income):,}")

# ============================================================================
# 4. Load Car Ownership Data
# ============================================================================

print("\n4. Processing car ownership data...")
cars_file = DESO_DIR / "deso_passenger_cars_in_traffic.csv"
df_cars = pd.read_csv(cars_file, skiprows=1, encoding='latin1')

print(f"   Raw shape: {df_cars.shape}")

# Rename columns
df_cars.columns = ['region', 'status', 'count']

# Filter to 'i trafik' (in traffic)
df_cars = df_cars[df_cars['status'].str.contains('trafik', na=False)].copy()

# Select and rename
df_cars = df_cars[['region', 'count']].rename(columns={
    'region': 'deso_code',
    'count': 'cars_total'
})

print(f"   Total DeSO zones with car data: {len(df_cars):,}")

# ============================================================================
# 5. Merge All Data
# ============================================================================

print("\n5. Merging all datasets...")

# Start with geometry
result = gdf.copy()
print(f"   Starting with {len(result):,} DeSO zones (geometry)")

# Merge birth background
result = result.merge(birth_wide, on='deso_code', how='left')
print(f"   After birth merge: {len(result):,} zones, {result['pop_total'].notna().sum():,} with data")

# Merge income
result = result.merge(df_income, on='deso_code', how='left')
print(f"   After income merge: {len(result):,} zones, {result['income_median_tkr'].notna().sum():,} with data")

# Merge cars
result = result.merge(df_cars, on='deso_code', how='left')
print(f"   After cars merge: {len(result):,} zones, {result['cars_total'].notna().sum():,} with data")

# Calculate derived metrics
result['cars_per_capita'] = (result['cars_total'] / result['pop_total']).round(3)
result['cars_per_1000'] = (result['cars_total'] / result['pop_total'] * 1000).round(1)

# ============================================================================
# 6. Reorder and Clean Columns
# ============================================================================

print("\n6. Finalizing dataset...")

# Define column order
essential_cols = [
    'deso_code',
    # Population
    'pop_total', 'pop_foreign_born', 'pct_foreign_born',
    # Birth regions
    'birth_sweden', 'birth_europe', 'birth_other',
    # Income
    'income_median_tkr', 'income_mean_tkr',
    'income_q1_pct', 'income_q2_pct', 'income_q3_pct', 'income_q4_pct',
    'income_n_persons',
    # Cars
    'cars_total', 'cars_per_capita', 'cars_per_1000',
    # Geometry
    'geometry'
]

# Filter to existing columns
final_cols = [c for c in essential_cols if c in result.columns]
result = result[final_cols]

# Fill missing numeric values with 0
numeric_cols = result.select_dtypes(include=[np.number]).columns
result[numeric_cols] = result[numeric_cols].fillna(0)

# ============================================================================
# 7. Save Outputs
# ============================================================================

print("\n7. Saving outputs...")

# Save as GeoPackage (with geometry)
output_gpkg = DESO_DIR / "deso_harmonized_2024.gpkg"
result.to_file(output_gpkg, driver='GPKG')
print(f"   ✓ Saved GeoPackage: {output_gpkg}")

# Save as Parquet (without geometry, for faster loading)
output_parquet = DESO_DIR / "deso_harmonized_2024.parquet"
result_df = pd.DataFrame(result.drop(columns='geometry'))
result_df.to_parquet(output_parquet, index=False)
print(f"   ✓ Saved Parquet: {output_parquet}")

# ============================================================================
# 8. Summary Statistics
# ============================================================================

print("\n" + "="*60)
print("HARMONIZATION SUMMARY")
print("="*60)

print(f"\nDataset Overview:")
print(f"  Total DeSO zones: {len(result):,}")
print(f"  CRS: {result.crs}")
print(f"  Columns: {len(result.columns)}")

print(f"\nData Completeness:")
for col in result.columns:
    if col == 'geometry':
        continue
    n_missing = result[col].isna().sum()
    n_zero = (result[col] == 0).sum()
    pct_valid = ((len(result) - n_missing - n_zero) / len(result) * 100)
    print(f"  {col:25s}: {pct_valid:5.1f}% valid ({n_missing:,} missing, {n_zero:,} zero)")

print(f"\nPopulation Statistics:")
print(f"  Total population: {result['pop_total'].sum():,.0f}")
print(f"  Mean per DeSO: {result['pop_total'].mean():.0f}")
print(f"  Median per DeSO: {result['pop_total'].median():.0f}")
print(f"  Foreign-born: {result['pop_foreign_born'].sum():,.0f} ({result['pct_foreign_born'].mean():.1f}% avg)")

print(f"\nIncome Statistics:")
print(f"  Median income (median): {result['income_median_tkr'].median():.1f} tkr")
print(f"  Mean income (mean): {result['income_mean_tkr'].mean():.1f} tkr")

print(f"\nCar Ownership:")
print(f"  Total cars: {result['cars_total'].sum():,.0f}")
print(f"  Cars per capita (mean): {result['cars_per_capita'].mean():.3f}")
print(f"  Cars per 1000 people (mean): {result['cars_per_1000'].mean():.1f}")

print(f"\nSample Data:")
print(result.head(3).drop(columns='geometry'))

print("\n" + "="*60)
print("✅ Harmonization complete!")
print("="*60)
