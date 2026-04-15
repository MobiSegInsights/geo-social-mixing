"""
Visitor Home CBGS Coverage Analysis

Analyzes the availability of visitor_home_cbgs data across POI categories
in the US weekly_patterns dataset.

Key questions:
1. What % of POIs have visitor_home_cbgs data by category?
2. Is missingness systematically biased by category?
3. Which categories should be retained for social mixing analysis?

Output:
- Category-level coverage statistics
- Recommendations for category filtering
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys


# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

US_WP_DIR = ROOT / "dbs/us_foot_traffic/weekly_patterns"
OUTPUT_DIR = ROOT / "dbs/us_foot_traffic"

print("=" * 70)
print("VISITOR_HOME_CBGS COVERAGE ANALYSIS")
print("=" * 70)

# =============================================================================
# 1. DISCOVER COLUMN NAMES
# =============================================================================

print("\n1. Discovering column names...")

sample_file = sorted(US_WP_DIR.glob("*.parquet"))[0]
df_sample = pd.read_parquet(sample_file)
df_sample = df_sample.head(100)  # Sample first 100 rows for column discovery
print(f"   Sample file: {sample_file.name}")
print(f"   Columns ({len(df_sample.columns)}):")
for col in df_sample.columns:
    print(f"     - {col}")

# Identify key columns (Advan naming convention)
# Standard SafeGraph -> Advan mapping:
#   PLACEKEY -> ID_STORE
#   RAW_VISIT_COUNTS -> VISIT_COUNTS
#   VISITOR_HOME_CBGS -> VISITOR_HOME_CBGS

# Find the visitor home column
cbgs_candidates = [c for c in df_sample.columns if 'HOME' in c.upper() or 'CBG' in c.upper()]
print(f"\n   Visitor home column candidates: {cbgs_candidates}")

# Set column names based on what we find
ID_COL = 'ID_STORE' if 'ID_STORE' in df_sample.columns else 'PLACEKEY'
VISIT_COL = 'VISIT_COUNTS' if 'VISIT_COUNTS' in df_sample.columns else 'RAW_VISIT_COUNTS'
CBGS_COL = 'VISITOR_HOME_CBGS'

print(f"\n   Using columns:")
print(f"     - ID: {ID_COL}")
print(f"     - Visits: {VISIT_COL}")
print(f"     - Home CBGS: {CBGS_COL}")

# =============================================================================
# 2. LOAD US WEEKLY PATTERNS DATA
# =============================================================================

print("\n2. Loading US weekly patterns data...")

us_files = sorted(US_WP_DIR.glob("*.parquet"))
print(f"   Found {len(us_files)} files")

# Load all files with relevant columns
cols_to_load = [ID_COL, 'TOP_CATEGORY', 'SUB_CATEGORY', VISIT_COL, CBGS_COL]
# Filter to columns that exist
cols_to_load = [c for c in cols_to_load if c in df_sample.columns]
print(f"   Loading columns: {cols_to_load}")

dfs = []
for i, f in enumerate(us_files):
    df = pd.read_parquet(f, columns=cols_to_load)
    dfs.append(df)
    if (i + 1) % 10 == 0:
        print(f"   Loaded {i + 1}/{len(us_files)} files...")

df_us = pd.concat(dfs, ignore_index=True)
del dfs

# Clean category columns
df_us['TOP_CATEGORY'] = df_us['TOP_CATEGORY'].str.strip()
df_us['SUB_CATEGORY'] = df_us['SUB_CATEGORY'].str.strip()

print(f"\n   Total records: {len(df_us):,}")

# =============================================================================
# 3. ANALYZE VISITOR_HOME_CBGS PRESENCE
# =============================================================================

print("\n3. Analyzing visitor_home_cbgs presence...")

# Check if visitor_home_cbgs is present (not null and not empty)
def has_valid_cbgs(val):
    """Check if visitor_home_cbgs has valid data."""
    if pd.isna(val):
        return False
    if isinstance(val, str):
        val = val.strip()
        if val == '' or val == '{}' or val == 'null' or val == '[]':
            return False
        # Try to parse as JSON and check if non-empty
        try:
            parsed = json.loads(val)
            if isinstance(parsed, dict):
                return len(parsed) > 0
            elif isinstance(parsed, list):
                return len(parsed) > 0
            return False
        except:
            return False
    if isinstance(val, dict):
        return len(val) > 0
    if isinstance(val, list):
        return len(val) > 0
    return False

def count_cbgs_tracts(val):
    """Count number of unique home tracts in CBGS data."""
    if pd.isna(val):
        return 0
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, dict):
                return len(parsed)
            elif isinstance(parsed, list):
                return len(parsed)
        except:
            return 0
    if isinstance(val, dict):
        return len(val)
    if isinstance(val, list):
        return len(val)
    return 0

print("   Computing coverage (this may take a minute)...")
df_us['has_cbgs'] = df_us[CBGS_COL].apply(has_valid_cbgs)
df_us['n_home_tracts'] = df_us[CBGS_COL].apply(count_cbgs_tracts)

total_records = len(df_us)
records_with_cbgs = df_us['has_cbgs'].sum()
overall_coverage = records_with_cbgs / total_records * 100

print(f"\n   Overall coverage:")
print(f"   - Records with visitor_home_cbgs: {records_with_cbgs:,} / {total_records:,}")
print(f"   - Coverage rate: {overall_coverage:.1f}%")

# Distribution of home tract counts
print(f"\n   Home tract count distribution (for records with CBGS):")
tract_dist = df_us[df_us['has_cbgs']]['n_home_tracts'].describe()
print(f"   - Mean: {tract_dist['mean']:.1f}")
print(f"   - Median: {tract_dist['50%']:.1f}")
print(f"   - Min: {tract_dist['min']:.0f}")
print(f"   - Max: {tract_dist['max']:.0f}")

# =============================================================================
# 4. COVERAGE BY SUB_CATEGORY
# =============================================================================

print("\n4. Coverage by SUB_CATEGORY...")

sub_cat_stats = df_us.groupby('SUB_CATEGORY').agg(
    total_records=('has_cbgs', 'count'),
    records_with_cbgs=('has_cbgs', 'sum'),
    total_visits=(VISIT_COL, 'sum'),
    mean_home_tracts=('n_home_tracts', 'mean')
).reset_index()

sub_cat_stats['coverage_pct'] = sub_cat_stats['records_with_cbgs'] / sub_cat_stats['total_records'] * 100
sub_cat_stats['pct_of_total'] = sub_cat_stats['total_records'] / total_records * 100
sub_cat_stats = sub_cat_stats.sort_values('total_records', ascending=False)

# Save full stats
sub_cat_stats.to_csv(OUTPUT_DIR / "sub_category_cbgs_coverage.csv", index=False)
print(f"   Saved: {OUTPUT_DIR / 'sub_category_cbgs_coverage.csv'}")

print(f"\n   Top 30 SUB_CATEGORY by volume:")
print(f"   {'Category':<50} {'Records':>8} {'Coverage':>8} {'Avg Tracts':>10}")
print("   " + "-" * 80)

for _, row in sub_cat_stats.head(30).iterrows():
    print(f"   {row['SUB_CATEGORY'][:50]:<50} {row['total_records']:>8,} {row['coverage_pct']:>7.1f}% {row['mean_home_tracts']:>10.1f}")

# =============================================================================
# 5. IDENTIFY LOW-COVERAGE CATEGORIES
# =============================================================================

print("\n5. Identifying categories with LOW visitor_home_cbgs coverage...")

# Define thresholds
MIN_COVERAGE_PCT = 50  # Minimum coverage to be considered reliable
MIN_RECORDS = 100  # Minimum records to be statistically meaningful

low_coverage = sub_cat_stats[
    (sub_cat_stats['coverage_pct'] < MIN_COVERAGE_PCT) &
    (sub_cat_stats['total_records'] >= MIN_RECORDS)
].sort_values('coverage_pct')

print(f"\n   Categories with <{MIN_COVERAGE_PCT}% coverage (min {MIN_RECORDS} records): {len(low_coverage)}")
print(f"   {'Category':<55} {'Records':>8} {'Coverage':>8}")
print("   " + "-" * 75)

for _, row in low_coverage.iterrows():
    print(f"   {row['SUB_CATEGORY'][:55]:<55} {row['total_records']:>8,} {row['coverage_pct']:>7.1f}%")

# =============================================================================
# 6. COVERAGE BY UNIFIED CATEGORY
# =============================================================================

print("\n6. Coverage by UNIFIED CATEGORY...")

import sys
sys.path.insert(0, str(ROOT))
from src.data.category_mapper import CategoryMapper

mapper = CategoryMapper()
df_us['unified_category'] = df_us['SUB_CATEGORY'].map(mapper.map_category)

unified_stats = df_us.groupby('unified_category').agg(
    total_records=('has_cbgs', 'count'),
    records_with_cbgs=('has_cbgs', 'sum'),
    total_visits=(VISIT_COL, 'sum'),
    mean_home_tracts=('n_home_tracts', 'mean')
).reset_index()

unified_stats['coverage_pct'] = unified_stats['records_with_cbgs'] / unified_stats['total_records'] * 100
unified_stats = unified_stats.sort_values('total_records', ascending=False)

print(f"\n   {'Unified Category':<25} {'Records':>10} {'With CBGS':>12} {'Coverage':>10} {'Avg Tracts':>12}")
print("   " + "-" * 75)

for _, row in unified_stats.iterrows():
    if pd.notna(row['unified_category']):
        print(f"   {row['unified_category']:<25} {row['total_records']:>10,} {row['records_with_cbgs']:>12,} {row['coverage_pct']:>9.1f}% {row['mean_home_tracts']:>12.1f}")

# Show excluded/unmapped
excluded_mask = df_us['unified_category'].isna()
excluded_stats = {
    'total': excluded_mask.sum(),
    'with_cbgs': df_us.loc[excluded_mask, 'has_cbgs'].sum(),
    'coverage': df_us.loc[excluded_mask, 'has_cbgs'].mean() * 100 if excluded_mask.any() else 0
}
print(f"\n   {'(excluded/unmapped)':<25} {excluded_stats['total']:>10,} {excluded_stats['with_cbgs']:>12,} {excluded_stats['coverage']:>9.1f}%")

# Save unified stats
unified_stats.to_csv(OUTPUT_DIR / "unified_category_cbgs_coverage.csv", index=False)
print(f"\n   Saved: {OUTPUT_DIR / 'unified_category_cbgs_coverage.csv'}")

# =============================================================================
# 7. IDENTIFY CATEGORIES TO EXCLUDE BASED ON CBGS COVERAGE
# =============================================================================

print("\n7. CATEGORIES TO EXCLUDE (low CBGS coverage)...")

# Categories in our unified schema with low coverage
unified_low = unified_stats[
    (unified_stats['coverage_pct'] < MIN_COVERAGE_PCT) &
    (unified_stats['unified_category'].notna())
]

if len(unified_low) > 0:
    print(f"\n   Unified categories with < {MIN_COVERAGE_PCT}% coverage:")
    for _, row in unified_low.iterrows():
        print(f"   - {row['unified_category']}: {row['coverage_pct']:.1f}% ({row['total_records']:,} records)")
else:
    print(f"\n   All unified categories have >= {MIN_COVERAGE_PCT}% coverage")

# SUB_CATEGORIES within retained unified categories that have low coverage
print(f"\n   SUB_CATEGORIES within unified schema with low coverage:")
mapped_subs = df_us[df_us['unified_category'].notna()].groupby('SUB_CATEGORY').agg(
    total=('has_cbgs', 'count'),
    with_cbgs=('has_cbgs', 'sum'),
    unified=('unified_category', 'first')
).reset_index()
mapped_subs['coverage'] = mapped_subs['with_cbgs'] / mapped_subs['total'] * 100
low_mapped = mapped_subs[(mapped_subs['coverage'] < MIN_COVERAGE_PCT) & (mapped_subs['total'] >= MIN_RECORDS)]

for _, row in low_mapped.sort_values('coverage').iterrows():
    print(f"   - {row['SUB_CATEGORY'][:50]}: {row['coverage']:.1f}% (unified: {row['unified']})")

# =============================================================================
# 8. SUMMARY AND RECOMMENDATIONS
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY AND RECOMMENDATIONS")
print("=" * 70)

high_coverage = sub_cat_stats[
    (sub_cat_stats['coverage_pct'] >= MIN_COVERAGE_PCT) &
    (sub_cat_stats['total_records'] >= MIN_RECORDS)
]

print(f"""
Overall Statistics:
  - Total POI-week records: {total_records:,}
  - Records with visitor_home_cbgs: {records_with_cbgs:,} ({overall_coverage:.1f}%)

Coverage Distribution:
  - SUB_CATEGORIES with >= 50% coverage: {len(high_coverage)}
  - SUB_CATEGORIES with < 50% coverage: {len(low_coverage)}

Unified Category Coverage:
""")

for _, row in unified_stats.sort_values('coverage_pct', ascending=False).iterrows():
    if pd.notna(row['unified_category']):
        status = "✓" if row['coverage_pct'] >= MIN_COVERAGE_PCT else "✗"
        print(f"  {status} {row['unified_category']}: {row['coverage_pct']:.1f}%")

print("""
Recommendations:
  1. Retain categories with >= 50% visitor_home_cbgs coverage
  2. For categories with < 50% coverage, consider:
     - Excluding from social mixing analysis, OR
     - Flagging as "limited data" in results
  3. Swedish data should be filtered to match retained US categories
  4. Document coverage limitations in methodology section
""")
