"""
Category Exclusion Analysis

Computes:
1. How much Swedish data gets excluded (K-12, childcare, transit)
2. Visit distribution alignment across unified categories
3. Comparison of TOP_CATEGORY, SUB_CATEGORY, unified_category

Run in Jupyter or as script.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.category_mapper import CategoryMapper, compute_exclusion_stats, compute_category_distribution

# Paths
SE_POI_DIR = ROOT / "dbs/poi_se/POI_se_2026_03_05"
SE_ASSIGNED = ROOT / "dbs/poi_assignment/stops_poi_assigned.parquet"
US_WP_DIR = ROOT / "dbs/us_foot_traffic/weekly_patterns"

print("=" * 70)
print("CATEGORY EXCLUSION & ALIGNMENT ANALYSIS")
print("=" * 70)

# =============================================================================
# 1. SWEDISH DATA ANALYSIS
# =============================================================================

print("\n" + "=" * 70)
print("1. SWEDISH DATA - EXCLUSION STATISTICS")
print("=" * 70)

# Load Swedish POI data
se_poi_files = sorted(SE_POI_DIR.glob("*.parquet"))
df_se_poi = pd.concat([pd.read_parquet(f) for f in se_poi_files], ignore_index=True)
df_se_poi['TOP_CATEGORY'] = df_se_poi['TOP_CATEGORY'].str.strip()
df_se_poi['SUB_CATEGORY'] = df_se_poi['SUB_CATEGORY'].str.strip()

print(f"\nSwedish POIs loaded: {len(df_se_poi):,}")

# Load Swedish assigned stops (only poi_id column to save memory)
df_se_assigned = pd.read_parquet(SE_ASSIGNED, columns=['poi_id'])
matched_stops = df_se_assigned[df_se_assigned['poi_id'].notna()]
print(f"Swedish matched stops: {len(matched_stops):,}")

# Count visits per POI
visits_per_poi = matched_stops.groupby('poi_id').size().reset_index(name='visit_count')
print(f"Unique POIs with visits: {len(visits_per_poi):,}")

# Merge with POI categories
df_se_visits = visits_per_poi.merge(
    df_se_poi[['PLACEKEY', 'TOP_CATEGORY', 'SUB_CATEGORY']],
    left_on='poi_id',
    right_on='PLACEKEY',
    how='left'
)

# Compute exclusion stats weighted by visits
se_stats = compute_exclusion_stats(df_se_visits, 'SUB_CATEGORY', 'visit_count')

print(f"\n--- Swedish Visit Distribution ---")
print(f"Total visits: {se_stats['total']:,.0f}")
print(f"Mapped to unified: {se_stats['mapped']:,.0f} ({se_stats['mapped_pct']:.1f}%)")
print(f"Excluded: {se_stats['excluded']:,.0f} ({se_stats['excluded_pct']:.1f}%)")
print(f"Unmapped: {se_stats['unmapped']:,.0f} ({se_stats['unmapped_pct']:.1f}%)")

print(f"\n--- Top Excluded Categories (Sweden) ---")
for cat, count in list(se_stats['excluded_breakdown'].items())[:10]:
    pct = count / se_stats['total'] * 100
    print(f"  {cat}: {count:,.0f} ({pct:.2f}%)")

# =============================================================================
# 2. US DATA ANALYSIS
# =============================================================================

print("\n" + "=" * 70)
print("2. US DATA - EXCLUSION STATISTICS")
print("=" * 70)

# Load US data (sample of files)
us_files = sorted(US_WP_DIR.glob("*.parquet"))[:5]
print(f"Loading {len(us_files)} US files...")

us_dfs = []
for f in us_files:
    df = pd.read_parquet(f, columns=['ID_STORE', 'TOP_CATEGORY', 'SUB_CATEGORY', 'VISIT_COUNTS'])
    us_dfs.append(df)

df_us = pd.concat(us_dfs, ignore_index=True)
df_us['TOP_CATEGORY'] = df_us['TOP_CATEGORY'].str.strip()
df_us['SUB_CATEGORY'] = df_us['SUB_CATEGORY'].str.strip()

print(f"US records loaded: {len(df_us):,}")

# Compute exclusion stats
us_stats = compute_exclusion_stats(df_us, 'SUB_CATEGORY', 'VISIT_COUNTS')

print(f"\n--- US Visit Distribution ---")
print(f"Total visits: {us_stats['total']:,.0f}")
print(f"Mapped to unified: {us_stats['mapped']:,.0f} ({us_stats['mapped_pct']:.1f}%)")
print(f"Excluded: {us_stats['excluded']:,.0f} ({us_stats['excluded_pct']:.1f}%)")
print(f"Unmapped: {us_stats['unmapped']:,.0f} ({us_stats['unmapped_pct']:.1f}%)")

print(f"\n--- Top Excluded Categories (US) ---")
for cat, count in list(us_stats['excluded_breakdown'].items())[:10]:
    pct = count / us_stats['total'] * 100
    print(f"  {cat}: {count:,.0f} ({pct:.2f}%)")

# =============================================================================
# 3. UNIFIED CATEGORY DISTRIBUTION COMPARISON
# =============================================================================

print("\n" + "=" * 70)
print("3. UNIFIED CATEGORY DISTRIBUTION COMPARISON")
print("=" * 70)

# Compute distributions
se_dist = compute_category_distribution(df_se_visits, 'SUB_CATEGORY', 'visit_count')
us_dist = compute_category_distribution(df_us, 'SUB_CATEGORY', 'RAW_VISIT_COUNTS')

# Normalize to percentages
se_total = sum(se_dist.values())
us_total = sum(us_dist.values())

mapper = CategoryMapper()
all_cats = mapper.UNIFIED_CATEGORIES

print(f"\n{'Category':<25} {'Sweden %':>10} {'US %':>10} {'Diff':>10}")
print("-" * 55)

for cat in all_cats:
    se_pct = se_dist.get(cat, 0) / se_total * 100 if se_total > 0 else 0
    us_pct = us_dist.get(cat, 0) / us_total * 100 if us_total > 0 else 0
    diff = se_pct - us_pct
    print(f"{mapper.get_category_description(cat):<25} {se_pct:>9.1f}% {us_pct:>9.1f}% {diff:>+9.1f}%")

# =============================================================================
# 4. EXAMPLE: PRESERVING ALL CATEGORY COLUMNS
# =============================================================================

print("\n" + "=" * 70)
print("4. EXAMPLE: CATEGORY COLUMN PRESERVATION")
print("=" * 70)

# Apply mapper preserving all columns
mapper = CategoryMapper()
df_example = df_se_visits.head(10).copy()
df_example = mapper.map_dataframe(df_example, 'SUB_CATEGORY', 'TOP_CATEGORY', 'unified_category')

print("\nSample with all category columns preserved:")
print(df_example[['poi_id', 'TOP_CATEGORY', 'SUB_CATEGORY', 'unified_category', 'visit_count']].to_string())

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
Swedish data exclusions:
  - K-12 Schools: ~{se_stats['excluded_breakdown'].get('Elementary and Secondary Schools', 0) / se_stats['total'] * 100:.1f}% of visits
  - Child Day Care: ~{se_stats['excluded_breakdown'].get('Child Day Care Services', 0) / se_stats['total'] * 100:.1f}% of visits
  - Urban Transit: ~{se_stats['excluded_breakdown'].get('Urban Transit Systems', 0) / se_stats['total'] * 100:.1f}% of visits
  - Total excluded: {se_stats['excluded_pct']:.1f}%

US data exclusions:
  - Total excluded: {us_stats['excluded_pct']:.1f}%

Both datasets will carry: TOP_CATEGORY, SUB_CATEGORY, unified_category
""")
