"""
Aggregate Swedish GPS Visits to Weekly POI Flows

Creates Swedish equivalent of US weekly_patterns data format.
Matches US column structure for cross-country comparability.

Each row = one (POI, week) combination, matching US weekly_patterns format.

Input:
    - dbs/poi_assignment/stops_poi_assigned.parquet
    - dbs/stops/*.parquet (for timestamps)
    - dbs/home_work/device_homes_deso_ipw.parquet
    - dbs/poi_se/POI_se_2026_03_05/*.parquet

Output:
    - dbs/sweden_weekly_patterns/sweden_weekly_patterns_{YYYY_MM}.parquet
      Format matches US weekly_patterns with VISITOR_HOME_CBGS equivalent

Usage:
    python aggregate_swedish_flows.py --year 2024 --month 3      # Single month
    python aggregate_swedish_flows.py --year 2024                 # Full year
    python aggregate_swedish_flows.py --start-date 2024-03-01 --end-date 2024-03-31
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import gc
import sys

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.category_mapper import CategoryMapper

# =============================================================================
# Configuration
# =============================================================================

# Paths
STOPS_ASSIGNED = ROOT_DIR / "dbs/poi_assignment/stops_poi_assigned.parquet"
STOPS_DIR = ROOT_DIR / "dbs/stops"  # Original stops with timestamps
DEVICE_HOMES = ROOT_DIR / "dbs/home_work/device_homes_deso_ipw.parquet"
POI_DIR = ROOT_DIR / "dbs/poi_se/POI_se_2026_03_05"
OUTPUT_DIR = ROOT_DIR / "dbs/sweden_weekly_patterns"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Aggregate Swedish GPS visits to weekly POI flows'
    )
    parser.add_argument(
        '--start-date', type=str,
        help='Start date (YYYY-MM-DD format)'
    )
    parser.add_argument(
        '--end-date', type=str,
        help='End date (YYYY-MM-DD format)'
    )
    parser.add_argument(
        '--year', type=int, default=2024,
        help='Year to process (default: 2024)'
    )
    parser.add_argument(
        '--month', type=int, default=None,
        help='Month to process (1-12). If not specified, process full year.'
    )
    return parser.parse_args()


def get_week_start(date):
    """Get the Monday of the week containing the given date."""
    return date - timedelta(days=date.weekday())


def get_week_id(date):
    """Get week identifier as YYYY-WNN format."""
    week_start = get_week_start(date)
    year, week_num, _ = week_start.isocalendar()
    return f"{year}-W{week_num:02d}"


def main():
    args = parse_args()

    # Determine date range
    if args.start_date and args.end_date:
        start_date = pd.to_datetime(args.start_date)
        end_date = pd.to_datetime(args.end_date)
        date_suffix = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
    elif args.month is not None:
        # Single month
        start_date = pd.Timestamp(args.year, args.month, 1)
        if args.month == 12:
            end_date = pd.Timestamp(args.year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = pd.Timestamp(args.year, args.month + 1, 1) - timedelta(days=1)
        date_suffix = f"{args.year}_{args.month:02d}"
    else:
        # Full year
        start_date = pd.Timestamp(args.year, 1, 1)
        end_date = pd.Timestamp(args.year, 12, 31)
        date_suffix = f"{args.year}"

    print("=" * 70)
    print("AGGREGATE SWEDISH GPS VISITS TO WEEKLY POI FLOWS")
    print("=" * 70)
    print(f"\nDate range: {start_date.date()} to {end_date.date()}")
    print(f"Output format matches US weekly_patterns (one row per POI-week)")

    # Output file name
    OUTPUT_FILE = OUTPUT_DIR / f"sweden_weekly_patterns_{date_suffix}.parquet"

    # =========================================================================
    # 1. Load Device Home Locations with IPW Weights
    # =========================================================================

    print("\n1. Loading device home locations...")

    df_homes = pd.read_parquet(
        DEVICE_HOMES,
        columns=['device_aid', 'home_deso_code', 'ipw_weight', 'has_home', 'has_building']
    )

    # Filter to devices with valid home DeSO
    df_homes = df_homes[
        df_homes['has_home'] &
        df_homes['has_building'] &
        df_homes['home_deso_code'].notna()
    ].copy()

    print(f"   Devices with valid home DeSO: {len(df_homes):,}")

    # Create lookup dict
    device_home_lookup = df_homes.set_index('device_aid')[['home_deso_code', 'ipw_weight']].to_dict('index')
    del df_homes
    gc.collect()

    # =========================================================================
    # 2. Load POI Metadata
    # =========================================================================

    print("\n2. Loading POI metadata...")

    poi_files = sorted(POI_DIR.glob("*.parquet"))
    poi_dfs = [pd.read_parquet(f) for f in poi_files]
    df_poi = pd.concat(poi_dfs, ignore_index=True)
    del poi_dfs

    # Clean category columns
    df_poi['TOP_CATEGORY'] = df_poi['TOP_CATEGORY'].str.strip()
    df_poi['SUB_CATEGORY'] = df_poi['SUB_CATEGORY'].str.strip()

    print(f"   Total POIs: {len(df_poi):,}")

    # Create POI lookup
    poi_columns = ['PLACEKEY', 'LOCATION_NAME', 'STREET_ADDRESS', 'CITY', 'REGION',
                   'POSTAL_CODE', 'LATITUDE', 'LONGITUDE', 'TOP_CATEGORY', 'SUB_CATEGORY',
                   'NAICS_CODE', 'BRAND']
    poi_columns = [c for c in poi_columns if c in df_poi.columns]
    df_poi_lookup = df_poi[poi_columns].drop_duplicates(subset=['PLACEKEY'])
    poi_metadata = df_poi_lookup.set_index('PLACEKEY').to_dict('index')

    del df_poi, df_poi_lookup
    gc.collect()

    # =========================================================================
    # 3. Load Original Stops with Timestamps
    # =========================================================================

    print("\n3. Loading original stops with timestamps...")

    # Find stop files - MUST use same order as assign_poi_tiered.py
    stop_files = sorted(STOPS_DIR.glob("*.parquet"))
    print(f"   Found {len(stop_files)} stop files")

    # Load all stops and create stop_id matching POI assignment logic
    stop_dfs = []
    global_stop_counter = 0

    for file_idx, f in enumerate(stop_files):
        df_stops = pd.read_parquet(f, columns=['device_aid', 'start'])
        n_stops = len(df_stops)

        # Create sequential stop_id (matching assign_poi_tiered.py logic)
        df_stops['stop_id'] = np.arange(global_stop_counter, global_stop_counter + n_stops)
        global_stop_counter += n_stops

        # Convert Unix timestamp to datetime
        df_stops['start_time'] = pd.to_datetime(df_stops['start'], unit='s')

        # Filter to date range if not full year 2024
        if not (start_date.year == 2024 and start_date.month == 1 and start_date.day == 1 and
                end_date.year == 2024 and end_date.month == 12 and end_date.day == 31):
            mask = (df_stops['start_time'] >= start_date) & (df_stops['start_time'] <= end_date + timedelta(days=1))
            df_stops = df_stops[mask]

        stop_dfs.append(df_stops[['stop_id', 'start_time']])

        if (file_idx + 1) % 10 == 0 or file_idx == len(stop_files) - 1:
            print(f"   File {file_idx + 1}/{len(stop_files)}: loaded {n_stops:,} stops")

    print(f"   Concatenating stop timestamps...")
    df_stop_times = pd.concat(stop_dfs, ignore_index=True)
    del stop_dfs
    gc.collect()

    print(f"   Total stops with timestamps: {len(df_stop_times):,}")

    # =========================================================================
    # 4. Load and Merge Assigned Stops
    # =========================================================================

    print("\n4. Loading assigned stops and merging with timestamps...")

    df_assigned = pd.read_parquet(STOPS_ASSIGNED, columns=['stop_id', 'device_aid', 'poi_id'])
    print(f"   Total assigned stop records: {len(df_assigned):,}")

    # Filter to stops with POI assignment
    df_assigned = df_assigned[df_assigned['poi_id'].notna()].copy()
    print(f"   Stops matched to POI: {len(df_assigned):,}")

    # Merge with timestamps
    print(f"   Merging with timestamps...")
    df_merged = df_assigned.merge(df_stop_times, on='stop_id', how='inner')
    print(f"   Stops with timestamps: {len(df_merged):,}")

    del df_assigned, df_stop_times
    gc.collect()

    if len(df_merged) == 0:
        print("   ERROR: No stops found after merging!")
        return

    # =========================================================================
    # 5. Add Home DeSO and Aggregate
    # =========================================================================

    print("\n5. Adding home locations and aggregating...")

    # Compute week start (Monday) - vectorized
    df_merged['week_start'] = df_merged['start_time'].dt.to_period('W-SUN').dt.start_time.dt.strftime('%Y-%m-%d')

    # Add home DeSO and IPW weight using vectorized merge
    df_homes_lookup = pd.DataFrame([
        {'device_aid': k, 'home_deso_code': v['home_deso_code'], 'ipw_weight': v['ipw_weight']}
        for k, v in device_home_lookup.items()
    ])

    df_merged = df_merged.merge(df_homes_lookup, on='device_aid', how='inner')
    print(f"   Stops with home DeSO: {len(df_merged):,}")

    del df_homes_lookup
    gc.collect()

    if len(df_merged) == 0:
        print("   ERROR: No stops with home DeSO!")
        return

    # Aggregate by (poi_id, week_start, home_deso_code)
    print(f"   Aggregating flows...")
    df_flows = df_merged.groupby(['poi_id', 'week_start', 'home_deso_code']).agg(
        raw_count=('device_aid', 'count'),
        weighted_count=('ipw_weight', 'sum')
    ).reset_index()

    print(f"   Unique (POI, week, DeSO) flows: {len(df_flows):,}")

    del df_merged
    gc.collect()

    print(f"   Unique (POI, week) combinations: {df_flows.groupby(['poi_id', 'week_start']).ngroups:,}")

    # =========================================================================
    # 6. Build Output DataFrame (US weekly_patterns format)
    # =========================================================================

    print("\n6. Building output DataFrame...")

    mapper = CategoryMapper()

    # Group by (poi_id, week_start) and build VISITOR_HOME_CBGS JSON
    print("   Aggregating to (POI, week) level...")

    poi_week_records = []
    for (poi_id, week_start), group in df_flows.groupby(['poi_id', 'week_start']):
        # Build VISITOR_HOME_CBGS JSON
        raw_dict = dict(zip(group['home_deso_code'], group['raw_count'].astype(int)))
        weighted_dict = dict(zip(group['home_deso_code'], group['weighted_count'].round(2)))

        poi_week_records.append({
            'poi_id': poi_id,
            'week_start': week_start,
            'VISIT_COUNTS': group['raw_count'].sum(),
            'WEIGHTED_VISIT_COUNTS': group['weighted_count'].sum(),
            'VISITOR_COUNTS': len(group),  # Unique home zones
            'VISITOR_HOME_CBGS': json.dumps(raw_dict),
            'VISITOR_HOME_CBGS_WEIGHTED': json.dumps(weighted_dict)
        })

    df_poi_week = pd.DataFrame(poi_week_records)

    del df_flows
    gc.collect()

    print(f"   POI-week records: {len(df_poi_week):,}")

    # Add POI metadata
    print("   Adding POI metadata...")

    meta_records = []
    for poi_id in df_poi_week['poi_id'].unique():
        meta = poi_metadata.get(poi_id, {})
        meta_records.append({
            'poi_id': poi_id,
            'LOCATION_NAME': meta.get('LOCATION_NAME', ''),
            'STREET_ADDRESS': meta.get('STREET_ADDRESS', ''),
            'CITY': meta.get('CITY', ''),
            'REGION': meta.get('REGION', ''),
            'POSTAL_CODE': meta.get('POSTAL_CODE', ''),
            'LATITUDE': meta.get('LATITUDE'),
            'LONGITUDE': meta.get('LONGITUDE'),
            'TOP_CATEGORY': meta.get('TOP_CATEGORY', ''),
            'SUB_CATEGORY': meta.get('SUB_CATEGORY', ''),
            'NAICS_CODE': meta.get('NAICS_CODE', ''),
            'BRAND': meta.get('BRAND', ''),
        })

    df_meta = pd.DataFrame(meta_records)

    # Merge
    df_output = df_poi_week.merge(df_meta, on='poi_id', how='left')

    # Add unified category
    df_output['unified_category'] = df_output['SUB_CATEGORY'].map(mapper.map_category)

    # Add week end (6 days after week start)
    df_output['DATE_RANGE_START'] = df_output['week_start']
    df_output['DATE_RANGE_END'] = pd.to_datetime(df_output['week_start']) + timedelta(days=6)
    df_output['DATE_RANGE_END'] = df_output['DATE_RANGE_END'].dt.strftime('%Y-%m-%d')

    # Rename columns
    df_output = df_output.rename(columns={'poi_id': 'PLACEKEY'})
    df_output['ID_STORE'] = df_output['PLACEKEY']

    # Add metadata
    df_output['ISO_COUNTRY_CODE'] = 'SE'

    # Reorder columns to match US format
    column_order = [
        'ID_STORE', 'PLACEKEY',
        'LOCATION_NAME', 'STREET_ADDRESS', 'CITY', 'REGION', 'POSTAL_CODE',
        'LATITUDE', 'LONGITUDE',
        'TOP_CATEGORY', 'SUB_CATEGORY', 'unified_category',
        'NAICS_CODE', 'BRAND',
        'VISIT_COUNTS', 'WEIGHTED_VISIT_COUNTS', 'VISITOR_COUNTS',
        'VISITOR_HOME_CBGS', 'VISITOR_HOME_CBGS_WEIGHTED',
        'DATE_RANGE_START', 'DATE_RANGE_END',
        'ISO_COUNTRY_CODE'
    ]
    df_output = df_output[[c for c in column_order if c in df_output.columns]]

    # Drop week_start (now have DATE_RANGE_START/END)
    if 'week_start' in df_output.columns:
        df_output = df_output.drop(columns=['week_start'])

    print(f"   Final records: {len(df_output):,}")

    # =========================================================================
    # 7. Summary Statistics
    # =========================================================================

    print("\n7. Summary statistics...")

    # Weeks covered
    weeks = df_output['DATE_RANGE_START'].unique()
    print(f"\n   Weeks covered: {len(weeks)}")
    for w in sorted(weeks):
        week_data = df_output[df_output['DATE_RANGE_START'] == w]
        print(f"   - {w}: {len(week_data):,} POIs, {week_data['VISIT_COUNTS'].sum():,.0f} visits")

    # Category distribution
    print("\n   Unified category distribution:")
    cat_dist = df_output.groupby('unified_category').agg({
        'PLACEKEY': 'nunique',
        'VISIT_COUNTS': 'sum',
        'VISITOR_COUNTS': 'mean'
    }).rename(columns={'PLACEKEY': 'unique_pois', 'VISITOR_COUNTS': 'avg_home_zones'})
    cat_dist['visit_pct'] = cat_dist['VISIT_COUNTS'] / cat_dist['VISIT_COUNTS'].sum() * 100
    cat_dist = cat_dist.sort_values('VISIT_COUNTS', ascending=False)

    print(f"\n   {'Category':<25} {'POIs':>8} {'Visits':>12} {'%':>8} {'Avg Zones':>10}")
    print("   " + "-" * 65)
    for cat, row in cat_dist.head(15).iterrows():
        if pd.notna(cat):
            print(f"   {str(cat):<25} {row['unique_pois']:>8,} {row['VISIT_COUNTS']:>12,.0f} "
                  f"{row['visit_pct']:>7.1f}% {row['avg_home_zones']:>10.1f}")

    # =========================================================================
    # 8. Save Output
    # =========================================================================

    print("\n8. Saving output...")

    df_output.to_parquet(OUTPUT_FILE, index=False)
    print(f"   ✓ Saved: {OUTPUT_FILE}")
    print(f"   File size: {OUTPUT_FILE.stat().st_size / 1e6:.1f} MB")

    # =========================================================================
    # 9. Final Summary
    # =========================================================================

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"""
Date Range: {start_date.date()} to {end_date.date()}
Weeks: {len(weeks)}

Output Format (matches US weekly_patterns):
  - One row per (POI, week) combination
  - DATE_RANGE_START/END: Week boundaries (Mon-Sun)
  - VISITOR_HOME_CBGS: JSON dict {{deso_code: visitor_count}}
  - All three category levels: TOP_CATEGORY, SUB_CATEGORY, unified_category

Statistics:
  - Total POI-week records: {len(df_output):,}
  - Unique POIs: {df_output['PLACEKEY'].nunique():,}
  - Total visits: {df_output['VISIT_COUNTS'].sum():,.0f}
  - Total weighted visits: {df_output['WEIGHTED_VISIT_COUNTS'].sum():,.0f}
  - POI-weeks with unified category: {df_output['unified_category'].notna().sum():,}

Output file: {OUTPUT_FILE}
""")

    print("=" * 70)
    print("✅ Swedish weekly patterns aggregation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
