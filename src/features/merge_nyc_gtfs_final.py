#!/usr/bin/env python
"""Merge all NYC GTFS files with fixes for problematic ones."""

import zipfile
from pathlib import Path
import pandas as pd
from tempfile import TemporaryDirectory

backup_dir = Path('dbs/gtfs_backup/new_york_updated')
output_file = Path('dbs/gtfs/new_york/merged_gtfs.zip')

# Files that need fixing (commuter rail with problematic structure)
NEEDS_FIX = {'gtfslirr.zip', 'gtfsmnr.zip'}

# Apply calendar fix to ALL files to ensure services run on all days
FIX_CALENDAR_ALL = True

def add_prefix(df, prefix, columns):
    """Add prefix to specified columns."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            mask = df[col].notna() & (df[col] != '')
            df.loc[mask, col] = prefix + '_' + df.loc[mask, col].astype(str)
    return df

def load_and_fix_gtfs(gtfs_path, prefix, needs_fix=False):
    """Load GTFS and apply fixes if needed."""
    tables = {}

    with zipfile.ZipFile(gtfs_path, 'r') as zf:
        file_list = zf.namelist()

        for txt_file in file_list:
            if not txt_file.endswith('.txt'):
                continue

            table_name = txt_file.replace('.txt', '')

            with zf.open(txt_file) as f:
                try:
                    df = pd.read_csv(f, dtype=str, low_memory=False)
                except:
                    continue

            if len(df) == 0:
                continue

            tables[table_name] = df

    # Get agency_id
    agency_id = 'UNKNOWN'
    if 'agency' in tables:
        agency_id = tables['agency']['agency_id'].iloc[0]

    # Apply fixes for problematic files (LIRR, Metro-North)
    if needs_fix:
        # Fix routes - add agency_id
        if 'routes' in tables and 'agency_id' not in tables['routes'].columns:
            tables['routes']['agency_id'] = agency_id

        # Fix stop_times - keep only essential columns
        if 'stop_times' in tables:
            essential = ['trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence']
            cols = [c for c in essential if c in tables['stop_times'].columns]
            tables['stop_times'] = tables['stop_times'][cols]

        # Remove problematic transfers
        if 'transfers' in tables:
            if 'from_trip_id' in tables['transfers'].columns:
                del tables['transfers']

    # Apply calendar fix to ALL files to ensure services run on all days
    if FIX_CALENDAR_ALL and 'trips' in tables:
        all_services = tables['trips']['service_id'].dropna().unique()
        tables['calendar'] = pd.DataFrame({
            'service_id': all_services,
            'monday': '1', 'tuesday': '1', 'wednesday': '1',
            'thursday': '1', 'friday': '1', 'saturday': '1', 'sunday': '1',
            'start_date': '20240101', 'end_date': '20271231',
        })
        # Remove calendar_dates to avoid conflicts
        if 'calendar_dates' in tables:
            del tables['calendar_dates']

    # Add prefixes
    for table_name, df in tables.items():
        if table_name == 'agency':
            tables[table_name] = add_prefix(df, prefix, ['agency_id'])
        elif table_name == 'routes':
            tables[table_name] = add_prefix(df, prefix, ['route_id', 'agency_id'])
        elif table_name == 'trips':
            tables[table_name] = add_prefix(df, prefix, ['trip_id', 'route_id', 'service_id', 'shape_id'])
        elif table_name == 'stops':
            tables[table_name] = add_prefix(df, prefix, ['stop_id', 'parent_station'])
        elif table_name == 'stop_times':
            tables[table_name] = add_prefix(df, prefix, ['trip_id', 'stop_id'])
        elif table_name == 'calendar':
            tables[table_name] = add_prefix(df, prefix, ['service_id'])
        elif table_name == 'calendar_dates':
            tables[table_name] = add_prefix(df, prefix, ['service_id'])
        elif table_name == 'shapes':
            tables[table_name] = add_prefix(df, prefix, ['shape_id'])
        elif table_name == 'transfers':
            tables[table_name] = add_prefix(df, prefix, ['from_stop_id', 'to_stop_id'])
        elif table_name == 'frequencies':
            tables[table_name] = add_prefix(df, prefix, ['trip_id'])

    return tables

# Collect all GTFS files
gtfs_files = sorted([f for f in backup_dir.glob('*.zip')
                     if 'merged' not in f.name and 'cleaned' not in f.name and 'backup' not in f.name])

print(f"Found {len(gtfs_files)} GTFS files:")
for f in gtfs_files:
    marker = " [NEEDS FIX]" if f.name in NEEDS_FIX else ""
    print(f"  - {f.name}{marker}")

# Load and merge all files
all_tables = {}

for gtfs_path in gtfs_files:
    prefix = gtfs_path.stem.replace('-', '_')
    needs_fix = gtfs_path.name in NEEDS_FIX

    print(f"\nProcessing: {gtfs_path.name} (prefix: {prefix})")

    tables = load_and_fix_gtfs(gtfs_path, prefix, needs_fix)

    for table_name, df in tables.items():
        print(f"  {table_name}: {len(df)} rows")
        if table_name not in all_tables:
            all_tables[table_name] = []
        all_tables[table_name].append(df)

# Merge and write
print(f"\n{'='*60}")
print("MERGING ALL TABLES")
print('='*60)

with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zf:
    for table_name, dfs in all_tables.items():
        merged = pd.concat(dfs, ignore_index=True)
        print(f"{table_name}: {len(merged):,} rows")
        csv_content = merged.to_csv(index=False)
        zf.writestr(f'{table_name}.txt', csv_content)

print(f"\nOutput: {output_file}")
print(f"Size: {output_file.stat().st_size / 1e6:.1f} MB")

# Quick validation
print(f"\n{'='*60}")
print("VALIDATION")
print('='*60)
with zipfile.ZipFile(output_file, 'r') as zf:
    with zf.open('trips.txt') as f:
        trips = pd.read_csv(f, dtype=str)
    with zf.open('stops.txt') as f:
        stops = pd.read_csv(f, dtype=str)
    with zf.open('routes.txt') as f:
        routes = pd.read_csv(f, dtype=str)

print(f"Total trips: {len(trips):,}")
print(f"Total stops: {len(stops):,}")
print(f"Total routes: {len(routes):,}")

# Check service coverage
with zipfile.ZipFile(output_file, 'r') as zf:
    with zf.open('calendar.txt') as f:
        cal = pd.read_csv(f, dtype=str)

trip_services = set(trips['service_id'].dropna())
cal_services = set(cal['service_id'].dropna())
orphans = trip_services - cal_services
print(f"Calendar services: {len(cal_services):,}")
print(f"Orphan services: {len(orphans)}")

if len(orphans) == 0:
    print("\n✓ All services have calendar entries!")
