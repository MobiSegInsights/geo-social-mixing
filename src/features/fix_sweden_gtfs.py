#!/usr/bin/env python
"""Create fixed Sweden GTFS: remove shapes, fix calendar, filter bad coords."""

import zipfile
from pathlib import Path
import pandas as pd

input_path = Path('dbs/gtfs_backup/sweden/sweden.zip')
output_path = Path('dbs/gtfs_backup/sweden/sweden_fixed.zip')

print(f"Input: {input_path} ({input_path.stat().st_size / 1e9:.2f} GB)")

# Files to exclude:
# - shapes.txt: 2.9 GB, not needed for routing
# - calendar_dates.txt: replaced by fixed calendar
# - transfers.txt: has from_trip_id/to_trip_id columns that break r5r
EXCLUDE_FILES = {'shapes.txt', 'calendar_dates.txt', 'transfers.txt'}

# Sweden bounding box
SWEDEN_LAT_MIN = 50.0  # Allow some buffer for southern ferries
SWEDEN_LAT_MAX = 70.0
SWEDEN_LON_MIN = 5.0   # Allow buffer for western connections
SWEDEN_LON_MAX = 30.0

# Supported route_types in r5r (standard GTFS + common extended types)
# Exclude taxi (1501), etc.
SUPPORTED_ROUTE_TYPES = {
    '0',    # Tram
    '1',    # Subway/Metro
    '2',    # Rail
    '3',    # Bus
    '4',    # Ferry
    '5',    # Cable tram
    '6',    # Aerial lift
    '7',    # Funicular
    '11',   # Trolleybus
    '12',   # Monorail
    # Extended types (100s-700s are generally supported)
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',  # Rail
    '200', '201', '202', '203', '204', '205', '206', '207', '208', '209',  # Coach
    '400', '401', '402', '403', '404', '405',  # Urban rail
    '700', '701', '702', '703', '704', '705', '706', '707', '708', '709',  # Bus
    '710', '711', '712', '713', '714', '715', '716', '717',
    '800',  # Trolleybus
    '900', '901', '902', '903', '904', '905', '906',  # Tram
    '1000', '1100', '1200', '1300', '1400',  # Water/air/other
}

with zipfile.ZipFile(input_path, 'r') as zf_in:
    files = [f for f in zf_in.namelist() if f not in EXCLUDE_FILES]
    print(f"\nProcessing {len(files)} files (excluding: {EXCLUDE_FILES})")

    # Load routes and filter unsupported types
    with zf_in.open('routes.txt') as f:
        routes = pd.read_csv(f, dtype=str)

    unsupported = ~routes['route_type'].isin(SUPPORTED_ROUTE_TYPES)
    print(f"Routes: {len(routes):,} total, {unsupported.sum():,} unsupported types removed")
    if unsupported.sum() > 0:
        print(f"  Removed types: {routes[unsupported]['route_type'].value_counts().to_dict()}")
    routes_fixed = routes[~unsupported]
    valid_route_ids = set(routes_fixed['route_id'])

    # Load trips and filter to valid routes
    with zf_in.open('trips.txt') as f:
        trips = pd.read_csv(f, dtype=str, low_memory=False)
    trips_valid_routes = trips[trips['route_id'].isin(valid_route_ids)]
    print(f"Trips: {len(trips):,} total, {len(trips_valid_routes):,} on valid routes")
    trips = trips_valid_routes

    all_services = trips['service_id'].dropna().unique()
    valid_trip_ids = set(trips['trip_id'])
    print(f"Services from trips: {len(all_services):,}")

    # Load stops to filter bad coordinates
    with zf_in.open('stops.txt') as f:
        stops = pd.read_csv(f, dtype=str)

    stops['stop_lat'] = pd.to_numeric(stops['stop_lat'], errors='coerce')
    stops['stop_lon'] = pd.to_numeric(stops['stop_lon'], errors='coerce')

    # Filter bad coordinates
    valid_coords = (
        (stops['stop_lat'] >= SWEDEN_LAT_MIN) & (stops['stop_lat'] <= SWEDEN_LAT_MAX) &
        (stops['stop_lon'] >= SWEDEN_LON_MIN) & (stops['stop_lon'] <= SWEDEN_LON_MAX)
    )
    bad_stops = stops[~valid_coords]
    good_stops = stops[valid_coords]

    print(f"Stops: {len(stops):,} total, {len(bad_stops):,} bad coords removed")

    # Get valid stop_ids
    valid_stop_ids = set(good_stops['stop_id'])

    # Create fixed calendar: all services run every day
    print(f"Calendar: creating fixed (all services run every day)")
    fixed_calendar = pd.DataFrame({
        'service_id': all_services,
        'monday': '1',
        'tuesday': '1',
        'wednesday': '1',
        'thursday': '1',
        'friday': '1',
        'saturday': '1',
        'sunday': '1',
        'start_date': '20240101',
        'end_date': '20271231',
    })

    # Write fixed GTFS
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf_out:
        for filename in files:
            print(f"  Processing: {filename}...", end=' ')

            if filename == 'stops.txt':
                # Write filtered stops
                csv_content = good_stops.to_csv(index=False)
                zf_out.writestr(filename, csv_content)
                print(f"{len(good_stops):,} rows")

            elif filename == 'calendar.txt':
                # Write fixed calendar (all services run every day)
                csv_content = fixed_calendar.to_csv(index=False)
                zf_out.writestr(filename, csv_content)
                print(f"{len(fixed_calendar):,} rows (FIXED: all days active)")

            elif filename == 'stop_times.txt':
                # Filter stop_times to valid stops AND valid trips
                with zf_in.open(filename) as f:
                    chunks = pd.read_csv(f, dtype=str, chunksize=500000, low_memory=False)
                    filtered_chunks = []
                    total_rows = 0
                    kept_rows = 0

                    for chunk in chunks:
                        total_rows += len(chunk)
                        valid_chunk = chunk[
                            chunk['stop_id'].isin(valid_stop_ids) &
                            chunk['trip_id'].isin(valid_trip_ids)
                        ]
                        kept_rows += len(valid_chunk)
                        filtered_chunks.append(valid_chunk)

                    filtered_st = pd.concat(filtered_chunks, ignore_index=True)

                csv_content = filtered_st.to_csv(index=False)
                zf_out.writestr(filename, csv_content)
                print(f"{kept_rows:,} / {total_rows:,} rows")

            elif filename == 'routes.txt':
                # Write filtered routes (unsupported types removed)
                csv_content = routes_fixed.to_csv(index=False)
                zf_out.writestr(filename, csv_content)
                print(f"{len(routes_fixed):,} rows")

            elif filename == 'trips.txt':
                # Remove shape_id references since we're excluding shapes
                trips_fixed = trips.copy()
                if 'shape_id' in trips_fixed.columns:
                    trips_fixed['shape_id'] = ''
                csv_content = trips_fixed.to_csv(index=False)
                zf_out.writestr(filename, csv_content)
                print(f"{len(trips_fixed):,} rows")

            else:
                # Copy other files as-is
                with zf_in.open(filename) as f:
                    content = f.read()
                zf_out.writestr(filename, content)

                # Count rows for txt files
                if filename.endswith('.txt'):
                    row_count = content.decode('utf-8').count('\n')
                    print(f"{row_count:,} rows")
                else:
                    print("copied")

print(f"\nOutput: {output_path}")
print(f"Size: {output_path.stat().st_size / 1e6:.1f} MB (was {input_path.stat().st_size / 1e6:.1f} MB)")
print(f"Reduction: {100 * (1 - output_path.stat().st_size / input_path.stat().st_size):.1f}%")

# Validation
print("\n" + "=" * 60)
print("VALIDATION")
print("=" * 60)

with zipfile.ZipFile(output_path, 'r') as zf:
    with zf.open('trips.txt') as f:
        trips_v = pd.read_csv(f, dtype=str, usecols=['service_id'])
    with zf.open('calendar.txt') as f:
        cal_v = pd.read_csv(f, dtype=str, usecols=['service_id'])

trip_services = set(trips_v['service_id'].dropna())
cal_services = set(cal_v['service_id'].dropna())
orphans = trip_services - cal_services

print(f"Trip services: {len(trip_services):,}")
print(f"Calendar services: {len(cal_services):,}")
print(f"Orphan services: {len(orphans)}")

if len(orphans) == 0:
    print("\n✓ All services have calendar entries!")
    print("✓ Sweden GTFS is ready for r5r!")
