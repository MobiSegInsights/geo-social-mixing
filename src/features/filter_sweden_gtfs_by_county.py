#!/usr/bin/env python
"""Filter Sweden GTFS to county-specific bounds using DeSO zones + 20km buffer."""

import zipfile
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import argparse

# Buffer distance (must match notebook 14)
BUFFER_METERS = 20000  # 20 km

COUNTY_NAMES = {
    '01': 'Stockholm', '03': 'Uppsala', '04': 'Södermanland', '05': 'Östergötland',
    '06': 'Jönköping', '07': 'Kronoberg', '08': 'Kalmar', '09': 'Gotland',
    '10': 'Blekinge', '12': 'Skåne', '13': 'Halland', '14': 'Västra Götaland',
    '17': 'Värmland', '18': 'Örebro', '19': 'Västmanland',
}

def get_county_buffer(county_code: str, deso_path: Path) -> gpd.GeoDataFrame:
    """Get buffered county boundary from DeSO zones."""
    deso_gdf = gpd.read_file(deso_path)
    deso_gdf['county_code'] = deso_gdf['deso_code'].str[:2]

    # Get DeSO zones in this county
    county_deso = deso_gdf[deso_gdf['county_code'] == county_code]

    if len(county_deso) == 0:
        raise ValueError(f"No DeSO zones found for county {county_code}")

    # Create convex hull and buffer by 20km
    # DeSO is in EPSG:3006 (SWEREF99 TM, meters)
    county_hull = county_deso.union_all().convex_hull
    buffered_hull = county_hull.buffer(BUFFER_METERS)

    return gpd.GeoDataFrame(geometry=[buffered_hull], crs=deso_gdf.crs)


def filter_gtfs_for_county(input_path: Path, output_path: Path, county_code: str, deso_path: Path):
    """Filter GTFS to county bounds using DeSO + 20km buffer."""
    county_name = COUNTY_NAMES.get(county_code, county_code)
    print(f"Filtering for {county_name} (county {county_code})")
    print(f"Buffer: {BUFFER_METERS / 1000:.0f} km")

    # Get county buffer geometry
    print("Loading DeSO boundaries...")
    buffer_gdf = get_county_buffer(county_code, deso_path)
    buffer_wgs84 = buffer_gdf.to_crs('EPSG:4326')
    buffer_geom = buffer_wgs84.geometry.iloc[0]

    with zipfile.ZipFile(input_path, 'r') as zf_in:
        # Load stops and filter by buffer
        print("Loading and filtering stops...")
        with zf_in.open('stops.txt') as f:
            stops = pd.read_csv(f, dtype=str)

        stops['stop_lat'] = pd.to_numeric(stops['stop_lat'], errors='coerce')
        stops['stop_lon'] = pd.to_numeric(stops['stop_lon'], errors='coerce')

        # Create GeoDataFrame and check intersection with buffer
        stops_gdf = gpd.GeoDataFrame(
            stops,
            geometry=gpd.points_from_xy(stops['stop_lon'], stops['stop_lat']),
            crs='EPSG:4326'
        )
        in_buffer = stops_gdf.geometry.within(buffer_geom)
        county_stops = stops[in_buffer].copy()
        valid_stop_ids = set(county_stops['stop_id'])
        print(f"Stops: {len(stops):,} total -> {len(county_stops):,} in county buffer")

        # Load and filter stop_times
        print("Filtering stop_times...")
        with zf_in.open('stop_times.txt') as f:
            chunks = pd.read_csv(f, dtype=str, chunksize=500000, low_memory=False)
            filtered_chunks = []
            total = 0
            kept = 0
            for chunk in chunks:
                total += len(chunk)
                valid = chunk[chunk['stop_id'].isin(valid_stop_ids)]
                kept += len(valid)
                filtered_chunks.append(valid)
            stop_times = pd.concat(filtered_chunks, ignore_index=True)
        print(f"Stop_times: {total:,} -> {kept:,}")

        # Get trips that have stop_times in county
        valid_trip_ids = set(stop_times['trip_id'].dropna().unique())
        print(f"Trips with county stops: {len(valid_trip_ids):,}")

        # Load and filter trips
        with zf_in.open('trips.txt') as f:
            trips = pd.read_csv(f, dtype=str, low_memory=False)
        county_trips = trips[trips['trip_id'].isin(valid_trip_ids)].copy()
        if 'shape_id' in county_trips.columns:
            county_trips['shape_id'] = ''
        print(f"Trips: {len(trips):,} -> {len(county_trips):,}")

        # Get routes and services used
        valid_route_ids = set(county_trips['route_id'].dropna())
        valid_service_ids = set(county_trips['service_id'].dropna())

        # Load and filter routes
        with zf_in.open('routes.txt') as f:
            routes = pd.read_csv(f, dtype=str)
        county_routes = routes[routes['route_id'].isin(valid_route_ids)]
        print(f"Routes: {len(routes):,} -> {len(county_routes):,}")

        # Get agencies used
        valid_agency_ids = set(county_routes['agency_id'].dropna()) if 'agency_id' in county_routes.columns else set()

        # Load agency
        with zf_in.open('agency.txt') as f:
            agency = pd.read_csv(f, dtype=str)
        if valid_agency_ids:
            county_agency = agency[agency['agency_id'].isin(valid_agency_ids)]
        else:
            county_agency = agency
        print(f"Agencies: {len(agency):,} -> {len(county_agency):,}")

        # Create fixed calendar
        print(f"Creating fixed calendar for {len(valid_service_ids):,} services")
        calendar = pd.DataFrame({
            'service_id': list(valid_service_ids),
            'monday': '1', 'tuesday': '1', 'wednesday': '1',
            'thursday': '1', 'friday': '1', 'saturday': '1', 'sunday': '1',
            'start_date': '20240101', 'end_date': '20271231',
        })

        # Load feed_info
        feed_info = None
        if 'feed_info.txt' in zf_in.namelist():
            with zf_in.open('feed_info.txt') as f:
                feed_info = pd.read_csv(f, dtype=str)

        # Write filtered GTFS
        print(f"\nWriting to {output_path}")
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf_out:
            zf_out.writestr('agency.txt', county_agency.to_csv(index=False))
            zf_out.writestr('stops.txt', county_stops.to_csv(index=False))
            zf_out.writestr('routes.txt', county_routes.to_csv(index=False))
            zf_out.writestr('trips.txt', county_trips.to_csv(index=False))
            zf_out.writestr('stop_times.txt', stop_times.to_csv(index=False))
            zf_out.writestr('calendar.txt', calendar.to_csv(index=False))
            if feed_info is not None:
                zf_out.writestr('feed_info.txt', feed_info.to_csv(index=False))

        print(f"Output size: {output_path.stat().st_size / 1e6:.1f} MB")
        print(f"\nDone! GTFS filtered for {county_name} using DeSO + {BUFFER_METERS/1000:.0f}km buffer")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--county', required=True, help='County code (e.g., 01, 12)')
    parser.add_argument('--input', default='dbs/gtfs_backup/sweden/sweden_fixed.zip')
    parser.add_argument('--output', help='Output path (default: dbs/gtfs/sweden_south/c_{county}/sweden_fixed.zip)')
    parser.add_argument('--deso', default='dbs/deso/deso_harmonized_2024.gpkg', help='Path to DeSO GeoPackage')
    args = parser.parse_args()

    # Normalize county code (remove c_ prefix if present)
    county_code = args.county.replace('c_', '').replace('c', '')

    if county_code not in COUNTY_NAMES:
        print(f"Unknown county: {county_code}")
        print(f"Available: {list(COUNTY_NAMES.keys())}")
        exit(1)

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else Path(f'dbs/gtfs/sweden_south/c_{county_code}/sweden_fixed.zip')
    deso_path = Path(args.deso)

    if not deso_path.exists():
        print(f"DeSO file not found: {deso_path}")
        exit(1)

    filter_gtfs_for_county(input_path, output_path, county_code, deso_path)
