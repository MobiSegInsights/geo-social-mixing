"""
Step 1.1.2: Assign POI to GPS Stops (Tiered Approach)

Multi-tier POI assignment using building footprints and distance thresholds
to ensure high-confidence matches for visitor diversity analysis.

Tiers:
  1. Inside building footprint (highest confidence) - uses POLYGON_WKT
  2. Within 20m of POI centroid (high confidence) - GPS uncertainty
  3. Within 50m of POI centroid (medium confidence)
  4. Within 100m of POI centroid (low confidence - flagged)
  >100m: No assignment (unmatched)

Usage:
    micromamba run -n geoenv python src/data/assign_poi_tiered.py

Input:
    - dbs/stops/*.parquet (GPS stops from infostop detection)
    - dbs/poi_se/POI_se_2026_03_05/*.parquet (SafeGraph POI data)
    - dbs/home_work/device_homes_deso_ipw.parquet (valid devices with home + building)

Output:
    - dbs/poi_assignment/stops_poi_assigned.parquet
      Columns: stop_id, device_aid, poi_id, poi_distance, confidence_tier,
               poi_category, poi_subcategory, assignment_method

Processing:
    - Filters to valid devices only (has_home=True AND has_building=True)
    - Excludes stops within 100m of device home locations
    - Processes in chunks for memory efficiency (file-by-file, then by chunk)
    - Writes results incrementally to avoid memory buildup
    - Sends Telegram notifications at key stages
"""

from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely import wkt
from shapely.geometry import Point
from shapely.strtree import STRtree
from scipy.spatial import cKDTree
from datetime import datetime
from tqdm import tqdm
import os
import requests
import gc

# Telegram credentials
TELEGRAM_TOKEN = os.getenv("TG_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TG_CHAT_ID")

def notify(msg: str):
    """Send a Telegram message if credentials exist."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=15)
    except Exception as e:
        print(f"Failed to notify: {e}")

# Paths
ROOT_DIR = Path(__file__).parent.parent.parent
STOPS_DIR = ROOT_DIR / "dbs" / "stops"
POI_DIR = ROOT_DIR / "dbs" / "poi_se" / "POI_se_2026_03_05"
HW_DIR = ROOT_DIR / "dbs" / "home_work"
OUTPUT_DIR = ROOT_DIR / "dbs" / "poi_assignment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Parameters
PROJ_CRS = 'EPSG:3006'  # SWEREF99 TM (Sweden)
TIER2_RADIUS = 20  # meters (high confidence)
TIER3_RADIUS = 50  # meters (medium confidence)
TIER4_RADIUS = 100  # meters (low confidence)
HOME_BUFFER = 100  # meters (exclude stops near home)
CHUNK_SIZE = 50_000  # Process stops in chunks (reduced for memory)

print("="*60)
print("Tiered POI Assignment for GPS Stops")
print("="*60)
print(f"\nTier 1: Building footprint intersection (highest confidence)")
print(f"Tier 2: Within {TIER2_RADIUS}m (high confidence - GPS uncertainty)")
print(f"Tier 3: Within {TIER3_RADIUS}m (medium confidence)")
print(f"Tier 4: Within {TIER4_RADIUS}m (low confidence)")
print(f"No match: >{TIER4_RADIUS}m")
print(f"\nChunk size: {CHUNK_SIZE:,} stops")

notify(f"🎯 POI assignment started at {datetime.now().isoformat()}\n"
       f"Tiers: Footprint | {TIER2_RADIUS}m | {TIER3_RADIUS}m | {TIER4_RADIUS}m")

# ============================================================================
# 1. Load POI Data (kept in memory - 658K is manageable)
# ============================================================================

print("\n1. Loading POI data...")
df_poi = pd.read_parquet(POI_DIR, engine='pyarrow')
print(f"   Loaded {len(df_poi):,} POIs")

# Filter to required columns and valid coordinates
poi_cols = ['PLACEKEY', 'LATITUDE', 'LONGITUDE', 'TOP_CATEGORY', 'SUB_CATEGORY',
            'LOCATION_NAME', 'POLYGON_WKT', 'CITY', 'NAICS_CODE']
df_poi = df_poi[poi_cols].copy()

# Remove POIs without coordinates or category
df_poi = df_poi[
    df_poi['LATITUDE'].notna() &
    df_poi['LONGITUDE'].notna() &
    df_poi['TOP_CATEGORY'].notna()
].copy()

print(f"   POIs with valid data: {len(df_poi):,}")
n_with_footprint = df_poi['POLYGON_WKT'].notna().sum()
print(f"   POIs with building footprint: {n_with_footprint:,} ({n_with_footprint/len(df_poi)*100:.1f}%)")

# Create GeoDataFrame with centroids
print("   Creating spatial index for centroids...")
gdf_poi = gpd.GeoDataFrame(
    df_poi,
    geometry=gpd.points_from_xy(df_poi['LONGITUDE'], df_poi['LATITUDE']),
    crs='EPSG:4326'
).to_crs(PROJ_CRS)

# Build KDTree for distance-based matching (Tiers 2-4)
poi_coords = np.column_stack([gdf_poi.geometry.x, gdf_poi.geometry.y])
poi_tree = cKDTree(poi_coords)
poi_ids = gdf_poi['PLACEKEY'].values
poi_categories = gdf_poi['TOP_CATEGORY'].values
poi_subcategories = gdf_poi['SUB_CATEGORY'].values

print(f"   ✓ KDTree built for {len(gdf_poi):,} POI centroids")

# Parse building footprints for Tier 1 (batch parsing)
print("\n   Parsing building footprints...")
df_poi_with_fp = df_poi[df_poi['POLYGON_WKT'].notna()].copy()

def batch_parse_wkt(wkt_series):
    """Parse WKT strings in batch with error handling."""
    results = []
    for wkt_str in wkt_series:
        try:
            geom = wkt.loads(wkt_str)
            results.append(geom)
        except:
            results.append(None)
    return results

footprint_geoms = batch_parse_wkt(df_poi_with_fp['POLYGON_WKT'].values)
df_poi_with_fp['footprint_geom'] = footprint_geoms

# Filter out failed parses
df_poi_with_fp = df_poi_with_fp[df_poi_with_fp['footprint_geom'].notna()].copy()

# Create GeoDataFrame with footprints
gdf_footprints = gpd.GeoDataFrame(
    df_poi_with_fp[['PLACEKEY', 'TOP_CATEGORY', 'SUB_CATEGORY']],
    geometry=df_poi_with_fp['footprint_geom'].values,
    crs='EPSG:4326'
).to_crs(PROJ_CRS)

# Build STRtree for efficient spatial queries (Tier 1)
footprint_tree = STRtree(gdf_footprints.geometry.values)
footprint_placekeys = gdf_footprints['PLACEKEY'].values
footprint_categories = gdf_footprints['TOP_CATEGORY'].values
footprint_subcategories = gdf_footprints['SUB_CATEGORY'].values

print(f"   ✓ STRtree built for {len(gdf_footprints):,} building footprints")

# Create centroid lookup for distance calculation
poi_centroid_lookup = dict(zip(gdf_poi['PLACEKEY'], gdf_poi.geometry))

notify(f"✅ Loaded {len(df_poi):,} POIs\n"
       f"  - {len(gdf_footprints):,} with footprints (STRtree)\n"
       f"  - KDTree for distance matching")

# ============================================================================
# 2. Load Valid Devices and Home Locations
# ============================================================================

print("\n2. Loading valid devices...")
hw_file = HW_DIR / "device_homes_deso_ipw.parquet"

if not hw_file.exists():
    print(f"   ERROR: Device homes file not found: {hw_file}")
    notify("⚠️ ERROR: Device homes file not found")
    exit(1)

df_hw = pd.read_parquet(hw_file)

# Focus on valid devices: has_home AND has_building
valid_devices = df_hw[df_hw['has_home'] & df_hw['has_building']].copy()
valid_device_set = set(valid_devices['device_aid'])
print(f"   Valid devices (home + building): {len(valid_devices):,}")

# Create home location lookup (projected coordinates)
print("   Creating home location lookup...")
home_gdf = gpd.GeoDataFrame(
    valid_devices[['device_aid', 'home_lat', 'home_lon']],
    geometry=gpd.points_from_xy(valid_devices['home_lon'], valid_devices['home_lat']),
    crs='EPSG:4326'
).to_crs(PROJ_CRS)

home_coords = dict(zip(
    home_gdf['device_aid'],
    zip(home_gdf.geometry.x, home_gdf.geometry.y)
))

print(f"   ✓ Home locations loaded for {len(home_coords):,} devices")

# Clean up
del df_hw, valid_devices, home_gdf
gc.collect()

# ============================================================================
# 3. Define Processing Functions
# ============================================================================

def process_stop_chunk(df_chunk, chunk_id):
    """
    Process a chunk of stops through all tiers.
    Returns DataFrame with POI assignments.
    """
    n_chunk = len(df_chunk)

    # Project stop coordinates
    gdf_chunk = gpd.GeoDataFrame(
        df_chunk,
        geometry=gpd.points_from_xy(df_chunk['lon'], df_chunk['lat']),
        crs='EPSG:4326'
    ).to_crs(PROJ_CRS)

    # Get projected coordinates
    stop_x = gdf_chunk.geometry.x.values
    stop_y = gdf_chunk.geometry.y.values
    stop_coords = np.column_stack([stop_x, stop_y])

    # Initialize result arrays
    result_poi_id = np.full(n_chunk, None, dtype=object)
    result_distance = np.full(n_chunk, np.nan)
    result_tier = np.zeros(n_chunk, dtype=int)
    result_category = np.full(n_chunk, None, dtype=object)
    result_subcategory = np.full(n_chunk, None, dtype=object)
    result_method = np.full(n_chunk, None, dtype=object)

    # Filter out home visits first
    is_home = np.zeros(n_chunk, dtype=bool)
    for i, (device, x, y) in enumerate(zip(df_chunk['device_aid'].values, stop_x, stop_y)):
        if device in home_coords:
            home_x, home_y = home_coords[device]
            dist_to_home = np.sqrt((x - home_x)**2 + (y - home_y)**2)
            if dist_to_home < HOME_BUFFER:
                is_home[i] = True

    # Process non-home stops
    non_home_mask = ~is_home
    non_home_indices = np.where(non_home_mask)[0]

    if len(non_home_indices) == 0:
        # All stops are home visits
        return pd.DataFrame({
            'stop_id': df_chunk['stop_id'].values,
            'device_aid': df_chunk['device_aid'].values,
            'poi_id': result_poi_id,
            'poi_distance': result_distance,
            'confidence_tier': result_tier,
            'poi_category': result_category,
            'poi_subcategory': result_subcategory,
            'assignment_method': result_method
        })

    # --- Tier 1: Footprint matching using STRtree ---
    unassigned_mask = non_home_mask.copy()

    for i in non_home_indices:
        if not unassigned_mask[i]:
            continue

        stop_point = gdf_chunk.geometry.iloc[i]

        # Query STRtree for potential matches
        candidate_indices = footprint_tree.query(stop_point)

        if len(candidate_indices) > 0:
            # Check actual containment
            for idx in candidate_indices:
                if gdf_footprints.geometry.iloc[idx].contains(stop_point):
                    # Found a match!
                    placekey = footprint_placekeys[idx]
                    centroid = poi_centroid_lookup.get(placekey)

                    result_poi_id[i] = placekey
                    result_distance[i] = stop_point.distance(centroid) if centroid else 0
                    result_tier[i] = 1
                    result_category[i] = footprint_categories[idx]
                    result_subcategory[i] = footprint_subcategories[idx]
                    result_method[i] = 'footprint'
                    unassigned_mask[i] = False
                    break

    # --- Tiers 2-4: Distance-based matching ---
    still_unassigned = np.where(unassigned_mask)[0]

    if len(still_unassigned) > 0:
        unassigned_coords = stop_coords[still_unassigned]

        # Query nearest POI
        dists, idxs = poi_tree.query(unassigned_coords, k=1, distance_upper_bound=TIER4_RADIUS)

        for j, i in enumerate(still_unassigned):
            if np.isfinite(dists[j]):
                dist = dists[j]
                poi_idx = idxs[j]

                result_poi_id[i] = poi_ids[poi_idx]
                result_distance[i] = dist
                result_category[i] = poi_categories[poi_idx]
                result_subcategory[i] = poi_subcategories[poi_idx]
                result_method[i] = 'distance'

                # Assign tier based on distance
                if dist <= TIER2_RADIUS:
                    result_tier[i] = 2
                elif dist <= TIER3_RADIUS:
                    result_tier[i] = 3
                else:
                    result_tier[i] = 4

    # Create result DataFrame
    return pd.DataFrame({
        'stop_id': df_chunk['stop_id'].values,
        'device_aid': df_chunk['device_aid'].values,
        'poi_id': result_poi_id,
        'poi_distance': result_distance,
        'confidence_tier': result_tier,
        'poi_category': result_category,
        'poi_subcategory': result_subcategory,
        'assignment_method': result_method
    })

# ============================================================================
# 4. Process Stop Files
# ============================================================================

print("\n3. Processing GPS stops...")

stop_files = sorted(STOPS_DIR.glob("*.parquet"))

if not stop_files:
    print(f"   ERROR: No parquet files found in {STOPS_DIR}")
    notify("⚠️ ERROR: No stop files found")
    exit(1)

print(f"   Found {len(stop_files)} stop file(s)")

# Statistics accumulators
total_stops = 0
total_valid_stops = 0
total_home_filtered = 0
tier_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

# Output file (write in append mode)
output_file = OUTPUT_DIR / "stops_poi_assigned.parquet"
all_results = []

# Process each stop file
for file_idx, stop_file in enumerate(stop_files):
    print(f"\n   Processing file {file_idx+1}/{len(stop_files)}: {stop_file.name}")

    # Load stop file
    df_stops_file = pd.read_parquet(stop_file)
    total_stops += len(df_stops_file)

    # Check column names and adapt
    if 'lat' not in df_stops_file.columns:
        if 'latitude' in df_stops_file.columns:
            df_stops_file = df_stops_file.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
        else:
            print(f"   WARNING: Cannot find lat/lon columns in {stop_file.name}")
            print(f"   Columns: {df_stops_file.columns.tolist()}")
            continue

    # Add stop_id if not present
    if 'stop_id' not in df_stops_file.columns:
        df_stops_file['stop_id'] = range(total_stops - len(df_stops_file), total_stops)

    # Filter to valid devices only
    df_stops_file = df_stops_file[df_stops_file['device_aid'].isin(valid_device_set)].copy()
    total_valid_stops += len(df_stops_file)

    if len(df_stops_file) == 0:
        print(f"     No valid device stops in this file")
        continue

    print(f"     Stops from valid devices: {len(df_stops_file):,}")

    # Process in chunks
    n_chunks = (len(df_stops_file) + CHUNK_SIZE - 1) // CHUNK_SIZE

    for chunk_idx in tqdm(range(n_chunks), desc=f"     Chunks", leave=False):
        start = chunk_idx * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, len(df_stops_file))
        df_chunk = df_stops_file.iloc[start:end].copy()

        # Process chunk
        result_chunk = process_stop_chunk(df_chunk, chunk_idx)
        all_results.append(result_chunk)

        # Update statistics
        for tier in range(5):
            tier_counts[tier] += (result_chunk['confidence_tier'] == tier).sum()

        # Memory cleanup
        del df_chunk, result_chunk

    # File-level cleanup
    del df_stops_file
    gc.collect()

    # Progress notification every 5 files
    if (file_idx + 1) % 5 == 0:
        notify(f"📊 Progress: {file_idx+1}/{len(stop_files)} files\n"
               f"Stops processed: {total_valid_stops:,}")

# ============================================================================
# 5. Combine and Save Results
# ============================================================================

print("\n4. Saving results...")

if not all_results:
    print("   ERROR: No results to save!")
    notify("⚠️ ERROR: No POI assignments generated")
    exit(1)

df_final = pd.concat(all_results, ignore_index=True)

# Save to parquet
df_final.to_parquet(output_file, index=False)
print(f"   ✓ Saved: {output_file}")
print(f"   File size: {output_file.stat().st_size / 1e6:.1f} MB")

# ============================================================================
# 6. Summary Statistics
# ============================================================================

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

total_home_filtered = tier_counts[0] - (len(df_final) - df_final['poi_id'].notna().sum())
# Recalculate tier counts from final data
tier_counts = {i: (df_final['confidence_tier'] == i).sum() for i in range(5)}

print(f"\nInput:")
print(f"  Total GPS stops (all files): {total_stops:,}")
print(f"  Stops from valid devices: {total_valid_stops:,}")
print(f"  Total POIs: {len(gdf_poi):,}")
print(f"  POIs with footprints: {len(gdf_footprints):,}")

print(f"\nFiltering:")
n_home = (df_final['confidence_tier'] == 0).sum() - (df_final['poi_id'].isna() & (df_final['confidence_tier'] == 0)).sum()
print(f"  Home visits excluded: (filtered during processing)")

print(f"\nAssignment Results:")
total_matched = (df_final['confidence_tier'] > 0).sum()
print(f"  Total assigned: {len(df_final):,}")
print(f"  Matched to POI: {total_matched:,} ({total_matched/len(df_final)*100:.1f}%)")
print(f"  Tier 1 (footprint): {tier_counts[1]:,} ({tier_counts[1]/len(df_final)*100:.1f}%)")
print(f"  Tier 2 (0-{TIER2_RADIUS}m): {tier_counts[2]:,} ({tier_counts[2]/len(df_final)*100:.1f}%)")
print(f"  Tier 3 ({TIER2_RADIUS}-{TIER3_RADIUS}m): {tier_counts[3]:,} ({tier_counts[3]/len(df_final)*100:.1f}%)")
print(f"  Tier 4 ({TIER3_RADIUS}-{TIER4_RADIUS}m): {tier_counts[4]:,} ({tier_counts[4]/len(df_final)*100:.1f}%)")
print(f"  Unmatched (>{TIER4_RADIUS}m or home): {tier_counts[0]:,} ({tier_counts[0]/len(df_final)*100:.1f}%)")

if total_matched > 0:
    matched = df_final[df_final['confidence_tier'] > 0]
    print(f"\nDistance Statistics (matched stops):")
    print(f"  Mean: {matched['poi_distance'].mean():.1f}m")
    print(f"  Median: {matched['poi_distance'].median():.1f}m")
    print(f"  Min: {matched['poi_distance'].min():.1f}m")
    print(f"  Max: {matched['poi_distance'].max():.1f}m")

    print(f"\nTop 10 POI Categories:")
    cat_counts = matched['poi_category'].value_counts().head(10)
    for cat, count in cat_counts.items():
        print(f"  {cat}: {count:,}")

print("\n" + "="*60)
print("✅ POI assignment complete!")
print("="*60)

notify(f"✅ POI assignment complete!\n\n"
       f"Processed: {total_valid_stops:,} stops\n"
       f"Matched: {total_matched:,} ({total_matched/len(df_final)*100:.1f}%)\n"
       f"  Tier 1: {tier_counts[1]:,}\n"
       f"  Tier 2: {tier_counts[2]:,}\n"
       f"  Tier 3: {tier_counts[3]:,}\n"
       f"  Tier 4: {tier_counts[4]:,}\n"
       f"  Unmatched: {tier_counts[0]:,}\n\n"
       f"Saved to: {output_file.name}")
