"""
Step 1.1.1c: Link Home Locations to Residential Buildings

Uses Overture Maps building data to link detected home locations
to precise residential buildings. Improves home location accuracy
and provides quality control.

Usage:
    python link_home_buildings.py

Input:
    - dbs/home_work/home_work_locations.parquet
    - Overture Maps buildings (downloaded on-the-fly)

Output:
    - dbs/home_work/home_work_buildings.parquet
      Adds: building_id, building_class, building_source, has_building
"""

from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely import wkb
import overturemaps
from dotenv import load_dotenv
load_dotenv()
import os
import requests
import traceback
from datetime import datetime

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
HW_DIR = ROOT_DIR / "dbs" / "home_work"
BUILDINGS_DIR = ROOT_DIR / "dbs" / "buildings"
BUILDINGS_DIR.mkdir(parents=True, exist_ok=True)

# Sweden bounding box (lon_min, lat_min, lon_max, lat_max)
SWEDEN_BBOX = (11.0273686052, 55.3617373725, 23.9033785336, 69.1062472602)

# Buffer distance in meters (for GPS uncertainty)
BUFFER_DISTANCE = 50

print("="*60)
print("Link Home Locations to Residential Buildings")
print("="*60)

notify(f"🏠 Building linkage started at {datetime.now().isoformat()}")

# ============================================================================
# 1. Load Home/Work Locations
# ============================================================================

print("\n1. Loading home/work locations...")
hw_file = HW_DIR / "home_work_locations.parquet"
df_hw = pd.read_parquet(hw_file)

print(f"   Loaded {len(df_hw):,} devices")
print(f"   Devices with home: {df_hw['has_home'].sum():,} ({df_hw['has_home'].mean()*100:.1f}%)")

# Filter to devices with home
df_homes = df_hw[df_hw['has_home']].copy()
print(f"   Processing {len(df_homes):,} devices with home")

# ============================================================================
# 2. Download or Load Building Data
# ============================================================================

buildings_file = BUILDINGS_DIR / "sweden_residential_buildings.parquet"

if buildings_file.exists():
    print(f"\n2. Loading cached building data from {buildings_file}...")
    gdf_buildings = gpd.read_parquet(buildings_file)
    print(f"   Loaded {len(gdf_buildings):,} residential buildings")
else:
    print("\n2. Downloading building data from Overture Maps...")
    print("   Downloading in regional chunks to avoid timeouts...")
    print(f"   Full bounding box: {SWEDEN_BBOX}")

    # Split Sweden into 3 regions to avoid network timeouts
    # South (Skåne, Göteborg area): lat 55-60
    # Central (Stockholm area): lat 60-65
    # North (Norrland): lat 65-69.1
    lon_min, lat_min, lon_max, lat_max = SWEDEN_BBOX

    regions = [
        ("South", (lon_min, 55.0, lon_max, 60.0)),
        ("Central", (lon_min, 60.0, lon_max, 65.0)),
        ("North", (lon_min, 65.0, lon_max, lat_max))
    ]

    dfs = []
    for region_name, bbox in regions:
        print(f"\n   Downloading {region_name} Sweden ({bbox})...")
        try:
            table = overturemaps.record_batch_reader("building", bbox).read_all()
            table = table.combine_chunks()
            df_region = table.to_pandas()
            print(f"     ✓ {region_name}: {len(df_region):,} buildings")
            dfs.append(df_region)
            notify(f"✅ Downloaded {region_name} Sweden: {len(df_region):,} buildings")
        except Exception as e:
            print(f"     ✗ Failed to download {region_name}: {e}")
            notify(f"⚠️ Failed to download {region_name} Sweden: {e}")
            raise

    # Combine all regions
    print(f"\n   Combining {len(dfs)} regions...")
    df = pd.concat(dfs, ignore_index=True)
    print(f"   Downloaded {len(df):,} total buildings")

    # Convert to GeoDataFrame
    gdf_buildings = gpd.GeoDataFrame(
        df,
        geometry=df['geometry'].apply(wkb.loads),
        crs="EPSG:4326"
    )

    # Extract relevant columns
    gdf_buildings = gdf_buildings[['id', 'geometry', 'sources', 'level',
                                   'subtype', 'class', 'height', 'names']]

    # Extract source safely
    def extract_source(sources):
        if sources is None:
            return None
        try:
            if isinstance(sources, list) and len(sources) > 0:
                return sources[0].get('dataset', None)
            elif hasattr(sources, '__len__') and len(sources) > 0:
                return sources[0].get('dataset', None)
        except:
            return None
        return None

    gdf_buildings['source'] = gdf_buildings['sources'].apply(extract_source)
    gdf_buildings = gdf_buildings[['id', 'source', 'level', 'subtype',
                                   'class', 'height', 'names', 'geometry']]

    # Check subtype distribution
    print(f"\n   Building subtypes:")
    subtype_dist = gdf_buildings['subtype'].value_counts(normalize=True)
    for subtype, pct in subtype_dist.head(10).items():
        print(f"     {subtype}: {pct*100:.1f}%")

    # Filter to residential buildings
    gdf_buildings = gdf_buildings[gdf_buildings['subtype'] == 'residential'].copy()
    print(f"\n   Filtered to {len(gdf_buildings):,} residential buildings ({len(gdf_buildings)/len(df)*100:.1f}%)")

    # Save for future use
    print(f"   Caching to {buildings_file}...")
    gdf_buildings.to_parquet(buildings_file)
    notify(f"✅ Downloaded and cached {len(gdf_buildings):,} residential buildings")

# ============================================================================
# 3. Buffer Buildings
# ============================================================================

print(f"\n3. Buffering buildings by {BUFFER_DISTANCE}m...")

# Transform to projected CRS (SWEREF99 TM) for accurate buffering
gdf_buildings = gdf_buildings.to_crs('EPSG:3006')
gdf_buildings['geometry'] = gdf_buildings['geometry'].buffer(BUFFER_DISTANCE)
gdf_buildings = gdf_buildings.to_crs('EPSG:4326')

print(f"   Buffered {len(gdf_buildings):,} buildings")

# ============================================================================
# 4. Convert Homes to GeoDataFrame
# ============================================================================

print("\n4. Converting home locations to GeoDataFrame...")

gdf_homes = gpd.GeoDataFrame(
    df_homes,
    geometry=gpd.points_from_xy(df_homes['home_lon'], df_homes['home_lat']),
    crs='EPSG:4326'
)

print(f"   Created GeoDataFrame with {len(gdf_homes):,} home locations")

# ============================================================================
# 5. Spatial Join
# ============================================================================

print("\n5. Performing spatial join (homes → buildings)...")
print("   This may take 5-10 minutes...")

# Spatial join with 'intersects' predicate
gdf_joined = gpd.sjoin(
    gdf_homes,
    gdf_buildings[['id', 'class', 'source', 'geometry']],
    how='left',
    predicate='intersects'
)

# Handle multiple matches (take first match)
gdf_joined = gdf_joined.drop_duplicates(subset=['device_aid'], keep='first')

print(f"   Join complete: {len(gdf_joined):,} devices")
notify(f"✅ Spatial join complete: {len(gdf_joined):,} devices processed")

# ============================================================================
# 6. Calculate Match Statistics
# ============================================================================

print("\n6. Analyzing match results...")

# Count matches
n_matched = gdf_joined['id'].notna().sum()
n_total = len(gdf_joined)
match_rate = n_matched / n_total * 100

print(f"\n   Match Statistics:")
print(f"     Total homes: {n_total:,}")
print(f"     Matched to buildings: {n_matched:,} ({match_rate:.1f}%)")
print(f"     Unmatched: {n_total - n_matched:,} ({100-match_rate:.1f}%)")

if n_matched > 0:
    print(f"\n   Building sources for matched homes:")
    source_dist = gdf_joined[gdf_joined['id'].notna()]['source'].value_counts()
    for source, count in source_dist.items():
        print(f"     {source}: {count:,} ({count/n_matched*100:.1f}%)")

    print(f"\n   Building classes for matched homes (top 10):")
    class_dist = gdf_joined[gdf_joined['id'].notna()]['class'].value_counts()
    for cls, count in class_dist.head(10).items():
        print(f"     {cls}: {count:,} ({count/n_matched*100:.1f}%)")

# ============================================================================
# 7. Merge Back with All Devices
# ============================================================================

print("\n7. Merging back with all devices...")

# Rename building columns
gdf_joined = gdf_joined.rename(columns={
    'id': 'building_id',
    'class': 'building_class',
    'source': 'building_source'
})

# Drop geometry and index columns
result = pd.DataFrame(gdf_joined.drop(columns=['geometry', 'index_right'], errors='ignore'))

# Add building match flag
result['has_building'] = result['building_id'].notna()

# Merge with devices without home (they won't have building matches)
df_no_home = df_hw[~df_hw['has_home']].copy()
df_no_home['building_id'] = None
df_no_home['building_class'] = None
df_no_home['building_source'] = None
df_no_home['has_building'] = False

# Combine
result = pd.concat([result, df_no_home], ignore_index=True)

print(f"   Total devices in output: {len(result):,}")

# ============================================================================
# 8. Reorder Columns and Save
# ============================================================================

print("\n8. Saving output...")

# Reorder columns
cols_order = [
    'device_aid',
    # Home
    'home_loc', 'home_lat', 'home_lon',
    'building_id', 'building_class', 'building_source', 'has_building',
    'has_home',
    # Work
    'work_loc', 'work_lat', 'work_lon',
    'has_work'
]
cols_order = [c for c in cols_order if c in result.columns]
result = result[cols_order]

# Save
output_file = HW_DIR / "home_work_buildings.parquet"
result.to_parquet(output_file, index=False)

print(f"   ✓ Saved: {output_file}")

# ============================================================================
# 9. Summary
# ============================================================================

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

print(f"\nDataset Overview:")
print(f"  Total devices: {len(result):,}")
print(f"  Devices with home: {result['has_home'].sum():,} ({result['has_home'].mean()*100:.1f}%)")
print(f"  Devices with building match: {result['has_building'].sum():,} ({result['has_building'].mean()*100:.1f}%)")
print(f"  Devices with work: {result['has_work'].sum():,} ({result['has_work'].mean()*100:.1f}%)")

print(f"\nMatch Rate (among devices with home):")
homes_only = result[result['has_home']]
print(f"  Building match rate: {homes_only['has_building'].mean()*100:.1f}%")

print(f"\nSample Data:")
print(result[result['has_building']].head(3))

print("\n" + "="*60)
print("✅ Building linkage complete!")
print("="*60)
print("\nNext steps:")
print("  1. Assign DeSO zones to home locations (with IPW)")
print("  2. POI matching")
print("  3. Aggregate to tract-to-POI flows")

notify(f"✅ Building linkage complete!\n"
       f"  - Total devices: {len(result):,}\n"
       f"  - Building match rate: {result['has_building'].mean()*100:.1f}%\n"
       f"  - Saved to: home_work_buildings.parquet")
