"""
Filter US Foot Traffic Data to Study Cities (Low-memory Pandas + Subprocess)

Uses one subprocess per batch and processes files one by one inside each worker
to reduce peak memory usage.

Usage:
    python src/data/filter_us_cities_simple.py
    python src/data/filter_us_cities_simple.py --batch-size 20
"""

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from pathlib import Path
import argparse
import sys
import time
import os
import requests
import subprocess
import json
import gc

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

TELEGRAM_TOKEN = os.getenv("TG_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TG_CHAT_ID")


def notify(msg: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID.strip(), "text": msg},
            timeout=15,
        )
    except Exception:
        pass


MSA_COUNTIES = {
    'new_york': {
        'msa_code': '35620',
        'msa_name': 'New York-Newark-Jersey City, NY-NJ-PA',
        'type': 'transit',
        'counties': [
            '36005', '36047', '36061', '36081', '36085', '36059', '36103',
            '36119', '36087', '36079', '34003', '34013', '34017', '34019',
            '34023', '34025', '34027', '34029', '34031', '34035', '34037',
            '34039', '42103',
        ]
    },
    'chicago': {
        'msa_code': '16980',
        'msa_name': 'Chicago-Naperville-Elgin, IL-IN-WI',
        'type': 'transit',
        'counties': [
            '17031', '17043', '17089', '17093', '17097', '17111', '17197',
            '17063', '18073', '18089', '18111', '18127', '55059',
        ]
    },
    'washington_dc': {
        'msa_code': '47900',
        'msa_name': 'Washington-Arlington-Alexandria, DC-VA-MD-WV',
        'type': 'transit',
        'counties': [
            '11001', '51013', '51059', '51107', '51153', '51510', '51600',
            '51610', '51683', '51685', '51061', '51177', '51179', '51187',
            '24009', '24017', '24021', '24031', '24033', '54037',
        ]
    },
    'houston': {
        'msa_code': '26420',
        'msa_name': 'Houston-The Woodlands-Sugar Land, TX',
        'type': 'car',
        'counties': [
            '48015', '48039', '48071', '48157', '48167', '48201', '48291',
            '48339', '48473',
        ]
    },
    'atlanta': {
        'msa_code': '12060',
        'msa_name': 'Atlanta-Sandy Springs-Alpharetta, GA',
        'type': 'car',
        'counties': [
            '13013', '13015', '13035', '13045', '13057', '13063', '13067',
            '13077', '13085', '13089', '13097', '13113', '13117', '13121',
            '13135', '13143', '13149', '13151', '13159', '13171', '13199',
            '13211', '13217', '13223', '13227', '13231', '13247', '13255',
            '13297',
        ]
    },
    'phoenix': {
        'msa_code': '38060',
        'msa_name': 'Phoenix-Mesa-Chandler, AZ',
        'type': 'car',
        'counties': ['04013', '04021'],
    }
}

ALL_COUNTIES = set()
COUNTY_TO_CITY = {}
for city, info in MSA_COUNTIES.items():
    for county in info['counties']:
        ALL_COUNTIES.add(county)
        COUNTY_TO_CITY[county] = city

from src.data.category_mapper import CategoryMapper
CATEGORY_MAPPING = CategoryMapper.CATEGORY_MAPPING


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input-dir',
        type=str,
        default=str(ROOT_DIR / 'dbs/us_foot_traffic/weekly_patterns')
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(ROOT_DIR / 'dbs/us_foot_traffic/cities')
    )
    parser.add_argument('--test', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=10)

    # Worker mode arguments
    parser.add_argument('--worker', action='store_true', help='Run as worker subprocess')
    parser.add_argument('--batch-idx', type=int, default=-1)
    parser.add_argument('--file-list', type=str, default='')

    return parser.parse_args()


def process_batch_worker(file_list: list, output_dir: Path, batch_idx: int) -> dict:
    """
    Worker function: process one file at a time for lower peak memory.
    Accumulates per-city data only within a batch, then writes one parquet per city per batch.
    """

    needed_cols = [
        "POI_CBG",
        "SUB_CATEGORY",
        "VISITOR_HOME_CBGS",
        "ID_STORE",
        "VISIT_COUNTS",
        "DATE_RANGE_START",
        "DATE_RANGE_END",
        "LATITUDE",
        "LONGITUDE",
    ]

    total = 0
    per_city = {}
    city_chunks = {city: [] for city in MSA_COUNTIES}

    for f in file_list:
        try:
            df = pd.read_parquet(f, columns=needed_cols)
        except Exception as e:
            print(f"WARNING: failed to read {f}: {e}", file=sys.stderr, flush=True)
            continue

        if df.empty:
            del df
            gc.collect()
            continue

        # Robust county extraction
        poi_cbg = pd.to_numeric(df["POI_CBG"], errors="coerce")
        county_fips = (
            poi_cbg.astype("Int64")
            .astype("string")
            .str.zfill(12)
            .str[:5]
        )

        mask = county_fips.isin(ALL_COUNTIES)
        if not mask.any():
            del df, poi_cbg, county_fips, mask
            gc.collect()
            continue

        df = df.loc[mask].copy()
        df["county_fips"] = county_fips.loc[mask].values
        df["study_city"] = df["county_fips"].map(COUNTY_TO_CITY)

        if "SUB_CATEGORY" in df.columns:
            df["SUB_CATEGORY"] = df["SUB_CATEGORY"].astype("string").str.strip()
            df["unified_category"] = df["SUB_CATEGORY"].map(CATEGORY_MAPPING)
        else:
            df["unified_category"] = pd.NA

        df = df.drop(columns=["county_fips"])

        matched = len(df)
        total += matched

        for city_name, df_city in df.groupby("study_city", sort=False):
            city_chunks[city_name].append(df_city.copy())
            per_city[city_name] = per_city.get(city_name, 0) + len(df_city)

        del df, poi_cbg, county_fips, mask
        gc.collect()

    # Write one file per city per batch
    for city_name, chunks in city_chunks.items():
        if not chunks:
            continue
        df_city_all = pd.concat(chunks, ignore_index=True)
        out_file = output_dir / city_name / f"part_{batch_idx:04d}.parquet"
        df_city_all.to_parquet(out_file, index=False, compression="snappy")
        del df_city_all
        gc.collect()

    del city_chunks
    gc.collect()

    return {"total": total, "per_city": per_city}


def run_worker_mode(args):
    output_dir = Path(args.output_dir)
    file_list = json.loads(args.file_list)

    result = process_batch_worker(file_list, output_dir, args.batch_idx)

    # stdout reserved for machine-readable JSON only
    print(json.dumps(result))


def compute_city_stats_streaming(city_dir: Path, city_name: str, city_info: dict) -> dict | None:
    """Compute city stats file by file to avoid concatenating all parts."""

    part_files = sorted(city_dir.glob("part_*.parquet"))
    if not part_files:
        return None

    needed_stats_cols = [
        "VISITOR_HOME_CBGS",
        "ID_STORE",
        "VISIT_COUNTS",
        "DATE_RANGE_START",
        "DATE_RANGE_END",
        "unified_category",
    ]

    total_records = 0
    total_visits = 0
    cbgs_valid = 0
    unified_nonnull = 0
    unique_pois = set()
    weeks = set()
    min_start = None
    max_end = None

    for f in part_files:
        try:
            df_part = pd.read_parquet(f, columns=needed_stats_cols)
        except Exception as e:
            print(f"WARNING: failed to read stats file {f}: {e}", flush=True)
            continue

        if df_part.empty:
            del df_part
            gc.collect()
            continue

        total_records += len(df_part)
        total_visits += int(df_part["VISIT_COUNTS"].sum())

        cbgs_valid += (
            df_part["VISITOR_HOME_CBGS"].notna()
            & (df_part["VISITOR_HOME_CBGS"] != "")
            & (df_part["VISITOR_HOME_CBGS"] != "{}")
        ).sum()

        unified_nonnull += df_part["unified_category"].notna().sum()

        unique_pois.update(df_part["ID_STORE"].dropna().unique().tolist())
        weeks.update(df_part["DATE_RANGE_START"].dropna().unique().tolist())

        part_min = df_part["DATE_RANGE_START"].min()
        part_max = df_part["DATE_RANGE_END"].max()

        if min_start is None or part_min < min_start:
            min_start = part_min
        if max_end is None or part_max > max_end:
            max_end = part_max

        del df_part
        gc.collect()

    if total_records == 0:
        return None

    return {
        'city': city_name,
        'msa_code': city_info['msa_code'],
        'msa_name': city_info['msa_name'],
        'city_type': city_info['type'],
        'total_records': total_records,
        'unique_pois': len(unique_pois),
        'total_visits': total_visits,
        'weeks': len(weeks),
        'date_start': str(min_start),
        'date_end': str(max_end),
        'cbgs_coverage_pct': round(cbgs_valid / total_records * 100, 1),
        'unified_category_coverage_pct': round(unified_nonnull / total_records * 100, 1),
    }


def run_orchestrator_mode(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for city in MSA_COUNTIES:
        (output_dir / city).mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("FILTER US FOOT TRAFFIC (Pandas + Subprocess)", flush=True)
    print("=" * 70, flush=True)

    start_time = time.time()
    notify("🚀 US City Filtering Started")

    parquet_files = sorted(input_dir.glob("*.parquet"))
    total_files = len(parquet_files)
    print(f"\nFound {total_files} files", flush=True)

    if args.test > 0:
        parquet_files = parquet_files[:args.test]
        print(f"TEST MODE: {len(parquet_files)} files", flush=True)

    batch_size = args.batch_size
    total_batches = (len(parquet_files) + batch_size - 1) // batch_size
    print(f"\n1. Processing {total_batches} batches of {batch_size} files...", flush=True)

    total_matched = 0
    city_totals = {city: 0 for city in MSA_COUNTIES}

    for batch_idx in range(total_batches):
        batch_start = time.time()

        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(parquet_files))
        batch_files = [str(f) for f in parquet_files[start_idx:end_idx]]

        result = subprocess.run(
            [
                sys.executable, __file__,
                '--worker',
                '--batch-idx', str(batch_idx),
                '--file-list', json.dumps(batch_files),
                '--output-dir', str(output_dir),
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        try:
            batch_result = json.loads(result.stdout.strip())
            matched = batch_result['total']
            total_matched += matched
            for city, count in batch_result.get('per_city', {}).items():
                city_totals[city] += count
        except json.JSONDecodeError:
            print(f"WARNING: could not parse batch {batch_idx} output", flush=True)
            if result.stdout:
                print("Worker stdout preview:", result.stdout[:500], flush=True)
            if result.stderr:
                print("Worker stderr preview:", result.stderr[:500], flush=True)
            matched = 0

        batch_time = time.time() - batch_start
        files_done = end_idx
        elapsed = time.time() - start_time
        rate = files_done / elapsed * 60 if elapsed > 0 else 0
        eta = (len(parquet_files) - files_done) / rate if rate > 0 else 0

        print(
            f"Batch {batch_idx+1}/{total_batches}: "
            f"{matched:,} records, {batch_time:.1f}s "
            f"(files {start_idx+1}-{end_idx}, ETA: {eta:.0f} min)",
            flush=True
        )

        if (batch_idx + 1) % 5 == 0:
            notify(
                f"📦 Batch {batch_idx+1}/{total_batches}\n"
                f"{total_matched:,} records\n"
                f"{elapsed/60:.1f} min"
            )

    print("\n2. Computing statistics...", flush=True)

    city_stats = []

    for city_name, city_info in MSA_COUNTIES.items():
        city_dir = output_dir / city_name
        stats = compute_city_stats_streaming(city_dir, city_name, city_info)

        if stats is None:
            print(f"  {city_name}: No data", flush=True)
            continue

        city_stats.append(stats)

        part_files = sorted(city_dir.glob("part_*.parquet"))
        size_mb = sum(f.stat().st_size for f in part_files) / 1e6

        print(
            f"  {city_name}: {stats['total_records']:,} records, "
            f"{stats['unique_pois']:,} POIs, {size_mb:.1f} MB",
            flush=True
        )

    print("\n3. Saving statistics...", flush=True)

    df_stats = pd.DataFrame(city_stats)
    df_stats.to_csv(output_dir / 'city_statistics.csv', index=False)

    if not df_stats.empty:
        print(
            df_stats[['city', 'city_type', 'total_records', 'unique_pois', 'total_visits']]
            .to_string(index=False),
            flush=True
        )

    total_time = time.time() - start_time
    total_records = sum(s['total_records'] for s in city_stats)
    total_visits = sum(s['total_visits'] for s in city_stats)

    print(f"\n{'='*70}\nCOMPLETE\n{'='*70}", flush=True)
    print(
        f"Cities: {len(city_stats)}, Records: {total_records:,}, Visits: {total_visits:,}",
        flush=True
    )
    print(f"Time: {total_time/60:.1f} minutes", flush=True)

    notify(
        f"🎉 Complete!\n"
        f"{len(city_stats)} cities\n"
        f"{total_records:,} records\n"
        f"{total_time/60:.1f} min"
    )


def main(args):
    if args.worker:
        run_worker_mode(args)
    else:
        run_orchestrator_mode(args)


if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
    except Exception as e:
        import traceback
        traceback.print_exc()
        if not args.worker:
            notify(f"❌ CRASHED: {str(e)[:200]}")
        raise