"""
Step 1.1.1b: Home & Work Detection for Swedish GPS Data

This script applies the HoWDe algorithm to classify detected stops as Home/Work/Other.
It processes stop data from step 1.1.1a (stop_detection.py).

Usage:
    python home_work_detection.py --batch 0          # Process single batch
    python home_work_detection.py --batch 0 10       # Process batches 0-9
    python home_work_detection.py --all              # Process all batches (0-49)
    python home_work_detection.py --merge            # Merge all outputs

Output:
    dbs/home_work/hw_{batch}.parquet
    Columns: device_aid, loc, start, end, location_type (H/W/O),
             detect_H_loc, detect_W_loc, latitude, longitude

References:
    - HoWDe: https://pypi.org/project/HoWDe/
"""

from pathlib import Path
from attrs import inspect
import pandas as pd
import numpy as np
import time
import os
import sys
import argparse
import traceback
from datetime import datetime

# PySpark imports
from pyspark.sql import SparkSession
from pyspark import SparkConf
import pyspark.sql.functions as F
from pyspark.sql.types import LongType

# Message handling
from dotenv import load_dotenv
load_dotenv()

import os, requests, traceback
from datetime import datetime

TELEGRAM_TOKEN = os.getenv("TG_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TG_CHAT_ID")

def notify(msg: str):
    """Send a Telegram message if credentials exist."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[notify skipped] Missing TG_BOT_TOKEN or TG_CHAT_ID")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=15)
    except Exception as e:
        print("Failed to notify:", e)

# HoWDe
import howde

# ============================================================================
# Configuration
# ============================================================================

# Paths
ROOT_DIR = Path(__file__).parent.parent.parent
STOPS_DIR = ROOT_DIR / "dbs" / "stops"
OUTPUT_DIR = ROOT_DIR / "dbs" / "home_work"

# Number of groups
NUM_GROUPS = 50

# HoWDe Configuration for Sweden
HOWDE_CONFIG = {
    # Time settings - Swedish GPS data has localtime already converted
    "is_time_local": True,
    "country": "Sweden",

    # Home hours definition (evening/night when people are at home)
    "start_hour_day": 6,      # Home hours: before 6am and after end_hour_day
    "end_hour_day": 24,       # Home hours: basically evening/night

    # Work hours definition (typical office hours)
    "start_hour_work": 8,     # Work hours start
    "end_hour_work": 18,      # Work hours end

    # Use full historical data (non-causal mode for better accuracy)
    "data_for_predict": False,
}

# HoWDe algorithm parameters
HOWDE_PARAMS = {
    # Temporal windows (days)
    "range_window_home": 14,   # Days to consider for home pattern
    "range_window_work": 28,   # Days to consider for work pattern

    # Density thresholds (higher = stricter detection)
    "dhn": 1,                  # Density multiplier for home
    "dn_H": 0.8,               # Density threshold for home (selected from grid search)
    "dn_W": 0.7,               # Density threshold for work

    # Frequency thresholds
    "hf_H": 0.7,               # Frequency threshold for home visits
    "hf_W": 0.4,               # Frequency threshold for work visits
    "df_W": 0.4,               # Default frequency for work

    # Output settings
    "output_format": "stop",   # Return stop-level labels
    "verbose": True,
}


# ============================================================================
# Spark Setup
# ============================================================================

def init_spark(app_name="HomeWorkDetection", driver_memory="56g", cores=18):
    """Initialize Spark session with optimized configuration."""
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    spark_conf = SparkConf().setMaster(f"local[{cores}]").setAppName(app_name)
    spark_conf.set("spark.executor.heartbeatInterval", "3600s")
    spark_conf.set("spark.network.timeout", "7200s")
    spark_conf.set("spark.sql.files.ignoreCorruptFiles", "true")
    spark_conf.set("spark.driver.memory", driver_memory)
    spark_conf.set("spark.driver.maxResultSize", "0")
    spark_conf.set("spark.executor.memory", "8g")
    spark_conf.set("spark.memory.fraction", "0.6")
    spark_conf.set("spark.sql.session.timeZone", "UTC")

    spark = SparkSession.builder.config(conf=spark_conf).getOrCreate()

    java_version = spark._jvm.System.getProperty("java.version")
    print(f"Java version: {java_version}")
    print(f"Spark Web UI: {spark.sparkContext.uiWebUrl}")

    return spark


# ============================================================================
# Home/Work Detection Class
# ============================================================================

class HomeWorkDetection:
    """
    Home/Work detection pipeline for Swedish GPS data.

    Processes stop data in batches, applying HoWDe algorithm
    to classify stops as Home/Work/Other.
    """

    def __init__(self, spark=None, stops_dir=STOPS_DIR, output_dir=OUTPUT_DIR):
        self.spark = spark or init_spark()
        self.stops_dir = Path(stops_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_stops(self, batch):
        """
        Load stop data for a batch and prepare for HoWDe.

        HoWDe expects columns: useruuid, loc, start, end
        Our stop data has: device_aid, loc, localtime, l_localtime
        """
        stop_file = self.stops_dir / f"stops_{batch}.parquet"
        if not stop_file.exists():
            raise FileNotFoundError(f"Stop file not found: {stop_file}")

        print(f"Loading stops from {stop_file}...")
        df = self.spark.read.parquet(str(stop_file))

        # Check required columns
        required_cols = ['device_aid', 'loc', 'localtime', 'l_localtime',
                         'latitude', 'longitude']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Prepare input for HoWDe:
        # - Rename device_aid -> useruuid
        # - Rename localtime -> start (local time epoch)
        # - Rename l_localtime -> end (local time epoch)
        # - Keep latitude/longitude for later joining
        # input_data = (df
        #     .withColumnRenamed('device_aid', 'useruuid')
        #     .withColumnRenamed('localtime', 'start')
        #     .withColumnRenamed('l_localtime', 'end')
        # )
        input_data = (
            df
            .drop("start", "end")  # drop old ones if present
            .withColumnRenamed("device_aid", "useruuid")
            .withColumn("start", F.col("localtime"))
            .withColumn("end",   F.col("l_localtime"))
            .drop("localtime", "l_localtime")
        )
        # Repartition by user for efficient groupBy operations
        input_data = input_data.repartition('useruuid')

        n_stops = df.count()
        n_users = df.select('device_aid').distinct().count()
        print(f"Loaded {n_stops:,} stops from {n_users:,} devices")

        return input_data

    def apply_howde(self, input_data):
        """
        Apply HoWDe algorithm to classify stops as Home/Work/Other.

        Parameters
        ----------
        input_data : pyspark.sql.DataFrame
            Stops with columns: useruuid, loc, start, end

        Returns
        -------
        pyspark.sql.DataFrame
            Labeled stops with additional columns:
            location_type (H/W/O), detect_H_loc, detect_W_loc
        """
        print("Applying HoWDe algorithm...")
        start_time = time.time()

        # Select only required columns for HoWDe
        howde_input = input_data.select("useruuid", "loc", "start", "end")

        # Apply HoWDe labelling
        #import inspect
        #print("HoWDe_labelling signature:", inspect.signature(howde.HoWDe_labelling))
        labeled_data = howde.HoWDe_labelling(
            howde_input,
            edit_config_default=HOWDE_CONFIG,
            range_window_home=HOWDE_PARAMS["range_window_home"],
            range_window_work=HOWDE_PARAMS["range_window_work"],

            # Map your old threshold params to the version you have:
            # dn_H / dn_W (density thresholds) -> C_days_H / C_days_W
            C_days_H=HOWDE_PARAMS["dn_H"],
            C_days_W=HOWDE_PARAMS["dn_W"],

            # hf_H / hf_W (frequency thresholds) -> f_hours_H / f_hours_W
            f_hours_H=HOWDE_PARAMS["hf_H"],
            f_hours_W=HOWDE_PARAMS["hf_W"],

            # df_W (default frequency for work) -> f_days_W
            f_days_W=HOWDE_PARAMS["df_W"],

            # Leave C_hours at default unless you know what it should be
            # C_hours=0.4,

            output_format=HOWDE_PARAMS["output_format"],
            verbose=HOWDE_PARAMS["verbose"],
        )

        elapsed = time.time() - start_time
        print(f"HoWDe labeling completed in {elapsed / 60:.1f} minutes")

        return labeled_data

    def process_batch(self, batch):
        """
        Process a single batch of stops through HoWDe.

        Parameters
        ----------
        batch : int
            Batch number (0-49)
        """
        print(f"\n{'='*60}")
        print(f"Processing batch {batch} - Home/Work Detection")
        print(f"{'='*60}")
        start_time = time.time()

        # Load and prepare stop data
        input_data = self.load_stops(batch)

        # Store coordinates for later joining
        coords_df = input_data.select(
            'useruuid', 'loc', 'start', 'end',
            'latitude', 'longitude'
        )

        # Apply HoWDe
        labeled_data = self.apply_howde(input_data)

        # Join back coordinates using useruuid, loc, start, end
        # HoWDe output has: useruuid, country, loc, date, start, end,
        #                   location_type, detect_H_loc, detect_W_loc
        result = labeled_data.join(
            coords_df,
            on=['useruuid', 'loc', 'start', 'end'],
            how='left'
        )

        # Rename useruuid back to device_aid for consistency
        result = result.withColumnRenamed('useruuid', 'device_aid')

        # Add batch identifier
        result = result.withColumn('batch', F.lit(batch))

        # Convert to Pandas and save
        df_result = result.toPandas()

        output_path = self.output_dir / f"hw_{batch}.parquet"
        print(f"Saving to {output_path}...")
        df_result.to_parquet(output_path, index=False)

        elapsed = (time.time() - start_time) / 60

        # Summary statistics
        n_stops = len(df_result)
        n_users = df_result['device_aid'].nunique()
        if 'location_type' in df_result.columns:
            type_counts = df_result['location_type'].value_counts()
            print(f"\nLocation type distribution:")
            for t, c in type_counts.items():
                print(f"  {t}: {c:,} ({100*c/n_stops:.1f}%)")

        print(f"\nBatch {batch} complete:")
        print(f"  - {n_stops:,} labeled stops")
        print(f"  - {n_users:,} unique devices")
        print(f"  - Time: {elapsed:.1f} minutes")
        print(f"  - Output: {output_path}")

    def process_all(self, start_batch=0, end_batch=NUM_GROUPS):
        """
        Process multiple batches.

        Parameters
        ----------
        start_batch : int
            First batch to process (inclusive)
        end_batch : int
            Last batch to process (exclusive)
        """
        total_start = time.time()

        for batch in range(start_batch, end_batch):
            # Check if stop file exists
            stop_file = self.stops_dir / f"stops_{batch}.parquet"
            if not stop_file.exists():
                print(f"Skipping batch {batch}: stop file not found")
                continue

            # Process with notification
            # notify(f"✅ HoWDe started {datetime.now().isoformat()} batch={batch}/{end_batch-1}\n")
            try:
                self.process_batch(batch)
                notify(f"✅ Batch {batch}/{end_batch-1} finished OK")
            except Exception as e:
                notify(f"❌ Batch {batch}/{end_batch-1} failed:\n{traceback.format_exc()[:3500]}")
                raise

        total_elapsed = (time.time() - total_start) / 60
        print(f"\n{'='*60}")
        print(f"All batches complete in {total_elapsed:.1f} minutes")
        print(f"{'='*60}")

    def merge_all(self, output_file="hw_all.parquet"):
        """
        Merge all batch outputs into a single file.

        Parameters
        ----------
        output_file : str
            Name of merged output file
        """
        print("Merging all home/work files...")

        all_files = list(self.output_dir.glob("hw_*.parquet"))
        all_files = [f for f in all_files if f.name != output_file]

        if not all_files:
            print("No home/work files found to merge.")
            return

        dfs = []
        for f in sorted(all_files):
            print(f"  Loading {f.name}...")
            dfs.append(pd.read_parquet(f))

        merged = pd.concat(dfs, ignore_index=True)

        output_path = self.output_dir / output_file
        merged.to_parquet(output_path, index=False)

        # Summary
        n_stops = len(merged)
        n_users = merged['device_aid'].nunique()

        print(f"\nMerged {len(all_files)} files:")
        print(f"  - Total stops: {n_stops:,}")
        print(f"  - Total devices: {n_users:,}")

        if 'location_type' in merged.columns:
            type_counts = merged['location_type'].value_counts()
            print(f"\nLocation type distribution:")
            for t, c in type_counts.items():
                print(f"  {t}: {c:,} ({100*c/n_stops:.1f}%)")

        print(f"\nOutput: {output_path}")

    def extract_home_work_locations(self, output_file="home_work_locations.parquet"):
        """
        Extract primary home and work locations for each device.

        This processes batch files incrementally to avoid loading all data at once.

        Creates a device-level summary with:
        - home_loc: Primary home location ID
        - home_lat, home_lon: Home coordinates
        - work_loc: Primary work location ID
        - work_lat, work_lon: Work coordinates
        """
        print("Extracting primary home/work locations...")
        print("Processing batches incrementally to avoid memory issues...")

        # Find all batch files
        batch_files = sorted(self.output_dir.glob("hw_*.parquet"))
        batch_files = [f for f in batch_files if f.name.startswith('hw_')
                       and f.name != 'hw_all.parquet']

        if not batch_files:
            print("No batch files found. Run home/work detection first.")
            return

        print(f"Found {len(batch_files)} batch files")

        # Accumulate visit counts across all batches
        home_counts = {}  # {(device_aid, loc): (count, lat, lon)}
        work_counts = {}  # {(device_aid, loc): (count, lat, lon)}
        all_devices = set()

        for i, batch_file in enumerate(batch_files, 1):
            print(f"  Processing {batch_file.name} ({i}/{len(batch_files)})...")

            # Load batch
            df = pd.read_parquet(batch_file,
                                 columns=['device_aid', 'loc', 'location_type',
                                         'latitude', 'longitude'])

            all_devices.update(df['device_aid'].unique())

            # Aggregate home stops
            home_batch = df[df['location_type'] == 'H']
            if len(home_batch) > 0:
                home_agg = (home_batch
                    .groupby(['device_aid', 'loc'])
                    .agg(
                        visits=('loc', 'count'),
                        latitude=('latitude', 'first'),
                        longitude=('longitude', 'first')
                    )
                )
                for idx, row in home_agg.iterrows():
                    key = idx  # (device_aid, loc)
                    if key in home_counts:
                        # Accumulate counts
                        old_count, lat, lon = home_counts[key]
                        home_counts[key] = (old_count + row['visits'], lat, lon)
                    else:
                        home_counts[key] = (row['visits'], row['latitude'], row['longitude'])

            # Aggregate work stops
            work_batch = df[df['location_type'] == 'W']
            if len(work_batch) > 0:
                work_agg = (work_batch
                    .groupby(['device_aid', 'loc'])
                    .agg(
                        visits=('loc', 'count'),
                        latitude=('latitude', 'first'),
                        longitude=('longitude', 'first')
                    )
                )
                for idx, row in work_agg.iterrows():
                    key = idx  # (device_aid, loc)
                    if key in work_counts:
                        old_count, lat, lon = work_counts[key]
                        work_counts[key] = (old_count + row['visits'], lat, lon)
                    else:
                        work_counts[key] = (row['visits'], row['latitude'], row['longitude'])

        print("\nFinding primary locations per device...")

        # Convert to DataFrames and find most frequent locations
        home_data = [
            {'device_aid': k[0], 'loc': k[1], 'visits': v[0],
             'latitude': v[1], 'longitude': v[2]}
            for k, v in home_counts.items()
        ]
        work_data = [
            {'device_aid': k[0], 'loc': k[1], 'visits': v[0],
             'latitude': v[1], 'longitude': v[2]}
            for k, v in work_counts.items()
        ]

        # Find primary home for each device
        if home_data:
            home_df = pd.DataFrame(home_data)
            home_df['rank'] = home_df.groupby('device_aid')['visits'].rank(
                method='first', ascending=False
            )
            home_primary = home_df[home_df['rank'] == 1][
                ['device_aid', 'loc', 'latitude', 'longitude']
            ].rename(columns={
                'loc': 'home_loc',
                'latitude': 'home_lat',
                'longitude': 'home_lon'
            })
        else:
            home_primary = pd.DataFrame(columns=['device_aid', 'home_loc', 'home_lat', 'home_lon'])

        # Find primary work for each device
        if work_data:
            work_df = pd.DataFrame(work_data)
            work_df['rank'] = work_df.groupby('device_aid')['visits'].rank(
                method='first', ascending=False
            )
            work_primary = work_df[work_df['rank'] == 1][
                ['device_aid', 'loc', 'latitude', 'longitude']
            ].rename(columns={
                'loc': 'work_loc',
                'latitude': 'work_lat',
                'longitude': 'work_lon'
            })
        else:
            work_primary = pd.DataFrame(columns=['device_aid', 'work_loc', 'work_lat', 'work_lon'])

        # Merge home and work
        result = pd.DataFrame({'device_aid': list(all_devices)})
        result = result.merge(home_primary, on='device_aid', how='left')
        result = result.merge(work_primary, on='device_aid', how='left')

        # Add detection flags
        result['has_home'] = result['home_loc'].notna()
        result['has_work'] = result['work_loc'].notna()

        output_path = self.output_dir / output_file
        result.to_parquet(output_path, index=False)

        # Summary
        n_total = len(result)
        n_home = result['has_home'].sum()
        n_work = result['has_work'].sum()
        n_both = (result['has_home'] & result['has_work']).sum()

        print(f"\nPrimary locations extracted:")
        print(f"  - Total devices: {n_total:,}")
        print(f"  - With home detected: {n_home:,} ({100*n_home/n_total:.1f}%)")
        print(f"  - With work detected: {n_work:,} ({100*n_work/n_total:.1f}%)")
        print(f"  - With both: {n_both:,} ({100*n_both/n_total:.1f}%)")
        print(f"\nOutput: {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Home/Work detection for Swedish GPS stops using HoWDe"
    )
    parser.add_argument(
        '--batch',
        type=int,
        nargs='+',
        help='Batch number(s) to process. Single number or start end range.'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all batches (0-49)'
    )
    parser.add_argument(
        '--merge',
        action='store_true',
        help='Merge all existing batch files into one'
    )
    parser.add_argument(
        '--extract',
        action='store_true',
        help='Extract primary home/work locations per device'
    )
    parser.add_argument(
        '--cores',
        type=int,
        default=18,
        help='Number of Spark cores to use (default: 18)'
    )
    parser.add_argument(
        '--memory',
        type=str,
        default='56g',
        help='Spark driver memory (default: 56g)'
    )

    args = parser.parse_args()

    # Initialize
    spark = init_spark(driver_memory=args.memory, cores=args.cores)
    hwd = HomeWorkDetection(spark=spark)

    if args.merge:
        hwd.merge_all()
    elif args.extract:
        hwd.extract_home_work_locations()
    elif args.all:
        hwd.process_all()
    elif args.batch:
        if len(args.batch) == 1:
            hwd.process_batch(args.batch[0])
        elif len(args.batch) == 2:
            hwd.process_all(start_batch=args.batch[0], end_batch=args.batch[1])
        else:
            for b in args.batch:
                hwd.process_batch(b)
    else:
        parser.print_help()

    spark.stop()


if __name__ == '__main__':
    main()
