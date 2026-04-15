"""
Step 1.1.1a: Stop Detection for Swedish GPS Data

This script applies the Infostop algorithm to detect stops from raw GPS traces.
Data is organized in 50 groups under D:\MAD_dbs\raw_data_se_24\format_parquet.

Usage:
    python stop_detection.py --batch 0          # Process single batch
    python stop_detection.py --batch 0 10       # Process batches 0-9
    python stop_detection.py --all              # Process all batches (0-49)

Output:
    dbs/stops/stops_{batch}.parquet
    Columns: device_aid, loc, start, end, latitude, longitude, size, batch,
             localtime, l_localtime (local time versions for HoWDe)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import time
import os
import sys
import argparse
from datetime import datetime

# PySpark imports
from pyspark.sql import SparkSession
from pyspark import SparkConf
import pyspark.sql.functions as F
from pyspark.sql.types import (
    StructType, StructField, IntegerType, DoubleType,
    StringType, LongType
)

# Infostop
from infostop import Infostop

# ============================================================================
# Configuration
# ============================================================================

# Paths
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_FOLDER = Path("/data/MAD_dbs/raw_data_se_24/format_parquet")
OUTPUT_DIR = ROOT_DIR / "dbs" / "stops"

# Infostop parameters
R1 = 30                    # meters - spatial radius for stop detection
R2 = 30                    # meters - spatial radius for merging nearby stops
MIN_STAYING_TIME = 15      # minutes - minimum stop duration
MAX_TIME_BETWEEN = 3       # hours - max gap between pings

# Sweden timezone offset (CET = UTC+1, CEST = UTC+2)
# For simplicity, we use UTC+1 (standard time)
# More precise handling would use pytz or zoneinfo
SWEDEN_UTC_OFFSET_SECONDS = 3600  # +1 hour

# Number of groups
NUM_GROUPS = 50


# ============================================================================
# Spark Setup
# ============================================================================

def init_spark(app_name="StopDetection", driver_memory="56g", cores=18):
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
# Infostop Processing
# ============================================================================

def infostop_per_user(key, data):
    """
    Apply Infostop algorithm to detect stops for a single user.

    Parameters
    ----------
    key : tuple
        (device_aid,) grouping key
    data : pd.DataFrame
        GPS records with columns: device_aid, timestamp, latitude, longitude

    Returns
    -------
    pd.DataFrame
        Detected stops with columns:
        device_aid, timestamp, latitude, longitude, loc,
        stop_latitude, stop_longitude, interval
    """
    model = Infostop(
        r1=R1,
        r2=R2,
        label_singleton=True,
        min_staying_time=MIN_STAYING_TIME * 60,  # Convert to seconds
        max_time_between=MAX_TIME_BETWEEN * 60 * 60,  # Convert to seconds
        min_size=2,
        min_spacial_resolution=0,
        distance_metric='haversine',
        weighted=False,
        weight_exponent=1,
        verbose=False,
    )

    # Remove abnormal GPS records (outside valid lat/lon range)
    x = data.loc[
        ~(
            ((data['latitude'] > 84) | (data['latitude'] < -80)) |
            ((data['longitude'] > 180) | (data['longitude'] < -180))
        ), :
    ]

    # Sort by timestamp and remove duplicates
    x = (x.sort_values(by='timestamp')
         .drop_duplicates(subset=['latitude', 'longitude', 'timestamp'])
         .reset_index(drop=True))
    x = x.dropna()

    if len(x) < 2:
        return pd.DataFrame(
            [],
            columns=['device_aid', 'timestamp', 'latitude', 'longitude',
                     'loc', 'stop_latitude', 'stop_longitude', 'interval']
        )

    # Handle large time gaps by interpolating timestamps
    # This helps Infostop handle discontinuous data
    x['t_seg'] = x['timestamp'].shift(-1)
    x.loc[x.index[-1], 't_seg'] = x.loc[x.index[-1], 'timestamp'] + 1

    max_gap_seconds = MAX_TIME_BETWEEN * 60 * 60
    x['n'] = x.apply(
        lambda row: range(
            int(row['timestamp']),
            min(int(row['t_seg']), int(row['timestamp']) + max_gap_seconds),
            max_gap_seconds - 1
        ),
        axis=1
    )
    x = x.explode('n')
    x['timestamp'] = x['n'].astype(float)
    x = x[['latitude', 'longitude', 'timestamp']].dropna()

    if len(x) < 2:
        return pd.DataFrame(
            [],
            columns=['device_aid', 'timestamp', 'latitude', 'longitude',
                     'loc', 'stop_latitude', 'stop_longitude', 'interval']
        )

    # Apply Infostop
    try:
        labels = model.fit_predict(x[['latitude', 'longitude', 'timestamp']].values)
    except Exception as e:
        return pd.DataFrame(
            [],
            columns=['device_aid', 'timestamp', 'latitude', 'longitude',
                     'loc', 'stop_latitude', 'stop_longitude', 'interval']
        )

    # Get stop centroids
    label_medians = model.compute_label_medians()

    x['loc'] = labels
    x['same_loc'] = x['loc'] == x['loc'].shift()
    x['little_time'] = (x['timestamp'] - x['timestamp'].shift() < max_gap_seconds)

    # Create interval IDs for continuous stays at same location
    x['interval'] = (~(x['same_loc'] & x['little_time'])).cumsum()

    # Map stop centroids
    latitudes = {k: v[0] for k, v in label_medians.items()}
    longitudes = {k: v[1] for k, v in label_medians.items()}
    x['stop_latitude'] = x['loc'].map(latitudes)
    x['stop_longitude'] = x['loc'].map(longitudes)
    x['device_aid'] = key[0]

    # Keep only actual stops (loc > 0, as -1 indicates moving/noise)
    x = x[x['loc'] > 0].copy()

    return x[['device_aid', 'timestamp', 'latitude', 'longitude',
              'loc', 'stop_latitude', 'stop_longitude', 'interval']]


# Output schema for applyInPandas
INFOSTOP_SCHEMA = StructType([
    StructField('loc', IntegerType()),
    StructField('timestamp', IntegerType()),
    StructField('interval', IntegerType()),
    StructField('latitude', DoubleType()),
    StructField('longitude', DoubleType()),
    StructField('device_aid', StringType()),
    StructField('stop_latitude', DoubleType()),
    StructField('stop_longitude', DoubleType()),
])


# ============================================================================
# Stop Detection Class
# ============================================================================

class StopDetection:
    """
    Stop detection pipeline for Swedish GPS data.

    Processes GPS data in batches (50 groups), applying Infostop
    to detect stops and preparing output for HoWDe home/work detection.
    """

    def __init__(self, spark=None, data_folder=DATA_FOLDER, output_dir=OUTPUT_DIR):
        self.spark = spark or init_spark()
        self.data_folder = Path(data_folder)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.file_paths_dict = {}

    def build_file_list(self):
        """Build list of parquet files for each group."""
        for grp in range(NUM_GROUPS):
            grp_path = self.data_folder / f"grp_{grp}"
            if grp_path.exists():
                files = list(grp_path.glob("*.parquet"))
                self.file_paths_dict[grp] = [str(f) for f in files]
                print(f"Group {grp}: {len(files)} files")
            else:
                print(f"Warning: Group {grp} not found at {grp_path}")

    def process_batch(self, batch):
        """
        Process a single batch of users.

        Parameters
        ----------
        batch : int
            Batch/group number (0-49)
        """
        if batch not in self.file_paths_dict:
            print(f"Error: Batch {batch} not in file list. Run build_file_list() first.")
            return

        print(f"\n{'='*60}")
        print(f"Processing batch {batch}")
        print(f"{'='*60}")
        start_time = time.time()

        file_paths = self.file_paths_dict[batch]
        print(f"Loading {len(file_paths)} files...")

        # Load GPS data
        df = self.spark.read.parquet(*file_paths).select(
            'device_aid', 'timestamp', 'latitude', 'longitude'
        )

        n_records = df.count()
        n_devices = df.select('device_aid').distinct().count()
        print(f"Loaded {n_records:,} records from {n_devices:,} devices")

        # Apply Infostop per user
        print("Applying Infostop stop detection...")
        stops = df.groupby('device_aid').applyInPandas(
            infostop_per_user,
            schema=INFOSTOP_SCHEMA
        )

        # Aggregate stops (combine continuous intervals at same location)
        print("Aggregating stops...")
        stop_locations = stops.groupby('device_aid', 'interval').agg(
            F.first('loc').alias('loc'),
            F.min('timestamp').alias('start'),
            F.max('timestamp').alias('end'),
            F.first('stop_latitude').alias('latitude'),
            F.first('stop_longitude').alias('longitude'),
            F.count('loc').alias('size')
        )

        # Add local time columns for HoWDe
        # HoWDe expects 'localtime' and 'l_localtime' in local timezone
        stop_locations = stop_locations.withColumn(
            'localtime',
            (F.col('start') + F.lit(SWEDEN_UTC_OFFSET_SECONDS)).cast(LongType())
        ).withColumn(
            'l_localtime',
            (F.col('end') + F.lit(SWEDEN_UTC_OFFSET_SECONDS)).cast(LongType())
        )

        # Convert to Pandas and save
        df_stops = stop_locations.toPandas()
        df_stops['batch'] = batch

        # Compute duration
        df_stops['duration_min'] = (df_stops['end'] - df_stops['start']) / 60

        output_path = self.output_dir / f"stops_{batch}.parquet"
        print(f"Saving to {output_path}...")
        df_stops.to_parquet(output_path, index=False)

        elapsed = (time.time() - start_time) / 60
        n_stops = len(df_stops)
        n_users = df_stops['device_aid'].nunique()

        print(f"\nBatch {batch} complete:")
        print(f"  - {n_stops:,} stops detected")
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
            self.process_batch(batch)

        total_elapsed = (time.time() - total_start) / 60
        print(f"\n{'='*60}")
        print(f"All batches complete in {total_elapsed:.1f} minutes")
        print(f"{'='*60}")

    def merge_all_stops(self, output_file="stops_all.parquet"):
        """
        Merge all batch outputs into a single file.

        Parameters
        ----------
        output_file : str
            Name of merged output file
        """
        print("Merging all stop files...")

        all_files = list(self.output_dir.glob("stops_*.parquet"))
        all_files = [f for f in all_files if f.name != output_file]

        if not all_files:
            print("No stop files found to merge.")
            return

        dfs = []
        for f in sorted(all_files):
            print(f"  Loading {f.name}...")
            dfs.append(pd.read_parquet(f))

        merged = pd.concat(dfs, ignore_index=True)

        output_path = self.output_dir / output_file
        merged.to_parquet(output_path, index=False)

        print(f"\nMerged {len(all_files)} files:")
        print(f"  - Total stops: {len(merged):,}")
        print(f"  - Total devices: {merged['device_aid'].nunique():,}")
        print(f"  - Output: {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stop detection for Swedish GPS data using Infostop"
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
    sd = StopDetection(spark=spark)
    sd.build_file_list()

    if args.merge:
        sd.merge_all_stops()
    elif args.all:
        sd.process_all()
    elif args.batch:
        if len(args.batch) == 1:
            sd.process_batch(args.batch[0])
        elif len(args.batch) == 2:
            sd.process_all(start_batch=args.batch[0], end_batch=args.batch[1])
        else:
            for b in args.batch:
                sd.process_batch(b)
    else:
        parser.print_help()

    spark.stop()


if __name__ == '__main__':
    main()
