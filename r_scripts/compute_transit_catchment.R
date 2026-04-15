# Transit Catchment Diversity - r5r Routing
#
# Usage:
#   Rscript r_scripts/compute_transit_catchment.R --city new_york --max_time 45 --mode WALK,TRANSIT
#   Rscript r_scripts/compute_transit_catchment.R --city new_york --mode WALK
#   Rscript r_scripts/compute_transit_catchment.R --city new_york --mode TRANSIT
#   Rscript r_scripts/compute_transit_catchment.R --city new_york --batch_size 10000

# ---- Force Java environment BEFORE loading r5r/rJava ----
options(java.parameters = "-Xmx48G")

candidate_java_homes <- c(
  "/usr/lib/jvm/java-21-openjdk-amd64",
  "/usr/lib/jvm/default-java",
  "/usr/lib/jvm/java-21-openjdk",
  "/usr/lib/jvm/temurin-21-jdk-amd64"
)

java_home <- candidate_java_homes[file.exists(candidate_java_homes)][1]

if (is.na(java_home)) {
  stop("Could not find a valid JAVA_HOME in known locations.")
}

libjvm_dir <- file.path(java_home, "lib", "server")
libjvm_file <- file.path(libjvm_dir, "libjvm.so")

if (!file.exists(libjvm_file)) {
  stop(sprintf("libjvm.so not found at: %s", libjvm_file))
}

old_ld <- Sys.getenv("LD_LIBRARY_PATH")
old_path <- Sys.getenv("PATH")

Sys.setenv(
  JAVA_HOME = java_home,
  PATH = paste(file.path(java_home, "bin"), old_path, sep = ":"),
  LD_LIBRARY_PATH = if (nzchar(old_ld)) {
    paste(libjvm_dir, old_ld, sep = ":")
  } else {
    libjvm_dir
  }
)

cat("JAVA_HOME =", Sys.getenv("JAVA_HOME"), "\n")

library(r5r)
library(sf)
library(data.table)
library(jsonlite)
library(optparse)
library(arrow)

option_list <- list(
  make_option(c("-c", "--city"), type = "character", default = "new_york",
              help = "City name [default: %default]"),
  make_option(c("-m", "--max_time"), type = "integer", default = 45,
              help = "Maximum trip duration in minutes [default: %default]"),
  make_option(c("--mode"), type = "character", default = "WALK",
              help = "Routing mode: WALK, TRANSIT, or WALK,TRANSIT [default: %default]"),
  make_option(c("--batch_size"), type = "integer", default = 50000,
              help = "Process origins in batches to handle errors [default: %default]"),
  make_option(c("--skip_errors"), type = "logical", default = TRUE,
              help = "Skip batches that fail instead of stopping [default: %default]")
)
opt <- parse_args(OptionParser(option_list = option_list))

# Parse mode string into vector
routing_modes <- strsplit(opt$mode, ",")[[1]]
routing_modes <- trimws(toupper(routing_modes))
cat(sprintf("Routing modes: %s\n", paste(routing_modes, collapse = ", ")))

config <- fromJSON("r_scripts/r5r_config.json")
city_config <- config$cities[[opt$city]]

cat(sprintf("Processing %s...\n", opt$city))

# Determine departure datetime and timezone
departure_dt <- if (!is.null(city_config$departure_datetime)) city_config$departure_datetime else config$parameters$departure_datetime
tz <- if (!is.null(city_config$timezone)) city_config$timezone else "America/New_York"
cat(sprintf("Departure datetime: %s (%s)\n", departure_dt, tz))

# Check if GTFS exists for transit mode
if ("TRANSIT" %in% routing_modes) {
  gtfs_files <- list.files(city_config$transport_network_dir, pattern = "\\.zip$", full.names = TRUE)
  if (length(gtfs_files) == 0) {
    stop("No GTFS files found. Transit mode requires GTFS data.")
  }
  cat(sprintf("Found %d GTFS file(s)\n", length(gtfs_files)))
}

# Build or load network
r5r_network <- build_network(
  data_path = city_config$transport_network_dir,
  verbose = TRUE,
  overwrite = FALSE
)

origins <- fread(city_config$poi_file)

# Load destinations - use config-specified file or default to tract_centroids.csv
destinations_file <- if (!is.null(city_config$destinations_file)) {
  city_config$destinations_file
} else {
  "dbs/routing/tract_centroids.csv"
}
cat(sprintf("Destinations file: %s\n", destinations_file))
destinations <- fread(destinations_file)

# Validate columns
required_cols <- c("id", "lat", "lon")
for (col in required_cols) {
  if (!col %in% names(origins)) stop(sprintf("Origins missing column: %s", col))
  if (!col %in% names(destinations)) stop(sprintf("Destinations missing column: %s", col))
}

# Ensure character IDs
origins[, id := as.character(id)]
destinations[, id := as.character(id)]

cat(sprintf("Origins (POIs): %d\n", nrow(origins)))
cat(sprintf("Destinations (tracts): %d\n", nrow(destinations)))

# Filter destinations by city with geographic buffer for border POIs
if ("city" %in% names(destinations)) {
  # Handle variants like "new_york_minimal" -> "new_york"
  city_filter <- gsub("_minimal$", "", opt$city)

  # Get core city tracts
  city_tracts <- destinations[city == city_filter]

  if (nrow(city_tracts) == 0) {
    stop(sprintf("No destinations found for city: %s", city_filter))
  }

  # Calculate bounding box of city tracts with buffer (~50km in degrees)
  buffer_deg <- 0.5  # ~50km buffer
  bbox_lat_min <- min(city_tracts$lat) - buffer_deg
  bbox_lat_max <- max(city_tracts$lat) + buffer_deg
  bbox_lon_min <- min(city_tracts$lon) - buffer_deg
  bbox_lon_max <- max(city_tracts$lon) + buffer_deg

  # Include all tracts within buffered bounding box (catches neighboring areas)
  destinations <- destinations[
    lat >= bbox_lat_min & lat <= bbox_lat_max &
    lon >= bbox_lon_min & lon <= bbox_lon_max
  ]

  cat(sprintf("Filtered destinations for %s with 50km buffer: %d tracts\n", city_filter, nrow(destinations)))
  cat(sprintf("  (core city: %d, added from buffer: %d)\n",
              nrow(city_tracts), nrow(destinations) - nrow(city_tracts)))
}

# ---- Validate origins against network coverage ----
outside_file <- file.path(city_config$output_dir, "origins_outside_network.csv")

if (file.exists(outside_file)) {
  # Use cached validation results
  cat("\nUsing cached origin validation...\n")
  origins_outside <- fread(outside_file)
  excluded_ids <- origins_outside$id

  n_before <- nrow(origins)
  origins <- origins[!id %in% excluded_ids]
  n_filtered <- n_before - nrow(origins)

  cat(sprintf("Filtered %d origins using cached file: %s\n", n_filtered, outside_file))
  rm(origins_outside)
  gc()

} else {
  # Run validation and cache results

  cat("\nValidating origins against network coverage...\n")

  # Get network street coverage to filter origins
  network_sf <- tryCatch({
    street_network_to_sf(r5r_network)
  }, error = function(e) {
    cat("Could not extract network coverage, skipping pre-validation\n")
    NULL
  })

  if (!is.null(network_sf) && "edges" %in% names(network_sf)) {
    # Get bounding box of the network
    edges_bbox <- sf::st_bbox(network_sf$edges)

    # Filter origins within network bounds (with small buffer)
    buffer <- 0.01  # ~1km buffer in degrees
    valid_mask <- origins$lon >= (edges_bbox["xmin"] - buffer) &
                  origins$lon <= (edges_bbox["xmax"] + buffer) &
                  origins$lat >= (edges_bbox["ymin"] - buffer) &
                  origins$lat <= (edges_bbox["ymax"] + buffer)

    n_outside <- sum(!valid_mask)
    if (n_outside > 0) {
      cat(sprintf("Filtered %d origins outside network coverage\n", n_outside))
      origins_outside <- origins[!valid_mask]
      origins <- origins[valid_mask]

      # Save excluded origins for reference
      fwrite(origins_outside, outside_file)
      cat(sprintf("Saved excluded origins to: %s\n", outside_file))
    } else {
      # Save empty file to indicate validation was done
      fwrite(data.table(id = character(0), lat = numeric(0), lon = numeric(0)), outside_file)
      cat("All origins within network coverage\n")
    }

    rm(network_sf)
    gc()
  }
}

# ---- Process in batches ----
n_origins <- nrow(origins)
batch_size <- opt$batch_size
n_batches <- ceiling(n_origins / batch_size)

cat(sprintf("\nProcessing %d origins in %d batches of %d\n", n_origins, n_batches, batch_size))

all_ttm <- list()
failed_batches <- c()

for (batch_idx in seq_len(n_batches)) {
  start_idx <- (batch_idx - 1) * batch_size + 1
  end_idx <- min(batch_idx * batch_size, n_origins)

  batch_origins <- origins[start_idx:end_idx]

  cat(sprintf("\n--- Batch %d/%d (origins %d-%d) ---\n", batch_idx, n_batches, start_idx, end_idx))

  batch_ttm <- tryCatch({
    travel_time_matrix(
      r5r_network,
      origins = batch_origins,
      destinations = destinations,
      mode = routing_modes,
      departure_datetime = as.POSIXct(
                                       if (!is.null(city_config$departure_datetime)) city_config$departure_datetime else config$parameters$departure_datetime,
                                       format = "%Y-%m-%d %H:%M:%S",
                                       tz = if (!is.null(city_config$timezone)) city_config$timezone else "America/New_York"),
      max_walk_time = config$parameters$max_walk_time,
      max_trip_duration = opt$max_time,
      time_window = config$parameters$time_window,
      percentiles = config$parameters$percentiles,
      progress = TRUE
    )
  }, error = function(e) {
    cat(sprintf("ERROR in batch %d: %s\n", batch_idx, e$message))

    if (opt$skip_errors) {
      cat("Skipping batch and continuing...\n")
      return(NULL)
    } else {
      stop(e)
    }
  })

  if (!is.null(batch_ttm) && nrow(batch_ttm) > 0) {
    all_ttm[[length(all_ttm) + 1]] <- batch_ttm
    cat(sprintf("Batch %d: %d OD pairs\n", batch_idx, nrow(batch_ttm)))
  } else {
    failed_batches <- c(failed_batches, batch_idx)
  }

  # Force garbage collection between batches
  gc()
}

# Combine results
if (length(all_ttm) == 0) {
  stop("All batches failed. No results to save.")
}

ttm <- rbindlist(all_ttm)
cat(sprintf("\nTotal travel time matrix rows: %d\n", nrow(ttm)))

if (length(failed_batches) > 0) {
  cat(sprintf("WARNING: %d batches failed: %s\n",
              length(failed_batches), paste(failed_batches, collapse = ", ")))
}

# Find travel time column
time_col <- NULL
for (col in c("travel_time_p50", "travel_time")) {
  if (col %in% names(ttm)) {
    time_col <- col
    break
  }
}
if (is.null(time_col)) {
  cat("Available columns: ", paste(names(ttm), collapse = ", "), "\n")
  stop("Could not find a travel time column in the r5r output.")
}

cat(sprintf("Using travel time column: %s\n", time_col))

# Filter reachable pairs
ttm_reachable <- ttm[!is.na(get(time_col)) & get(time_col) <= opt$max_time]
cat(sprintf("Reachable OD pairs within %d min: %d\n", opt$max_time, nrow(ttm_reachable)))

# Find ID columns dynamically
origin_id_col <- NULL
for (col in c("from_id", "fromId", "id")) {
  if (col %in% names(ttm_reachable)) {
    origin_id_col <- col
    break
  }
}
if (is.null(origin_id_col)) {
  cat("TTM columns: ", paste(names(ttm_reachable), collapse = ", "), "\n")
  stop("Could not find origin ID column.")
}

dest_id_col <- NULL
for (col in c("to_id", "toId")) {
  if (col %in% names(ttm_reachable)) {
    dest_id_col <- col
    break
  }
}
if (is.null(dest_id_col)) {
  stop("Could not find destination ID column.")
}

cat(sprintf("Using columns: origin=%s, dest=%s, time=%s\n", origin_id_col, dest_id_col, time_col))

# Aggregate by origin
catchment_summary <- ttm_reachable[
  ,
  .(
    reachable_tract_count = as.integer(uniqueN(get(dest_id_col))),
    reachable_tract_ids = paste(sort(unique(get(dest_id_col))), collapse = "|"),
    min_travel_time = as.numeric(min(get(time_col), na.rm = TRUE)),
    median_travel_time = as.numeric(median(get(time_col), na.rm = TRUE))
  ),
  by = c(origin_id_col)
]

# Rename to poi_id
setnames(catchment_summary, origin_id_col, "poi_id")
catchment_summary[, poi_id := as.character(poi_id)]

# Merge with all POIs to include those with no reachable tracts
poi_base <- unique(origins[, .(poi_id = as.character(id))])
catchment_summary <- merge(poi_base, catchment_summary, by = "poi_id", all.x = TRUE)

# Fill NA values
catchment_summary[is.na(reachable_tract_count), reachable_tract_count := 0L]
catchment_summary[is.na(reachable_tract_ids), reachable_tract_ids := ""]

# Add POI coordinates
poi_coords <- unique(origins[, .(poi_id = as.character(id), lat, lon)])
catchment_summary <- merge(catchment_summary, poi_coords, by = "poi_id", all.x = TRUE)

# Add metadata
catchment_summary[, `:=`(
  city = opt$city,
  max_time_min = opt$max_time,
  routing_mode = paste(routing_modes, collapse = "+")
)]

# Save outputs
mode_suffix <- tolower(gsub(",", "_", opt$mode))
output_file <- file.path(
  city_config$output_dir,
  sprintf("catchment_summary_%s_%dmin.parquet", mode_suffix, opt$max_time)
)
write_parquet(catchment_summary, output_file)
cat(sprintf("\nSaved summary: %s (%d POIs)\n", output_file, nrow(catchment_summary)))

od_output_file <- file.path(
  city_config$output_dir,
  sprintf("catchment_od_pairs_%s_%dmin.parquet", mode_suffix, opt$max_time)
)
write_parquet(ttm_reachable, od_output_file)
cat(sprintf("Saved OD pairs: %s (%d pairs)\n", od_output_file, nrow(ttm_reachable)))

# Summary stats
cat("\n=== Summary ===\n")
cat(sprintf("Total POIs: %d\n", nrow(catchment_summary)))
cat(sprintf("POIs with reachable tracts: %d (%.1f%%)\n",
            sum(catchment_summary$reachable_tract_count > 0),
            100 * mean(catchment_summary$reachable_tract_count > 0)))
cat(sprintf("Mean reachable tracts per POI: %.1f\n", mean(catchment_summary$reachable_tract_count)))
if (sum(!is.na(catchment_summary$median_travel_time)) > 0) {
  cat(sprintf("Median travel time: %.1f min\n",
              median(catchment_summary$median_travel_time, na.rm = TRUE)))
}
if (length(failed_batches) > 0) {
  cat(sprintf("\nWARNING: %d batches failed (origins may be missing from results)\n", length(failed_batches)))
}

stop_r5(r5r_network)
cat("\nDone.\n")
