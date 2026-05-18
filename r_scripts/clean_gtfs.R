# Clean GTFS - Fix common issues that cause r5r to crash
# Handles: empty stop_ids, missing stops, empty tables, malformed files, ID conflicts
#
# Usage:
#   Rscript r_scripts/clean_gtfs.R --city washington_dc
#   Rscript r_scripts/clean_gtfs.R --city chicago --merge_and_clean
#   Rscript r_scripts/clean_gtfs.R --city chicago --merge_and_clean --rail_only

if (!requireNamespace("gtfstools", quietly = TRUE)) {
  install.packages("gtfstools")
}

library(gtfstools)
library(data.table)
library(jsonlite)
library(optparse)

option_list <- list(
  make_option(c("-c", "--city"), type = "character", default = "washington_dc"),
  make_option(c("-i", "--input"), type = "character", default = "merged_gtfs.zip"),
  make_option(c("-o", "--output"), type = "character", default = "cleaned_gtfs.zip"),
  make_option(c("--merge_and_clean"), action = "store_true", default = FALSE,
              help = "Merge individual GTFS files with cleaning (use when merged file has issues)"),
  make_option(c("--rail_only"), action = "store_true", default = FALSE,
              help = "Only include rail services (route_type 0,1,2)")
)
opt <- parse_args(OptionParser(option_list = option_list))

config <- fromJSON("r_scripts/r5r_config.json")
city_config <- config$cities[[opt$city]]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

clean_single_gtfs <- function(gtfs, prefix = NULL) {
  # Helper to add prefix only if not already present (avoids double-prefixing)
  add_prefix_safe <- function(x, prefix) {
    prefix_pattern <- paste0("^", prefix, "_")
    already_has <- grepl(prefix_pattern, as.character(x))
    ifelse(already_has, x, paste0(prefix, "_", x))
  }

  # Add prefix to IDs to avoid conflicts when merging
  if (!is.null(prefix) && nzchar(prefix)) {
    cat(sprintf("  Adding prefix '%s' to IDs...\n", prefix))

    # Prefix route_id
    if (!is.null(gtfs$routes) && nrow(gtfs$routes) > 0) {
      gtfs$routes[, route_id := add_prefix_safe(route_id, prefix)]
    }

    # Prefix trip_id and update route_id reference
    if (!is.null(gtfs$trips) && nrow(gtfs$trips) > 0) {
      gtfs$trips[, trip_id := add_prefix_safe(trip_id, prefix)]
      gtfs$trips[, route_id := add_prefix_safe(route_id, prefix)]
      if ("shape_id" %in% names(gtfs$trips) && any(!is.na(gtfs$trips$shape_id))) {
        gtfs$trips[!is.na(shape_id), shape_id := add_prefix_safe(shape_id, prefix)]
      }
      if ("service_id" %in% names(gtfs$trips)) {
        gtfs$trips[, service_id := add_prefix_safe(service_id, prefix)]
      }
    }

    # Prefix stop_id
    if (!is.null(gtfs$stops) && nrow(gtfs$stops) > 0) {
      gtfs$stops[, stop_id := add_prefix_safe(stop_id, prefix)]
      if ("parent_station" %in% names(gtfs$stops) && any(!is.na(gtfs$stops$parent_station) & gtfs$stops$parent_station != "")) {
        gtfs$stops[!is.na(parent_station) & parent_station != "", parent_station := add_prefix_safe(parent_station, prefix)]
      }
    }

    # Prefix stop_times references
    if (!is.null(gtfs$stop_times) && nrow(gtfs$stop_times) > 0) {
      gtfs$stop_times[, trip_id := add_prefix_safe(trip_id, prefix)]
      gtfs$stop_times[, stop_id := add_prefix_safe(stop_id, prefix)]
    }

    # Prefix shapes
    if (!is.null(gtfs$shapes) && nrow(gtfs$shapes) > 0) {
      gtfs$shapes[, shape_id := add_prefix_safe(shape_id, prefix)]
    }

    # Prefix calendar
    if (!is.null(gtfs$calendar) && nrow(gtfs$calendar) > 0) {
      gtfs$calendar[, service_id := add_prefix_safe(service_id, prefix)]
    }

    # Prefix calendar_dates
    if (!is.null(gtfs$calendar_dates) && nrow(gtfs$calendar_dates) > 0) {
      gtfs$calendar_dates[, service_id := add_prefix_safe(service_id, prefix)]
    }

    # Prefix frequencies
    if (!is.null(gtfs$frequencies) && nrow(gtfs$frequencies) > 0) {
      gtfs$frequencies[, trip_id := add_prefix_safe(trip_id, prefix)]
    }

    # Prefix transfers
    if (!is.null(gtfs$transfers) && nrow(gtfs$transfers) > 0) {
      gtfs$transfers[, from_stop_id := add_prefix_safe(from_stop_id, prefix)]
      gtfs$transfers[, to_stop_id := add_prefix_safe(to_stop_id, prefix)]
    }

    # Prefix agency
    if (!is.null(gtfs$agency) && nrow(gtfs$agency) > 0 && "agency_id" %in% names(gtfs$agency)) {
      gtfs$agency[, agency_id := add_prefix_safe(agency_id, prefix)]
      if (!is.null(gtfs$routes) && "agency_id" %in% names(gtfs$routes)) {
        gtfs$routes[, agency_id := add_prefix_safe(agency_id, prefix)]
      }
    }
  }

  # Remove empty tables that can cause issues
  empty_tables <- c()
  for (tbl_name in names(gtfs)) {
    tbl <- gtfs[[tbl_name]]
    if (is.data.frame(tbl) && nrow(tbl) == 0) {
      empty_tables <- c(empty_tables, tbl_name)
    }
  }
  if (length(empty_tables) > 0) {
    cat(sprintf("  Removing empty tables: %s\n", paste(empty_tables, collapse = ", ")))
    for (tbl_name in empty_tables) {
      gtfs[[tbl_name]] <- NULL
    }
  }

  # Fix stop_times with empty or NA stop_id
  if (!is.null(gtfs$stop_times) && nrow(gtfs$stop_times) > 0) {
    bad_stop_times <- gtfs$stop_times[is.na(stop_id) | stop_id == "" | stop_id == '""' | stop_id == "NA"]
    if (nrow(bad_stop_times) > 0) {
      cat(sprintf("  Removing %d stop_times with empty/invalid stop_id\n", nrow(bad_stop_times)))
      gtfs$stop_times <- gtfs$stop_times[!is.na(stop_id) & stop_id != "" & stop_id != '""' & stop_id != "NA"]
    }
  }

  # Fix stop_times with empty or NA trip_id
  if (!is.null(gtfs$stop_times) && nrow(gtfs$stop_times) > 0) {
    bad_trip_times <- gtfs$stop_times[is.na(trip_id) | trip_id == "" | trip_id == '""']
    if (nrow(bad_trip_times) > 0) {
      cat(sprintf("  Removing %d stop_times with empty trip_id\n", nrow(bad_trip_times)))
      gtfs$stop_times <- gtfs$stop_times[!is.na(trip_id) & trip_id != "" & trip_id != '""']
    }
  }

  # Remove stop_times referencing non-existent stops
  if (!is.null(gtfs$stop_times) && !is.null(gtfs$stops) && nrow(gtfs$stop_times) > 0 && nrow(gtfs$stops) > 0) {
    valid_stops <- unique(gtfs$stops$stop_id)
    invalid_refs <- gtfs$stop_times[!stop_id %in% valid_stops]
    if (nrow(invalid_refs) > 0) {
      cat(sprintf("  Removing %d stop_times referencing non-existent stops\n", nrow(invalid_refs)))
      gtfs$stop_times <- gtfs$stop_times[stop_id %in% valid_stops]
    }
  }

  # Remove trips with no stop_times
  if (!is.null(gtfs$trips) && !is.null(gtfs$stop_times) && nrow(gtfs$trips) > 0) {
    valid_trips <- unique(gtfs$stop_times$trip_id)
    orphan_trips <- gtfs$trips[!trip_id %in% valid_trips]
    if (nrow(orphan_trips) > 0) {
      cat(sprintf("  Removing %d trips with no stop_times\n", nrow(orphan_trips)))
      gtfs$trips <- gtfs$trips[trip_id %in% valid_trips]
    }
  }

  # Remove routes with no trips
  if (!is.null(gtfs$routes) && !is.null(gtfs$trips) && nrow(gtfs$routes) > 0) {
    valid_routes <- unique(gtfs$trips$route_id)
    orphan_routes <- gtfs$routes[!route_id %in% valid_routes]
    if (nrow(orphan_routes) > 0) {
      cat(sprintf("  Removing %d routes with no trips\n", nrow(orphan_routes)))
      gtfs$routes <- gtfs$routes[route_id %in% valid_routes]
    }
  }

  # Remove stops with missing coordinates
  if (!is.null(gtfs$stops) && nrow(gtfs$stops) > 0) {
    if ("stop_lat" %in% names(gtfs$stops) && "stop_lon" %in% names(gtfs$stops)) {
      bad_coords <- gtfs$stops[is.na(stop_lat) | is.na(stop_lon) | stop_lat == "" | stop_lon == ""]
      if (nrow(bad_coords) > 0) {
        cat(sprintf("  Removing %d stops with missing coordinates\n", nrow(bad_coords)))
        gtfs$stops <- gtfs$stops[!is.na(stop_lat) & !is.na(stop_lon) & stop_lat != "" & stop_lon != ""]
      }
    }
  }

  # Clean up unused shapes
  if (!is.null(gtfs$shapes) && !is.null(gtfs$trips) && nrow(gtfs$shapes) > 0) {
    if ("shape_id" %in% names(gtfs$trips)) {
      valid_shapes <- unique(gtfs$trips$shape_id)
      valid_shapes <- valid_shapes[!is.na(valid_shapes) & valid_shapes != ""]
      if (length(valid_shapes) > 0) {
        orphan_shapes <- nrow(gtfs$shapes) - nrow(gtfs$shapes[shape_id %in% valid_shapes])
        if (orphan_shapes > 0) {
          cat(sprintf("  Removing %d orphan shape records\n", orphan_shapes))
          gtfs$shapes <- gtfs$shapes[shape_id %in% valid_shapes]
        }
      }
    }
  }

  # Clean up frequencies
  if (!is.null(gtfs$frequencies) && nrow(gtfs$frequencies) > 0) {
    if (!is.null(gtfs$trips)) {
      valid_trips <- unique(gtfs$trips$trip_id)
      gtfs$frequencies <- gtfs$frequencies[trip_id %in% valid_trips]
    }
  }

  return(gtfs)
}

merge_gtfs_manual <- function(gtfs_list) {
  # Manual merge that handles schema differences better
  merged <- list()

  # Get all table names across all GTFS files
  all_tables <- unique(unlist(lapply(gtfs_list, names)))

  for (tbl_name in all_tables) {
    # Collect this table from all GTFS files that have it
    tables <- lapply(gtfs_list, function(g) {
      if (tbl_name %in% names(g) && !is.null(g[[tbl_name]]) && nrow(g[[tbl_name]]) > 0) {
        return(g[[tbl_name]])
      }
      return(NULL)
    })
    tables <- Filter(Negate(is.null), tables)

    if (length(tables) > 0) {
      # Use fill=TRUE to handle different columns
      merged[[tbl_name]] <- rbindlist(tables, fill = TRUE)
    }
  }

  class(merged) <- c("dt_gtfs", "gtfs", "list")
  return(merged)
}

# ============================================================================
# MAIN
# ============================================================================

if (opt$merge_and_clean) {
  # Merge individual GTFS files with cleaning
  gtfs_dir <- city_config$transport_network_dir
  gtfs_files <- list.files(gtfs_dir, pattern = "\\.zip$", full.names = TRUE)
  gtfs_files <- gtfs_files[!grepl("merged|cleaned|backup", basename(gtfs_files))]

  cat(sprintf("Found %d GTFS files to merge and clean:\n", length(gtfs_files)))
  for (f in gtfs_files) {
    cat(sprintf("  - %s\n", basename(f)))
  }

  gtfs_list <- list()

  for (gtfs_file in gtfs_files) {
    fname <- basename(gtfs_file)
    prefix <- gsub("\\.zip$", "", fname)
    prefix <- gsub("[^a-zA-Z0-9]", "_", prefix)  # Clean prefix
    prefix <- substr(prefix, 1, 20)  # Limit length

    cat(sprintf("\n--- Processing %s ---\n", fname))

    gtfs <- tryCatch({
      suppressWarnings(read_gtfs(gtfs_file))
    }, error = function(e) {
      cat(sprintf("  ERROR reading: %s\n", e$message))
      return(NULL)
    })

    if (is.null(gtfs)) {
      next
    }

    # Basic stats
    n_routes <- if (!is.null(gtfs$routes)) nrow(gtfs$routes) else 0
    n_stops <- if (!is.null(gtfs$stops)) nrow(gtfs$stops) else 0
    n_trips <- if (!is.null(gtfs$trips)) nrow(gtfs$trips) else 0
    cat(sprintf("  Routes: %d, Stops: %d, Trips: %d\n", n_routes, n_stops, n_trips))

    # Filter to rail only if requested
    if (opt$rail_only && !is.null(gtfs$routes) && nrow(gtfs$routes) > 0) {
      rail_types <- c(0, 1, 2)  # Tram, Subway, Rail
      if ("route_type" %in% names(gtfs$routes)) {
        rail_routes <- gtfs$routes$route_id[gtfs$routes$route_type %in% rail_types]
        if (length(rail_routes) == 0) {
          cat("  Skipping: No rail routes\n")
          next
        }
        gtfs <- tryCatch({
          filter_by_route_id(gtfs, rail_routes)
        }, error = function(e) {
          cat(sprintf("  Warning: Could not filter routes: %s\n", e$message))
          gtfs
        })
        cat(sprintf("  Filtered to %d rail routes\n", length(rail_routes)))
      }
    }

    # Clean and add prefix
    gtfs <- clean_single_gtfs(gtfs, prefix = prefix)

    if (!is.null(gtfs$routes) && nrow(gtfs$routes) > 0) {
      gtfs_list[[fname]] <- gtfs
    } else {
      cat("  Skipping: No routes after cleaning\n")
    }
  }

  if (length(gtfs_list) == 0) {
    stop("No valid GTFS files to merge")
  }

  cat(sprintf("\n=== Merging %d cleaned GTFS files ===\n", length(gtfs_list)))

  merged_gtfs <- tryCatch({
    merge_gtfs_manual(gtfs_list)
  }, error = function(e) {
    cat(sprintf("Manual merge failed: %s\n", e$message))
    cat("Trying gtfstools merge...\n")
    merge_gtfs(gtfs_list)
  })

  # Final cleaning pass on merged result
  cat("\n=== Final cleaning pass ===\n")
  merged_gtfs <- clean_single_gtfs(merged_gtfs, prefix = NULL)

  output_file <- file.path(gtfs_dir, opt$output)

} else {
  # Clean existing merged file
  input_file <- file.path(city_config$transport_network_dir, opt$input)
  output_file <- file.path(city_config$transport_network_dir, opt$output)

  cat(sprintf("Reading GTFS: %s\n", input_file))
  merged_gtfs <- read_gtfs(input_file)

  cat("\n=== Cleaning GTFS ===\n")
  merged_gtfs <- clean_single_gtfs(merged_gtfs, prefix = NULL)
}

# Final stats
cat("\n=== Final GTFS stats ===\n")
cat(sprintf("Routes: %d\n", if (!is.null(merged_gtfs$routes)) nrow(merged_gtfs$routes) else 0))
cat(sprintf("Trips: %d\n", if (!is.null(merged_gtfs$trips)) nrow(merged_gtfs$trips) else 0))
cat(sprintf("Stop times: %d\n", if (!is.null(merged_gtfs$stop_times)) nrow(merged_gtfs$stop_times) else 0))
cat(sprintf("Stops: %d\n", if (!is.null(merged_gtfs$stops)) nrow(merged_gtfs$stops) else 0))

if (!is.null(merged_gtfs$routes) && "route_type" %in% names(merged_gtfs$routes)) {
  cat("\nRoute types:\n")
  route_types <- table(merged_gtfs$routes$route_type)
  type_names <- c("0" = "Tram", "1" = "Subway", "2" = "Rail", "3" = "Bus",
                  "4" = "Ferry", "5" = "Cable", "6" = "Gondola", "7" = "Funicular")
  for (rt in names(route_types)) {
    type_name <- if (rt %in% names(type_names)) type_names[rt] else "Other"
    cat(sprintf("  %s (%s): %d routes\n", rt, type_name, route_types[rt]))
  }
}

# Write cleaned GTFS
cat(sprintf("\nWriting: %s\n", output_file))
write_gtfs(merged_gtfs, output_file)
cat(sprintf("Done! File size: %.1f MB\n", file.size(output_file) / 1e6))

cat("\n=== Next steps ===\n")
cat(sprintf("1. rm -f %s/network.dat %s/*.mapdb*\n",
            city_config$transport_network_dir, city_config$transport_network_dir))
cat(sprintf("2. Update config or rename: mv %s %s/merged_gtfs.zip\n",
            output_file, city_config$transport_network_dir))
cat("3. Re-run compute_transit_catchment.R\n")
