# GTFS Validation and Merging using gtfstools
# https://ipeagit.github.io/gtfstools/
#
# Usage:
#   Rscript r_scripts/validate_merge_gtfs.R --city new_york
#   Rscript r_scripts/validate_merge_gtfs.R --city new_york --validate_only
#   Rscript r_scripts/validate_merge_gtfs.R --city new_york --output merged_gtfs.zip

# Install gtfstools if needed
if (!requireNamespace("gtfstools", quietly = TRUE)) {
  install.packages("gtfstools")
}

library(gtfstools)
library(data.table)
library(jsonlite)
library(optparse)

option_list <- list(
  make_option(c("-c", "--city"), type = "character", default = "new_york",
              help = "City name [default: %default]"),
  make_option(c("-o", "--output"), type = "character", default = NULL,
              help = "Output filename for merged GTFS [default: merged_gtfs.zip]"),
  make_option(c("--validate_only"), action = "store_true", default = FALSE,
              help = "Only validate, don't merge"),
  make_option(c("--filter_date"), type = "character", default = NULL,
              help = "Filter services to specific date (YYYY-MM-DD)"),
  make_option(c("--rail_only"), action = "store_true", default = FALSE,
              help = "Only include rail services (route_type 0,1,2)")
)
opt <- parse_args(OptionParser(option_list = option_list))

# Load config
config <- fromJSON("r_scripts/r5r_config.json")
city_config <- config$cities[[opt$city]]

if (is.null(city_config)) {
  stop(sprintf("City '%s' not found in config", opt$city))
}

gtfs_dir <- city_config$transport_network_dir
cat(sprintf("GTFS directory: %s\n", gtfs_dir))

# Find all GTFS zip files
gtfs_files <- list.files(gtfs_dir, pattern = "\\.zip$", full.names = TRUE)
gtfs_files <- gtfs_files[!grepl("merged", basename(gtfs_files))]  # Exclude existing merged files

if (length(gtfs_files) == 0) {
  stop("No GTFS zip files found")
}

cat(sprintf("\nFound %d GTFS files:\n", length(gtfs_files)))
for (f in gtfs_files) {
  cat(sprintf("  - %s (%.1f MB)\n", basename(f), file.size(f) / 1e6))
}

# ============================================================================
# VALIDATION
# ============================================================================
cat("\n" , strrep("=", 60), "\n")
cat("VALIDATING GTFS FILES\n")
cat(strrep("=", 60), "\n")

gtfs_list <- list()
validation_results <- list()

for (gtfs_file in gtfs_files) {
  fname <- basename(gtfs_file)
  cat(sprintf("\n--- %s ---\n", fname))

  # Read GTFS
  gtfs <- tryCatch({
    read_gtfs(gtfs_file)
  }, error = function(e) {
    cat(sprintf("  ERROR reading: %s\n", e$message))
    return(NULL)
  })

  if (is.null(gtfs)) {
    validation_results[[fname]] <- "READ_ERROR"
    next
  }

  # Basic stats
  n_routes <- if (!is.null(gtfs$routes)) nrow(gtfs$routes) else 0
  n_stops <- if (!is.null(gtfs$stops)) nrow(gtfs$stops) else 0
  n_trips <- if (!is.null(gtfs$trips)) nrow(gtfs$trips) else 0

  cat(sprintf("  Routes: %d, Stops: %d, Trips: %d\n", n_routes, n_stops, n_trips))

  # Check route types
  if (!is.null(gtfs$routes) && "route_type" %in% names(gtfs$routes)) {
    route_types <- table(gtfs$routes$route_type)
    type_names <- c("0" = "Tram", "1" = "Subway", "2" = "Rail", "3" = "Bus",
                    "4" = "Ferry", "5" = "Cable", "6" = "Gondola", "7" = "Funicular")
    for (rt in names(route_types)) {
      type_name <- if (rt %in% names(type_names)) type_names[rt] else "Other"
      cat(sprintf("    Route type %s (%s): %d routes\n", rt, type_name, route_types[rt]))
    }
  }

  # Check service dates
  if (!is.null(gtfs$calendar) && nrow(gtfs$calendar) > 0) {
    # Handle both numeric and character date formats
    start_col <- gtfs$calendar$start_date
    end_col <- gtfs$calendar$end_date
    if (is.numeric(start_col)) {
      start_dates <- as.Date(as.character(start_col), format = "%Y%m%d")
      end_dates <- as.Date(as.character(end_col), format = "%Y%m%d")
    } else {
      start_dates <- as.Date(start_col, format = "%Y%m%d")
      end_dates <- as.Date(end_col, format = "%Y%m%d")
    }
    cat(sprintf("  Service period: %s to %s\n",
                min(start_dates, na.rm = TRUE),
                max(end_dates, na.rm = TRUE)))
  } else if (!is.null(gtfs$calendar_dates) && nrow(gtfs$calendar_dates) > 0) {
    date_col <- gtfs$calendar_dates$date
    if (is.numeric(date_col)) {
      dates <- as.Date(as.character(date_col), format = "%Y%m%d")
    } else {
      # Remove quotes if present
      date_col <- gsub('"', '', as.character(date_col))
      dates <- as.Date(date_col, format = "%Y%m%d")
    }
    cat(sprintf("  Service dates: %s to %s (%d entries)\n",
                min(dates, na.rm = TRUE),
                max(dates, na.rm = TRUE),
                length(dates)))
  } else {
    cat("  Service dates: NOT FOUND\n")
  }

  # Validate using gtfstools (skip detailed validation, just check structure)
  # Note: validate_gtfs() in newer versions requires output_path, so we do basic checks
  validation_status <- "VALID"

  # Basic structural validation
  required_files <- c("agency", "routes", "trips", "stops", "stop_times")
  missing_files <- required_files[!required_files %in% names(gtfs)]

  if (length(missing_files) > 0) {
    cat(sprintf("  WARNING: Missing required tables: %s\n", paste(missing_files, collapse = ", ")))
    validation_status <- "ISSUES"
  }

  # Check for empty tables
  empty_tables <- c()
  for (tbl in names(gtfs)) {
    if (is.data.frame(gtfs[[tbl]]) && nrow(gtfs[[tbl]]) == 0) {
      empty_tables <- c(empty_tables, tbl)
    }
  }
  if (length(empty_tables) > 0) {
    cat(sprintf("  WARNING: Empty tables: %s\n", paste(empty_tables, collapse = ", ")))
    validation_status <- "ISSUES"
  }

  # Check service dates
  has_calendar <- !is.null(gtfs$calendar) && nrow(gtfs$calendar) > 0
  has_calendar_dates <- !is.null(gtfs$calendar_dates) && nrow(gtfs$calendar_dates) > 0

  if (!has_calendar && !has_calendar_dates) {
    cat("  WARNING: No calendar or calendar_dates table\n")
    validation_status <- "ISSUES"
  }

  if (validation_status == "VALID") {
    cat("  Validation: PASSED (basic checks)\n")
  }

  validation_results[[fname]] <- validation_status

  # Filter to rail only if requested
  if (opt$rail_only && !is.null(gtfs$routes)) {
    rail_types <- c(0, 1, 2)  # Tram, Subway, Rail
    rail_routes <- gtfs$routes$route_id[gtfs$routes$route_type %in% rail_types]

    if (length(rail_routes) == 0) {
      cat("  Skipping: No rail routes\n")
      next
    }

    gtfs <- filter_by_route_id(gtfs, rail_routes)
    cat(sprintf("  Filtered to %d rail routes\n", length(rail_routes)))
  }

  # Filter by date if requested
  if (!is.null(opt$filter_date)) {
    filter_date <- as.Date(opt$filter_date)
    gtfs <- tryCatch({
      filter_by_service_date(gtfs, filter_date)
    }, error = function(e) {
      cat(sprintf("  Warning: Could not filter by date: %s\n", e$message))
      gtfs
    })
  }

  gtfs_list[[fname]] <- gtfs
}

# ============================================================================
# VALIDATION SUMMARY
# ============================================================================
cat("\n", strrep("=", 60), "\n")
cat("VALIDATION SUMMARY\n")
cat(strrep("=", 60), "\n\n")

valid_count <- sum(unlist(validation_results) == "VALID")
issue_count <- sum(unlist(validation_results) == "ISSUES")
error_count <- sum(unlist(validation_results) == "READ_ERROR")

cat(sprintf("Valid: %d, With issues: %d, Read errors: %d\n", valid_count, issue_count, error_count))
cat(sprintf("Files to merge: %d\n", length(gtfs_list)))

if (opt$validate_only) {
  cat("\n--validate_only specified, skipping merge\n")
  quit(status = 0)
}

# ============================================================================
# MERGE
# ============================================================================
if (length(gtfs_list) == 0) {
  stop("No valid GTFS files to merge")
}

cat("\n", strrep("=", 60), "\n")
cat("MERGING GTFS FILES\n")
cat(strrep("=", 60), "\n\n")

cat(sprintf("Merging %d GTFS files...\n", length(gtfs_list)))

# Merge all GTFS files
merged_gtfs <- tryCatch({
  # merge_gtfs takes a list of gtfs objects
  if (length(gtfs_list) == 1) {
    gtfs_list[[1]]
  } else {
    merge_gtfs(gtfs_list)
  }
}, error = function(e) {
  cat(sprintf("ERROR merging: %s\n", e$message))
  return(NULL)
})

if (is.null(merged_gtfs)) {
  stop("Failed to merge GTFS files")
}

# Stats on merged GTFS
cat("\nMerged GTFS stats:\n")
cat(sprintf("  Routes: %d\n", nrow(merged_gtfs$routes)))
cat(sprintf("  Stops: %d\n", nrow(merged_gtfs$stops)))
cat(sprintf("  Trips: %d\n", nrow(merged_gtfs$trips)))
cat(sprintf("  Stop times: %d\n", nrow(merged_gtfs$stop_times)))

if (!is.null(merged_gtfs$routes) && "route_type" %in% names(merged_gtfs$routes)) {
  cat("\nRoute types in merged GTFS:\n")
  route_types <- table(merged_gtfs$routes$route_type)
  type_names <- c("0" = "Tram", "1" = "Subway", "2" = "Rail", "3" = "Bus",
                  "4" = "Ferry", "5" = "Cable", "6" = "Gondola", "7" = "Funicular")
  for (rt in names(route_types)) {
    type_name <- if (rt %in% names(type_names)) type_names[rt] else "Other"
    cat(sprintf("  %s (%s): %d routes\n", rt, type_name, route_types[rt]))
  }
}

# Save merged GTFS
output_file <- if (!is.null(opt$output)) {
  file.path(gtfs_dir, opt$output)
} else {
  file.path(gtfs_dir, "merged_gtfs.zip")
}

cat(sprintf("\nWriting merged GTFS to: %s\n", output_file))

write_gtfs(merged_gtfs, output_file)

cat(sprintf("Done! Merged file size: %.1f MB\n", file.size(output_file) / 1e6))

# Final validation of merged file
cat("\nValidating merged GTFS...\n")
merged_validation <- tryCatch({
  validate_gtfs(merged_gtfs)
}, error = function(e) {
  cat(sprintf("Validation error: %s\n", e$message))
  return(NULL)
})

if (!is.null(merged_validation) && nrow(merged_validation) == 0) {
  cat("Merged GTFS validation: PASSED\n")
} else if (!is.null(merged_validation)) {
  cat(sprintf("Merged GTFS validation: %d issues\n", nrow(merged_validation)))
}

cat("\n", strrep("=", 60), "\n")
cat("COMPLETE\n")
cat(strrep("=", 60), "\n")
