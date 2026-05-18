# The spatial structure of social mixing

This repository contains code and analysis for the paper:

> **Where diverse populations gather: Transit accessibility and the spatial structure of social mixing**  
> Yuan Liao  

## Overview

Urban venues serve as arenas for social mixing, where individuals from different socioeconomic backgrounds share space during daily activities. This study examines how transit accessibility influences visitor diversity at points of interest (POIs) across major cities in Sweden and the United States.

Using mobile phone GPS traces from Sweden and aggregated foot traffic data from the US (2024), we compute visitor diversity indices based on the socioeconomic composition of visitors' home neighborhoods. We employ spatial regression models and geographically weighted regression (GWR) to test whether transit catchment diversity predicts visitor diversity, and examine the spatial heterogeneity of this relationship.

## Study Areas
- **Sweden**: Stockholm, Gothenburg, Malmö, Uppsala, Västerås, Örebro, Linköping, Helsingborg, Lund (9 cities)
- **United States**: New York, Washington DC, Atlanta (3 cities)

## Repository Structure

```
geo-social-mixing/
├── config/                     # Configuration files
│   └── unified_poi_categories.yaml   # POI category harmonization
│
├── data/                       # Data documentation (see data/README.md)
│
├── r_scripts/                  # R scripts for transit routing
│   ├── clean_gtfs.R           # GTFS data cleaning
│   ├── compute_transit_catchment.R   # Transit accessibility computation
│   └── validate_merge_gtfs.R  # GTFS validation
│
├── src/                        # Python source code
│   ├── data/                  # Data processing and harmonization
│   ├── features/              # Feature engineering (diversity, spatial controls)
│   ├── analysis/              # Spatial analysis (LISA, hotspots, correlations)
│   ├── models/                # Regression models (OLS, SLM, GWR, robustness)
│   └── visualization/         # Plotting utilities (GWR coefficients)
│
├── notebooks/                  # Jupyter notebooks
│   ├── 01-14_*.ipynb         # Data preparation and exploration
│   ├── 15_compute_diversity_metrics.ipynb   # Core diversity computation
│   └── 16_publication_figures.ipynb         # Manuscript figures and tables
│
├── outputs/                    # Results
│   ├── figures/               # Publication figures
│   ├── tables/                # Result tables
│   ├── phase2/                # Spatial regression results
│   ├── phase3/                # GWR results
│   └── robustness/            # Robustness check results
│
└── .devcontainer/              # Docker development environment
    └── environment.yml        # Conda environment specification
```

## Installation

### Using Docker (Recommended)

The repository includes a Dev Container configuration for VS Code:

1. Install [Docker](https://www.docker.com/) and [VS Code](https://code.visualstudio.com/)
2. Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
3. Open the repository in VS Code
4. Click "Reopen in Container" when prompted

### Manual Installation

```bash
# Create conda environment
conda env create -f .devcontainer/environment.yml
conda activate geoenv

# Install R packages (for transit routing)
R -e "install.packages(c('r5r', 'data.table', 'gtfstools'), repos='https://cloud.r-project.org')"
```

## Data Access

See [`data/README.md`](data/README.md) for detailed information on data sources and access.

**Summary of data sources:**
- **Sweden POIs**: SafeGraph Global Places $^1$
- **Sweden mobility**: Anonymized GPS trajectories (restricted access)
- **US foot traffic**: Advan Weekly Patterns Plus $^2$
- **US Census**: American Community Survey 2024 5-year estimates
- **Transit networks**: GTFS feeds from respective transit agencies

1 SafeGraph. (2022). Global Places (POI) & Geometry [Dataset]. Dewey Data. https://doi.org/10.82551/SMXB-1K04

2 Advan Research. (2025). Foot Traffic / Weekly Patterns Plus [Dataset]. Dewey Data. https://doi.org/10.82551/C103-N851

## Usage

### Core Analysis Pipeline

The analysis proceeds in the below steps with notebooks in the brackets serving data exploration and verification purposes:

**Swedish mobility data processing (Steps 1-6)**
1. `src/data/stop_detection.py` — Detect stays from GPS traces (`notebooks/01`)
2. `src/data/home_work_detection.py` — Identify home/work locations (`notebooks/02`)
3. `src/data/link_home_buildings.py` — Link homes to building footprints
4. `src/data/harmonize_deso.py` — Harmonize DeSO zone boundaries (`notebooks/03`)
5. `src/data/assign_home_deso_ipw.py` — Assign home DeSO with population weights
6. `src/data/assign_poi_tiered.py` — Tiered POI assignment (`notebooks/05`)

**POI and US data preparation (Steps 7-9)**

7. `notebooks/04` — Download SafeGraph POIs for Sweden (manual)
8. `notebooks/06` — Download US foot traffic data
9. `notebooks/07` — POI category alignment → `src/data/category_mapper.py`

**Flow aggregation and filtering (Steps 10-12)**

10. `src/data/aggregate_swedish_flows.py` — Aggregate Swedish tract-to-POI flows (`notebooks/10`)
11. `src/data/filter_us_cities.py` — Filter US data to study cities (`notebooks/11`)
12. `notebooks/12` — Download US census data

**Transit catchment and diversity metrics (Steps 13-15)**

13. `notebooks/13-14` — Transit catchment data preparation → `r_scripts/` (r5r routing)
14. `src/features/filter_sweden_gtfs_by_county.py` — Filter Swedish GTFS by county
15. `notebooks/15` — Compute diversity metrics using `src/features/diversity_metrics.py`

**Feature engineering (Steps 16-17)**

16. `src/features/compute_geographic_catchment.py` — Geographic (walkshed) catchment diversity
17. `src/features/compute_spatial_controls.py` — Distance to center, POI density, transit proximity

**Spatial analysis and modeling (Steps 18-21)**

18. `src/analysis/compute_lisa_clusters.py` — LISA cluster analysis
19. `src/models/run_spatial_spillover_analysis.py` — OLS and spatial lag models
20. `src/models/run_mgwr_analysis.py` — Geographically weighted regression
21. `src/analysis/hotspot_transit_proximity.py` — Transit proximity hotspot analysis

**Robustness checks (Step 22)**

22. `src/models/run_stepwise_robustness.py` — Stepwise OLS with incremental controls

**Results and figures (Step 23)**

23. `notebooks/16` — Statistical tests, figures, and tables for publication

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Yuan Liao  
Department of Human Geography, Lund University  
Email: yuan.liao@keg.lu.se

## Acknowledgments

This research is funded by the Swedish Research Council (Project Number 2022-06215).

The author acknowledges Jorge Gil for providing the mobile phone application data (Sweden).
