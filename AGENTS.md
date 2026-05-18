# AGENTS.md

## Project Overview

This project analyzes the **spatial geography of social mixing at points of interest (POIs)**, examining how transit accessibility shapes the geographic distribution of visitor diversity across urban space. It is a cross-national comparison between Swedish cities (Stockholm, Gothenburg, Malmö, Uppsala, Västerås, Örebro, Linköping, Helsingborg, Lund) and US cities (New York, Washington DC, Atlanta).

**Target output**: Original research article for *Frontiers in Big Data* — Special Issue on Social Impacts of Human Mobility.

**Core research questions**:
1. How does the spatial structure of transit accessibility shape the geographic distribution of socially mixed venues, and does this relationship exhibit spatial spillover effects?
2. Does the spatial pattern of the transit–mixing relationship differ systematically between transit-oriented and car-dependent cities?

## Key Concepts

- **Visitor diversity**: Entropy of socioeconomic composition of a POI's visitors (from origin tract income distributions)
- **Transit catchment diversity**: Entropy of residential population reachable within 45 minutes by public transit from a POI
- **Spatial spillover**: The hypothesis that visitor diversity at a POI depends on diversity at nearby POIs, particularly those near transit infrastructure
- **Transit nodes as spatial bridges**: Theoretical framework—transit concentrates flows from diverse neighborhoods at common nodes, creating pedestrian-scale dispersal of diversity into surrounding venues

## Project Structure

```
geo-social-mixing/
├── config/                 # Configuration files
│   └── unified_poi_categories.yaml  # POI category harmonization
├── data/                   # Data documentation (see data/README.md)
├── dbs/                    # Raw/processed data (gitignored)
├── src/
│   ├── data/              # Data loading and harmonization
│   ├── features/          # Feature engineering (diversity indices)
│   ├── analysis/          # LISA clustering, hotspot analysis
│   └── models/            # OLS, SLM, GWR regression models
├── r_scripts/             # R code for transit routing (r5r)
├── notebooks/             # Exploration and analysis notebooks
├── outputs/               # Results (figures, tables, manuscript)
└── .devcontainer/         # Docker development environment
```

## Data Sources

| Dataset | Location | Format | Notes |
|---------|----------|--------|-------|
| Swedish GPS traces (2024) | `dbs/stops/` | Parquet | Processed stay detections |
| Swedish census (DeSO) | `dbs/deso/` | Parquet | Population, income by zone |
| Swedish GTFS | `dbs/gtfs/` | ZIP | Samtrafiken transit schedules |
| Swedish weekly patterns | `dbs/sweden_weekly_patterns/` | Parquet | Aggregated POI visits |
| US foot traffic | `dbs/us_foot_traffic/` | Parquet | Advan Weekly Patterns Plus |
| US census (ACS) | `dbs/us_census/` | Parquet | Tract-level socioeconomics |
| US GTFS | `dbs/gtfs/` | ZIP | City-specific transit feeds |
| POIs (Sweden) | `dbs/poi_se/` | Parquet | SafeGraph Global Places |
| Transit routing | `dbs/routing/` | Parquet | r5r catchment outputs |

## Technical Stack

**Python** (primary):
- `pandas`, `geopandas` — Data manipulation
- `pysal`, `spreg`, `libpysal`, `esda` — Spatial analysis and regression
- `mgwr` — Geographically Weighted Regression
- `infostop` — Stay detection from GPS traces
- `pyspark` — Large-scale Swedish data processing

**R** (transit routing only):
- `r5r` — GTFS-based transit accessibility computation
- Called from Python via subprocess or rpy2

## Key Parameters

Defined in `config/config.yaml`:

```yaml
stay_detection:
  r1: 30          # meters (spatial threshold)
  tmin: 15        # minutes (minimum stay duration)

transit:
  max_travel_time: 45    # minutes (primary analysis)
  max_walk_distance: 800 # meters

spatial_weights:
  knn_k: 10
  distance_band: 800     # meters
  transit_proximity: 800 # meters

poi:
  min_visits: 50         # minimum for inclusion
  min_origin_tracts: 5   # minimum for reliable diversity
```

## Execution Phases

### Phase 1: Data Harmonization & Feature Engineering (Months 1-2)

**Goal**: Create unified analysis dataset with POI-level visitor diversity, transit catchment diversity, and spatial weights matrices.

**Key tasks**:
1. Process Swedish GPS → stays → POI visits → tract-to-POI flows
2. Load and filter US foot traffic data
3. Harmonize to common schema (country, city, origin_tract, poi_id, visits)
4. Compute visitor diversity index (entropy) for each POI
5. Compute transit catchment diversity via r5r routing
6. Compute walkshed neighborhood diversity (800m buffer)
7. Construct spatial weights: W₁ (distance), W₂ (transit-proximity), W₃ (transit-network)
8. Assemble final `analysis_dataset.parquet`

**Validation**: ~50K-100K POIs per city, VIF < 5, no missing core variables

### Phase 2: Descriptive & Global Spatial Models (Months 3-4)

**Goal**: Establish spatial autocorrelation patterns and estimate global spatial regression models.

**Key tasks**:
1. Map visitor diversity distributions, compute Moran's I
2. LISA cluster analysis (High-High, Low-Low clusters)
3. OLS baseline → test residual spatial autocorrelation
4. LM diagnostics → choose SLM vs SEM
5. Estimate Spatial Lag Models with W₁, W₂, W₃
6. **Critical test**: Is ρ(W₂) > ρ(W₁)? (supports transit-specific spillover)
7. Estimate spillover distance (test distance bands 200m–2000m)

**Key output**: `slm_comparison.csv` with ρ coefficients across weights specifications

### Phase 3: GWR & Hotspot Identification (Month 5)

**Goal**: Capture spatial heterogeneity in transit–mixing relationship.

**Key tasks**:
1. Estimate GWR with adaptive bandwidth (AICc selection)
2. Extract local coefficient surfaces for transit catchment
3. Quantify heterogeneity: CV of local coefficients by city/country
4. **Hypothesis test**: US cities have higher coefficient variance than Swedish cities
5. Identify transit mixing hotspots (high local β, significant, clustered)
6. Estimate MGWR to compare bandwidth scales across predictors

**Key output**: `gwr_local_coefficients.parquet`, hotspot maps

### Phase 4: Robustness & Writing (Months 6-7)

**Goal**: Validate findings and produce publication-ready outputs.

**Robustness checks**:
- Alternative DV (dissimilarity index vs entropy)
- Alternative spatial weights (distance bands 400m, 1000m)
- VIF and endogeneity diagnostics

## Common Commands

```bash
# Initialize environment (using devcontainer or manual setup)
micromamba create -f .devcontainer/environment.yml -n geoenv
micromamba activate geoenv

# Run spatial analysis
python src/analysis/compute_lisa_clusters.py
python src/models/run_spatial_spillover_analysis.py
python src/models/run_mgwr_analysis.py

# Run R transit routing
Rscript r_scripts/compute_transit_catchment.R --city stockholm
```

## Code Style & Conventions

- **Python**: Follow PEP 8, use type hints, docstrings (Google style)
- **File naming**: snake_case for all files and variables
- **Data files**: Parquet for tabular data, GeoPackage for spatial data
- **Outputs**: Figures as PNG (300 DPI), tables as CSV and LaTeX

## Important Implementation Notes

### Swedish Data Processing
The Swedish GPS data is individual-level and must be aggregated to tract-to-POI flows to match the US data structure. Use population weights (IPW) during aggregation to correct for sampling bias.

```python
# Key aggregation logic
flows = (device_visits
    .merge(device_homes[['device_id', 'home_deso_id', 'population_weight']])
    .groupby(['home_deso_id', 'poi_id'])
    .agg(weighted_visits=('population_weight', 'sum'))
    .reset_index())
```

### Spatial Weights Construction
Two types of weights matrices test different spillover mechanisms:

```python
# W₁: Distance-decay with cutoff (800m, α=1)
# w_ij = 1/d_ij if d_ij ≤ 800m, else 0
W1 = build_distance_decay_weights(pois_gdf, cutoff_m=800, alpha=1)

# W₂: Transit-proximity (POIs connected if both near same transit stop within 400m)
W2 = build_transit_proximity_weights(pois_gdf, transit_stops, threshold=400)
```

### GWR Estimation
Use `mgwr` package with adaptive bandwidth:

```python
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW

# Bandwidth selection
selector = Sel_BW(coords, y, X, kernel='adaptive_bisquare')
bw = selector.search(criterion='AICc')

# Model estimation
model = GWR(coords, y, X, bw, kernel='adaptive_bisquare')
results = model.fit()
```

### Transit Accessibility (R)
The r5r routing must be run in R. Call from Python:

```python
import subprocess
subprocess.run([
    'Rscript', 'r_scripts/compute_transit_catchment.R',
    '--city', city,
    '--max_time', '45',
    '--output', f'data/interim/{city}/catchment.parquet'
])
```

## Debugging Tips

- **Memory issues with Swedish data**: Process by city, use Dask or PySpark for stay detection
- **r5r crashes**: Ensure Java heap size is set (`options(java.parameters = "-Xmx8G")`)
- **Spatial weights errors**: Check for duplicate POI coordinates (jitter if needed)
- **GWR convergence**: If bandwidth selection fails, try fixed bandwidth first

## Key Validation Checks

| Checkpoint | Expected Value | Notes |
|------------|----------------|-------|
| Swedish device count | ~6.5M | Full 2024 data |
| Home detection rate | ~71% | HoWDe algorithm |
| POI match rate | ~46% | Tiered spatial assignment |
| Moran's I (birth, US) | 0.67-0.83 | Strong spatial clustering |
| Moran's I (birth, Sweden) | 0.08-0.24 | Weak spatial clustering |
| SLM ρ stationarity | \|ρ\| < 1 | Required for valid interpretation |
| GWR R² (US, birth) | 0.67-0.80 | Higher fit than Sweden |

## References

Key methodological references embedded in the analysis:

- Moro et al. (2021) *Nature Communications* — Experienced income segregation decomposition
- Nilforoshan et al. (2023) *Nature* — City hubs and segregation
- Liao et al. (2025) *npj Sustainable Mobility and Transport* — Limited mobility and segregation (basis for Swedish data processing)
- Fotheringham et al. (2002) — GWR methodology
- Anselin (1988) — Spatial econometrics and LM diagnostics

## Contact & Context

This project extends prior work on segregation in Sweden (Liao et al. 2025) by:
1. Moving from individual-level to POI-level analysis
2. Adding cross-national comparison with US cities
3. Introducing spatially explicit methods (spatial lag, GWR)
4. Testing transit-specific spillover mechanisms

The key novelty is demonstrating *where* transit reshapes social mixing geography, not just *whether* it does.