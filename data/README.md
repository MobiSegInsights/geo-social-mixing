# Data Sources and Access

This document describes the data sources used in this study and how to access them.

## Overview

The analysis requires the following data types:
1. **Points of Interest (POIs)**: Venue locations and categories
2. **Mobility data**: Visitor flows to POIs
3. **Socioeconomic data**: Population characteristics by census zone
4. **Transit network data**: GTFS feeds for public transit routing

## Data Sources

### Sweden

#### POI Data
- **Source**: SafeGraph Global Places
- **Access**: Commercial license required
- **Coverage**: Comprehensive POI database with categories and coordinates

SafeGraph. (2022). Global Places (POI) & Geometry [Dataset]. Dewey Data. https://doi.org/10.82551/SMXB-1K04

#### Mobility Data
- **Source**: Anonymized GPS trajectories from mobile phone applications
- **Access**: Restricted; contact the author for collaboration inquiries
- **Coverage**: ~6.5 million devices throughout 2024
- **Processing**: Stay detection via Infostop algorithm, home detection via HoWDe

#### Socioeconomic Data
- **Source**: Statistics Sweden (SCB) register-based statistics
- **Access**: Public via [SCB](https://www.scb.se/)
- **Geography**: DeSO zones (Demografiska statistikområden)
- **Variables**: Birth background (Sweden-born, Europe-born, non-Europe-born), income quartiles

#### Transit Network
- **Source**: Samtrafiken AB
- **Access**: Public via [Trafiklab](https://www.trafiklab.se/)
- **Format**: GTFS static feeds
- **Coverage**: All Swedish public transit operators

### United States

#### POI and Mobility Data
- **Source**: Advan Weekly Patterns Plus
- **Access**: Commercial license required
- **Coverage**: New York, Washington DC, Atlanta metropolitan areas
- **Period**: January 2024 - January 2025 (53 weeks)
- **Variables**: Weekly visitor counts, visitor home CBG distributions

Advan Research. (2025). Foot Traffic / Weekly Patterns Plus [Dataset]. Dewey Data. https://doi.org/10.82551/C103-N851

#### Socioeconomic Data
- **Source**: American Community Survey (ACS) 2024 5-year estimates
- **Access**: Public via [Census Bureau API](https://www.census.gov/data/developers/data-sets/acs-5year.html)
- **Geography**: Census tracts
- **Variables**: 
  - Household income distribution (Table B19001)
  - Nativity and citizenship (Table B05001)
  - Race/ethnicity (Table B03002)

#### Transit Networks

| City | Agency | Source |
|------|--------|--------|
| New York | Multi | [Link](https://data.ny.gov/Transportation/MTA-General-Transit-Feed-Specification-GTFS-Static/fgm6-ccue/about_data) |
| Washington DC | Multi | [Link](https://developer.wmata.com/) |
| Atlanta | Multi | [Link](https://opendata.atlantaregional.com/datasets/0e58b8cd1a4248e3a019cca4fc79a919/about) |

## Data Schema

### Processed Mobility Data

After preprocessing, mobility data is harmonized to a common schema:

| Column | Type | Description |
|--------|------|-------------|
| `poi_id` | string | Unique POI identifier |
| `origin_zone` | string | Home census zone (DeSO/tract) |
| `visits` | float | Weighted visit count |
| `week` | date | Week of observation |

### Diversity Metrics

Computed at the POI level:

| Column | Type | Description |
|--------|------|-------------|
| `poi_id` | string | Unique POI identifier |
| `visitor_birth_entropy` | float | Normalized entropy of visitor birth backgrounds |
| `visitor_income_entropy` | float | Normalized entropy of visitor income quartiles |
| `residential_birth_entropy` | float | Entropy of POI's home zone |
| `residential_income_entropy` | float | Entropy of POI's home zone |
| `catchment_birth_entropy` | float | Transit catchment population entropy |
| `catchment_income_entropy` | float | Transit catchment population entropy |

### Transit Catchment

Computed via r5r routing:

| Column | Type | Description |
|--------|------|-------------|
| `poi_id` | string | POI identifier |
| `reachable_zones` | list | Census zones reachable within 45 min |
| `catchment_population` | float | Total population in catchment |

## File Organization

Raw data should be placed in the `dbs/` directory (gitignored):

```
dbs/
├── sweden_weekly_patterns/     # Processed Swedish mobility data
├── us_foot_traffic/            # US Advan Weekly Patterns
│   └── cities/                 # Filtered by metropolitan area
├── us_census/                  # ACS tract-level data
├── deso/                       # Swedish DeSO zone data
├── gtfs/                       # Transit feeds by city
├── routing/                    # r5r routing outputs
└── poi_se/                     # Swedish POI data
```

## Reproduction Notes

1. **Transit catchments**: Computed using r5r with 45-minute travel time threshold
2. **Diversity indices**: Normalized Shannon entropy (0-1 scale)
3. **Spatial weights**: Distance-decay (800m cutoff) and transit-proximity (400m)
4. **Minimum thresholds**: 50 visits and 5 origin zones per POI

## Contact

For data access inquiries, contact Yuan Liao (yuan.liao@keg.lu.se).
