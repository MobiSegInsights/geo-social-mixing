"""
Diversity Metrics for Visitor Mixing Analysis

Two sets of metrics for income and birth background:
- Set 1: Entropy-based (multi-category evenness)
- Set 2: ICE-based (bipolar concentration)

Applied to three contexts:
- Residential: Tract/DeSO where POI is located
- Transit Catchment: Reachable tracts within travel time
- Actual Visitors: From foot traffic weighted by home tract

"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Union, Literal

# =============================================================================
# ENTROPY-BASED METRICS (Set 1)
# =============================================================================

def entropy(proportions: np.ndarray, normalize: bool = False) -> float:
    """
    Compute Shannon entropy from proportions.

    H = -Σ p_i * log(p_i)

    Parameters
    ----------
    proportions : array-like
        Proportions that sum to 1 (or will be normalized)
    normalize : bool
        If True, return H / log(k) to get range [0, 1]

    Returns
    -------
    float
        Entropy value. Returns 0 if all mass in one category.
    """
    p = np.asarray(proportions, dtype=float)

    # Handle edge cases
    if p.sum() == 0:
        return np.nan

    # Normalize to ensure sum = 1
    p = p / p.sum()

    # Store original category count BEFORE filtering zeros (for normalization)
    n_categories = len(p)

    # Filter out zeros (0 * log(0) = 0 by convention)
    p = p[p > 0]

    H = -np.sum(p * np.log(p))

    if normalize and n_categories > 1:
        H = H / np.log(n_categories)

    return H


def entropy_income_quartiles(q1: float, q2: float, q3: float, q4: float,
                              normalize: bool = False) -> float:
    """
    Compute entropy across 4 income quartiles.

    Parameters
    ----------
    q1, q2, q3, q4 : float
        Population/visitor shares in each income quartile
    normalize : bool
        If True, normalize by log(4) for range [0, 1]

    Returns
    -------
    float
        Entropy value. Max entropy (uniform) = log(4) ≈ 1.386 or 1.0 if normalized
    """
    return entropy([q1, q2, q3, q4], normalize=normalize)


def entropy_birth_binary(p_native: float, p_foreign: float,
                         normalize: bool = False) -> float:
    """
    Compute binary entropy for native vs foreign-born.

    Parameters
    ----------
    p_native : float
        Proportion native-born
    p_foreign : float
        Proportion foreign-born (outside EU for Sweden, all foreign for US)
    normalize : bool
        If True, normalize by log(2) for range [0, 1]

    Returns
    -------
    float
        Entropy value. Max entropy (50-50) = log(2) ≈ 0.693 or 1.0 if normalized
    """
    return entropy([p_native, p_foreign], normalize=normalize)


# =============================================================================
# ICE-BASED METRICS (Set 2) - Bipolar, No Normalization
# =============================================================================

def ice_bipolar(p_high: float, p_low: float) -> float:
    """
    Compute bipolar ICE without normalization.

    ICE = (p_high - p_low) / (p_high + p_low)

    Parameters
    ----------
    p_high : float
        Proportion in "advantaged" group (e.g., Q4 income, native-born)
    p_low : float
        Proportion in "disadvantaged" group (e.g., Q1 income, foreign-born)

    Returns
    -------
    float
        ICE value in range [-1, +1]
        +1 = all in high group
        -1 = all in low group
         0 = equal proportions
    """
    total = p_high + p_low
    if total == 0:
        return np.nan
    return (p_high - p_low) / total


def ice_income(p_q4: float, p_q1: float) -> float:
    """
    Compute ICE for income: highest (Q4) vs lowest (Q1) quartile.

    ICE_income = (p_Q4 - p_Q1) / (p_Q4 + p_Q1)

    Parameters
    ----------
    p_q4 : float
        Proportion in highest income quartile
    p_q1 : float
        Proportion in lowest income quartile

    Returns
    -------
    float
        +1 = all high income, -1 = all low income, 0 = equal mix
    """
    return ice_bipolar(p_q4, p_q1)


def ice_birth(p_native: float, p_foreign: float) -> float:
    """
    Compute ICE for birth background: native vs foreign-born.

    ICE_birth = (p_native - p_foreign) / (p_native + p_foreign)

    Parameters
    ----------
    p_native : float
        Proportion native-born (Sweden-born for SE, native for US)
    p_foreign : float
        Proportion foreign-born (outside EU for SE, all foreign for US)

    Returns
    -------
    float
        +1 = all native, -1 = all foreign, 0 = equal mix
    """
    return ice_bipolar(p_native, p_foreign)


# =============================================================================
# AGGREGATION FUNCTIONS FOR DIFFERENT CONTEXTS
# =============================================================================

def compute_tract_composition_sweden(deso_df: pd.DataFrame, deso_code: str) -> dict:
    """
    Get demographic composition for a single Swedish DeSO zone.

    Parameters
    ----------
    deso_df : pd.DataFrame
        DeSO data with columns: deso_code, birth_sweden, birth_europe, birth_other,
        income_q1_pct, income_q2_pct, income_q3_pct, income_q4_pct, pop_total
    deso_code : str
        DeSO zone identifier

    Returns
    -------
    dict
        Composition with keys: p_native, p_foreign, p_q1, p_q2, p_q3, p_q4, pop_total
    """
    row = deso_df[deso_df['deso_code'] == deso_code]
    if len(row) == 0:
        return None
    row = row.iloc[0]

    # Birth: Sweden-born vs Other (outside EU), excluding Europe-born
    total_birth = row['birth_sweden'] + row['birth_other']
    if total_birth > 0:
        p_native = row['birth_sweden'] / total_birth
        p_foreign = row['birth_other'] / total_birth
    else:
        p_native = p_foreign = np.nan

    # Income quartiles (already percentages, normalize to sum to 1)
    q_total = row['income_q1_pct'] + row['income_q2_pct'] + row['income_q3_pct'] + row['income_q4_pct']
    if q_total > 0:
        p_q1 = row['income_q1_pct'] / q_total
        p_q2 = row['income_q2_pct'] / q_total
        p_q3 = row['income_q3_pct'] / q_total
        p_q4 = row['income_q4_pct'] / q_total
    else:
        p_q1 = p_q2 = p_q3 = p_q4 = np.nan

    return {
        'p_native': p_native,
        'p_foreign': p_foreign,
        'p_q1': p_q1,
        'p_q2': p_q2,
        'p_q3': p_q3,
        'p_q4': p_q4,
        'pop_total': row['pop_total']
    }


def compute_tract_composition_us(tract_df: pd.DataFrame, geoid: str) -> dict:
    """
    Get demographic composition for a single US census tract.

    Parameters
    ----------
    tract_df : pd.DataFrame
        US census data with columns: GEOID, native_born, foreign_born,
        median_household_income, income_quintile, total_population
    geoid : str
        Census tract GEOID

    Returns
    -------
    dict
        Composition with keys: p_native, p_foreign, p_q1, p_q2, p_q3, p_q4,
        median_income, income_quintile, pop_total
    """
    row = tract_df[tract_df['GEOID'] == geoid]
    if len(row) == 0:
        return None
    row = row.iloc[0]

    # Birth: Native vs Foreign
    total_birth = row['native_born'] + row['foreign_born']
    if total_birth > 0:
        p_native = row['native_born'] / total_birth
        p_foreign = row['foreign_born'] / total_birth
    else:
        p_native = p_foreign = np.nan

    # Income: Store quintile for aggregation (will need to aggregate across tracts)
    # Individual tract has single median income, not quartile distribution
    return {
        'p_native': p_native,
        'p_foreign': p_foreign,
        'median_income': row['median_household_income'],
        'income_quintile': row['income_quintile'],
        'pop_total': row['total_population']
    }


def aggregate_composition_weighted(compositions: list[dict],
                                    weights: Optional[list[float]] = None) -> dict:
    """
    Aggregate demographic compositions across multiple tracts with weights.

    Parameters
    ----------
    compositions : list of dict
        List of composition dicts from compute_tract_composition_*
    weights : list of float, optional
        Weights for each tract (e.g., visit counts). If None, use population.

    Returns
    -------
    dict
        Aggregated composition
    """
    if not compositions:
        return None

    # Filter out None values
    compositions = [c for c in compositions if c is not None]
    if not compositions:
        return None

    # Default weights: population
    if weights is None:
        weights = [c.get('pop_total', 1) for c in compositions]

    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()  # Normalize

    result = {}

    # Aggregate birth proportions
    p_native = np.nansum([w * c['p_native'] for w, c in zip(weights, compositions)])
    p_foreign = np.nansum([w * c['p_foreign'] for w, c in zip(weights, compositions)])

    # Renormalize
    total = p_native + p_foreign
    if total > 0:
        result['p_native'] = p_native / total
        result['p_foreign'] = p_foreign / total
    else:
        result['p_native'] = result['p_foreign'] = np.nan

    # Aggregate income quartiles (if available - Sweden style)
    if 'p_q1' in compositions[0]:
        p_q1 = np.nansum([w * c['p_q1'] for w, c in zip(weights, compositions)])
        p_q2 = np.nansum([w * c['p_q2'] for w, c in zip(weights, compositions)])
        p_q3 = np.nansum([w * c['p_q3'] for w, c in zip(weights, compositions)])
        p_q4 = np.nansum([w * c['p_q4'] for w, c in zip(weights, compositions)])

        q_total = p_q1 + p_q2 + p_q3 + p_q4
        if q_total > 0:
            result['p_q1'] = p_q1 / q_total
            result['p_q2'] = p_q2 / q_total
            result['p_q3'] = p_q3 / q_total
            result['p_q4'] = p_q4 / q_total
        else:
            result['p_q1'] = result['p_q2'] = result['p_q3'] = result['p_q4'] = np.nan

    return result


def aggregate_composition_us_income_quartiles(tract_df: pd.DataFrame,
                                               geoids: list[str],
                                               weights: Optional[list[float]] = None) -> dict:
    """
    Aggregate US tract compositions and compute income quartile distribution.

    Since US tracts have median income (not quartile shares), we:
    1. Assign each tract to a quartile based on study-area income distribution
    2. Aggregate to get quartile proportions across tracts

    Parameters
    ----------
    tract_df : pd.DataFrame
        US census data with income_quintile column
    geoids : list of str
        List of tract GEOIDs to aggregate
    weights : list of float, optional
        Weights for each tract

    Returns
    -------
    dict
        Aggregated composition with p_q1, p_q2, p_q3, p_q4, p_native, p_foreign
    """
    # Filter to relevant tracts
    subset = tract_df[tract_df['GEOID'].isin(geoids)].copy()
    if len(subset) == 0:
        return None

    # Match weights to tracts
    if weights is not None:
        weight_map = dict(zip(geoids, weights))
        subset['weight'] = subset['GEOID'].map(weight_map).fillna(0)
    else:
        subset['weight'] = subset['total_population']

    total_weight = subset['weight'].sum()
    if total_weight == 0:
        return None

    # Aggregate birth background
    birth_weighted = subset[['native_born', 'foreign_born']].multiply(subset['weight'], axis=0)
    total_native = birth_weighted['native_born'].sum()
    total_foreign = birth_weighted['foreign_born'].sum()
    total_birth = total_native + total_foreign

    if total_birth > 0:
        p_native = total_native / total_birth
        p_foreign = total_foreign / total_birth
    else:
        p_native = p_foreign = np.nan

    # Aggregate income: map quintiles to quartiles and compute shares
    # Q1_lowest, Q2, Q3, Q4, Q5_highest -> combine Q4+Q5 for Q4
    quintile_map = {
        'Q1_lowest': 'q1',
        'Q2': 'q2',
        'Q3': 'q3',
        'Q4': 'q4',
        'Q5_highest': 'q4'  # Combine top two quintiles
    }
    subset['quartile'] = subset['income_quintile'].map(quintile_map)

    quartile_weights = subset.groupby('quartile')['weight'].sum()
    q_total = quartile_weights.sum()

    if q_total > 0:
        p_q1 = quartile_weights.get('q1', 0) / q_total
        p_q2 = quartile_weights.get('q2', 0) / q_total
        p_q3 = quartile_weights.get('q3', 0) / q_total
        p_q4 = quartile_weights.get('q4', 0) / q_total
    else:
        p_q1 = p_q2 = p_q3 = p_q4 = np.nan

    return {
        'p_native': p_native,
        'p_foreign': p_foreign,
        'p_q1': p_q1,
        'p_q2': p_q2,
        'p_q3': p_q3,
        'p_q4': p_q4
    }


# =============================================================================
# COMPUTE ALL METRICS FOR A POI
# =============================================================================

def compute_diversity_metrics(composition: dict) -> dict:
    """
    Compute all diversity metrics (Set 1 & Set 2) from a composition dict.

    Parameters
    ----------
    composition : dict
        Must contain: p_native, p_foreign, p_q1, p_q2, p_q3, p_q4

    Returns
    -------
    dict
        All metrics:
        - entropy_birth, entropy_birth_norm
        - entropy_income, entropy_income_norm
        - ice_birth
        - ice_income
    """
    if composition is None:
        return {
            'entropy_birth': np.nan,
            'entropy_birth_norm': np.nan,
            'entropy_income': np.nan,
            'entropy_income_norm': np.nan,
            'ice_birth': np.nan,
            'ice_income': np.nan
        }

    return {
        # Set 1: Entropy
        'entropy_birth': entropy_birth_binary(
            composition['p_native'],
            composition['p_foreign'],
            normalize=False
        ),
        'entropy_birth_norm': entropy_birth_binary(
            composition['p_native'],
            composition['p_foreign'],
            normalize=True
        ),
        'entropy_income': entropy_income_quartiles(
            composition['p_q1'],
            composition['p_q2'],
            composition['p_q3'],
            composition['p_q4'],
            normalize=False
        ),
        'entropy_income_norm': entropy_income_quartiles(
            composition['p_q1'],
            composition['p_q2'],
            composition['p_q3'],
            composition['p_q4'],
            normalize=True
        ),
        # Set 2: ICE
        'ice_birth': ice_birth(
            composition['p_native'],
            composition['p_foreign']
        ),
        'ice_income': ice_income(
            composition['p_q4'],
            composition['p_q1']
        )
    }


# =============================================================================
# CONVENIENCE FUNCTIONS FOR DATAFRAME OPERATIONS
# =============================================================================

def add_diversity_metrics_to_df(df: pd.DataFrame,
                                 p_native_col: str = 'p_native',
                                 p_foreign_col: str = 'p_foreign',
                                 p_q1_col: str = 'p_q1',
                                 p_q2_col: str = 'p_q2',
                                 p_q3_col: str = 'p_q3',
                                 p_q4_col: str = 'p_q4',
                                 prefix: str = '') -> pd.DataFrame:
    """
    Add diversity metric columns to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with composition columns
    p_native_col, p_foreign_col : str
        Column names for birth proportions
    p_q1_col, ..., p_q4_col : str
        Column names for income quartile proportions
    prefix : str
        Prefix for new column names (e.g., 'residential_', 'catchment_', 'visitor_')

    Returns
    -------
    pd.DataFrame
        DataFrame with added metric columns
    """
    df = df.copy()

    # Set 1: Entropy
    df[f'{prefix}entropy_birth'] = df.apply(
        lambda r: entropy_birth_binary(r[p_native_col], r[p_foreign_col], normalize=False),
        axis=1
    )
    df[f'{prefix}entropy_birth_norm'] = df.apply(
        lambda r: entropy_birth_binary(r[p_native_col], r[p_foreign_col], normalize=True),
        axis=1
    )
    df[f'{prefix}entropy_income'] = df.apply(
        lambda r: entropy_income_quartiles(r[p_q1_col], r[p_q2_col], r[p_q3_col], r[p_q4_col], normalize=False),
        axis=1
    )
    df[f'{prefix}entropy_income_norm'] = df.apply(
        lambda r: entropy_income_quartiles(r[p_q1_col], r[p_q2_col], r[p_q3_col], r[p_q4_col], normalize=True),
        axis=1
    )

    # Set 2: ICE
    df[f'{prefix}ice_birth'] = df.apply(
        lambda r: ice_birth(r[p_native_col], r[p_foreign_col]),
        axis=1
    )
    df[f'{prefix}ice_income'] = df.apply(
        lambda r: ice_income(r[p_q4_col], r[p_q1_col]),
        axis=1
    )

    return df


# =============================================================================
# METRIC INTERPRETATION HELPERS
# =============================================================================

def interpret_entropy(value: float, normalized: bool = False) -> str:
    """Interpret entropy value."""
    if np.isnan(value):
        return "No data"

    if normalized:
        if value > 0.9:
            return "Very high diversity (near uniform)"
        elif value > 0.7:
            return "High diversity"
        elif value > 0.5:
            return "Moderate diversity"
        elif value > 0.3:
            return "Low diversity"
        else:
            return "Very low diversity (concentrated)"
    else:
        return f"Entropy = {value:.3f}"


def interpret_ice(value: float, dimension: Literal['birth', 'income'] = 'birth') -> str:
    """Interpret ICE value."""
    if np.isnan(value):
        return "No data"

    if dimension == 'birth':
        high_label, low_label = "native", "foreign-born"
    else:
        high_label, low_label = "high-income", "low-income"

    if value > 0.6:
        return f"Strongly {high_label} concentrated"
    elif value > 0.2:
        return f"Moderately {high_label} concentrated"
    elif value > -0.2:
        return "Mixed/balanced"
    elif value > -0.6:
        return f"Moderately {low_label} concentrated"
    else:
        return f"Strongly {low_label} concentrated"
