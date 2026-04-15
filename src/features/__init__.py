"""Feature engineering modules."""

from .diversity_metrics import (
    # Set 1: Entropy-based
    entropy,
    entropy_income_quartiles,
    entropy_birth_binary,
    # Set 2: ICE-based
    ice_bipolar,
    ice_income,
    ice_birth,
    # Aggregation
    compute_tract_composition_sweden,
    compute_tract_composition_us,
    aggregate_composition_weighted,
    aggregate_composition_us_income_quartiles,
    # All-in-one
    compute_diversity_metrics,
    add_diversity_metrics_to_df,
)

__all__ = [
    'entropy',
    'entropy_income_quartiles',
    'entropy_birth_binary',
    'ice_bipolar',
    'ice_income',
    'ice_birth',
    'compute_tract_composition_sweden',
    'compute_tract_composition_us',
    'aggregate_composition_weighted',
    'aggregate_composition_us_income_quartiles',
    'compute_diversity_metrics',
    'add_diversity_metrics_to_df',
]
