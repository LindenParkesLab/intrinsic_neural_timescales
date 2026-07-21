"""Project entry point for spatial null models (intrinsic-neural-timescales).

The generic null-generating machinery — geodesic/Euclidean distance bases, the per-hemisphere
BrainSMASH surrogate generator, and the variogram-null inference helpers — lives in
``snaplab_tools.nulls`` so it can be shared across projects (see that package's docstring for the
geodesic-vs-Euclidean rationale and the per-hemisphere handling).

This module re-exports that API so the rest of the project imports nulls from one place, and adds
the conveniences that are specific to *this* project (cache-directory convention, etc.).
"""

import os

# Re-export the shared API (single import surface for the project).
from snaplab_tools.nulls import (
    get_null_p,
    load_distance_matrix,
    build_geodesic_distance_matrix,
    generate_surrogates,
    corr_with_null,
    corr_with_covar_null,
    residualize,
    correlate_family,
    network_enrichment,
)

__all__ = [
    "get_null_p",
    "load_distance_matrix",
    "build_geodesic_distance_matrix",
    "generate_surrogates",
    "corr_with_null",
    "corr_with_covar_null",
    "correlate_family",
    "network_enrichment",
    "residualize",
    "surrogate_cache_dir",
]

# --------------------------------------------------------------------------------------------
# Project-specific conveniences
# --------------------------------------------------------------------------------------------
def surrogate_cache_dir(processor, dataset):
    """BrainSMASH surrogate cache directory for this project.

    The directory is created if absent. Pass the returned path as the cache_dir
    argument to generate_surrogates / correlate_family.

    Parameters
    ----------
    processor : object
        Any object exposing an output_dir attribute; the cache lives under it.
    dataset : str
        Dataset name, used as a subdirectory.

    Returns
    -------
    str
        Path <processor.output_dir>/results/<dataset>/brainsmash_nulls.
    """
    path = os.path.join(processor.output_dir, "results", dataset, "brainsmash_nulls")
    os.makedirs(path, exist_ok=True)
    return path
