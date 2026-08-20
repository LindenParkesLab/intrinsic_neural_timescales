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
# NB: get_null_p and residualize are generic statistics and now live in snaplab_tools.stats
# (they are no longer re-exported by snaplab_tools.nulls); everything else stays in .nulls.
from snaplab_tools.stats import get_null_p, residualize
from snaplab_tools.nulls import (
    load_distance_matrix,
    build_geodesic_distance_matrix,
    generate_surrogates,
    corr_with_null,
    corr_with_covar_null,
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
    "null_basis",
    "require_full_parcellation",
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


def null_basis(processor, dataset, n_regions=400, kind="geodesic", atlas="schaefer"):
    """Everything a notebook needs to build spatial nulls: distance basis + surrogate cache.

    fig3 and fig4 both need the same three objects before they can call generate_surrogates, and
    they must be the SAME three or the two notebooks silently build their nulls on different bases
    (and stop sharing the on-disk cache). One call, one basis.

    Parameters
    ----------
    processor : object
        Any object exposing an output_dir attribute (the Timescales processor).
    dataset : str
        Dataset name; selects the cache subdirectory.
    n_regions : int
        Parcellation resolution.
    kind : {'geodesic', 'euclidean'}
        Distance basis. 'geodesic' is per-hemisphere (cross-hemisphere entries are NaN), which is
        what makes a hemisphere-restricted map yield hemisphere-restricted surrogates.
    atlas : {'schaefer', 'glasser'}
        Parcellation family.

    Returns
    -------
    (distance_matrix, hemi, cache_dir) : (ndarray, ndarray of '<U1', str)
    """
    distance_matrix, hemi = load_distance_matrix(n_regions=n_regions, kind=kind, atlas=atlas)
    return distance_matrix, hemi, surrogate_cache_dir(processor, dataset)


def require_full_parcellation(n_present, n_regions, what, hint=""):
    """Raise unless ``n_present`` covers the whole parcellation.

    Spatial nulls and surface panels are built over all ``n_regions`` parcels, so anything derived
    from a region-masked fit (fig3's ``apply_tsnr_mask``) misaligns against them -- usually not at
    the point of the mistake but several steps later, as a broadcasting error or, worse, a silently
    mismatched correlation. Fail at the boundary instead.

    Parameters
    ----------
    n_present : int
        Number of regions actually carried by the object being checked.
    n_regions : int
        Number the parcellation should have.
    what : str
        Name of the object, used in the message.
    hint : str
        Optional remedy appended to the message.
    """
    if int(n_present) != int(n_regions):
        raise ValueError(
            f"{what} carries {int(n_present)} regions but the full {int(n_regions)}-parcel "
            f"parcellation is required (spatial nulls and surface panels are built over all "
            f"parcels).{(' ' + hint) if hint else ''}"
        )
