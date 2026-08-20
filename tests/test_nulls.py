"""
Tests for the project null-model layer (src/nulls.py).

The generic null-generating machinery is tested in snaplab_tools/tests/test_nulls.py. Here we
only check the project-specific surface: that src.nulls re-exports the shared API (so the rest of
the project has one import point) and that the project cache-dir convention is correct.
"""

import os

import pytest

from src import nulls


class TestReExport:
    def test_public_api_is_present(self):
        for name in [
            "get_null_p",
            "load_distance_matrix",
            "build_geodesic_distance_matrix",
            "generate_surrogates",
            "corr_with_null",
            "corr_with_covar_null",
            "correlate_family",
            "network_enrichment",
            "residualize",
        ]:
            assert hasattr(nulls, name), f"src.nulls is missing re-export: {name}"

    def test_reexports_are_the_shared_objects(self):
        import snaplab_tools.nulls as shared

        assert nulls.generate_surrogates is shared.generate_surrogates
        assert nulls.correlate_family is shared.correlate_family
        assert nulls.load_distance_matrix is shared.load_distance_matrix


class TestSurrogateCacheDir:
    def test_builds_and_creates_project_convention_path(self, tmp_path):
        class _Proc:
            output_dir = str(tmp_path)

        path = nulls.surrogate_cache_dir(_Proc(), "HCP-D")
        assert path == os.path.join(str(tmp_path), "results", "HCP-D", "brainsmash_nulls")
        assert os.path.isdir(path)  # created on demand
