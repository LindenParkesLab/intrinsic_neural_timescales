"""Send/receive communication asymmetry for undirected weighted structural connectomes.

This module computes, per parcel and per subject, a directional-asymmetry scalar for
three *asymmetric* communication measures. Even though the input structural connectome
(SC) is symmetric, each of these measures produces an asymmetric pairwise communication
matrix C (C[i, j] != C[j, i] in general), which lets us separate a parcel's tendency to
*send* (broadcast) from its tendency to *receive*. The per-parcel asymmetry scalar is the
downstream quantity of interest (e.g. to correlate against a developmental cortical map).

Measures (all yield asymmetric C on a symmetric SC):
    1. Navigation efficiency
    2. Search information
    3. Mean first-passage time, reported as diffusion efficiency (1 / MFPT)

Backend: netneurotools 0.3.0. The return semantics relied on below are:

    navigation_wu(nav_dist_mat, sc_mat)
        -> (nav_sr, nav_sr_node, nav_path_len, nav_path_hop, nav_paths)
        `nav_path_len` is scored in the units of `nav_dist_mat`; unreachable/failed
        greedy-routing pairs are np.inf and the diagonal is 0. `sc_mat` is used only to
        establish connectedness.
    search_information(W, D, has_memory=False)
        -> SI (N, N); diagonal is NaN, no-valid-path edges are np.inf, and the result is
        not guaranteed symmetric even for symmetric input. `D` is a weight-to-length remap.
    diffusion_efficiency(W)
        -> (GE_diff, E_diff); E_diff = 1 / mean_first_passage_time(W) with diagonal 0.

Conventions:
    - Communication matrix C indexed C[i, j] = communication FROM i TO j. This matches the
      netneurotools convention for all three functions above (row = source, col = target).
    - send(i)    = nanmean over j != i of C[i, j]   (row i,   excluding diagonal)
    - receive(j) = nanmean over i != j of C[i, j]   (column j, excluding diagonal)
    - asymmetry(k) = send(k) - receive(k)
    - Diagonal is excluded from every reduction.

Polarity (oriented so higher = more efficient BEFORE reducing, so asymmetry > 0 means
broadcaster-leaning consistently across measures):
    - Navigation efficiency : already an efficiency  -> identity (used as-is)
    - Diffusion efficiency  : already an efficiency  -> identity (used as-is)
    - Search information     : a cost in bits         -> negated (-SI) for the reduction
Raw (non-oriented) matrices are retained alongside the oriented ones.
"""

import hashlib
import os
import warnings

import numpy as np
import pandas as pd
import scipy as sp
import scipy.sparse.csgraph  # noqa: F401  # ensures sp.sparse.csgraph is importable
import netneurotools.metrics as nnt

# Small positive constant for the search-information weight-to-length remapping
# L = -log10(W / (max(W) + eps)); keeps the argument of log10 in (0, 1] for present edges.
_SI_EPS = 1e-10

# Canonical measure keys used throughout the returned data structures.
_MEASURES = ("navigation_efficiency", "search_information", "diffusion_efficiency")
_COMPONENTS = ("send", "receive", "asymmetry")

# Orientation metadata: documents, per measure, how the raw quantity was mapped to an
# "efficiency" (higher = better) before the send/receive/asymmetry reduction.
_ORIENTATION = {
    "navigation_efficiency": {
        "raw_quantity": "navigation efficiency = 1 / navigation_wu path length (routed and "
                        "scored on Euclidean distance D)",
        "orientation": "identity",
        "negated": False,
        "higher_is_more_efficient": True,
    },
    "search_information": {
        "raw_quantity": "search information (bits); a communication COST",
        "orientation": "negated (-SI)",
        "negated": True,
        "higher_is_more_efficient": True,
    },
    "diffusion_efficiency": {
        "raw_quantity": "diffusion efficiency = 1 / mean first-passage time",
        "orientation": "identity",
        "negated": False,
        "higher_is_more_efficient": True,
    },
}


def _largest_component(W):
    """Return sorted node indices of the largest connected component of W's binary support.

    Connectedness is assessed on the binary support (W != 0) treated as undirected. For a
    fully connected graph this returns all node indices.
    """
    support = sp.sparse.csr_matrix((W != 0).astype(np.int8))
    n_comp, labels = sp.sparse.csgraph.connected_components(support, directed=False)
    if n_comp == 1:
        return np.arange(W.shape[0]), n_comp
    # Largest component = the label with the most members.
    counts = np.bincount(labels)
    keep_label = int(np.argmax(counts))
    return np.where(labels == keep_label)[0], n_comp


def _scatter(sub_mat, idx, n):
    """Place an (m, m) submatrix computed over `idx` back into an (n, n) NaN-filled matrix."""
    full = np.full((n, n), np.nan, dtype=float)
    full[np.ix_(idx, idx)] = sub_mat
    return full


def _nav_efficiency_matrix(Wsub, Dsub):
    """Navigation efficiency C[i, j] = 1 / navigation path length (from i to j).

    Route and score on the Euclidean distance matrix D (canonical). Failed greedy-routing
    pairs (np.inf path length) become NaN and are thus excluded from downstream means.
    NOTE: navigation_wu can also be scored on an alternative length matrix via
    get_navigation_path_length(nav_paths, alt_dist_mat); we default to D for both.
    """
    # navigation_wu(nav_dist_mat, sc_mat): sc_mat only used to establish connectedness.
    _, _, nav_path_len, _, _ = nnt.navigation_wu(Dsub, Wsub)
    nav_path_len = np.asarray(nav_path_len, dtype=float)
    nav_path_len[np.isinf(nav_path_len)] = np.nan  # failed pairs -> NaN
    np.fill_diagonal(nav_path_len, np.nan)          # exclude self path (len 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 / nav_path_len


def _search_information_matrix(Wsub):
    """Raw search information C[i, j] (bits) for communication from i to j.

    Length matrix L = -log10(W / (max(W) + eps)); zero weights map to +inf length (an
    absent edge). Returns the raw SI cost (orientation/negation applied by the caller).
    No-valid-path edges (np.inf) are converted to NaN.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        # Zero weights -> log10(0) = -inf -> length +inf (absent edge), as intended.
        L = -np.log10(Wsub / (Wsub.max() + _SI_EPS))
    SI = np.asarray(nnt.search_information(Wsub, L), dtype=float)
    SI[np.isinf(SI)] = np.nan  # no valid shortest path -> NaN
    # search_information already sets the diagonal to NaN.
    return SI


def _diffusion_efficiency_matrix(Wsub):
    """Diffusion efficiency C[i, j] = 1 / MFPT(i -> j); already an efficiency."""
    _, E_diff = nnt.diffusion_efficiency(Wsub)
    E_diff = np.asarray(E_diff, dtype=float)
    E_diff[np.isinf(E_diff)] = np.nan
    np.fill_diagonal(E_diff, np.nan)  # diffusion_efficiency fills diagonal with 0
    return E_diff


def _send_receive_asymmetry(C):
    """Reduce an oriented (N, N) communication matrix to per-parcel send/receive/asymmetry.

    Diagonal must already be NaN. Parcels dropped upstream (all-NaN rows/cols) come back as
    NaN via nanmean; the associated "empty slice" warning is suppressed intentionally.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        send = np.nanmean(C, axis=1)     # row means: from i to all j
        receive = np.nanmean(C, axis=0)  # column means: from all i to j
    asymmetry = send - receive
    return send, receive, asymmetry


def compute_send_receive(W, D, connected_policy="warn"):
    """Per-parcel send/receive/asymmetry for a single symmetric connectome.

    Parameters
    ----------
    W : (N, N) ndarray
        Symmetric, weighted structural connectome for one subject. It is the CALLER's
        responsibility to apply any subject-level weight normalization beforehand; no
        normalization is applied here.
    D : (N, N) ndarray
        Shared Euclidean distance matrix (group Schaefer centroids), used for navigation.
    connected_policy : {'warn', 'raise'}, optional
        Behavior when W's binary support is not a single connected component:
            'warn'  (default) -> emit a warning, restrict all measures to the largest
                     connected component, and return NaN for the dropped parcels.
            'raise' -> raise a ValueError.

    Returns
    -------
    dict
        {
          'send':       {measure: (N,) ndarray},
          'receive':    {measure: (N,) ndarray},
          'asymmetry':  {measure: (N,) ndarray},
          'cortex_mean':{measure: {'send','receive','asymmetry'}},  # whole-cortex nanmeans
          'matrices':   {measure: {'raw': (N,N), 'oriented': (N,N)}},
          'orientation': orientation-metadata dict (per measure),
          'disconnected': bool,      # True if the largest-component path was taken
          'n_components': int,       # number of connected components in the binary support
          'largest_component_size': int,
        }
    """
    W = np.asarray(W, dtype=float)
    D = np.asarray(D, dtype=float)
    n = W.shape[0]
    if W.shape != (n, n):
        raise ValueError(f"W must be square; got {W.shape}.")
    if D.shape != (n, n):
        raise ValueError(f"D must match W shape {(n, n)}; got {D.shape}.")

    idx, n_comp = _largest_component(W)
    disconnected = idx.shape[0] < n
    if disconnected:
        if connected_policy == "raise":
            raise ValueError(
                f"Binary support of W is not a single connected component "
                f"({n_comp} components; largest has {idx.shape[0]}/{n} parcels)."
            )
        elif connected_policy == "warn":
            warnings.warn(
                f"Binary support of W has {n_comp} connected components; restricting to the "
                f"largest ({idx.shape[0]}/{n} parcels). Dropped parcels return NaN.",
                stacklevel=2,
            )
        else:
            raise ValueError(
                f"connected_policy must be 'warn' or 'raise'; got {connected_policy!r}."
            )

    Wsub = W[np.ix_(idx, idx)]
    Dsub = D[np.ix_(idx, idx)]

    # Raw (natural-form) matrices per measure, restricted to the connected component.
    raw_sub = {
        "navigation_efficiency": _nav_efficiency_matrix(Wsub, Dsub),
        "search_information": _search_information_matrix(Wsub),
        "diffusion_efficiency": _diffusion_efficiency_matrix(Wsub),
    }

    # Oriented (higher = more efficient) matrices: negate only the search-information cost.
    oriented_sub = {
        "navigation_efficiency": raw_sub["navigation_efficiency"],
        "search_information": -raw_sub["search_information"],
        "diffusion_efficiency": raw_sub["diffusion_efficiency"],
    }

    send, receive, asymmetry = {}, {}, {}
    cortex_mean, matrices = {}, {}
    for measure in _MEASURES:
        C_raw = _scatter(raw_sub[measure], idx, n)
        C_oriented = _scatter(oriented_sub[measure], idx, n)
        s, r, a = _send_receive_asymmetry(C_oriented)
        send[measure], receive[measure], asymmetry[measure] = s, r, a
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            cortex_mean[measure] = {
                "send": float(np.nanmean(s)),
                "receive": float(np.nanmean(r)),
                "asymmetry": float(np.nanmean(a)),
            }
        matrices[measure] = {"raw": C_raw, "oriented": C_oriented}

    return {
        "send": send,
        "receive": receive,
        "asymmetry": asymmetry,
        "cortex_mean": cortex_mean,
        "matrices": matrices,
        "orientation": _ORIENTATION,
        "disconnected": bool(disconnected),
        "n_components": int(n_comp),
        "largest_component_size": int(idx.shape[0]),
    }


# --------------------------------------------------------------------------------------------
# Per-subject on-disk cache
# --------------------------------------------------------------------------------------------
# Each subject slice is independent, so we cache the REDUCED result (send/receive/asymmetry per
# measure + the connectedness metadata) — not the large N x N communication matrices, which are
# not consumed by run_over_subjects. The cache is content-addressed: the key hashes the subject's
# connectome slice, the shared distance matrix, and connected_policy, so regenerating the
# connectome (e.g. a normalization change) automatically invalidates stale entries rather than
# silently returning wrong numbers. Bumping _CACHE_VERSION invalidates everything on purpose.
_CACHE_VERSION = "v1"


def _hash_array(arr):
    """Stable hex digest of an array's raw bytes (contiguous float64)."""
    return hashlib.sha1(np.ascontiguousarray(arr, dtype=np.float64).tobytes()).hexdigest()


def _subject_cache_key(W_slice, d_hash, connected_policy):
    """8-char content-address for one subject: hashes W slice + D + policy + cache version."""
    h = hashlib.sha1()
    h.update(_CACHE_VERSION.encode())
    h.update(_hash_array(W_slice).encode())
    h.update(d_hash.encode())
    h.update(str(connected_policy).encode())
    return h.hexdigest()[:8]


def _reduce_result(res):
    """Strip a compute_send_receive() result down to the cache-worthy (small) payload."""
    return {
        "send": {m: res["send"][m] for m in _MEASURES},
        "receive": {m: res["receive"][m] for m in _MEASURES},
        "asymmetry": {m: res["asymmetry"][m] for m in _MEASURES},
        "disconnected": bool(res["disconnected"]),
        "n_components": int(res["n_components"]),
        "largest_component_size": int(res["largest_component_size"]),
    }


def _save_subject_cache(path, reduced):
    """Atomically write a reduced result to `path` (tmp file + os.replace)."""
    arrays = {f"{comp}__{m}": reduced[comp][m] for comp in _COMPONENTS for m in _MEASURES}
    arrays["_meta"] = np.array(
        [reduced["disconnected"], reduced["n_components"], reduced["largest_component_size"]],
        dtype=float,
    )
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)  # file object -> no ".npz" suffix appended
    os.replace(tmp, path)


def _load_subject_cache(path):
    """Inverse of _save_subject_cache; returns the same dict shape as _reduce_result."""
    with np.load(path) as data:
        reduced = {comp: {m: data[f"{comp}__{m}"] for m in _MEASURES} for comp in _COMPONENTS}
        meta = data["_meta"]
    reduced["disconnected"] = bool(meta[0])
    reduced["n_components"] = int(meta[1])
    reduced["largest_component_size"] = int(meta[2])
    return reduced


def _compute_or_load(W_slice, D, subject_id, connected_policy, cache_dir, d_hash):
    """Return the reduced send/receive result for one subject, using the cache when available."""
    if cache_dir is None:
        return _reduce_result(
            compute_send_receive(W_slice, D, connected_policy=connected_policy)
        )
    key = _subject_cache_key(W_slice, d_hash, connected_policy)
    path = os.path.join(cache_dir, f"{subject_id}__{key}.npz")
    if os.path.exists(path):
        return _load_subject_cache(path)
    reduced = _reduce_result(
        compute_send_receive(W_slice, D, connected_policy=connected_policy)
    )
    _save_subject_cache(path, reduced)
    return reduced


def run_over_subjects(W_stack, D, subject_ids, connected_policy="warn", cache_dir=None):
    """Compute send/receive/asymmetry for every subject slice of a connectome stack.

    Parameters
    ----------
    W_stack : (N, N, S) ndarray
        Stack of symmetric connectomes; W_stack[:, :, s] is subject s.
    D : (N, N) ndarray
        Shared Euclidean distance matrix, applied to all subjects.
    subject_ids : sequence of length S
        Subject identifiers aligned BY POSITION to axis 2 of W_stack. Correct id/slice
        ordering is the CALLER's responsibility; this function only asserts the length
        match and carries the subject_id (not the integer index) into all outputs.
    connected_policy : {'warn', 'raise'}, optional
        Passed through to compute_send_receive for each slice.
    cache_dir : str or None, optional
        If given, per-subject reduced results are cached on disk under `cache_dir` (created if
        needed) as ``{subject_id}__{key}.npz``, where `key` is a content hash of the subject's
        connectome slice, `D`, and `connected_policy`. On a subsequent call, unchanged subjects
        are loaded instead of recomputed, so adding/removing subjects only recomputes the new
        ones. `None` (default) disables caching (original behavior). The large N x N
        communication matrices are never cached — only the send/receive/asymmetry payload that
        this function returns.

    Returns
    -------
    long_df : pandas.DataFrame
        Tidy long-format table with columns
        [subject_id, parcel, measure, send, receive, asymmetry]. `parcel` is the 0-indexed
        parcel number (0..N-1).
    stacked : dict
        {measure: {component: (S, N) ndarray}} with component in {'send','receive',
        'asymmetry'}; rows are aligned by position to `subject_ids`.
    summary : dict
        {'n_disconnected': {measure: int}, 'n_disconnected_subjects': int, 'n_subjects': int}.
        Because disconnection is a per-subject property of W (shared across measures), the
        per-measure counts are identical; they are reported per measure for convenience.
    """
    # Ordering contract: subject_ids[s] describes W_stack[:, :, s]. We verify only length;
    # any deeper id/slice correspondence must be guaranteed upstream.
    assert len(subject_ids) == W_stack.shape[2], (
        f"len(subject_ids)={len(subject_ids)} != W_stack.shape[2]={W_stack.shape[2]}"
    )

    n_subjects = W_stack.shape[2]
    n_parcels = W_stack.shape[0]

    # Hash the shared distance matrix once; it feeds every subject's cache key.
    d_hash = _hash_array(D) if cache_dir is not None else ""
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    stacked = {
        measure: {comp: np.full((n_subjects, n_parcels), np.nan, dtype=float)
                  for comp in _COMPONENTS}
        for measure in _MEASURES
    }
    n_disconnected = {measure: 0 for measure in _MEASURES}
    n_disconnected_subjects = 0

    long_records = []
    parcels = np.arange(n_parcels)
    for s in range(n_subjects):
        subject_id = subject_ids[s]
        res = _compute_or_load(
            W_stack[:, :, s], D, subject_id, connected_policy, cache_dir, d_hash
        )

        if res["disconnected"]:
            n_disconnected_subjects += 1
            for measure in _MEASURES:
                n_disconnected[measure] += 1

        for measure in _MEASURES:
            send = res["send"][measure]
            receive = res["receive"][measure]
            asymmetry = res["asymmetry"][measure]
            stacked[measure]["send"][s, :] = send
            stacked[measure]["receive"][s, :] = receive
            stacked[measure]["asymmetry"][s, :] = asymmetry
            long_records.append(pd.DataFrame({
                "subject_id": subject_id,
                "parcel": parcels,
                "measure": measure,
                "send": send,
                "receive": receive,
                "asymmetry": asymmetry,
            }))

    long_df = pd.concat(long_records, ignore_index=True)

    summary = {
        "n_disconnected": n_disconnected,
        "n_disconnected_subjects": n_disconnected_subjects,
        "n_subjects": n_subjects,
    }
    return long_df, stacked, summary
