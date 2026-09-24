import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Any
from tqdm import tqdm
from nilearn.glm.first_level import make_first_level_design_matrix
from collections import defaultdict

from src.data import DataLoader, ScanInfo, parse_scan_string, build_scan_string
from snaplab_tools.stats import partial_pearsonr
# INT estimation lives entirely in snaplab_tools -- this project defines no estimator of its own.
# What this module owns is the per-scan caching layer around them
# (compute_autocorrelation_function / compute_timescale_measures). Import these two from
# snaplab_tools.timescales directly rather than through here; they are not re-exported.
from snaplab_tools.timescales import TIMESCALE_METHODS, compute_acf
import logging
logger = logging.getLogger(__name__)


# ==================== Run vocabulary ====================
#: 3T rest runs. HCP-YA uses LR/RL phase encoding, HCP-D uses PA.
REST_RUNS_3T = ['rfMRIREST1LR', 'rfMRIREST1RL', 'rfMRIREST2LR', 'rfMRIREST2RL',   # HCP-YA
                'rfMRIREST1PA', 'rfMRIREST2PA']                                   # HCP-D
#: 7T rest runs (HCP-YA), alternating PA/AP.
REST_RUNS_7T = ['rfMRIREST17TPA', 'rfMRIREST27TAP', 'rfMRIREST37TPA', 'rfMRIREST47TAP']
#: 7T movie runs (HCP-YA), alternating AP/PA -- note the offset against REST_RUNS_7T.
MOVIE_RUNS_7T = ['tfMRIMOVIE17TAP', 'tfMRIMOVIE27TPA', 'tfMRIMOVIE37TPA', 'tfMRIMOVIE47TAP']
REST_RUNS_MEG = ['3Restin', '4Restin', '5Restin']
TASK_RUNS_MEG = ['6Wrkmem', '7Wrkmem']

#: Task runs grouped into the condition they are averaged into: ``[(runs, condition_key), ...]``.
TASK_RUN_GROUPS = [
    # HCP-YA — LR/RL counterbalanced pairs
    (['tfMRIWMLR',          'tfMRIWMRL'],          'task-WM'),
    (['tfMRIEMOTIONLR',     'tfMRIEMOTIONRL'],     'task-EMOTION'),
    (['tfMRIGAMBLINGLR',    'tfMRIGAMBLINGRL'],    'task-GAMBLING'),
    (['tfMRILANGUAGELR',    'tfMRILANGUAGERL'],    'task-LANGUAGE'),
    (['tfMRIRELATIONALLR',  'tfMRIRELATIONALRL'],  'task-RELATIONAL'),
    (['tfMRISOCIALLR',      'tfMRISOCIALRL'],      'task-SOCIAL'),
    # HCP-D — single PA direction (no LR/RL counterpart)
    (['tfMRICARITPA'],                              'task-CARIT'),
    (['tfMRIEMOTIONPA'],                            'task-EMOTION'),
    (['tfMRIGUESSINGPA'],                           'task-GUESSING'),
]

#: Condition key for the 7T movies, which TASK_RUN_GROUPS does not cover because
#: _apply_scan_groupings collapses them with a separate flag (``average_7t_movies``).
MOVIE_GROUP = (MOVIE_RUNS_7T, 'task-Movies')

# ==================== Rest-control window offset ====================
REST_CONTROL_OFFSET_SECONDS = {
    ('HCP-YA',    'MRI3T'): 126.0,
    ('HCP-YA',    'MRI7T'): 0.0,
    ('HCP-YA',    'MEG'):   0.0,
    ('HCP-D',     'MRI3T'): 0.0,
    ('Penn_LEAD', 'MRI3T'): 0.0,
}

#: Offset for a (dataset, modality) that has not been measured.
REST_CONTROL_OFFSET_DEFAULT = 0.0


def rest_control_offset_seconds(dataset, modality):
    """Scan-onset offset in seconds for a dataset's matched rest control window.

    Parameters
    ----------
    dataset : str
        Dataset name, as :class:`Timescales` carries it.
    modality : str
        Modality token, e.g. ``'MRI3T'``. HCP-YA needs it: its 3T runs carry a transient and its
        7T runs do not, so a dataset-only table would be wrong for one of them.

    Returns
    -------
    float
        Offset in seconds, or :data:`REST_CONTROL_OFFSET_DEFAULT` when the pair is not listed.
    """
    return float(REST_CONTROL_OFFSET_SECONDS.get((dataset, modality),
                                                 REST_CONTROL_OFFSET_DEFAULT))


def rest_control_offset_frames(dataset, modality, tr):
    """The offset above, in frames of a scan sampled at ``tr``.

    Parameters
    ----------
    dataset, modality : str
        Passed to :func:`rest_control_offset_seconds`.
    tr : float
        Repetition time in seconds of the run the window is cut from.

    Returns
    -------
    int
        ``round(offset_seconds / tr)``. HCP-YA 3T: 126 / 0.72 = 175 frames exactly.

    Raises
    ------
    ValueError
        If ``tr`` is not positive -- a zero TR would silently give a zero offset.
    """
    tr = float(tr)
    if not tr > 0:
        raise ValueError(f'tr must be positive to convert an offset to frames, got {tr!r}')
    return int(round(rest_control_offset_seconds(dataset, modality) / tr))

REST_WINDOW_MATCH = {
    ('HCP-YA', 'MRI3T'): {'WM':       ('tfMRIWMLR', 'tfMRIWMRL')},
    ('HCP-D',  'MRI3T'): {'CARIT':    ('tfMRICARITPA',),
                          'EMOTION':  ('tfMRIEMOTIONPA',),
                          'GUESSING': ('tfMRIGUESSINGPA',)},
}

#: Output-type token prefix for a length-matched rest window.
REST_WINDOW_TOKEN = 'restlen'

#: Written into every length-matched ACF file so it cannot be misread once it leaves its directory.
REST_WINDOW_NOTE = ("REST timeseries cut to one window the length of this subject's matched task "
                    "run, starting at the dataset's scan-onset offset. Nothing projected out.")


def rest_window_token(match):
    """Token of the rest windows matched to ``match``, e.g. ``'CARIT'`` -> ``'restlen-CARIT'``.

    The hyphenated token-value form of ``restctrl-drift`` and ``bp-0.01-0.08hz``: a rest run is cut
    to a different length per matched task, so each set of windows needs its own name.
    """
    if not isinstance(match, str) or not match:
        raise ValueError(f'a length-matched rest token needs a match name (e.g. CARIT); got {match!r}')
    return f'{REST_WINDOW_TOKEN}-{match}'


def rest_window_output_type(base, match):
    """Append the length-matched-rest token, e.g. ``autocorr`` -> ``autocorr_restlen-CARIT``.

    Same last slot as ``_resids`` and ``_restctrl``: a file is at most one of them. ``base``
    must already carry ``_resampled`` and any frequency-filter token.
    """
    return f'{base}_{rest_window_token(match)}'


def rest_window_matches(dataset, modality=None):
    """Names of the tasks this dataset's rest runs are matched to, over every covered modality or one."""
    names = []
    for (ds, mod), table in REST_WINDOW_MATCH.items():
        if ds == dataset and (modality is None or mod == modality):
            names += [m for m in table if m not in names]
    return tuple(names)


def rest_window_covered(dataset, modality, match=None):
    """True when rest runs of this dataset x modality have length-matched windows (to ``match``, if given)."""
    table = REST_WINDOW_MATCH.get((dataset, modality), {})
    return bool(table) if match is None else match in table


def rest_window_match_for(dataset, task):
    """The match whose windows are as long as ``task``, e.g. ``('HCP-D', 'tfMRICARITPA') -> 'CARIT'``.

    The one place a consumer turns the task it analyses into the rest it contrasts it with, so fig3,
    fig4 and hcpd_behavior cannot pick different windows for the same task.

    Raises
    ------
    ValueError
        If no rest window of this dataset is matched to ``task``.
    """
    available = {}
    for (ds, _), table in REST_WINDOW_MATCH.items():
        if ds != dataset:
            continue
        for name, tasks in table.items():
            if task in tasks:
                return name
            available[name] = tasks
    raise ValueError(f'no length-matched rest for {dataset} {task}; matched tasks: {available or "none"}. '
                     'See src.timescales.REST_WINDOW_MATCH.')


class Timescales(DataLoader):
    """Per-scan INT estimation with a disk cache, on top of :class:`~src.data.DataLoader`.

    This class owns the *caching and bookkeeping* around INT estimation, not the estimation
    itself — every estimator comes from :mod:`snaplab_tools.timescales` (see the module header).
    What it adds is: a filename convention that encodes every field identifying a scan, so
    sessions/spaces/runs never collide; load-or-compute-then-save for each scan; and the task-GLM
    step that produces the residual timeseries the paper's INTs are estimated from.

    A typical run, in order::

        proc = Timescales(dataset='HCP-YA', timeseries_data=ts)
        proc.regress_task_structure()                       # task scans -> residuals
        proc.compute_autocorrelation_function(use_residuals=True)
        proc.compute_timescale_measures(method='auc_zero_crossing')

    Each ``compute_*`` call fills the correspondingly named attribute *and* writes one file per
    scan, so re-running is cheap and a partially completed run resumes rather than restarts.
    Pass ``force_recompute=True`` to ignore the cache.

    Attributes
    ----------
    timeseries_data : dict or None
        ``scan_key -> (n_parcels, n_timepoints)``, as passed in.
    scans : list of str
        The scan keys, in insertion order.
    timeseries_residuals, task_events, design_matrices, task_events_acf, task_contrasts : dict
        Set by :meth:`regress_task_structure`; ``None`` per scan for rest scans and failures.
    autocorr_dict : dict
        Set by :meth:`compute_autocorrelation_function`; ``scan_key -> (n_regions, n_lags)``.
    timescales_dict : dict
        Set by :meth:`compute_timescale_measures`; ``scan_key -> (n_regions,)`` in seconds.
    data_dict : dict
        Set by :meth:`get_precomputed_outputs` when loading results computed in an earlier session.

    Notes
    -----
    Output filenames are built by :meth:`_get_output_filepath` from the scan's identifying fields
    plus an ``output_type`` tag. The tag encodes the processing that produced the file
    (``_resids``, ``_resampled``, a frequency-filter token), so variants coexist on disk rather
    than overwriting one another. Renaming a tag orphans its cache.
    """

    def __init__(
        self,
        dataset: str,
        timeseries_data: dict,
        load_behavior: bool = True,
        verbose: bool = True
    ):
        """Initialise Timescales for a dataset and its per-scan timeseries.

        Parameters
        ----------
        dataset : str
            Dataset name (passed to DataLoader).
        timeseries_data : dict
            Mapping scan_key -> (n_parcels, n_timepoints) array. May be None.
        load_behavior : bool
            Whether DataLoader should load behavioral data on init.
        verbose : bool
            Print an initialisation summary.
        """
        # Call the parent class's constructor
        super().__init__(dataset=dataset, load_behavior=load_behavior, verbose=verbose)

        self.timeseries_data = timeseries_data
        if timeseries_data is not None:
            self.scans = list(timeseries_data.keys())
        self.autocorr_rest_control = None
        self.autocorr_rest_control_drift = None
        self.rest_control_meta = None
        self.rest_window_span = None
        self.rest_window_meta = None

        if verbose:
            print(f"\n\tInitialized Timescales for dataset: {dataset}")
            if timeseries_data is not None:
                print(f"\tNumber of scans: {len(self.scans)}")
            print(f"\tOutput directory: {self.output_dir}")

    # ============== Save/Load Infrastructure ==============
    def _get_output_filepath(
        self,
        subject_id: str,
        output_type: str,
        task: str,
        modality: str,
        seg: str,
        space: str = None,
        session: str = None,
        run: str = None,
        suffix: str = '.npy',
        mode: str = 'load'
    ) -> Path:
        """Generate a standardized filepath for subject-specific outputs (key-value naming).

        The filename includes every field that uniquely identifies a scan so that
        outputs for different sessions, spaces, or runs never collide.

        Parameters
        ----------
        subject_id : str
            Subject identifier.
        output_type : str
            Type of output (e.g. 'autocorr', 'timescale_acf').
        task : str
            Task/run label (e.g. 'rfMRIREST1LR').
        modality : str
            Modality string (e.g. 'MRI3T').
        seg : str
            Atlas/parcellation name.
        space : str
            Template space (e.g. 'fsLR', 'MNI152NLin2009cAsym').
        session : str
            Session label (e.g. '1', '01'); required for longitudinal data.
        run : str
            Run label (e.g. '01'); included when present.
        suffix : str
            File extension.
        mode : {'load', 'save'}
            'save' creates parent dirs; 'load' does not.

        Returns
        -------
        Path
            Path to the output file.
        """
        # Create output subdirectory structure
        output_subdir = self.output_dir / 'data' / 'timescales_outputs' / self.dataset / output_type
        if mode == 'save':
            output_subdir.mkdir(parents=True, exist_ok=True)

        # Build filename — include all fields that uniquely identify a scan
        filename = build_scan_string(
            sub=subject_id,
            ses=session,
            modality=modality,
            task=task,
            space=space,
            seg=seg,
            run=run,
            stat=output_type,
        ) + suffix

        return output_subdir / filename

    def _save_subject_output(
        self,
        data: Any,
        subject_id: str,
        output_type: str,
        task: str,
        modality: str,
        seg: str = None,
        space: str = None,
        session: str = None,
        run: str = None,
        metadata: Dict = None,
    ) -> Path:
        """Save subject-specific output data with key-value naming."""
        if data.ndim == 1:
            suffix = '.txt'
        elif data.ndim > 1:
            suffix = '.npy'

        filepath = self._get_output_filepath(
            subject_id=subject_id,
            output_type=output_type,
            task=task,
            modality=modality,
            seg=seg,
            space=space,
            session=session,
            run=run,
            suffix=suffix,
            mode='save',
        )
        
        # Save data
        if suffix == '.npy' and metadata is not None:
            np.savez(filepath.with_suffix('.npz'), 
                     data=data, metadata=metadata)
        elif suffix == '.npy' and metadata is None:
            np.save(filepath, data)
        elif suffix == '.txt':
            np.savetxt(filepath, data)
        
        return filepath

    def _load_subject_output(
        self,
        subject_id: str,
        output_type: str,
        task: str,
        modality: str,
        seg: str = None,
        space: str = None,
        session: str = None,
        run: str = None,
    ) -> Optional[Any]:
        """Load previously saved subject-specific output with key-value naming."""
        suffix = '.npy'
        filepath = self._get_output_filepath(
            subject_id=subject_id,
            output_type=output_type,
            task=task,
            modality=modality,
            seg=seg,
            space=space,
            session=session,
            run=run,
            suffix=suffix,
            mode='load',
        )
        
        # Check for .npz if .npy doesn't exist
        if not filepath.exists() and suffix == '.npy':
            filepath = filepath.with_suffix('.npz')
            # if that doesn't find the file, try .txt
            if not filepath.exists():
                filepath = filepath.with_suffix('.txt')
        
        if not filepath.exists():
            return None
        
        # Load data
        try:
            if filepath.suffix == '.npz':
                loaded = np.load(filepath, allow_pickle=True)
                return loaded['data']
            elif filepath.suffix == '.npy':
                return np.load(filepath)
            elif filepath.suffix == '.txt':
                return np.loadtxt(filepath)
        except Exception as e:
            logger.warning("Could not load %s: %s", filepath, e)
            return None

    def _parse_scan_string(self, scan_key: str) -> Dict[str, str]:
        """Parse a scan key using key-value format.

        Parameters
        ----------
        scan_key : str
            Scan-key string, e.g.
            "sub-100307_modality-MRI3T_task-rfMRIREST1LR_seg-Schaefer4007".

        Returns
        -------
        dict
            Parsed key-value pairs (see parse_scan_string).
        """
        return parse_scan_string(scan_key)

    # ============== Computation Methods ==============
    def _rest_control_source(self, kind: str) -> dict:
        """The control ACF dict for ``kind``, or a message saying what has not been run yet."""
        if kind not in REST_CONTROL_KINDS:
            raise ValueError(f"unknown rest-control kind {kind!r}; "
                             f"available: {sorted(REST_CONTROL_KINDS)}")
        source = (self.autocorr_rest_control if kind == 'full'
                  else self.autocorr_rest_control_drift)
        if source is None:
            raise ValueError(
                f"no {kind!r} rest control available. Call regress_task_structure() and then "
                "build_rest_controls() before requesting use_rest_control.")
        return source

    def _run_for_all_scans(
        self,
        compute_fn,
        output_type: str,
        use_residuals: bool = False,
        use_rest_control: str = None,
        use_rest_window: str = None,
        metadata_fn=None,
        force_recompute: bool = False,
        save_outputs: bool = True,
        verbose: bool = True,
    ) -> dict:
        """
        Shared iteration loop for all per-scan compute methods.

        For each scan in self.scans:
          1. Try to load from cache (unless force_recompute).
          2. If not cached, call compute_fn(timeseries_scan, kv_dict) → result.
          3. Optionally save the result.

        Parameters
        ----------
        compute_fn : callable
            Signature: (timeseries_scan: np.ndarray, kv_dict: dict) -> np.ndarray
            Should raise on failure; exceptions are caught here and logged.
        output_type : str
            Used for cache file naming and logging.
        use_residuals : bool
            If True and condition == 'task', use self.timeseries_residuals instead
            of self.timeseries_data.
        use_rest_control : {'full', 'drift'} or None
            Third source: the matched rest control built by :meth:`build_rest_controls`. Rest
            scans are skipped entirely (they have no control), so they neither hit the cache nor
            count as failures. The values in those dicts are already ACFs -- see the note on
            :attr:`autocorr_rest_control` -- so ``compute_fn`` receives an ACF rather than a
            timeseries on this path. Mutually exclusive with ``use_residuals``.
        use_rest_window : str or None
            Fourth source: the rest windows matched to this task name (a key of
            ``self.rest_window_span``), resolved by :meth:`resolve_rest_windows`. Only covered
            rest scans have one; task scans and uncovered rest scans are skipped, not counted as
            misses or failures. The window is a timeseries, so ``compute_fn`` runs exactly as for
            full-length rest.
        metadata_fn : callable or None
            ``(scan_key, kv_dict) -> dict``, the metadata written alongside each result. None
            uses the default identifying dict. The control path needs per-scan metadata (which
            rest run, which window), which a fixed dict cannot carry.
        force_recompute : bool
            Skip cache lookup if True.
        save_outputs : bool
            Persist results to disk if True.
        verbose : bool
            Print summary counts when True.

        Returns
        -------
        dict mapping scan_key -> result array (or None on failure)
        """
        results = dict.fromkeys(self.scans)
        n_loaded = n_computed = n_failed = 0

        for scan_key in tqdm(self.scans, desc=f"Computing {output_type}"):
            kv_dict = self._parse_scan_string(scan_key)
            subject_id = kv_dict.get('sub', '')
            modality   = kv_dict.get('modality', '')
            condition  = kv_dict.get('condition', '')
            task       = kv_dict.get('task', '')
            seg        = kv_dict.get('seg', '')
            space      = kv_dict.get('space')
            session    = kv_dict.get('ses')
            run        = kv_dict.get('run')
            tr         = kv_dict.get('tr', '')

            # A rest scan has no matched control, so it is not a cache miss and not a failure.
            if use_rest_control is not None and condition != 'task':
                results[scan_key] = None
                continue
            # Likewise only covered rest scans have a length-matched window.
            if use_rest_window is not None and scan_key not in self.rest_window_span[use_rest_window]:
                results[scan_key] = None
                continue

            if not force_recompute:
                cached = self._load_subject_output(
                    subject_id=subject_id, output_type=output_type,
                    task=task, modality=modality, seg=seg,
                    space=space, session=session, run=run,
                )
                if cached is not None:
                    results[scan_key] = cached
                    n_loaded += 1
                    continue

            try:
                if use_rest_control is not None:
                    timeseries_scan = self._rest_control_source(use_rest_control)[scan_key]
                elif use_rest_window is not None:
                    start, stop = self.rest_window_span[use_rest_window][scan_key]
                    timeseries_scan = np.asarray(self.timeseries_data[scan_key])[:, start:stop]
                elif use_residuals and condition == 'task':
                    timeseries_scan = self.timeseries_residuals[scan_key]
                else:
                    timeseries_scan = self.timeseries_data[scan_key]

                result = compute_fn(timeseries_scan, kv_dict)
                results[scan_key] = result
                n_computed += 1

                if save_outputs:
                    metadata = (metadata_fn(scan_key, kv_dict)
                                if metadata_fn is not None
                                else {'output_type': output_type, 'tr': tr, 'modality': modality,
                                      'dataset': self.dataset})
                    self._save_subject_output(
                        data=result,
                        subject_id=subject_id,
                        output_type=output_type,
                        task=task,
                        modality=modality,
                        seg=seg,
                        space=space,
                        session=session,
                        run=run,
                        metadata=metadata,
                    )
            except Exception as e:
                logger.warning("Failed to compute %s for %s: %s", output_type, scan_key, e)
                results[scan_key] = None
                n_failed += 1

        if verbose:
            print(f"\n{output_type} computation complete:")
            print(f"  Loaded from cache: {n_loaded}")
            print(f"  Newly computed:    {n_computed}")
            if n_failed:
                print(f"  Failed:            {n_failed}")
            print(f"  Total:             {len(self.scans)}")

        return results

    def compute_autocorrelation_function(
        self,
        use_fft: bool = False,
        force_recompute: bool = False,
        save_outputs: bool = True,
        suffix: str = None,
        use_residuals: bool = False,
        use_rest_control: str = None,
        use_rest_window: str = None,
        lowpass: float = None,
        highpass: float = None,
        resampled: bool = False,
        verbose: bool = True,
    ):
        """Compute (or load) the ACF for every scan into self.autocorr_dict.

        Runs compute_acf per scan via _run_for_all_scans, caching results to disk.
        lowpass/highpass affect only the output_type name (filtering is applied at
        load time, not here).

        Parameters
        ----------
        use_fft : bool
            Use the FFT-based ACF path in compute_acf.
        force_recompute : bool
            Ignore cached files and recompute.
        save_outputs : bool
            Persist results to disk.
        suffix : str or None
            Extra tag appended to output_type.
        use_residuals : bool
            Use self.timeseries_residuals for task scans instead of raw timeseries.
        use_rest_control : {'full', 'drift'} or None
            Cache the matched rest control instead, appending ``_restctrl`` / ``_restctrl-drift``.
            The ACFs come from :meth:`build_rest_controls`; this call caches them and writes their
            metadata. The control inherits ``resampled`` and the frequency-filter tokens from this
            call, so pass the same values used for the task side -- a control built without them
            is silently comparing unlike things.
        use_rest_window : str or None
            Cache the rest windows matched to this task name instead (e.g. ``'CARIT'``), appending
            ``_restlen-CARIT``. Requires :meth:`resolve_rest_windows`; call once per match. Each
            window goes through the same ``compute_acf`` call as full-length rest, so the two
            cannot drift apart. Inherits ``resampled`` and the frequency-filter tokens from this
            call, as the controls do.
        lowpass, highpass : float or None
            Used only for output naming (see above).
        resampled : bool
            Used only for output naming: append '_resampled' so ACFs computed from
            TR-resampled timeseries are stored separately from native-TR ACFs.
        verbose : bool
            Print summary counts.
        """
        # Token order mirrors the processing order: resampling is applied before
        # filtering, so '_resampled' precedes the frequency-filter token.
        output_type = 'autocorr'
        if resampled:
            output_type = f'{output_type}_resampled'
        if highpass is not None and lowpass is not None:
            output_type = f'{output_type}_bp-{highpass}-{lowpass}hz'
        elif highpass is not None:
            output_type = f'{output_type}_hp-{highpass}hz'
        elif lowpass is not None:
            output_type = f'{output_type}_lp-{lowpass}hz'
        if suffix:
            output_type = f'{output_type}_{suffix}'
        if use_residuals and use_rest_control is not None:
            raise ValueError('use_residuals and use_rest_control are mutually exclusive: a file '
                             'is either the task residual or the rest control, never both')
        if use_residuals:
            output_type = f'{output_type}_resids'
        if use_rest_control is not None:
            output_type = rest_control_output_type(output_type, use_rest_control)
        if use_rest_window is not None:
            if use_residuals or use_rest_control is not None:
                raise ValueError('use_rest_window is mutually exclusive with use_residuals and '
                                 'use_rest_control: a file is a task residual, a rest control or '
                                 'a length-matched rest window, never two of them')
            if not isinstance(use_rest_window, str):
                raise ValueError(f'use_rest_window takes the name of a matched task (one of '
                                 f'{list(rest_window_matches(self.dataset))}), not {use_rest_window!r}')
            if getattr(self, 'rest_window_span', None) is None:
                raise ValueError('no length-matched rest windows; call resolve_rest_windows() '
                                 'before requesting use_rest_window')
            if use_rest_window not in self.rest_window_span:
                raise ValueError(f'no length-matched rest windows named {use_rest_window!r} for '
                                 f'{self.dataset}; resolved: {sorted(self.rest_window_span)}')
            output_type = rest_window_output_type(output_type, use_rest_window)

        # lowpass/highpass are used for output naming only; filtering is applied at load time via load_batch
        def _compute(timeseries_scan, kv_dict):
            # atleast_2d because compute_acf mirrors its input's dimensionality, and the cache
            # format (and every downstream estimator) assumes (n_regions, n_lags).
            return np.atleast_2d(compute_acf(timeseries_scan, use_fft=use_fft))

        def _passthrough(acf, kv_dict):
            # build_rest_controls already reduced the control window to an ACF and threw the
            # window away, so there is nothing left to compute here -- this call exists for the
            # caching, the metadata and the token.
            return np.atleast_2d(acf)

        def _control_metadata(scan_key, kv_dict):
            meta = dict(self.rest_control_meta.get(scan_key, {}))
            meta['output_type'] = output_type
            meta['control_kind'] = REST_CONTROL_KINDS[use_rest_control][1]
            meta['control_note'] = REST_CONTROL_NOTE
            return meta

        def _window_metadata(scan_key, kv_dict):
            meta = dict(self.rest_window_meta[use_rest_window].get(scan_key, {}))
            meta['output_type'] = output_type
            return meta

        if use_rest_control is not None:
            metadata_fn = _control_metadata
        elif use_rest_window is not None:
            metadata_fn = _window_metadata
        else:
            metadata_fn = None

        self.autocorr_dict = self._run_for_all_scans(
            compute_fn=_passthrough if use_rest_control is not None else _compute,
            output_type=output_type,
            use_residuals=use_residuals,
            use_rest_control=use_rest_control,
            use_rest_window=use_rest_window,
            metadata_fn=metadata_fn,
            force_recompute=force_recompute,
            save_outputs=save_outputs,
            verbose=verbose,
        )

    def compute_timescale_measures(
        self,
        method: str = 'zero_crossing',
        force_recompute: bool = False,
        save_outputs: bool = True,
        suffix: str = None,
        use_residuals: bool = False,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Compute timescale measures from the autocorrelation functions.

        Applies the chosen method from TIMESCALE_METHODS to each scan's ACF in
        self.autocorr_dict, storing results in self.timescales_dict (with disk
        caching). Requires compute_autocorrelation_function to have been run.

        Parameters
        ----------
        method : str
            Key into TIMESCALE_METHODS (e.g. 'zero_crossing', 'auc_zero_crossing').
        force_recompute : bool
            Ignore cached files and recompute.
        save_outputs : bool
            Persist results to disk.
        suffix : str or None
            Extra tag appended to output_type.
        use_residuals : bool
            Accepted for API symmetry; ignored here (residual handling is upstream
            in compute_autocorrelation_function).
        verbose : bool
            Print summary counts.
        """
        if not hasattr(self, 'autocorr_dict') or self.autocorr_dict is None:
            raise ValueError("Must compute autocorrelation first")

        if method not in TIMESCALE_METHODS:
            raise ValueError(
                f"Unknown timescale method '{method}'. "
                f"Available: {list(TIMESCALE_METHODS)}"
            )
        method_fn = TIMESCALE_METHODS[method]

        output_type = f'timescale_{method}'
        if suffix:
            output_type = f'{output_type}_{suffix}'

        # compute_timescale_measures keeps its own loop (rather than using
        # _run_for_all_scans) because it needs to look up autocorr_dict by
        # scan_key rather than using timeseries_data/timeseries_residuals.
        results = dict.fromkeys(self.scans)
        n_loaded = n_computed = n_failed = 0

        for scan_key in tqdm(self.scans, desc=f"Computing {method} timescales"):
            kv_dict = self._parse_scan_string(scan_key)
            subject_id = kv_dict.get('sub', '')
            modality   = kv_dict.get('modality', '')
            task       = kv_dict.get('task', '')
            seg        = kv_dict.get('seg', '')
            space      = kv_dict.get('space')
            session    = kv_dict.get('ses')
            run        = kv_dict.get('run')
            tr         = kv_dict.get('tr', '')

            if not force_recompute:
                cached = self._load_subject_output(
                    subject_id=subject_id, output_type=output_type,
                    task=task, modality=modality, seg=seg,
                    space=space, session=session, run=run,
                )
                if cached is not None:
                    results[scan_key] = cached
                    n_loaded += 1
                    continue

            acf = self.autocorr_dict.get(scan_key)
            if acf is None:
                results[scan_key] = None
                continue

            try:
                # No tr fallback: every estimator multiplies a lag count by tr, so a missing
                # tr would silently return lags labelled as seconds. Fail the scan instead.
                if not tr:
                    raise ValueError(f"scan key carries no usable tr: {scan_key!r}")
                result = method_fn(acf, float(tr))
                results[scan_key] = result
                n_computed += 1

                if save_outputs:
                    self._save_subject_output(
                        data=result,
                        subject_id=subject_id,
                        output_type=output_type,
                        task=task,
                        modality=modality,
                        seg=seg,
                        space=space,
                        session=session,
                        run=run,
                        metadata={'method': method, 'tr': tr},
                    )
            except Exception as e:
                logger.warning("Failed to compute %s for %s: %s", method, scan_key, e)
                results[scan_key] = None
                n_failed += 1

        self.timescales_dict = results

        if verbose:
            print(f"\n{method} timescale computation complete:")
            print(f"  Loaded from cache: {n_loaded}")
            print(f"  Newly computed:    {n_computed}")
            if n_failed:
                print(f"  Failed:            {n_failed}")
            print(f"  Total:             {len(self.scans)}")

    def compute_timeseries_measures(
        self,
        method: str = 'tsnr',
        force_recompute: bool = False,
        save_outputs: bool = True,
        use_residuals: bool = False,
        resampled: bool = False,
        verbose: bool = True,
    ):
        """Compute a per-scan timeseries metric into self.data_dict.

        Supports 'tsnr' (temporal SNR) and 'var' (temporal variance), applied per
        scan via _run_for_all_scans with disk caching.

        Parameters
        ----------
        method : {'tsnr', 'var'}
            Metric to compute.
        force_recompute : bool
            Ignore cached files and recompute.
        save_outputs : bool
            Persist results to disk.
        use_residuals : bool
            Use self.timeseries_residuals for task scans instead of raw timeseries.
        resampled : bool
            Used only for output naming: append '_resampled' so metrics computed from
            TR-resampled timeseries are stored separately from native-TR metrics.
        verbose : bool
            Print summary counts.
        """
        output_type = f'metric_{method}'
        if resampled:
            output_type = f'{output_type}_resampled'
        if use_residuals:
            output_type = f'{output_type}_resids'

        _METRIC_FNS = {
            'tsnr': calculate_tsnr,
            'var':  lambda ts: np.var(ts, axis=1),
        }
        if method not in _METRIC_FNS:
            raise ValueError(f"Unknown method '{method}'. Available: {list(_METRIC_FNS)}")
        metric_fn = _METRIC_FNS[method]

        def _compute(timeseries_scan, kv_dict):
            return metric_fn(timeseries_scan)

        self.data_dict = self._run_for_all_scans(
            compute_fn=_compute,
            output_type=output_type,
            use_residuals=use_residuals,
            force_recompute=force_recompute,
            save_outputs=save_outputs,
            verbose=verbose,
        )

    # ==================== Deconfounding Functions ====================
    def regress_task_structure(self):
        """Regress task-evoked structure out of every task scan's timeseries.

        For each task scan, builds a nilearn design matrix from its events
        (HRF-convolved for fMRI; raw for MEG), fits a per-region GLM, and keeps the
        residuals. Also stores the events, design matrices, per-condition beta
        contrast maps, the flattened (all-conditions-merged) task regressor, and its
        ACF. Rest scans and any failures are stored as None.

        Sets self.timeseries_residuals, self.task_events, self.design_matrices,
        self.task_regressor_flat, self.task_events_acf, and self.task_contrasts (all
        keyed by scan_key).

        The flattened regressor is kept, not only its ACF: it is the curve the
        supplementary task-regression figure plots as the design's own
        autocorrelation, and rebuilding it in a notebook would let the notebook's
        HRF/drift settings drift away from the ones actually used here.
        """
        timeseries_residuals = dict().fromkeys(self.timeseries_data)
        task_events = dict().fromkeys(self.timeseries_data)
        design_matrices = dict().fromkeys(self.timeseries_data)
        task_regressor_flat = dict().fromkeys(self.timeseries_data)
        task_events_acf = dict().fromkeys(self.timeseries_data)
        task_contrasts = dict().fromkeys(self.timeseries_data)

        for scan_key in tqdm(self.scans, desc=f"Regressing task structure out of timeseries"):
            # Parse scan key
            kv_dict = self._parse_scan_string(scan_key)
            condition = kv_dict.get('condition', '')
            modality = kv_dict.get('modality', '')
            tr = kv_dict.get('tr', '')

            if condition == 'task':
                try:
                    timeseries = self.timeseries_data[scan_key]
                    n_timepoints = timeseries.shape[1]

                    if modality == 'MEG':
                        hrf_model = None
                        drift_model = None
                    elif modality == 'MRI3T' or modality == 'MRI7T':
                        hrf_model = 'glover'
                        drift_model = 'cosine'

                    # Reconstruct ScanInfo so get_task_events receives the correct type.
                    scan_info = ScanInfo(
                        subject_id=kv_dict.get('sub', ''),
                        task=kv_dict.get('task', ''),
                        space=kv_dict.get('space', ''),
                        seg=kv_dict.get('seg', ''),
                        modality=modality,
                        tr=float(tr) if tr else 0.0,
                        session=kv_dict.get('ses'),
                        run=kv_dict.get('run'),
                    )
                    # Create design matrix using nilearn
                    tsk_evnts = self.get_task_events(scan_info)
                    # replace all trials with a single trial types
                    # tsk_evnts['trial_type'] = 'trial'
                    design_matrix = create_design_matrix_nilearn(tsk_evnts, n_timepoints, tr,
                                                                 hrf_model=hrf_model, drift_model=drift_model
                                                                 )
                    # Fit GLM using nilearn approach
                    timeseries_residuals[scan_key], predicted, betas = fit_glm(timeseries, design_matrix)
                    task_events[scan_key] = tsk_evnts
                    design_matrices[scan_key] = design_matrix

                    # Extract task-based contrasts (beta maps) for each condition.
                    # Nuisance columns (drift_*, constant) are excluded; each remaining
                    # column's beta vector is a parcel-wise contrast map.
                    dm_columns = design_matrix.columns.tolist()
                    task_cols = [col for col in dm_columns
                                 if not col.startswith('drift_') and col != 'constant']
                    task_contrasts[scan_key] = {
                        col: betas[:, dm_columns.index(col)] for col in task_cols
                    }

                    # get autocorrelation of task events
                    tsk_evnts_flat = tsk_evnts.copy()
                    tsk_evnts_flat['trial_type'] = 'trial'
                    design_matrix_flat = create_design_matrix_nilearn(tsk_evnts_flat, n_timepoints, tr,
                                                                      hrf_model=hrf_model, drift_model=drift_model
                                                                      )
                    task_regressor_flat[scan_key] = design_matrix_flat['trial'].values
                    # atleast_2d keeps this a (1, n_lags) array, as it was before compute_acf
                    # moved to snaplab_tools (which returns 1-D for a 1-D input).
                    task_events_acf[scan_key] = np.atleast_2d(
                        compute_acf(design_matrix_flat['trial'].values))
                except Exception:
                    timeseries_residuals[scan_key] = None
                    task_events[scan_key] = None
                    design_matrices[scan_key] = None
                    task_regressor_flat[scan_key] = None
                    task_events_acf[scan_key] = None
                    task_contrasts[scan_key] = None
            else:
                timeseries_residuals[scan_key] = None
                task_events[scan_key] = None
                design_matrices[scan_key] = None
                task_regressor_flat[scan_key] = None
                task_events_acf[scan_key] = None
                task_contrasts[scan_key] = None

        self.timeseries_residuals = timeseries_residuals
        self.task_events = task_events
        self.design_matrices = design_matrices
        self.task_regressor_flat = task_regressor_flat
        self.task_events_acf = task_events_acf
        self.task_contrasts = task_contrasts
    
    def resolve_rest_windows(self, verbose: bool = True):
        """Fix every length-matched window of every covered rest scan (see REST_WINDOW_MATCH).

        Sets
        ----
        self.rest_window_span : dict
            ``match -> {scan_key -> (start, stop)}``, covered rest scans only. Every match of every
            covered modality present has an entry, empty when no subject has that task run.
        self.rest_window_meta : dict
            ``match -> {scan_key -> metadata}``, written into each length-matched ACF file.

        Raises
        ------
        ValueError
            If there is no timeseries data, or a window does not fit inside its rest run.
        """
        if getattr(self, 'timeseries_data', None) is None:
            raise ValueError('no timeseries data; length-matched windows are cut from loaded runs')

        groups = defaultdict(lambda: {'task': {}, 'rest': []})
        for scan_key in self.scans:
            kv = self._parse_scan_string(scan_key)
            if not rest_window_covered(self.dataset, kv.get('modality', '')):
                continue
            group = (kv.get('sub', ''), kv.get('ses'), kv.get('modality', ''))
            if kv.get('condition') == 'rest':
                groups[group]['rest'].append((scan_key, kv))
            elif kv.get('condition') == 'task':
                groups[group]['task'][kv.get('task', '')] = scan_key

        spans, meta, n_skipped = {}, {}, defaultdict(int)
        for (_, _, modality), members in groups.items():
            for match, tasks in REST_WINDOW_MATCH[(self.dataset, modality)].items():
                spans.setdefault(match, {})
                meta.setdefault(match, {})
                task_keys = [members['task'].get(t) for t in tasks]
                lengths = [int(np.asarray(self.timeseries_data[k]).shape[1]) for k in task_keys
                           if k is not None and self.timeseries_data.get(k) is not None]
                if not lengths:                     # no run of this task -> nothing to match to
                    n_skipped[match] += len(members['rest'])
                    continue
                n_target = min(lengths)
                for scan_key, kv in members['rest']:
                    rest = self.timeseries_data.get(scan_key)
                    if rest is None:
                        n_skipped[match] += 1
                        continue
                    tr = float(kv.get('tr'))
                    offset_seconds = rest_control_offset_seconds(self.dataset, modality)
                    offset_frames = rest_control_offset_frames(self.dataset, modality, tr)
                    n_rest = int(np.asarray(rest).shape[1])
                    if offset_frames + n_target > n_rest:
                        raise ValueError(
                            f'{scan_key}: offset {offset_frames} + {match}-matched {n_target} frames '
                            f'exceeds the {n_rest}-frame rest run; REST_WINDOW_MATCH or the offset '
                            f'table is wrong for {self.dataset} {modality}')
                    spans[match][scan_key] = (int(offset_frames), int(offset_frames + n_target))
                    meta[match][scan_key] = {
                        'window_kind':       'length_matched_rest',
                        'window_note':       REST_WINDOW_NOTE,
                        'rest_match':        match,
                        'matched_task':      list(tasks),
                        'target_frames':     int(n_target),
                        'source_window':     [int(offset_frames), int(offset_frames + n_target)],
                        'source_run_frames': int(n_rest),
                        'offset_seconds':    float(offset_seconds),
                        'offset_frames':     int(offset_frames),
                        'tr':                kv.get('tr', ''),
                        'modality':          modality,
                        'dataset':           self.dataset,
                    }
        self.rest_window_span, self.rest_window_meta = spans, meta
        if verbose:
            for match, by_scan in spans.items():
                print(f'Length-matched rest windows ({rest_window_token(match)}): {len(by_scan)} '
                      f'resolved, {n_skipped[match]} rest scan(s) skipped (no {match} run or no '
                      f'rest data)')
        return spans

    def build_rest_controls(self, int_method: str = 'auc_zero_crossing', use_fft: bool = True,
                            dtype=np.float32, verbose: bool = True):
        """Build the matched rest control for every task scan, as ACFs.

        Two controls are built from the same design in one pass:

        ``full``
            the complete design -- task regressors, drift basis and constant -- via ``fit_glm``.
        ``drift``
            the drift basis and constant only, via ``drift_only_control``. This is what separates
            "the cosine basis did it" from "the task regressors did it", and on these data the
            drift term is the larger of the two for every condition except WM.

        Parameters
        ----------
        int_method : str
            Passed to ``drift_only_control``; only its ``acf`` output is kept, so this affects
            nothing that is cached. Left configurable so it matches the analysis if that ever
            changes.
        use_fft : bool
            Passed to ``compute_acf``. Pass what the task side used.
        dtype : numpy dtype
            Storage dtype for the ACFs. float32 by default: ACF values are bounded in [-1, 1] and
            every estimator integrates or thresholds them, so float32 is far beyond the precision
            of the estimate and halves the cache.
        verbose : bool
            Print a summary, including how many designs had to be truncated.

        Sets
        ----
        self.autocorr_rest_control, self.autocorr_rest_control_drift : dict
            ``scan_key -> (n_regions, n_lags)`` ACF, keyed by the **task** scan. None for rest
            scans and for any task scan whose control could not be built.
        self.rest_control_meta : dict
            ``scan_key -> metadata dict``, the per-file provenance required by the contract.

        Raises
        ------
        ValueError
            If the offset leaves no window in a paired rest run, or if a non-zero offset leaves
            too little room for a length-matched control. Both mean the offset table is wrong for
            this dataset, so they are raised rather than counted as per-scan failures.

        """
        if getattr(self, 'design_matrices', None) is None:
            raise ValueError('no design matrices; call regress_task_structure() first')

        control_full = dict.fromkeys(self.scans)
        control_drift = dict.fromkeys(self.scans)
        control_meta = {}

        # Group by subject x session x modality: pairing is only meaningful inside an
        # acquisition. A 3T task must never be paired to a 7T rest run, and Penn LEAD's two
        # sessions must not borrow from each other.
        groups = defaultdict(lambda: {'task': [], 'rest': []})
        parsed = {}
        for scan_key in self.scans:
            kv = self._parse_scan_string(scan_key)
            parsed[scan_key] = kv
            group = (kv.get('sub', ''), kv.get('ses'), kv.get('modality', ''))
            if kv.get('condition') == 'task':
                groups[group]['task'].append(scan_key)
            elif kv.get('condition') == 'rest':
                groups[group]['rest'].append(scan_key)

        n_built = n_truncated = n_skipped = n_failed = 0
        for group, members in tqdm(groups.items(), desc='Building matched rest controls'):
            rest_key_of = {parsed[k].get('task', ''): k for k in members['rest']}
            pairing = pair_rest_runs([parsed[k].get('task', '') for k in members['task']],
                                     [parsed[k].get('task', '') for k in members['rest']])

            for scan_key in members['task']:
                kv = parsed[scan_key]
                design = self.design_matrices.get(scan_key)
                events = self.task_events.get(scan_key)
                pair = pairing.get(kv.get('task', ''))
                if design is None or pair is None:
                    n_skipped += 1
                    continue

                rest_task, pe_matched = pair
                rest_scan = self.timeseries_data.get(rest_key_of.get(rest_task))
                if rest_scan is None:
                    n_skipped += 1
                    continue

                # The window is resolved outside the try: a window that does not fit is a
                # wrong offset table, not a scan that happened to fail, and it must not be
                # swallowed into the failure count.
                tr = float(kv.get('tr'))
                modality = kv.get('modality', '')
                offset_seconds = rest_control_offset_seconds(self.dataset, modality)
                offset_frames = rest_control_offset_frames(self.dataset, modality, tr)
                n_rest = int(np.asarray(rest_scan).shape[1])
                # Actual run length, never the nominal one: 14 of HCP-YA's 3,864 3T rest runs
                # are shorter than 1200 frames and one 7T run is 887 rather than 900.
                if offset_frames >= n_rest:
                    raise ValueError(
                        f'{scan_key}: rest-control offset {offset_seconds} s '
                        f'({offset_frames} frames) leaves no window in a {n_rest}-frame '
                        f'{rest_task} run')
                n_frames = min(design.shape[0], n_rest - offset_frames)
                truncated = n_frames < design.shape[0]
                if truncated and offset_frames > 0:
                    # Truncation is tolerated only at offset 0, where it is the known 7T-movie
                    # case. Combined with a non-zero offset it would mean the offset itself had
                    # eaten the task's frames, which is a bad offset rather than a short run.
                    raise ValueError(
                        f'{scan_key}: offset {offset_frames} + task {design.shape[0]} frames '
                        f'exceeds the {n_rest}-frame {rest_task} run; the offset table gives '
                        f'this dataset/modality no room for a length-matched control')

                try:
                    if truncated:
                        # 7T movies only: the movie is longer than a 7T rest run. Rebuild rather
                        # than slice the design -- the cosine basis is defined over the run
                        # length, so a sliced drift column is not the basis for this window.
                        design = create_design_matrix_nilearn(
                            events, n_frames, tr,
                            hrf_model=None if modality == 'MEG' else 'glover',
                            drift_model=None if modality == 'MEG' else 'cosine')
                    window = np.asarray(rest_scan, dtype=float)[
                        :, offset_frames:offset_frames + n_frames]

                    resid_full, _, _ = fit_glm(window, design)
                    control_full[scan_key] = np.atleast_2d(
                        compute_acf(resid_full, use_fft=use_fft)).astype(dtype)
                    del resid_full

                    drift = drift_only_control(window, design, tr, int_method=int_method,
                                               use_fft=use_fft)
                    control_drift[scan_key] = np.atleast_2d(drift['acf']).astype(dtype)
                    del drift, window

                    task_cols, nuisance_cols = split_design_columns(design)
                    control_meta[scan_key] = {
                        'design_from_task': kv.get('task', ''),
                        'source_rest_task': rest_task,
                        'source_window':    [int(offset_frames),
                                             int(offset_frames + n_frames)],
                        'source_run_frames': int(n_rest),
                        'offset_seconds':   float(offset_seconds),
                        'offset_frames':    int(offset_frames),
                        'design_truncated': bool(truncated),
                        'pe_matched':       bool(pe_matched),
                        'n_task_cols':      len(task_cols),
                        'n_drift_cols':     len([c for c in nuisance_cols if c != 'constant']),
                        'tr':               kv.get('tr', ''),
                        'modality':         kv.get('modality', ''),
                        'dataset':          self.dataset,
                    }
                    n_built += 1
                    n_truncated += int(truncated)
                except Exception as e:
                    logger.warning("Failed to build rest control for %s: %s", scan_key, e)
                    control_full[scan_key] = None
                    control_drift[scan_key] = None
                    n_failed += 1

        self.autocorr_rest_control = control_full
        self.autocorr_rest_control_drift = control_drift
        self.rest_control_meta = control_meta

        if verbose:
            offsets = sorted({(m['offset_seconds'], m['offset_frames'])
                              for m in control_meta.values()})
            print("\nMatched rest controls built:")
            print(f"  Built:              {n_built}")
            print("  Window offset:      " + (', '.join(f'{s:g}s ({f} frames)'
                                                        for s, f in offsets) or 'n/a'))
            print(f"  Design truncated:   {n_truncated}")
            if n_skipped:
                print(f"  Skipped (no design or no paired rest run): {n_skipped}")
            if n_failed:
                print(f"  Failed:             {n_failed}")

    # ==================== Analysis Functions ====================        
    def match_registry_to_behavior(self):
        """Subset behavioral data to the subjects present in the scan registry.

        Builds self.df_behavior_matched: one row per unique registry subject that
        also appears in self.df_behavior, preserving registry order.
        """
        df_behavior_matched = pd.DataFrame(columns=self.df_behavior.columns)
        df_behavior_matched.index.name = 'subject_id'
        
        for subject_id in list(dict.fromkeys(scan.subject_id for scan in self.scan_registry)):
            subject_id = str(subject_id)
            if subject_id in self.df_behavior.index and subject_id not in df_behavior_matched.index:
                df_behavior_matched.loc[subject_id] = self.df_behavior.loc[subject_id]
            elif subject_id in self.df_behavior.index and subject_id in df_behavior_matched.index:
                pass
            
        self.df_behavior_matched = df_behavior_matched
        
    def get_precomputed_outputs(
        self,
        output_type: str = 'zero_crossing'
    ) -> Dict[str, Any]:
        """Load previously saved per-scan outputs into self.data_dict.

        Iterates the scan registry and loads each scan's saved output_type file
        (if present) via _load_subject_output; scans with no saved file map to None.

        Parameters
        ----------
        output_type : str
            Output type to load (e.g. 'zero_crossing', 'timescale_tau').
        """
        data_dict = {}

        for scan_key in tqdm([str(scan) for scan in self.scan_registry], desc=f"Loading {output_type}"):
            # Parse scan key
            kv_dict = self._parse_scan_string(scan_key)
            
            subject_id = kv_dict.get('sub', '')
            modality   = kv_dict.get('modality', '')
            task       = kv_dict.get('task', '')
            seg        = kv_dict.get('seg', '')
            space      = kv_dict.get('space')
            session    = kv_dict.get('ses')
            run        = kv_dict.get('run')

            # Check for existing output
            loaded_data = self._load_subject_output(
                subject_id=subject_id,
                output_type=output_type,
                task=task,
                modality=modality,
                seg=seg,
                space=space,
                session=session,
                run=run,
            )
            
            data_dict[scan_key] = loaded_data
            
        self.data_dict = data_dict

# ==================== Standalone Functions ====================
def create_design_matrix_nilearn(
    events,
    n_timepoints,
    tr,
    hrf_model='glover',
    drift_model='cosine',
    high_pass=0.01,
    add_regs=None
    ):
    """Create a first-level design matrix using nilearn.

    Parameters
    ----------
    events : pd.DataFrame
        Events with columns onset, duration, trial_type. Must be non-empty -- see Raises.
    n_timepoints : int
        Number of timepoints (frames).
    tr : float
        Repetition time in seconds.
    hrf_model : str
        HRF model ('glover', 'spm', 'glover + derivative', ...).
    drift_model : str or None
        Drift model ('polynomial', 'cosine', or None).
    high_pass : float
        High-pass filter cutoff in Hz.
    add_regs : ndarray or None
        Additional regressors.

    Returns
    -------
    pd.DataFrame
        nilearn design matrix (task regressors, drift terms, constant).

    Raises
    ------
    ValueError
        If ``events`` is None or empty. nilearn accepts that and returns a perfectly valid
        drift-and-constant design with **no task regressors at all**, so a caller whose events
        failed to resolve gets a complete, plausible-looking set of results that measures the
        drift basis and calls it task regression. ``DataLoader.get_task_events`` returns None for
        any id it cannot resolve -- a malformed subject id is enough -- and that has already
        produced one full run of wrong numbers in ``sfig_drift_spec``. There is no legitimate
        caller here that wants a task-free design: ask for ``drift_model`` alone if that is what
        is meant, or use :func:`drift_only_control`.
    """
    if events is None or len(events) == 0:
        raise ValueError(
            'create_design_matrix_nilearn received no task events '
            f'({"None" if events is None else "an empty frame"}). nilearn would build a design '
            'with drift and constant but no task regressors, and every downstream number would '
            'look fine. Check that the subject id reached get_task_events in the form its '
            'handler expects.')

    # Create frame times
    frame_times = np.arange(n_timepoints) * tr
    
    # Use nilearn to create design matrix
    design_matrix = make_first_level_design_matrix(
        frame_times=frame_times,
        events=events,
        hrf_model=hrf_model,
        drift_model=drift_model,
        high_pass=high_pass,
        add_regs=add_regs
    )
    
    return design_matrix

def fit_glm(data, design_matrix):
    """Fit a GLM per region, masking NaN and zero-variance regions.

    Regions with any NaN or zero temporal variance are excluded from the fit and
    returned as NaN in all three outputs.

    Parameters
    ----------
    data : ndarray
        2D array (n_parcels, n_timepoints).
    design_matrix : pd.DataFrame
        nilearn design matrix.

    Returns
    -------
    residuals : ndarray
        Residual timeseries after GLM fitting (NaN for excluded regions).
    predicted : ndarray
        Predicted timeseries (NaN for excluded regions).
    betas : ndarray
        Regression coefficients, shape (n_parcels, n_regressors).
    """
    
    # Convert design matrix to numpy array
    X = design_matrix.values
    n_parcels, n_timepoints = data.shape
    
    # Initialize outputs
    residuals = np.full_like(data, np.nan)
    predicted = np.full_like(data, np.nan)
    betas = np.full((n_parcels, X.shape[1]), np.nan)
    
    # Create mask for valid regions (no NaNs and non-zero variance)
    has_nan = np.any(np.isnan(data), axis=1)
    has_zero_variance = np.var(data, axis=1) == 0
    valid_mask = ~(has_nan | has_zero_variance)
    
    if np.sum(valid_mask) == 0:
        # No valid regions - return all NaNs
        return residuals, predicted, betas
    
    # Fit GLM only for valid regions
    data_valid = data[valid_mask, :]
    
    # Fit GLM using least squares
    betas_valid, residuals_sum, rank, s = np.linalg.lstsq(X, data_valid.T, rcond=None)
    
    # Compute predicted and residual signals for valid regions
    predicted_valid = np.dot(X, betas_valid).T
    residuals_valid = data_valid - predicted_valid
    
    # Place results back into full arrays
    residuals[valid_mask, :] = residuals_valid
    predicted[valid_mask, :] = predicted_valid
    betas[valid_mask, :] = betas_valid.T
    
    return residuals, predicted, betas

def calculate_tsnr(timeseries_matrix):
    """
    Calculate temporal signal-to-noise ratio (tSNR) for ROI timeseries data.
    
    Parameters
    ----------
    timeseries_matrix : numpy.ndarray
        2D array with shape (n_rois, n_timepoints) where each row represents
        an ROI and each column represents a timepoint.
    
    Returns
    -------
    tsnr : numpy.ndarray
        1D array with shape (n_rois,) containing tSNR values for each ROI.
        tSNR = mean(signal) / std(signal) across time for each ROI.
    
    Notes
    -----
    - Higher tSNR values indicate better signal quality
    - ROIs with zero standard deviation will result in inf values
    - Consider masking out ROIs with very low signal or high motion
    """
    # Calculate mean and standard deviation across time (axis=1)
    mean_signal = np.mean(timeseries_matrix, axis=1)
    std_signal = np.std(timeseries_matrix, axis=1, ddof=1)  # ddof=1 for sample std
    
    # Calculate tSNR, handling division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        tsnr = mean_signal / std_signal
    
    return tsnr

# ==================== Standalone Utility Functions (for use in notebooks) ====================
def split_design_columns(design_matrix):
    """Split a nilearn design matrix into its task columns and its nuisance columns.

    One definition of "which column is a task regressor", used by the GLM contrast extraction,
    by the drift-only baseline the supplementary figure is built on, and by the notebook that
    reports how many of each there are. Written once here so the baseline cannot end up
    regressing a different column set than the figure says it did.

    Parameters
    ----------
    design_matrix : pd.DataFrame
        nilearn design matrix.

    Returns
    -------
    (task_cols, nuisance_cols) : (list of str, list of str)
        `nuisance_cols` is the ``drift_*`` basis plus ``constant``, in the matrix's own order.
    """
    columns = [str(c) for c in design_matrix.columns]
    nuisance_cols = [c for c in columns if c.startswith('drift_') or c == 'constant']
    task_cols = [c for c in columns if c not in nuisance_cols]
    return task_cols, nuisance_cols


def drift_only_control(raw_data, design_matrix, tr, int_method='auc_zero_crossing',
                       use_fft=True):
    """Refit a task GLM with its nuisance columns alone, and measure what that costs the INT.

    A design built by ``create_design_matrix_nilearn`` carries a cosine drift basis and a
    constant alongside the condition regressors, so its residual has been high-pass filtered on
    top of whatever filtering the timeseries already received at load time. When the input was
    only linearly detrended (``DataLoader.load_batch(detrend=True, frequency_filter=False)``)
    that cosine basis is *new* filtering, and a raw-versus-residual INT comparison charges its
    effect to task regression: on HCP-YA working memory it is about a third of the difference.

    The residual returned here is the fix, not just the diagnostic. Using it as the "before"
    side of the figure puts the identical drift basis on both sides, so what is left between
    them is the task regressors and nothing else.

    Parameters
    ----------
    raw_data : (n_regions, n_timepoints) ndarray
        The same timeseries the full GLM is fit to.
    design_matrix : pd.DataFrame
        The full nilearn design matrix. Only its ``drift_*`` and ``constant`` columns are used.
    tr : float
        Repetition time in seconds.
    int_method : str
        Key of ``TIMESCALE_METHODS``; use the one the rest of the analysis uses.
    use_fft : bool
        Passed to ``compute_acf``.

    Returns
    -------
    dict
        ``residual`` (n_regions, n_timepoints) the drift-only residual -- the figure's
        baseline; ``acf`` (n_regions, n_lags) its ACF; ``int_drift`` (n_regions,) its INT;
        ``int_raw`` (n_regions,) the INT of the untouched input, for the comparison;
        ``delta`` (n_regions,) ``int_drift - int_raw``; ``median``, ``q25``, ``q75`` of the
        finite deltas; ``nuisance_cols`` the columns actually regressed; ``n_valid`` regions.
    """
    if int_method not in TIMESCALE_METHODS:
        raise ValueError(f"unknown int_method {int_method!r}; "
                         f"available: {sorted(TIMESCALE_METHODS)}")
    _, nuisance_cols = split_design_columns(design_matrix)
    if not nuisance_cols:
        raise ValueError('design matrix has no drift_* or constant columns to control for')

    int_fn = TIMESCALE_METHODS[int_method]
    raw_data = np.asarray(raw_data, dtype=float)
    acf_raw = np.atleast_2d(compute_acf(raw_data, use_fft=use_fft))
    int_raw = np.asarray(int_fn(acf_raw, tr), dtype=float)

    residual, _, _ = fit_glm(raw_data, design_matrix[nuisance_cols])
    acf_drift = np.atleast_2d(compute_acf(residual, use_fft=use_fft))
    int_drift = np.asarray(int_fn(acf_drift, tr), dtype=float)

    delta = int_drift - int_raw
    finite = np.isfinite(delta)
    return {
        'residual': residual,
        'acf': acf_drift,
        'int_drift': int_drift,
        'int_raw': int_raw,
        'delta': delta,
        'median': float(np.median(delta[finite])) if finite.any() else np.nan,
        'q25': float(np.percentile(delta[finite], 25)) if finite.any() else np.nan,
        'q75': float(np.percentile(delta[finite], 75)) if finite.any() else np.nan,
        'nuisance_cols': nuisance_cols,
        'n_valid': int(finite.sum()),
    }


#: Phase-encoding tokens a task name may end in. Matched case-sensitively: UCLA's ``scap`` ends
#: in a lowercase "ap" and is not an AP-encoded scan.
PE_DIRECTIONS = ('LR', 'RL', 'AP', 'PA')

#: Control kind -> (output_type token, ``control_kind`` metadata value).
REST_CONTROL_KINDS = {
    'full':  ('restctrl',       'matched_rest_control'),
    'drift': ('restctrl-drift', 'drift_only_rest_control'),
}

#: Written into every control file so the file cannot be misread once it leaves its directory.
REST_CONTROL_NOTE = ("REST timeseries with this task run's design projected out. "
                     "Not a measurement of rest during the task. Contains no task.")


def phase_encoding_direction(task):
    """Phase-encoding direction of a task name, or None when it does not encode one.

    Read as the last two characters, case-sensitively, which is what separates HCP's
    ``tfMRIGUESSINGPA`` (PA-encoded) from UCLA's ``scap`` (not AP-encoded). Datasets whose task
    names carry no direction (``rest``, ``nback``, ``reversallearning``) return None and pair by
    index instead -- see :func:`pair_rest_runs`.

    Parameters
    ----------
    task : str
        Task/run label as it appears in the scan registry.

    Returns
    -------
    str or None
        One of ``PE_DIRECTIONS``, or None.
    """
    tail = str(task)[-2:]
    return tail if tail in PE_DIRECTIONS else None


def condition_of(task):
    """The condition a task run is averaged into, from the shared run vocabulary.

    Uses :data:`TASK_RUN_GROUPS` and :data:`MOVIE_GROUP` -- the same lists
    :func:`_apply_scan_groupings` averages by -- so a control is always built for the grouping the
    notebooks actually use. A run that belongs to no group is its own condition, which is the
    right answer for single-run datasets (Penn LEAD's ``nback``, PNC's ``frac2back``).

    Parameters
    ----------
    task : str
        Task/run label.

    Returns
    -------
    str
        Condition key, e.g. ``'task-WM'``.
    """
    for runs, condition in list(TASK_RUN_GROUPS) + [MOVIE_GROUP]:
        if task in runs:
            return condition
    return f'task-{task}'


def pair_rest_runs(task_runs, rest_runs):
    """Pair each task run with the rest run its matched control is built from.

    **Phase encoding first, then index within the condition.** For a task run, the candidate pool
    is the rest runs sharing its phase-encoding direction; the run picked is the one at the task
    run's ordinal position *within its own condition*, cycling if the condition has more runs than
    the pool holds. Where no direction can be read on either side, the pool is every rest run and
    the pairing is index-only.

    Matching phase encoding matches susceptibility distortion and dropout, so the control shares
    the task run's geometry as well as its design and its length. Index-only pairing is not a
    substitute and the 7T movies are the proof: the movie runs are AP, PA, PA, AP while the 7T
    rest runs are PA, AP, PA, AP, so pairing by position mismatches ``tfMRIMOVIE17TAP`` and
    ``tfMRIMOVIE27TPA`` and gets the other two right by luck. On HCP-YA 3T the two rules agree
    entirely, because ``…LR`` sorts before ``…RL`` on both sides -- which is exactly why the
    intent has to be written down rather than inferred from a case where it does not bite.

    Parameters
    ----------
    task_runs : sequence of str
        The subject's task run labels, in registry order.
    rest_runs : sequence of str
        The subject's rest run labels *of the same modality*, in registry order. Pairing a 3T task
        to a 7T rest run would defeat the point of the control.

    Returns
    -------
    dict
        ``task_run -> (rest_run, pe_matched)``. ``pe_matched`` is False when the fallback was
        used, and is recorded in each control file's metadata. Task runs are omitted when
        ``rest_runs`` is empty.

    Examples
    --------
    >>> pair_rest_runs(['tfMRIWMLR', 'tfMRIWMRL'],
    ...                ['rfMRIREST1LR', 'rfMRIREST1RL', 'rfMRIREST2LR', 'rfMRIREST2RL'])
    {'tfMRIWMLR': ('rfMRIREST1LR', True), 'tfMRIWMRL': ('rfMRIREST1RL', True)}
    """
    rest_runs = list(rest_runs)
    if not rest_runs:
        return {}

    by_direction = defaultdict(list)
    for rest_run in rest_runs:
        by_direction[phase_encoding_direction(rest_run)].append(rest_run)

    pairing = {}
    seen = defaultdict(int)      # (condition, direction) -> runs of this kind already paired
    for task_run in task_runs:
        direction = phase_encoding_direction(task_run)
        pool = by_direction.get(direction) if direction is not None else None
        pe_matched = bool(pool)
        if not pe_matched:
            pool = rest_runs
        key = (condition_of(task_run), direction if pe_matched else None)
        pairing[task_run] = (pool[seen[key] % len(pool)], pe_matched)
        seen[key] += 1
    return pairing


def rest_control_output_type(base, kind):
    """Append a control token to an ``output_type``, e.g. ``autocorr`` -> ``autocorr_restctrl``.

    Kept next to :data:`REST_CONTROL_KINDS` so the token cannot be spelled two ways. The token
    goes last, in the slot ``_resids`` occupies for the task side -- both answer "what was
    projected out of what", and a file never carries both.

    Parameters
    ----------
    base : str
        Output type built so far, including any ``_resampled`` and frequency-filter tokens. Those
        must already be present: a control that does not inherit them is silently comparing
        unlike things.
    kind : {'full', 'drift'}
        Which control.

    Returns
    -------
    str
    """
    if kind not in REST_CONTROL_KINDS:
        raise ValueError(f"unknown rest-control kind {kind!r}; "
                         f"available: {sorted(REST_CONTROL_KINDS)}")
    return f'{base}_{REST_CONTROL_KINDS[kind][0]}'


def compute_deflections(rest_timescales, task_timescales, absolute=False):
    """Compute task-vs-rest timescale deflections (task minus rest).

    Elementwise, so it works on any pair of matching-shaped arrays — typically 1-D
    (n_subjects,) for a single region, or 2-D (n_subjects, n_regions).

    Parameters
    ----------
    rest_timescales, task_timescales : array-like
        Matching-shaped timescale values. Series/DataFrame are converted to arrays.
    absolute : bool
        Return |task - rest| rather than the signed difference. Note that task
        shortens INT in ~95% of subject x region cells (HCP-YA WM), so the signed
        deflection is near-unidirectional and abs() is close to a sign flip; the
        signed value is directly interpretable as "amount of INT shortening".

    Returns
    -------
    ndarray
        Deflections, same shape as the inputs.

    Notes
    -----
    This function deliberately has no rest-INT control. Regressing rest out of
    (task - rest) is algebraically identical to residualising task on rest — the
    subtraction cancels exactly — which silently turns a deflection into a residual
    and destroys the S-A structure that fig2 tests. Control for rest INT at the
    analysis step instead (see compute_regional_corrs's `covariate`).
    """
    if isinstance(rest_timescales, (pd.Series, pd.DataFrame)):
        rest_timescales = rest_timescales.to_numpy()
    if isinstance(task_timescales, (pd.Series, pd.DataFrame)):
        task_timescales = task_timescales.to_numpy()

    rest_timescales = np.asarray(rest_timescales, dtype=float)
    task_timescales = np.asarray(task_timescales, dtype=float)

    if rest_timescales.shape != task_timescales.shape:
        raise ValueError(
            f"rest_timescales and task_timescales must have the same shape; "
            f"got {rest_timescales.shape} and {task_timescales.shape}."
        )

    int_deflections = task_timescales - rest_timescales
    return np.abs(int_deflections) if absolute else int_deflections


def compute_regional_corrs(rest_data, task_data, behavior_data,
                           covariate=None, absolute=False, verbose=False, return_pval=False):
    """Correlate rest, task, and deflection timescales with a behavior, per region.

    For each region, correlates the behavior with rest, task, and the task-vs-rest
    deflection across subjects. Subjects with missing behavior are dropped first;
    regions with < 3 valid subjects are skipped. All three correlations for a region
    use the same subject set, so they are directly comparable.

    Parameters
    ----------
    rest_data, task_data : ndarray
        (n_subjects, n_regions) timescale matrices.
    behavior_data : ndarray
        (n_subjects,) behavioral values.
    covariate : ndarray or None
        (n_subjects,) or (n_subjects, k) subject-level covariates. When given, every
        correlation becomes a partial correlation controlling for them (df = n - 2 - k).
        Build the matrix with src.gams.build_nuisance so fig2 and fig3/fig4 keep one
        definition of the nuisance set.

        Every column must be SUBJECT-level — one value per subject, constant across
        regions — e.g. mean rest INT (np.nanmean(rest_data, axis=1)), age, sex, head
        motion. Such a covariate removes a global confound without reshaping the regional
        map the S-A test then reads.

        Do NOT pass a region-specific rest vector. rest_data[:, r] carries the S-A
        gradient, so regressing it out strips the S-A structure these correlations are
        used to test — empirically it collapses fig2's S-A coupling from r ~ +0.56 to
        r ~ -0.18 (see the diagnostics cell in timescales_analysis_fig2.ipynb). That
        mistake is shape-legal, so no check here can catch it; passing the whole
        (n_subjects, n_regions) matrix IS caught.
    absolute : bool
        Use |task - rest| instead of the signed deflection (see compute_deflections).
    verbose : bool
        Print the valid-subject count.
    return_pval : bool
        Also return the three p-value arrays.

    Returns
    -------
    tuple of ndarray
        (rest_corr, task_corr, deflection_corr), each (n_regions,); plus the
        matching p-value arrays when return_pval is True.
    """
    n_regions = rest_data.shape[1]
    rest_correlations       = np.full(n_regions, np.nan)
    task_correlations       = np.full(n_regions, np.nan)
    deflection_correlations = np.full(n_regions, np.nan)
    rest_pvalues            = np.full(n_regions, np.nan)
    task_pvalues            = np.full(n_regions, np.nan)
    deflection_pvalues      = np.full(n_regions, np.nan)

    behavior_data = np.asarray(behavior_data, dtype=float)

    beh_valid      = ~np.isnan(behavior_data)
    behavior_clean = behavior_data[beh_valid]
    rest_clean     = rest_data[beh_valid, :]
    task_clean     = task_data[beh_valid, :]

    covariate_clean = None
    if covariate is not None:
        covariate = np.asarray(covariate, dtype=float)
        if covariate.ndim not in (1, 2) or covariate.shape[0] != behavior_data.shape[0]:
            raise ValueError(
                f"covariate must be (n_subjects,) or (n_subjects, k) matching behavior_data "
                f"{behavior_data.shape}; got {covariate.shape}. Pass subject-level values "
                f"such as np.nanmean(rest_data, axis=1), age or sex — not a regional vector."
            )
        if covariate.ndim == 2 and covariate.shape[1] == n_regions:
            raise ValueError(
                f"covariate has one column per region ({n_regions}), so it looks like a whole "
                f"regional matrix rather than subject-level covariates. Regressing out "
                f"region-wise rest INT strips the S-A structure these correlations test."
            )
        covariate_clean = covariate[beh_valid]

    if verbose:
        print(f"  Valid subjects: {np.sum(beh_valid)}/{len(behavior_data)}")

    deflections = compute_deflections(rest_clean, task_clean, absolute=absolute)

    for region_idx in range(n_regions):
        rest_region = rest_clean[:, region_idx]
        task_region = task_clean[:, region_idx]

        # Joint mask so rest / task / deflection all describe the same subject set.
        # (isnan(deflection) is exactly isnan(rest) | isnan(task).)
        valid = ~(np.isnan(rest_region) | np.isnan(task_region))
        if valid.sum() < 3:
            continue

        behavior_v  = behavior_clean[valid]
        covariate_v = None if covariate_clean is None else covariate_clean[valid]

        rest_correlations[region_idx], rest_pvalues[region_idx], _ = partial_pearsonr(
            rest_region[valid], behavior_v, covariate_v)
        task_correlations[region_idx], task_pvalues[region_idx], _ = partial_pearsonr(
            task_region[valid], behavior_v, covariate_v)
        deflection_correlations[region_idx], deflection_pvalues[region_idx], _ = partial_pearsonr(
            deflections[valid, region_idx], behavior_v, covariate_v)

    if return_pval:
        return (rest_correlations, task_correlations, deflection_correlations,
                rest_pvalues, task_pvalues, deflection_pvalues)
    else:
        return (rest_correlations, task_correlations, deflection_correlations)


def filter_data_dict(data_dict, data_dict_residuals=None, remove_subject_level=True):
    """
    Filter data dictionary by removing scans with missing or problematic data.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary of scan data
    data_dict_residuals : dict, optional
        Dictionary of residualized data to replace task scans
    remove_subject_level : bool, default=True
        If True, remove all scans for a subject if any scan has issues.
        If False, only remove the specific problematic scan.
    
    Returns
    -------
    data_dict : dict
        Filtered data dictionary
    data_dict_residuals : dict, optional
        Only returned if data_dict_residuals was provided
    """
    
    remove_these_keys = []
    
    # Check for None values
    for scan_key in data_dict.keys():
        if data_dict[scan_key] is None:
            if remove_subject_level:
                sub_key = scan_key.split('_')[0]
                remove_these_keys.append([k for k in data_dict.keys() if sub_key in k])
            else:
                remove_these_keys.append([scan_key])
    
    # Optionally replace task data with residualized data
    if data_dict_residuals is not None:
        for scan_key in data_dict.keys():
            if 'condition-task' in scan_key:
                try:
                    data_dict[scan_key] = data_dict_residuals[scan_key].copy()
                except Exception:
                    if remove_subject_level:
                        sub_key = scan_key.split('_')[0]
                        remove_these_keys.append([k for k in data_dict.keys() if sub_key in k])
                    else:
                        remove_these_keys.append([scan_key])
    
    # Flatten list and remove duplicates
    remove_these_keys = list(dict.fromkeys([item for sublist in remove_these_keys for item in sublist]))
    
    # Remove marked keys
    for remove_key in remove_these_keys:
        if remove_key in data_dict:
            del data_dict[remove_key]
    
    if data_dict_residuals is None:
        return data_dict
    else:
        return data_dict, data_dict_residuals

def _apply_scan_groupings(
    data_dict: dict,
    average_3t_rest: bool = False,
    average_7t_rest: bool = False,
    average_all_rest: bool = False,
    average_7t_movies: bool = False,
    average_3t_task_pairs: bool = False,
    average_3t_tasks: bool = False,
    average_meg_rest: bool = False,
    average_meg_task: bool = False,
    statistic: str = 'mean',
) -> dict:
    """
    Apply scan-grouping logic to a data_dict.

    All keys in data_dict are expected to be scan-key strings; values are
    1-D arrays of shape (n_regions,).  Returns a new dict with the same
    structure after the requested grouping operations are applied.

    statistic : {'mean', 'median'}
        How runs are collapsed into a condition. 'mean' is the default and the historical
        behaviour. It makes no difference for a two-run pair (the median of two values is their
        mean), so it only bites where a group has three or more runs -- HCP-YA's four rest runs,
        or average_3t_tasks over twelve.

    Raises ValueError on mutually-exclusive flag combinations, or an unknown statistic.
    """
    if statistic not in ('mean', 'median'):
        raise ValueError(f"statistic must be 'mean' or 'median'; got {statistic!r}")
    _reduce = np.nanmean if statistic == 'mean' else np.nanmedian
    if average_3t_rest and average_all_rest:
        raise ValueError("average_3t_rest and average_all_rest are mutually exclusive")
    if average_7t_rest and average_all_rest:
        raise ValueError("average_7t_rest and average_all_rest are mutually exclusive")
    if average_3t_task_pairs and average_3t_tasks:
        raise ValueError("average_3t_task_pairs and average_3t_tasks are mutually exclusive")

    out = dict(data_dict)  # shallow copy — we replace keys, not mutate values

    def _avg_group(keys_to_merge, new_key):
        """Collapse a list of keys into new_key with `statistic`; remove originals."""
        arrays = [out[k] for k in keys_to_merge if k in out and out[k] is not None]
        if not arrays:
            return
        out[new_key] = _reduce(np.stack(arrays, axis=0), axis=0)
        for k in keys_to_merge:
            out.pop(k, None)

    def _keys_matching(patterns):
        return [k for k in list(out) if any(p in k for p in patterns)]

    _3t_rest_tasks  = REST_RUNS_3T
    _7t_rest_tasks  = REST_RUNS_7T
    _7t_movie_tasks = MOVIE_RUNS_7T
    _3t_task_pairs  = TASK_RUN_GROUPS
    _meg_rest_tasks = REST_RUNS_MEG
    _meg_task_tasks = TASK_RUNS_MEG

    if average_3t_rest:
        _avg_group(_keys_matching(_3t_rest_tasks), 'task-REST_3T')
    if average_7t_rest:
        _avg_group(_keys_matching(_7t_rest_tasks), 'task-REST_7T')
    if average_all_rest:
        _avg_group(_keys_matching(_3t_rest_tasks + _7t_rest_tasks), 'task-Rest')
    if average_7t_movies:
        _avg_group(_keys_matching(_7t_movie_tasks), 'task-Movies')
    if average_3t_task_pairs:
        for task_names, new_key in _3t_task_pairs:
            _avg_group(_keys_matching(task_names), new_key)
    if average_3t_tasks:
        all_3t_task_names = [t for pair, _ in _3t_task_pairs for t in pair]
        _avg_group(_keys_matching(all_3t_task_names), 'task-Tasks')
    if average_meg_rest:
        _avg_group(_keys_matching(_meg_rest_tasks), 'task-MEG_Rest')
    if average_meg_task:
        _avg_group(_keys_matching(_meg_task_tasks), 'task-MEG_Task')

    return out


def create_ints_dataframe_custom_grouping(
    data_dict,
    n_regions=400,
    average_3t_rest=False,
    average_7t_rest=False,
    average_all_rest=False,
    average_7t_movies=False,
    average_3t_task_pairs=False,
    average_3t_tasks=False,
    average_meg_rest=False,
    average_meg_task=False,
    sort_by_median=False,
    statistic='mean',
):
    """
    Create a DataFrame (regions × scan-types) of group-level INTs.

    Parameters
    ----------
    data_dict : dict
        scan_key → 1-D array of shape (n_regions,)
    n_regions : int
        Number of brain regions (used only to initialise the output DataFrame)
    average_* : bool
        Scan-grouping flags — see _apply_scan_groupings for semantics.
    sort_by_median : bool
        Sort columns by descending median across regions. This is a sort KEY and is unrelated to
        `statistic`: it orders columns the same way whichever statistic produced their values.
    statistic : {'mean', 'median'}
        How subjects are collapsed into one value per region, and how runs are collapsed into a
        condition (it is passed to :func:`_apply_scan_groupings`, so both steps use the same one
        and the result is never a mixture). 'mean' is the default and the historical behaviour.
        The median is the more robust summary where a few subjects carry extreme INTs, which is
        what an ACF-based estimator produces for a scan that barely decorrelates.
    """
    if statistic not in ('mean', 'median'):
        raise ValueError(f"statistic must be 'mean' or 'median'; got {statistic!r}")
    _reduce = np.nanmean if statistic == 'mean' else np.nanmedian
    def _extract_scan_name(scan_key):
        """Return the task-entity value from a scan key string."""
        for part in scan_key.split('_'):
            if part.startswith('task-'):
                return part[len('task-'):]
        return scan_key

    # Group by scan type (strip per-subject info), collapsing subjects with `statistic`
    by_scan_type = defaultdict(list)
    for scan_key, arr in data_dict.items():
        if arr is not None:
            by_scan_type[_extract_scan_name(scan_key)].append(arr)

    scan_type_stat = {
        name: _reduce(np.stack(arrs, axis=0), axis=0)
        for name, arrs in by_scan_type.items()
    }

    # Apply grouping, with the same statistic so the two steps cannot disagree
    scan_type_stat = _apply_scan_groupings(
        scan_type_stat,
        average_3t_rest=average_3t_rest,
        average_7t_rest=average_7t_rest,
        average_all_rest=average_all_rest,
        average_7t_movies=average_7t_movies,
        average_3t_task_pairs=average_3t_task_pairs,
        average_3t_tasks=average_3t_tasks,
        average_meg_rest=average_meg_rest,
        average_meg_task=average_meg_task,
        statistic=statistic,
    )

    df = pd.DataFrame(scan_type_stat, index=range(n_regions))

    if sort_by_median:
        order = df.median(axis=0).sort_values(ascending=False).index
        df = df[order]

    return df

def sort_and_group_timescales(timescales, brain_map, ntile=10):
    """Sort regions by a brain map and average them into equal-sized ntiles.

    Reorders regions by ascending brain_map value, then averages consecutive
    equal-sized blocks into `ntile` bins (nanmean across regions in each bin).

    Parameters
    ----------
    timescales : ndarray
        (n_subjects, n_regions) values.
    brain_map : ndarray
        (n_regions,) values used to sort regions (e.g. the S-A axis).
    ntile : int
        Number of equal-sized bins.

    Returns
    -------
    ndarray
        (n_subjects, ntile) block-averaged values.
    """
    n_nodes = len(brain_map)
    brain_map_sort_idx = np.argsort(brain_map)
    
    timescales_sorted = timescales[:, brain_map_sort_idx]
    timescales_sorted_dec = np.zeros((timescales_sorted.shape[0], ntile))

    step_size = int(n_nodes / ntile)
    counter = 0
    for i in np.arange(0, n_nodes, step_size):
        # timescales_sorted_dec[:, counter] = timescales_sorted[:, i:i+step_size].mean(axis=1)
        timescales_sorted_dec[:, counter] = np.nanmean(timescales_sorted[:, i:i+step_size], axis=1)
        counter += 1
        
    return timescales_sorted_dec

def convert_data_dict(data_dict, scan_registry, brain_map=None, verbose=True):
    """Regroup a flat scan_key -> array dict into a task -> stacked-array dict.

    For each unique task in the registry, stacks that task's per-subject arrays.
    If brain_map is given, each task's stack is additionally sorted and binned via
    sort_and_group_timescales.

    Parameters
    ----------
    data_dict : dict
        scan_key -> 1-D region array.
    scan_registry : list of ScanInfo
        Registry used to enumerate tasks and their scan keys.
    brain_map : ndarray or None
        If given, apply sort_and_group_timescales per task.
    verbose : bool
        Print each task's output shape.

    Returns
    -------
    dict
        task -> (n_subjects, n_regions) array, or (n_subjects, ntile) if brain_map
        was provided.
    """
    unique_tasks = list(dict.fromkeys(scan.task for scan in scan_registry))
    data_dict_converted = dict.fromkeys(unique_tasks)

    for task in data_dict_converted.keys():
        sub_list = [str(scan) for scan in scan_registry if scan.task == task]
        extracted_data = {key: data_dict[key] for key in sub_list if key in data_dict}
        extracted_data = np.array(list(extracted_data.values()))

        if brain_map is not None:
            extracted_data = sort_and_group_timescales(extracted_data, brain_map)

        data_dict_converted[task] = extracted_data
        
        if verbose:
            print(task, extracted_data.shape)
            
    return data_dict_converted

def average_data_dict_converted(
    data_dict_converted,
    average_7t_rest=False,
    average_3t_rest=False,
    average_all_rest=False,
    average_7t_movies=False,
    average_3t_task_pairs=False,
    average_3t_tasks=False,
    average_meg_rest=False,
    average_meg_task=False,
):
    """Average scan groups within a task-keyed dict produced by convert_data_dict."""
    # data_dict_converted is keyed by task name; values are (n_subjects, n_regions) arrays.
    # _apply_scan_groupings._keys_matching uses substring matching, so it works whether
    # the key is a full scan string or just a bare task name like 'rfMRIREST1LR'.
    flat = {task: arr for task, arr in data_dict_converted.items() if arr is not None}

    return _apply_scan_groupings(
        flat,
        average_3t_rest=average_3t_rest,
        average_7t_rest=average_7t_rest,
        average_all_rest=average_all_rest,
        average_7t_movies=average_7t_movies,
        average_3t_task_pairs=average_3t_task_pairs,
        average_3t_tasks=average_3t_tasks,
        average_meg_rest=average_meg_rest,
        average_meg_task=average_meg_task,
    )
