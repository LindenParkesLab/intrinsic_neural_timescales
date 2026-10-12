"""Machine-written results summaries for the figure notebooks.

Each of the four figure notebooks ends with a single cell that collates its results into one
Markdown document and **overwrites** it on every run, so the file on disk always describes the
notebook's current configuration. The documents are written for a reader (human or LLM) doing
the write-up: every number that appears in a panel, or that a sentence of the manuscript would
cite, is stated in text so it never has to be read back off a figure.

The formatting lives here rather than in the notebooks for the usual reason: four notebooks
writing the same document format from four hand-rolled f-string blocks would drift. Only the
*content* is notebook-specific.

Nothing here computes a statistic. A report cell may re-derive a quantity the figure cell
printed but did not keep (fig1's paired t-tests, say), but the derivation stays in the notebook
next to the figure it belongs to; this module only lays the result out.

Tables are emitted as GitHub-flavoured Markdown by hand -- ``DataFrame.to_markdown`` needs
``tabulate``, which is not a dependency of this project and is not worth becoming one.
"""

import os
import datetime

import numpy as np
import pandas as pd

from src.data import _find_project_root
from src.plotting import FIGURE_SCRATCH_BASE

# Where summaries are written. Since 2026-09-17 that is the NAS, in a ``results_summaries``
# directory beside the panels the same notebooks export, so a figure and the document describing it
# travel together and neither lives in the repo. ``FIGURE_SCRATCH_BASE`` is imported from
# ``src.plotting`` rather than restated here: two copies of the NAS root is the kind of thing that
# drifts apart.
#
# Set to None to fall back to ``<project root>/results_summaries``, resolved at call time by
# ``_find_project_root``, which walks up from the CWD looking for setup.py -- the same resolver
# behind ``Timescales.output_dir``. That fallback is deliberately NOT derived from ``__file__``:
# every figure notebook starts with ``pip install .``, so the module that actually runs lives in the
# environment's site-packages and ``__file__`` points there, not at the repo. That is exactly how
# the first version of this wrote fig1.md into ``.../site-packages/results_summaries/``.
SUMMARY_DIR = os.path.join(FIGURE_SCRATCH_BASE, 'results_summaries')


def summary_path(stem, out_dir=None):
    """Path this notebook's summary is written to, creating the directory.

    Parameters
    ----------
    stem : str
        File stem, e.g. ``'fig3'``. ``.md`` is appended.
    out_dir : str or None
        Directory to write into. Defaults to :data:`SUMMARY_DIR` (the NAS), or
        ``<project root>/results_summaries`` when that is set to None.

    Returns
    -------
    str
        Absolute path to ``<out_dir>/<stem>.md``.
    """
    if out_dir is None:
        out_dir = SUMMARY_DIR or os.path.join(str(_find_project_root()), 'results_summaries')
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f'{stem}.md')


# ==============================================================================
# Value formatting
# ==============================================================================

def fmt_num(value, nd=3):
    """A number as a short string: ``nd`` significant-ish digits, NaN/None as 'n/a'.

    Small magnitudes fall back to scientific notation rather than printing as '0.000', which
    would silently turn a tiny effect into no effect.
    """
    if value is None:
        return 'n/a'
    if isinstance(value, (str, bool, np.bool_)):
        return str(value)
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(v):
        return 'n/a'
    if v != 0 and abs(v) < 10 ** -nd:
        return f'{v:.1e}'
    return f'{v:.{nd}f}'


def fmt_p(p, floor=None):
    """A p-value as a string, honouring the resolution of the test that produced it.

    ``floor`` is the finest resolution the test can express (``1 / n_perms`` for a permutation
    test). A p at or below it is reported as ``< floor`` rather than as a number the test cannot
    actually resolve -- and never as 0.
    """
    if p is None:
        return 'n/a'
    try:
        p = float(p)
    except (TypeError, ValueError):
        return str(p)
    if not np.isfinite(p):
        return 'n/a'
    if floor is not None and p <= floor:
        return f'< {floor:.1g}'
    if p < 1e-4:
        return f'{p:.1e}'
    return f'{p:.4f}'


def fmt_ci(lo, hi, nd=3):
    """A confidence interval as ``[lo, hi]``."""
    return f'[{fmt_num(lo, nd)}, {fmt_num(hi, nd)}]'


def fmt_range(values, nd=3):
    """``min-max`` of an array, NaNs ignored."""
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).any():
        return 'n/a'
    return f'{fmt_num(np.nanmin(values), nd)} to {fmt_num(np.nanmax(values), nd)}'


def describe(values, nd=3, name=None):
    """One-line descriptive summary of a vector: n, mean, SD, median, range."""
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 'n/a (no finite values)'
    line = (f'n={v.size}, mean={fmt_num(v.mean(), nd)}, SD={fmt_num(v.std(ddof=1) if v.size > 1 else np.nan, nd)}, '
            f'median={fmt_num(np.median(v), nd)}, range {fmt_num(v.min(), nd)} to {fmt_num(v.max(), nd)}')
    return f'{name}: {line}' if name else line


# ==============================================================================
# The report builder
# ==============================================================================

def _escape(text):
    """Make a string safe to sit inside a Markdown table cell.

    A bare ``|`` ends the cell and silently shifts every column after it -- ``peak |r|`` is enough
    to break a whole table -- and a newline ends the row. Both are escaped/flattened rather than
    stripped, so the value the caller passed is still the value the reader sees.
    """
    return str(text).replace('|', '\\|').replace('\n', ' ')


class ResultsReport:
    """Accumulates Markdown blocks, then writes them to one file.

    The document is rewritten in full on every ``write()``, so a summary can never be a mixture
    of two runs' numbers -- which is the whole point of it being machine-written.

    Parameters
    ----------
    title : str
        Document title (an H1).
    notebook : str
        Filename of the notebook that produces this document, recorded in the provenance line so
        a summary read on its own still names what made it.
    path : str
        Where to write. Use :func:`summary_path`.
    subtitle : str or None
        One or two sentences under the title saying what the figure shows.
    """

    def __init__(self, title, notebook, path, subtitle=None):
        self.title = title
        self.notebook = notebook
        self.path = path
        self.subtitle = subtitle
        self._blocks = []

    # -- block writers ---------------------------------------------------------
    def section(self, title, level=2):
        """Start a section. ``level`` is the Markdown heading depth (2 = ``##``)."""
        self._blocks.append(f'{"#" * level} {title}')
        return self

    def para(self, text):
        """A paragraph of prose."""
        self._blocks.append(str(text).strip())
        return self

    def note(self, text):
        """A block-quoted caveat -- the things a reader must not write up without."""
        body = '\n'.join(f'> {line}' for line in str(text).strip().splitlines())
        self._blocks.append(body)
        return self

    def bullets(self, items):
        """A bullet list from an iterable of strings."""
        items = [str(i) for i in items if i is not None]
        if items:
            self._blocks.append('\n'.join(f'- {i}' for i in items))
        return self

    def kv(self, mapping, nd=3):
        """A bullet list of ``**key** — value`` pairs, for configuration and provenance."""
        rows = [f'- **{k}** — {v if isinstance(v, str) else fmt_num(v, nd)}'
                for k, v in mapping.items() if v is not None]
        if rows:
            self._blocks.append('\n'.join(rows))
        return self

    def table(self, df, index=False, nd=3, index_label=''):
        """A DataFrame as a Markdown table.

        Floats go through :func:`fmt_num`; everything else is stringified as-is, so a column of
        pre-formatted strings (p-values, CIs, significance stars) survives untouched.
        """
        df = pd.DataFrame(df).copy()
        if index:
            df = df.reset_index()
            if index_label:
                df = df.rename(columns={df.columns[0]: index_label})

        def _cell(v):
            if isinstance(v, (float, np.floating)):
                return _escape(fmt_num(v, nd))
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                return 'n/a'
            return _escape(str(v))

        header = [_escape(str(c)) for c in df.columns]
        rows = [[_cell(v) for v in row] for row in df.itertuples(index=False, name=None)]
        widths = [max(len(header[i]), *(len(r[i]) for r in rows)) if rows else len(header[i])
                  for i in range(len(header))]
        out = ['| ' + ' | '.join(h.ljust(w) for h, w in zip(header, widths)) + ' |',
               '| ' + ' | '.join('-' * w for w in widths) + ' |']
        out += ['| ' + ' | '.join(c.ljust(w) for c, w in zip(r, widths)) + ' |' for r in rows]
        self._blocks.append('\n'.join(out))
        return self

    # -- output ----------------------------------------------------------------
    def render(self):
        """The full document as a string."""
        stamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        head = [f'# {self.title}']
        if self.subtitle:
            head.append(self.subtitle.strip())
        head.append(f'*Generated {stamp} by `{self.notebook}`. This file is overwritten on every '
                    'run of that notebook — edit the notebook, not this document.*')
        return '\n\n'.join(head + self._blocks).rstrip() + '\n'

    def write(self, verbose=True):
        """Write (overwriting) and return the path."""
        text = self.render()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'w') as f:
            f.write(text)
        if verbose:
            print(f'Results summary ({len(text.splitlines())} lines, '
                  f'{len(text) / 1024:.1f} kB) -> {self.path}')
        return self.path


# ==============================================================================
# Age-grid significance windows
# ==============================================================================

def sig_spans(mask, x):
    """Contiguous ``True`` runs of ``mask`` as ``(start, end)`` pairs of ``x``.

    fig3's panel E and fig4's coupling traces both mark significance per age point and both need
    to report it as age *windows* rather than as a count of grid points, so the conversion lives
    here once. Endpoints are grid values, not the interpolated zero crossings ``plot_r_age``
    prints on the panel; they agree to within one grid step.
    """
    mask = np.asarray(mask, dtype=bool)
    x = np.asarray(x, dtype=float)
    spans, start = [], None
    for i, hit in enumerate(mask):
        if hit and start is None:
            start = i
        elif not hit and start is not None:
            spans.append((x[start], x[i - 1]))
            start = None
    if start is not None:
        spans.append((x[start], x[-1]))
    return spans


def fmt_spans(mask, x, nd=1, unit='yr', empty='none'):
    """:func:`sig_spans` as a readable string, e.g. ``'8.9–12.4, 15.0–21.9 yr'``."""
    spans = sig_spans(mask, x)
    if not spans:
        return empty
    body = ', '.join(f'{a:.{nd}f}–{b:.{nd}f}' for a, b in spans)
    return f'{body} {unit}' if unit else body
