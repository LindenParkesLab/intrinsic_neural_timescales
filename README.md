# Intrinsic neural timescales

## Background

Intrinsic neural timescales (INTs) are estimated from the autocorrelation of resting-state and task fMRI,
parcel by parcel (the area under the autocorrelation function up to its first zero crossing), in four
datasets: HCP-YA, HCP-D, WAND and Penn LEAD. The analyses ask how task engagement changes INTs and their
alignment with the sensorimotor–association (S-A) axis (Figures 1–2), how those changes develop with age
in HCP-D (Figure 3), and which neurobiological maps track that development (Figure 4).

## Reference paper

Parkes L., Sydnor V.J., Howell A., France J.M., Westbrook A., & Larsen B. *Task-evoked flexibility of intrinsic neural timescales is hierarchically organized, behaviorally relevant, and refined across two developmental stages* [bioRxiv](https://www.biorxiv.org/).

## Repository layout

```text
src/                     the analysis package, imported as `src`
  config.py              per-dataset processing parameters (tasks, sessions, subject lists, denoising)
  configs/datasets/      one YAML per dataset: data location, file naming, TR, behaviour/events handlers
  data.py                scan registry, timeseries loading and denoising, behaviour and task events
  timescales.py          task regression, ACFs and INTs, matched rest controls, on-disk cache
  notebook.py            loading and comparator helpers shared by the figure notebooks
  gams.py                age GAMs, derivatives and bootstraps (Figures 3–4)
  nulls.py               BrainSMASH spatial-autocorrelation nulls
  supporting_data.py     S-A axis, parcel geometry, gene expression, myelin
  plotting.py            figure panels
scripts/                 the notebooks that produce every result (see below)
tests/                   pytest suite
```

## Installation

Python ≥ 3.10 (developed on 3.11). From the repository root:

```bash
conda create -n int python=3.11
conda activate int
pip install -e ".[test]"
```

This also installs [snaplab_tools](https://github.com/LindenParkesLab/snaplab_tools) (ACF and INT
estimators, GAMs, spatial nulls, plotting), pinned in `setup.py`.

## Data

No data are distributed with this repository. You need:

- **Parcellated timeseries** (Schaefer 400, 7 networks) for each dataset: HCP-YA (S1200, 3T and 7T),
  HCP-D, WAND and Penn LEAD. HCP-YA and HCP-D are read as fsLR `.ptseries.nii`; WAND and Penn LEAD as XCP-D `.tsv`. Point `paths.base` in each
  `src/configs/datasets/<dataset>.yaml` at your copy; the YAML also gives the expected directory and file names.
- **Behaviour, demographics and task events** from each dataset's release, read by the handlers in
  `src/data.py`, and the subject list named by `subject_ids_file` in `src/config.py` in each dataset's base
  directory.
- **Supporting maps** in `data/supporting_data/`: the S-A axis (`schaefer400-7_sa-axis.npy`), parcel
  centroids, cytoarchitecture and microstructure maps, Allen Human Brain Atlas expression
  (`expression_schaefer400-7.csv`) with brain-gene lists (`gao_brain_genes.txt`), cell-type maps, and the
  HCP-YA 7T movie timing files (`movie_data/`). These are derived from public sources, some of which are available
  from the authors on request.

Outputs are written under the repository root: `scan_registry_*.csv`, the per-scan cache in
`data/timescales_outputs/`, and GAM caches, CSVs and surrogate nulls in `results/<dataset>/`. Figure
panels go to the directory in `$INT_FIGURE_DIR`.

## Running the analyses

Notebooks are in `scripts/`. They were run in this order:

| step | notebook | produces |
| --- | --- | --- |
| 1 | `timescales_subject_lists.ipynb` | low-motion subject lists (mean FD ≤ 0.2 mm) |
| 2 | `timescales_process.ipynb` | scan registries and the INT cache; run once per configuration in its header table |
| 3 | `timescales_analysis_fig1_super.ipynb` | Figure 1: task vs rest INTs and their S-A coupling, four datasets |
| 3 | `timescales_analysis_sfig1.ipynb` | HCP-YA per scan, and against matched rest controls |
| 3 | `timescales_analysis_fig2.ipynb` | Figure 2: HCP-YA INT deflections, S-A structure and behaviour |
| 3 | `timescales_analysis_fig3.ipynb` | Figure 3: HCP-D development (fits every GAM and writes the cache fig4 reads) |
| 3 | `timescales_analysis_fig4.ipynb` | Figure 4: gene-expression and myelin coupling to the developmental rate |
| 3 | `timescales_analysis_sfig_rest_control.ipynb` | deflections against matched rest controls, HCP-YA and HCP-D |
| 3 | `timescales_analysis_sfig_task_regression.ipynb` | single-subject worked example of task regression |
| 3 | `timescales_schematic.ipynb` | schematic figure (synthetic data) |
| 4 | `timescales_analysis_data_summary.ipynb` | sample sizes, demographics and head motion |

## Tests

```bash
pytest
```

Unit tests run anywhere. Integration tests read the real data and are skipped when a dataset's
`paths.base` does not exist.

## Citation

If you use this code, please cite the paper above.

## License

BSD 3-Clause — see [LICENSE](LICENSE).
