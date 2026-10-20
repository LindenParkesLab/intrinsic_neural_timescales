from setuptools import find_packages, setup

# The snaplab_tools release this code was run with (the first with tau_expfit_free and acf_ratio_21).
SNAPLAB_TOOLS_TAG = 'v0.1.1'

setup(
    name="intrinsic_neural_timescales",
    # The importable package is ``src`` (``from src.timescales import ...``); tests are not installed.
    packages=find_packages(include=["src", "src.*"]),
    package_data={
        'src': ['configs/datasets/*.yaml'],
    },
    python_requires=">=3.10",
    license="BSD-3-Clause",
    install_requires=[
        f"snaplab_tools[surface] @ git+https://github.com/LindenParkesLab/snaplab_tools.git@{SNAPLAB_TOOLS_TAG}",
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "statsmodels",
        "patsy",
        "nibabel",
        "nilearn",
        "matplotlib",
        "seaborn",
        "adjustText",
        "pyyaml",
        "tqdm",
    ],
    extras_require={
        "test": ["pytest"],
    },
)
