from setuptools import find_packages, setup

setup(
    name="src",
    packages=find_packages(),
    package_data={
        'src': ['configs/datasets/*.yaml'],
    },
)
