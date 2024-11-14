from setuptools import find_packages, setup

setup(
    name="mtobjects",
    packages=find_packages(),
    entry_points={"console_scripts": ["mto = cli:main"]},
    version="0.0.1",
)
