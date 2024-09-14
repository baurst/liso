from setuptools import find_packages, setup

setup(
    name="nuscenes-devkit",
    packages=find_packages(include=("nuscenes",), exclude=("nuimages", "tutorials")),
    zip_safe=False,
)
