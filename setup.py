from setuptools import setup, find_packages

setup(
    name="dual-computational-systems",
    py_modules=["dual_computational_systems"],
    version="0.0.1",
    packages=find_packages(exclude=("datasets",)),
    include_package_data=True,
)
