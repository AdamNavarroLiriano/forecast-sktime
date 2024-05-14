from setuptools import find_packages, setup

PROJECT = "timeseries_forecast"
VERSION = "0.0.1"


setup(
    name=PROJECT,
    version=VERSION,
    packages=find_packages(where="src", exclude=["tests"]),
    description="Library for easily fitting and predicting time series",
    long_description_content_type="text/markdown",
    author="Adam Navarro Liriano",
    author_email="a.navarro0608@gmail.com",
    zip_safe=False,
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">3.6",
)
