from setuptools import setup, find_packages

with open("README.md", "r") as f:
    description = f.read()

setup(
    name='ligrad',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        "numpy>=1.17",
        "scipy>=1.5",
        "astropy>=4.0",
        "pylightcurve>=2.0"
    ],
    long_description=description,
    long_description_content_type="text/markdown",
)