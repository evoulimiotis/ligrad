from setuptools import setup, find_packages

with open("README.md", "r") as f:
    readme_content = f.read()

setup(
    name='ligrad',
    version='1.1.5',
    packages=find_packages(),
    install_requires=[
        "numpy>=1.17",
        "scipy>=1.5",
        "astropy>=4.0",
        "pylightcurve>=2.0"
    ],
    
    author="Eleftherios Voulimiotis",
    author_email="evoulimi@physics.auth.gr",
    
    description="A Python package for generating transit light-curves of gravity-darkened, oblate stars",
    long_description=readme_content,
    long_description_content_type="text/markdown",
)