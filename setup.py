from setuptools import setup

# from gpuparallel import __version__
__version__ = "1.0"

setup(
    name="gpuparallel",
    version=__version__,
    description="Joblib-like interface for parallel GPU computations (e.g. data preprocessing)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://gpuparallel.readthedocs.io/en/latest/?badge=latest",
    author="Vladimir Ivashkin",
    author_email="illusionww@gmail.com",
    license="MIT",
    packages=["gpuparallel"],
    install_requires=[
        "tqdm",
        "billiard>=3.6.3.0,<4.0",
    ],
)
