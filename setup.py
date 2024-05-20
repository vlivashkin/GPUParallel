from setuptools import setup

from gpuparallel import __version__

setup(
    name='gpuparallel',
    version=__version__,
    description='Joblib-like interface for parallel GPU computations (e.g. data preprocessing)',
    url='',
    author='Vladimir Ivashkin',
    author_email='illusionww@gmail.com',
    license='MIT',
    packages=['gpuparallel'],
    install_requires=[
        'tqdm',
    ],
)
