from setuptools import setup

setup(
    name='gpuparallel',
    version='0.0.1',
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
