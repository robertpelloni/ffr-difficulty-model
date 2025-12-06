from setuptools import find_packages, setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='stepmania-difficulty-predictor',
    packages=find_packages(),
    version='0.2.0',
    description='A library to predict the difficulty of StepMania (.sm) files.',
    author='Wilson Cheung',
    license='MIT',
    install_requires=required
)
