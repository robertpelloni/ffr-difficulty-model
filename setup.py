from setuptools import find_packages, setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='stepmania-difficulty-predictor',
    packages=find_packages(),
    version='0.3.0',
    description='A library to predict the difficulty of StepMania (.sm) files.',
    author='Wilson Cheung',
    license='MIT',
    install_requires=required,
    package_data={'stepmania_difficulty_predictor': ['model/random_forest_regressor.p']},
    include_package_data=True,
)
