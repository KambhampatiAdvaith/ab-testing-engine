from setuptools import setup, find_packages

setup(
    name='ab-testing-engine',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'matplotlib>=3.4.0',
    ],
    python_requires='>=3.8',
    author='AB Testing Engine',
    description='Advanced A/B Testing & User Segmentation Engine',
)
