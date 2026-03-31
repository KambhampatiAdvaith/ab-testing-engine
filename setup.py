from setuptools import setup, find_packages

setup(
    name='ab-testing-engine',
    version='0.2.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'matplotlib>=3.4.0',
        'fastapi>=0.104.0',
        'uvicorn>=0.24.0',
        'sqlalchemy>=2.0.0',
        'psycopg2-binary>=2.9.0',
        'pydantic>=2.0.0',
    ],
    python_requires='>=3.8',
    author='AB Testing Engine',
    description='Advanced A/B Testing & User Segmentation Engine with FastAPI backend',
)
