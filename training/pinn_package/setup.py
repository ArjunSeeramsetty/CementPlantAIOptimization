from setuptools import setup, find_packages


setup(
    name="cement-pinn-trainer",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "scikit-learn>=1.0",
        "numpy>=1.21",
    ],
    description="Minimal trainer package for Vertex AI POC",
    author="Cement AI Platform",
)


