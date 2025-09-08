from setuptools import setup, find_packages


setup(
    name="cement-ai-platform",
    version="0.1.0",
    description="Generative AI-driven optimization platform for cement plants",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.24",
        "scipy>=1.11",
        "pydantic>=2.5",
        "python-dotenv>=1.0",
        "google-cloud-bigquery>=3.14",
        "google-cloud-storage>=2.14",
        "google-cloud-aiplatform>=1.66",
        "google-auth>=2.30",
    ],
)



