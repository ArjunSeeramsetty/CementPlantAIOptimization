#!/usr/bin/env python3
"""
Setup script for JK Cement Digital Twin Platform
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "JK Cement Digital Twin Platform"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="cement-ai-platform",
    version="1.0.0",
    description="AI-powered Digital Twin Platform for JK Cement",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="JK Cement AI Team",
    author_email="ai-team@jkcement.com",
    url="https://github.com/ArjunSeeramsetty/CementPlantAIOptimization",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Manufacturing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "bandit>=1.7.0",
            "safety>=2.0.0",
        ],
        "streamlit": [
            "streamlit>=1.28.0",
            "plotly>=5.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cement-ai-platform=cement_ai_platform.agents.jk_cement_platform:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)