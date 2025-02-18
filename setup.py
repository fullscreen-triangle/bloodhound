from setuptools import setup, find_packages
from pathlib import Path

# Read requirements
def read_requirements(filename: str) -> list:
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read the README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="bloodhound",
    version="1.0.0",
    description="A comprehensive platform for scientific experiment management and analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Kundai Sachikonye",
    author_email="kundai.f.sachikonye@gmail.com",
    url="https://github.com/fullscreen-triangle/bloodhound",
    packages=find_packages(exclude=["tests*", "docs*"]),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": [
            "black",
            "isort",
            "flake8",
            "mypy",
            "pytest",
            "pytest-asyncio",
            "pytest-cov",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
        ],
    },
    entry_points={
        "console_scripts": [
            "bloodhound=backend.app_manager:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
    ],
    project_urls={
        "Documentation": "https://bloodhound.readthedocs.io/",
        "Source": "https://github.com/fullscreen-triangle/bloodhound",
        "Tracker": "https://github.com/fullscreen-triangle/bloodhound/issues",
    },
)
