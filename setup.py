"""
Setup script for SQL GRPO Training package.
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = [
        line.strip() 
        for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="sql-grpo-training",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="GRPO training for text-to-SQL generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sql-grpo-training",
    packages=find_packages(exclude=["tests", "tests.*", "scripts"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.11,<3.13",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.0.0",
        ],
        "flash-attn": [
            "flash-attn>=2.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sql-grpo-train=scripts.train:main",
        ],
    },
)
