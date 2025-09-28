from setuptools import setup, find_packages

setup(
    name="test-time-gym",
    version="0.1.0",
    description="A safe simulation environment for LLM agent test-time learning",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.29.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "pydantic>=2.0.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "jsonlines>=3.1.0",
        "tqdm>=4.62.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.991",
        ]
    },
    python_requires=">=3.8",
    author="Test-Time Gym Team",
    author_email="test@example.com",
    url="https://github.com/your-org/test-time-gym",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)