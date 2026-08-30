from setuptools import setup, find_packages

setup(
    name="ak-visual-intelligence",
    version="0.1.0",
    description="AK Consultants Visual Intelligence Platform",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.9",
)
