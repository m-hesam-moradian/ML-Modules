from setuptools import setup, find_packages

setup(
    name="ml-project",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A machine learning project for fraud detection.",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "scikit-learn",
        "openpyxl"
    ],
    python_requires='>=3.6',
)