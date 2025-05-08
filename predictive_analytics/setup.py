from setuptools import setup, find_packages

setup(
    name="predictive_analytics",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "flask>=2.0.0",
        "flask-cors>=3.0.0",
        "prometheus-client>=0.9.0",
        "kafka-python>=2.0.0",
        "prophet>=1.1.0",
        "scikit-learn>=0.24.0",
        "pulp>=2.4",
        "redis>=3.5.0",
        "pandas>=1.2.0",
        "numpy>=1.19.0",
    ],
    python_requires=">=3.8",
) 