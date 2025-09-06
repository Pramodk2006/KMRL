from setuptools import setup, find_packages

setup(
    name="kmrl-intellifleet",
    version="1.0.0",
    description="AI-Driven Train Induction Planning System for KMRL",
    author="SIH Team",
    packages=find_packages(),
    install_requires=[
        "ortools>=9.7.2996",
        "pandas>=2.0.0",
        "streamlit>=1.28.0",
        "lightgbm>=4.0.0",
        "pymoo>=0.6.0",
        "plotly>=5.15.0",
    ],
    python_requires=">=3.8",
)