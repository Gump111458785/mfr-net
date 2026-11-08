from setuptools import setup, find_packages

setup(
    name="mfr_net",
    version="1.0.0",
    description="Multi-Scale Feature Rectification and Uncertainty Calibration for Visual Classification",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.64.0",
        "tabulate>=0.8.9",
        "scipy>=1.7.0",
    ],
)
