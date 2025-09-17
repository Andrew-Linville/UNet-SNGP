from setuptools import setup, find_packages

setup(
    name="unet-concrete-rewrite",
    version="0.1.0",
    description="U-Net training pipeline for concrete segmentation",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.11",
)
