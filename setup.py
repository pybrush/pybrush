from setuptools import setup, find_packages

setup(
    name="pybrush",
    version="0.1.0",
    author="Snehil",
    author_email="pybrush@gmail.com",
    description="A Python library for Machine Unlearning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/pybrush/pybrush",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "torch",
        "tensorflow",
        "scikit-learn"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)