from setuptools import setup, find_packages

setup(
    name="your_toolbox",
    use_scm_version={
        "write_to": "your_toolbox/_version.py"
    },
    setup_requires=["setuptools_scm"],
    author="Your Name",
    author_email="your.email@example.com",
    description="Utility functions for neuroimaging and Alzheimer's research",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/your_toolbox",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "nibabel",
        "pandas",
        "scipy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
