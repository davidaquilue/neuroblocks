from setuptools import setup, find_packages

setup(
    name="neuroblocks",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    author="David Aquilué-Llorens",
    author_email="david.aquilue@upf.edu",
    description="Utility functions for neuroimaging and Alzheimer's research",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/davidaquilue/neuroblocks",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)