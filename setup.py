import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="iwc2tb",
    version="0.0.1",
    author="Inderpreet Kaur",
    description="Scripts to read and analyse ARTS simulations (Radar2Tb) ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SEE-MOF/IWC2TB",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GNU Affero",
        "Operating System :: OS Independent",
    ],
    install_requires=[ "numpy", "scipy", "matplotlib"],
    python_requires=">=3.6",
    include_package_data=True,
    package_data={
        '': ['*.ini']
    }
)

