# https://packaging.python.org/tutorials/packaging-projects/

import setuptools
import versioneer

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="opencgs",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Arved Enders-Seidlitz",
    author_email="arved.enders-seidlitz@ikz-berlin.de",
    description="Open source crystal growth simulations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nemocrys/opencgs",
    packages=["opencgs", "opencgs.geo", "opencgs.test"],
    include_package_data=True,
    package_data={"": ["data/*.yml"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        #'gmsh',
        "meshio",
        "pandas",
        "pyyaml",
        "pyelmer",
    ],
    python_requires=">=3.7",
)
