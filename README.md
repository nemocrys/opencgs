# opencgs

Validated open source crystal growth simulation.

The project is developed and maintained by the [**Model experiments group**](https://www.ikz-berlin.de/en/research/materials-science/section-fundamental-description#c486) at the Leibniz Institute for Crystal Growth (IKZ).

### Referencing
If you use this code in your research, please cite our article (available with open access):

> A. Enders-Seidlitz, J. Pal, and K. Dadzis, Development and validation of a thermal simulation for the Czochralski crystal growth process using model experiments *Journal of Crystal Growth*, In Press. https://doi.org/10.1088/1757-899X/1223/1/012003.
> 
## Overview

This package provides an interface to set up crystal growth simulations using open source software. Currently, the focus is set on Czochralski growth (and some of the functionality is tailored to it). Nevertheless, opencgs may also be useful for other applications. opencgs is validated within the [NEMOCRYS project](https://www.researchgate.net/project/NEMOCRYS-Next-Generation-Multiphysical-Models-for-Crystal-Growth-Processes) using model experiments.

Currently, the focus of opencgs is on 2D axisymmetric thermal simulation (including inductive heating) using the software Elmer. Further extension, e.g., for 3D domains or thermal stresses computation is planned.

A paper showing the capability of opencgs for Czochralski growth simulation has been submitted.

## Usage

opencgs simulations are setup in Python, for an example refer to [test-cz-induction](https://github.com/nemocrys/test-cz-induction).

## Installation

opencgs itself is provided in form of a Python package and can be installed with pip:

```
git clone https://github.com/nemocrys/opencgs
cd opencgs
pip install -e .
```

However, this only provides the basic functionality and does not install the required solvers. For a full description of the configuration for refer to *Docker/Dockerfile* or directly use our docker image (next section).

## Docker image

A [Docker](https://docs.docker.com/get-docker/) image with the complete opencgs setup is provided on [dockerhub](https://hub.docker.com/r/nemocrys/opencgs). To run an interactive session mapping your working directory into the container on Linux use:

```
docker run -it --rm -v $PWD:/home/workdir -e LOCAL_UID=$(id -u $USER) -e LOCAL_GID=$(id -g $USER) nemocrys/opencgs:latest bash

```

On Windows use:

```
docker run -it --rm -v ${PWD}:/home/workdir nemocrys/opencgs:latest bash
```

## Documentation

An extensive documentation is still under construction.

## Support

In case of questions just open an issue or contact Arved Enders-Seidlitz.

## Acknowledgements

[This project](https://www.researchgate.net/project/NEMOCRYS-Next-Generation-Multiphysical-Models-for-Crystal-Growth-Processes) has received funding from the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation programme (grant agreement No 851768).

<img src="https://raw.githubusercontent.com/nemocrys/opencgs/master/EU-ERC.png">

## Contribution

Any help to improve this package is very welcome!
