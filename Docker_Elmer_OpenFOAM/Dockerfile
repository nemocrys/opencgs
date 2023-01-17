# USAGE INSTSRUCTIONS
# 
# Build image (run this in the main directory (where .git folder is)):
# docker build -t nemocrys/opencgs_elmer_of:vx.x.x -f ./Docker_Elmer_OpenFOAM/Dockerfile .
# or
# docker build --no-cache -t nemocrys/opencgs_elmer_of:vX.X.X -f ./Docker_Elmer_OpenFOAM/Dockerfile .
#
# Run image:
# docker run -it --rm nemocrys/opencgs_elmer_of:vX.X.X bash
#
# Run image and map current directory to home/workdir on Linux:
# docker run -it --rm -v $PWD:/home/workdir -e LOCAL_UID=$(id -u $USER) -e LOCAL_GID=$(id -g $USER) nemocrys/opencgs_elmer_of:vX.X.X bash
# and on Windows:
# docker run -it --rm -v ${PWD}:/home/workdir nemocrys/opencgs_elmer_of:vX.X.X bash

FROM ubuntu:20.04

# Labels
LABEL maintainer="Arved Wintzer <arved.wintzer@ikz-berlin.de>"
LABEL description="opencgs computation environment"

ARG ELMER_COMMIT=d3b69300319f2aa696ca9cc08a9bb465008425fd
ARG GMSH_VERSION=4.10.5

ARG BUILD_ESSENTIAL_VERSION=12.8*
ARG CMAKE_VERSION=3.16.*
ARG GIT_VERSION=1:2.25.*
ARG LIBBLAS_DEV_VERSION=3.9.*
ARG LIBLAPPACK_DEV_VERSION=3.9.*
ARG LIBMUMPS_DEV_VERSION=5.2.*
ARG LIBPARMETIS_DEV_VERSION=4.0.*
ARG MPICH_VERSION=3.3.*

ARG PYTHON_VERSION=3.8.*
ARG PYTHON_PIP_VERSION=20.0.*
ARG MESHIO_VERSION=5.3.*
ARG NUMPY_VERSION=1.23.*
ARG MATPLOTLIB_VERSION=3.5.*
ARG SCIPY_VERSION=1.9.*
ARG PANDAS_VERSION=1.4.*
ARG PYYAML_VERSION=6.0.*
ARG PYELMER_VERSION=1.0.*
ARG OBJECTGMSH_VERSION=0.9

# deactivate interations -> no question for geolocation
ENV DEBIAN_FRONTEND=noninteractive 

# Set the working directory to /home
WORKDIR /home

# Add packages to compile Elmer
# check available versions with 'apt-cache madison <package>'
RUN apt-get update && \
    apt-get -y install \
    build-essential=${BUILD_ESSENTIAL_VERSION} \
    cmake=${CMAKE_VERSION} \
    git=${GIT_VERSION} \
    libblas-dev=${LIBBLAS_DEV_VERSION}\
	liblapack-dev=${LIBLAPPACK_DEV_VERSION}\
	libmumps-dev=${LIBMUMPS_DEV_VERSION}\
	libparmetis-dev=${LIBPARMETIS_DEV_VERSION}\
	mpich=${MPICH_VERSION}

# Clone the Elmer source code and compile

RUN mkdir elmer && \
    cd elmer && \
    git clone https://github.com/ElmerCSC/elmerfem.git && \
    cd elmerfem && \
    git checkout ${ELMER_COMMIT} && \
    cd .. && \
    mkdir build && \
    mkdir install && \
  	cd build && \
    cmake ../elmerfem \
        -DCMAKE_INSTALL_PREFIX=../install \
        -DCMAKE_C_COMPILER=/usr/bin/gcc \
        -DCMAKE_Fortran_COMPILER=/usr/bin/gfortran \
        -DWITH_LUA=TRUE\
        -DWITH_MPI:BOOL=TRUE \
        -DWITH_OpenMP=True\
        -DWITH_Mumps:BOOL=TRUE \
        -DWITH_Hypre:BOOL=FALSE \
        -DWITH_Trilinos:BOOL=FALSE \
        -DWITH_ELMERGUI:BOOL=FALSE \
        -DWITH_ElmerIce:BOOL=FALSE && \
	make install

# Add Elmer to path
ENV PATH=$PATH:/home/elmer/install/bin

# Install python and required packages
RUN apt-get update && \
    apt-get -y install \
    python3=${PYTHON_VERSION} \
    python3-pip=${PYTHON_PIP_VERSION} &&\
    pip3 install \
        meshio==${MESHIO_VERSION} \
        numpy==${NUMPY_VERSION} \
        matplotlib==${MATPLOTLIB_VERSION} \
        scipy==${SCIPY_VERSION} \
        pandas==${PANDAS_VERSION} \
        pyyaml==${PYYAML_VERSION} \
        pyelmer==${PYELMER_VERSION} \
        objectgmsh==${OBJECTGMSH_VERSION}

# Install gmsh & dependencies
# istructions found here: https://fenicsproject.discourse.group/t/problem-with-installation-of-latest-gmsh-4-6-0-via-pip/4078/2
RUN apt-get update && \
    apt-get install -y \
        wget \
        libglu1 \
        libxrender1  \
        libxcursor-dev \
        libxft-dev \
        libxinerama-dev && \
    wget -nc  http://gmsh.info/bin/Linux/gmsh-${GMSH_VERSION}-Linux64-sdk.tgz && \
    tar -xf gmsh-${GMSH_VERSION}-Linux64-sdk.tgz && \
    rm gmsh-${GMSH_VERSION}-Linux64-sdk.tgz
ENV PYTHONPATH=/home/gmsh-${GMSH_VERSION}-Linux64-sdk/lib:$PYTHONPATH
ENV PATH=/home/gmsh-${GMSH_VERSION}-Linux64-sdk/bin:$PATH

# Install opencgs from this repository
COPY  . /home/opencgs
RUN cd /home/opencgs &&\
    pip3 install .

# Install additional utilities
RUN apt-get update && \
    apt-get install -y mc && \
    mkdir scripts && \
    cd scripts && \
    echo "source /usr/lib/mc/mc-wrapper.sh" >> mc && \
    chmod +x mc
# ENV PATH=/home/scripts:$PATH

# setup openFOAM
RUN apt-get update && \
        apt-get install -y \
        vim \
        ssh \
        sudo \
        wget \
        git \
        software-properties-common ;\
        rm -rf /var/lib/apt/lists/*


# Dependencies for Ubuntu 20.04 LTS
RUN apt-get update && \
        apt-get install -y \
        apt-utils \
        git-core \
        build-essential \
        binutils-dev \
        cmake \
        flex \
        zlib1g-dev \
        libncurses5-dev \
        curl \
        bison \
        libxt-dev \
        rpm \
        mercurial \
        graphviz \
        python \
        python-dev  \
        gcc-7 \
        g++-7 \
        mpich

RUN pip3 install \
    pyFoam==2021.*

# get source code
RUN wget -q -O - https://dl.openfoam.com/add-debian-repo.sh | bash && \
    apt-get install -y ca-certificates && \
    apt-get update
        
SHELL ["/bin/bash", "-c"]

RUN apt-get update && \ 
    apt-get install -y openfoam2112-default && \
    source /usr/lib/openfoam/openfoam2112/etc/bashrc 

# change to bash shell, else source does not work
SHELL ["/bin/bash", "-c"]


# setup workingdirectory for mapping

RUN mkdir /home/workdir
WORKDIR /home/workdir

# modify user id and group
# see https://techflare.blog/permission-problems-in-bind-mount-in-docker-volume/
RUN apt-get update && \
    apt-get -y install gosu 
ENTRYPOINT [ "/home/opencgs/Docker_Elmer_OpenFOAM/entrypoint.sh" ]
