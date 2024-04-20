# USAGE INSTSRUCTIONS
# 
# Build image (run this in the main directory (where .git folder is)):
# docker build -t nemocrys/opencgs:vX.X.X -f ./Docker/Dockerfile .
# or
# docker build --no-cache -t nemocrys/opencgs:vX.X.X -f ./Docker/Dockerfile .
#
# Run image:
# docker run -it --rm nemocrys/opencgs:vX.X.X bash
#
# Run image and map current directory to home/workdir on Linux:
# docker run -it --rm -v $PWD:/home/workdir -e LOCAL_UID=$(id -u $USER) -e LOCAL_GID=$(id -g $USER) nemocrys/opencgs:vX.X.X bash
# and on Windows:
# docker run -it --rm -v ${PWD}:/home/workdir nemocrys/opencgs:vX.X.X bash

FROM ubuntu:20.04

# Labels
LABEL maintainer="Arved Wintzer <arved.wintzer@ikz-berlin.de>"
LABEL description="opencgs computation environment"

ARG ELMER_COMMIT=c37d486543bf09fbc6adecb7fcd46f90d945fbd5
ARG ELMER_TO_FOAM_COMMIT=61b9089ea6ccc161023ee55da437254c366cabf0
ARG NEMOFOAM_UTILS_COMMIT=fc29cf9f3071a28895be5156f1d917b05125ea65
ARG GMSH_VERSION=4.11.1

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
ARG PYELMER_VERSION=1.1.*
ARG OBJECTGMSH_VERSION=0.*
ARG NEMOBLOCK_VERSION=1.0.*
ARG PYVISTA_VERSION=0.39

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
        objectgmsh==${OBJECTGMSH_VERSION} \
        nemoblock==${NEMOBLOCK_VERSION} \
        pyvista==${PYVISTA_VERSION}

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
        
# change to bash shell, else source does not work
SHELL ["/bin/bash", "-c"]

RUN apt-get update && \ 
    apt-get install -y openfoam2212-default && \
    source /usr/lib/openfoam/openfoam2212/etc/bashrc 

RUN cd /home && \
    git clone https://github.com/nemocrys/elmerToFoamBC && \
    cd elmerToFoamBC && \
    git checkout ${ELMER_TO_FOAM_COMMIT} && \
    source /usr/lib/openfoam/openfoam2212/etc/bashrc && \
    wmake elmerToFoamBC

# install nemoFoam-utils
# submodule installation of elmerToFaomBC doesn't worke because of foam_user_appbin
RUN cd /home &&\
    source /usr/lib/openfoam/openfoam2212/etc/bashrc  &&\
    git clone https://github.com/nemocrys/nemoFoam-utils && \
    cd nemoFoam-utils && \
    git checkout ${NEMOFOAM_UTILS_COMMIT} && \
    cd src && \
    source /usr/lib/openfoam/openfoam2212/etc/bashrc && \
    chmod +x Allwmake && \
    ./Allwmake

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
    chmod +x mc && \
    apt-get install -y tree


# setup workingdirectory for mapping
RUN mkdir /home/workdir
WORKDIR /home/workdir

# quality of life improvements (for root user)
RUN cd ~ &&\
        echo 'bind '"'"'"\e[A": history-search-backward'"'" >> $HOME/.bashrc && \
        echo 'bind '"'"'"\e[B": history-search-forward'"'" >> $HOME/.bashrc && \
        echo 'LS_COLORS=$LS_COLORS:'"'"'di=32:'"'"' ; export LS_COLORS' >> $HOME/.bashrc && \
        echo "alias of2212='source /usr/lib/openfoam/openfoam2212/etc/bashrc '" >> $HOME/.bashrc && \
        echo "alias mc='source /usr/lib/mc/mc-wrapper.sh'" >> $HOME/.bashrc

# OpenFOAM enviroment variables
RUN export WM_USER_PROJECT_DIR=/home/workdir/user-foam && \
    export FOAM_RUN=/home/workdir/run

# modify user id and group
# see https://techflare.blog/permission-problems-in-bind-mount-in-docker-volume/
RUN apt-get update && \
    apt-get -y install gosu 
ENTRYPOINT [ "/home/opencgs/Docker_Elmer_OpenFOAM/entrypoint.sh" ]
