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
LABEL maintainer="Arved Enders-Seidlitz <arved.enders-seidlitz@ikz-berlin.de>"
LABEL description="opencgs computation environment"

# deactivate interations -> no question for geolocation
ENV DEBIAN_FRONTEND=noninteractive 

# Set the working directory to /home
WORKDIR /home

# Add packages to compile Elmer
# check available versions with 'apt-cache madison <package>'
RUN apt-get update && \
    apt-get -y install \
    build-essential=12.8* \
    cmake=3.16.* \
    git=1:2.25.* \
    libblas-dev=3.9.*\
	liblapack-dev=3.9.*\
	libmumps-dev=5.2.*\
	libparmetis-dev=4.0.*
	# mpich=3.3.*  # install this package if you want to use mpi

# Clone the Elmer source code and compile
RUN mkdir elmer && \
    cd elmer && \
    git clone -b release-9.0 https://github.com/ElmerCSC/elmerfem.git && \
    mkdir build && \
    mkdir install && \
  	cd build && \
    cmake ../elmerfem \
        -DCMAKE_INSTALL_PREFIX=../install \
        -DCMAKE_C_COMPILER=/usr/bin/gcc \
        -DCMAKE_Fortran_COMPILER=/usr/bin/gfortran \
        -DWITH_MPI:BOOL=FALSE \
        -DWITH_Mumps:BOOL=TRUE \
        -DWITH_Hypre:BOOL=FALSE \
        -DWITH_Trilinos:BOOL=FALSE \
        -DWITH_ELMERGUI:BOOL=FALSE \
        -DWITH_ElmerIce:BOOL=FALSE && \
	make install

# Add Elmer to path
ENV PATH=$PATH:/home/elmer/install/bin

# Install python and required packages
RUN apt-get -y install \
    python3=3.8.* \
    python3-pip=20.0.* &&\
    pip3 install \
        meshio==4.4.*\
        numpy==1.20.*\
        matplotlib==3.4.*\
        scipy==1.6.*\
        pandas==1.2.*\
        pyyaml==5.4.*\
        pyelmer==0.3.*\
        objectgmsh==0.*

# Install gmsh & dependencies
# istructions found here: https://fenicsproject.discourse.group/t/problem-with-installation-of-latest-gmsh-4-6-0-via-pip/4078/2
ARG GMSH_VERSION=4.8.4
RUN apt-get install -y \
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
    pip3 install -e .

# Install additional utilities
RUN apt-get install -y mc && \
    mkdir scripts && \
    cd scripts && \
    echo "source /usr/lib/mc/mc-wrapper.sh" >> mc && \
    chmod +x mc

ENV PATH=/home/scripts:$PATH
RUN mkdir /home/workdir

# # create user
# RUN adduser --disabled-password --gecos '' user
# USER user

WORKDIR /home/workdir

# modify user id and group
# see https://techflare.blog/permission-problems-in-bind-mount-in-docker-volume/
RUN apt-get -y install gosu 
ENTRYPOINT [ "/home/opencgs/Docker/entrypoint.sh" ]
