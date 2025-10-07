ARG BASE=nvidia/cuda:12.3.1-devel-ubuntu22.04
FROM $BASE

RUN apt-get update && apt-get install -y \
        bc \
        wget \
        git \
        ccache \
        libfftw3-dev \
        $ADDITIONAL_PACKAGES \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN git config --global --add safe.directory "*"

# Install newer CMake manually
ARG CMAKE_VERSION=3.23.2
ARG build_dir=/tmp/build

ENV CMAKE_DIR=/opt/cmake \
    FFTWDIR=/usr
RUN CMAKE_SCRIPT=cmake-${CMAKE_VERSION}-linux-x86_64.sh && \
    mkdir -p ${build_dir} && mkdir -p ${CMAKE_DIR} \
    cd ${build_dir} && \
    wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/${CMAKE_SCRIPT} && \
    sh ${CMAKE_SCRIPT} --skip-license --prefix=${CMAKE_DIR} && \
    cd ${HOME} && rm -rf ${build_dir}

# Build a CUDA-enabled OpenMPI. This is probably not necessary for a compile-only test.
# Install CUDA-aware OpenMPI
ARG OPENMPI_VERSION=4.1.6
ARG build_dir=/tmp/build

RUN mkdir -p ${build_dir} && cd ${build_dir} && \
    wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-${OPENMPI_VERSION}.tar.gz && \
    tar -xzf openmpi-${OPENMPI_VERSION}.tar.gz && cd openmpi-${OPENMPI_VERSION} && \
    ./configure --with-cuda --enable-mpi-cxx --prefix=/opt/openmpi && \
    make -j"$(nproc)" && make install && \
    ldconfig && \
    cd / && rm -rf ${build_dir}

ENV PATH=/opt/openmpi/bin:${CMAKE_DIR}/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/openmpi/lib:$LD_LIBRARY_PATH

WORKDIR /work
CMD ["bash"]
