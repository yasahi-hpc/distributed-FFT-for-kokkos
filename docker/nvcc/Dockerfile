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

ENV FFTWDIR "/usr"

RUN git config --global --add safe.directory "*"

# Install newer CMake manually
ARG CMAKE_VERSION=3.23.2
ARG build_dir=/tmp/build

ENV CMAKE_DIR=/opt/cmake
RUN CMAKE_SCRIPT=cmake-${CMAKE_VERSION}-linux-x86_64.sh && \
    mkdir -p ${build_dir} && mkdir -p ${CMAKE_DIR} \
    cd ${build_dir} && \
    wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/${CMAKE_SCRIPT} && \
    sh ${CMAKE_SCRIPT} --skip-license --prefix=${CMAKE_DIR} && \
    cd ${HOME} && rm -rf ${build_dir}
ENV PATH=${CMAKE_DIR}/bin:$PATH

# Install MPI
# Build a CUDA-enabled OpenMPI. This is probably not necessary for a compile-only test.
#Download MPI
RUN wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.6.tar.bz2 && \
    tar -xf openmpi-4.1.6.tar.bz2 && \
    cd openmpi-4.1.6 && ./configure --prefix=/usr/local --with-cuda=/usr/local/cuda && \
    make -C openmpi-4.1.6 -j$(nproc) all

RUN make -C openmpi-4.1.6 all install && ldconfig

# Print some MPI Info
RUN which mpirun && mpirun --version &&
    which mpicxx && mpicxx --version &&
    ompi_info --parsable --all | grep mpi_built_with_cuda_support:value:true

WORKDIR /work
CMD ["bash"]