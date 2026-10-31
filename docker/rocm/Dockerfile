ARG ROCM_VERSION=7.1.1
FROM rocm/dev-ubuntu-24.04:${ROCM_VERSION}-complete

RUN apt-get update && apt-get install -y \
        kmod \
        wget \
        ccache \
        git \
        file \
        hipfft \
        rocfft \
        libfftw3-dev \
        openmpi-bin \
        libopenmpi-dev \
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

# Make sure MPI is discoverable
ENV PATH=${CMAKE_DIR}/bin:/usr/lib/x86_64-linux-gnu/openmpi/bin:/opt/rocm/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/openmpi/lib:/opt/rocm/hipfft/lib:$LD_LIBRARY_PATH
ENV CMAKE_PREFIX_PATH=/usr/lib/x86_64-linux-gnu/openmpi:/opt/rocm/hip/:/opt/rocm/:$CMAKE_PREFIX_PATH

# Install rocSHMEM from source: the rocSHMEM backend needs rocSHMEM 3.3 or
# later (rocshmem_alltoallmem_on_stream), and the ROCm releases ship older
# ones (3.1.0 in ROCm 7.1.1). rocSHMEM 3.3.0 is in projects/rocshmem of
# ROCm/rocm-systems at the tag therock-7.12. It is built with the IPC conduit
# only, as a static library with relocatable device code, for the GPU of the
# CI builds (Kokkos_ARCH_VEGA90A).
ARG ROCSHMEM_TAG=therock-7.12
ARG ROCSHMEM_GPU_TARGETS=gfx90a

ENV ROCSHMEM_DIR=/opt/rocshmem
RUN git clone --depth 1 --branch ${ROCSHMEM_TAG} --filter=blob:none --sparse \
        https://github.com/ROCm/rocm-systems.git ${build_dir}/rocm-systems && \
    cd ${build_dir}/rocm-systems && \
    git sparse-checkout set projects/rocshmem && \
    cmake -S projects/rocshmem -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=${ROCSHMEM_DIR} \
        -DGPU_TARGETS=${ROCSHMEM_GPU_TARGETS} \
        -DUSE_IPC=ON \
        -DUSE_RO=OFF \
        -DUSE_GDA=OFF \
        -DBUILD_TOOLS=OFF \
        && \
    cmake --build build --parallel $(nproc) && \
    cmake --install build && \
    cd ${HOME} && rm -rf ${build_dir}

ENV CMAKE_PREFIX_PATH=${ROCSHMEM_DIR}:$CMAKE_PREFIX_PATH

WORKDIR /work
CMD ["bash"]
