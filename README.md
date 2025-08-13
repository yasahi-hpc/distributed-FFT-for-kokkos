# distributed-FFT-for-kokkos

The distributed FFT interface for kokkos users based on [kokkos](https://github.com/kokkos/kokkos) and [kokkos-fft](https://github.com/kokkos/kokkos-fft).

## Using
First of all, you need to clone this repo.

```bash
git clone --recursive https://github.com/yasahi-hpc/distributed-FFT-for-kokkos.git
```

### Prerequisites

To use `distributed-FFT-for-kokkos`, we need the following:
* `CMake 3.22+`
* `Kokkos 4.5+`
* `kokkos-fft 0.4+`
* `gcc 11.0.0+` (CPUs)
* `IntelLLVM 2025.0.0+` (CPUs, Intel GPUs)
* `nvcc 11.0.0+` (NVIDIA GPUs)
* `rocm 5.6.0+` (AMD GPUs)

### Compile and run

For compilation, we basically rely on the CMake options for Kokkos. For example, the compile options for MI250X GPU is as follows.

```bash
cmake -B build \
      -DCMAKE_CXX_COMPILER=hipcc \
      -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_TESTS=ON \
      -DENABLE_EXAMPLES=ON \
      -DKokkos_ENABLE_HIP=ON \
      -DKokkos_ARCH_AMD_GFX90A=ON \
      -DCMAKE_EXE_LINKER_FLAGS="${PE_MPICH_GTL_DIR_amd_gfx90a} ${PE_MPICH_GTL_LIBS_amd_gfx90a}" 
cmake --build build -j 8
```

To run the tests, please run the following command.

```bash
cd build
ctest --output-on-failure
```

To run an example, please run the following command for 16 GPUs with 2 nodes.

```bash
export MPICH_GPU_SUPPORT_ENABLED=1
cd build
srun --ntasks-per-node 8 --cpus-per-task 1 --threads-per-core 1 --gpu-bind closest examples/navier-stokes-MPI-batched/navier-stokes-MPI-batched -px 16 -Re 1600 -dt 0.001 -nx 1024 -nbiter 10
```

## LICENSE

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`distributed-FFT-for-kokkos` is distributed under the MIT license.
