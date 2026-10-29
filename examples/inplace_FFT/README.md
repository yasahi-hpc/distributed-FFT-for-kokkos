# In-place distributed FFT

This example runs a 3D real-to-complex (R2C) transform in place: the input
and the output Views point to the same allocation. The input is split into
slabs along x and the output into slabs along y. It then runs the backward
transform and checks both results against an out-of-place plan.

```bash
cmake -B build -DENABLE_EXAMPLES=ON ...
cmake --build build --target inplace_FFT
mpiexec -n 4 ./build/examples/inplace_FFT/inplace_FFT
```

Any number of processes works. The global extents `(16, 15, 14)` are not
divisible by every process count, so the ranks can have different local
extents.

## How to write an in-place transform

1. **Compute the local extents** of the input and output on each topology
   with `compute_local_extents_and_starts`.
2. **Allocate one buffer** of `compute_inplace_allocation_size` elements of
   the output value type.
3. **Build both Views as unmanaged Views** on `buffer.data()`, each with its
   own local extents.
4. **Create the plan** with these two Views. The plan is in-place because
   the two Views start at the same address. `Plan::is_inplace()` returns
   `true`.
5. **Run `execute`** with the same two Views: forward from `u` to `u_hat`,
   and backward from `u_hat` to `u`.

```cpp
auto [in_extents, in_starts] =
    KokkosFFT::Distributed::compute_local_extents_and_starts(
        gin_extents, in_topology, MPI_COMM_WORLD);
auto [out_extents, out_starts] =
    KokkosFFT::Distributed::compute_local_extents_and_starts(
        gout_extents, out_topology, MPI_COMM_WORLD);

const std::size_t buffer_size =
    KokkosFFT::Distributed::compute_inplace_allocation_size<
        double, Kokkos::complex<double>>(in_extents, out_extents);
View1D<Kokkos::complex<double>> buffer("buffer", buffer_size);

UnmanagedView3D<double> u(reinterpret_cast<double*>(buffer.data()), in0, in1,
                          in2);
UnmanagedView3D<Kokkos::complex<double>> u_hat(buffer.data(), out0, out1,
                                               out2);

KokkosFFT::Distributed::Plan plan(exec, u, u_hat, axes, in_topology,
                                  out_topology, MPI_COMM_WORLD);
KokkosFFT::Distributed::execute(plan, u, u_hat, KokkosFFT::Direction::forward);
KokkosFFT::Distributed::execute(plan, u_hat, u, KokkosFFT::Direction::backward);
```

## What in-place means here

- **The Views have the same shapes as out-of-place.** Each View is dense in
  its own local extents.
  - Before `forward`, the data is `u(i, j, k)` in the layout of `u`.
  - After `forward`, it is `u_hat(i, j, k)` in the layout of `u_hat`.
  - `forward` overwrites the input, and `backward` overwrites the output.
- **The real View is not padded.** This differs from the in-place R2C of
  single-node Kokkos-FFT (its example `08_inplace_FFT`). There, the
  innermost extent of the real data is padded to `2 * (n / 2 + 1)`. Here,
  `u` is `(nx / p, ny, nz)`, not `(nx / p, ny, 2 * (nz / 2 + 1))`. The input
  and output live on different topologies, so their local extents differ
  anyway, and padding would not line them up.
- **The plan does not need to run an in-place FFT library call.** It only
  needs the two Views to point to the same data. Internally, the plan moves
  the data through its own communication buffers. It allocates nothing
  more than an out-of-place plan, with one exception: an R2C or C2R
  transform that needs neither communication nor transposes (e.g. on one
  process) copies its input into one internal buffer of the input size.
- **The Views must either start at the same address or not overlap.**
  Views that overlap partially make the plan constructor throw.
- **The placement is fixed at construction.** An in-place plan must be
  executed with Views that point to the same data, and an out-of-place plan
  with Views that do not. Otherwise `execute` throws.

## The size of the allocation

The buffer must hold the larger of the local input and the local output
**on every rank**. This is not always the output: with uneven extents, the
input can be the larger one on some ranks. In this example at 4 processes:

| rank | input `u` (double)    | output `u_hat` (complex) | buffer      |
| ---- | --------------------- | ------------------------ | ----------- |
| 0–2  | `(4, 15, 14)`: 6720 B | `(16, 4, 8)`: 8192 B     | 512 complex |
| 3    | `(4, 15, 14)`: 6720 B | `(16, 3, 8)`: 6144 B     | 420 complex |

`compute_inplace_allocation_size<InValueType, OutValueType>(in_extents,
out_extents)` returns this size in elements of `OutValueType`. The plan
cannot check it, because an unmanaged View does not know the size of the
allocation behind it. A buffer that is too small is overwritten past its
end without an error.

## Limitations

- **rocFFT:** with `PRIORITIZE_TPL_PLAN_IF_AVAILABLE`, an in-place plan does
  not use rocFFT. It uses the internal plans instead. Constructing a
  `TplPlan` directly with rocFFT and in-place Views throws.
- **cuFFTMp** supports in-place plans.
