import time
import numpy as np

def identity_test(nx=16, ny=16, nz=16):
    """
    Test the performance of FFT on a 3D identity matrix.
    """
    # Create a 3D identity matrix
    A = np.random.rand(nx, ny, nz)
    A_ref = np.copy(A)

    # Perform FFT
    start_time = time.time()

    # X-slab (nx, ny, nz) FFT2 along Y and Z
    A_hat = np.fft.fft2(A, axes=(-2, -1))

    # Transpose (nx, ny, nz) -> (ny, nz, nx)
    A_hat = np.transpose(A_hat, (1, 2, 0))

    # Y-slab (ny, nz, nx) FFT along X
    A_hat = np.fft.fft(A_hat, axis=-1)

    # Y-slab (ny, nz, nx) FFT along X (with normalization)
    A_hat = np.fft.ifft(A_hat, axis=-1)

    # Transpose (ny, nz, nx) -> (nx, ny, nz)
    A_hat = np.transpose(A_hat, (2, 0, 1))

    # X-slab (nx, ny, nz) IFFT2 along Y and Z
    A = np.fft.ifft2(A_hat, axes=(-2, -1))
    end_time = time.time()

    # Check if the result is close to the expected output
    assert np.allclose(A, A_ref), "Input matrix has been modified."

    return end_time - start_time

if __name__ == '__main__':
    s = identity_test(nx=16, ny=16, nz=16)
    print(f"3D identity test took {s} [s]")
