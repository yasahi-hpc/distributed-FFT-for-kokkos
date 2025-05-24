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

    # Z-pencil (nx, ny, nz) FFT along Z
    A_hat = np.fft.fft(A, axis=-1)

    # Transpose (nx, ny, nz) -> (nx, nz, ny)
    A_hat = np.transpose(A_hat, (0, 2, 1))

    # Y-pencil (nx, nz, ny) FFT along Y
    A_hat = np.fft.fft(A_hat, axis=-1)

    # Transpose (nx, nz, ny) -> (ny, nz, nx)
    A_hat = np.transpose(A_hat, (2, 1, 0))

    # X-pencil (ny, nz, nx) FFT along X
    A_hat = np.fft.fft(A_hat, axis=-1)

    # X-pencil (ny, nz, nx) IFFT along X
    A_hat = np.fft.ifft(A_hat, axis=-1)

    # transpose (ny, nz, nx) -> (nx, nz, ny)
    A_hat = np.transpose(A_hat, (2, 1, 0))

    # Y-pencil (nx, nz, ny) IFFT along Y
    A_hat = np.fft.ifft(A_hat, axis=-1)

    # Transpose (nx, nz, ny) -> (nx, ny, nz)
    A_hat = np.transpose(A_hat, (0, 2, 1))

    # Z-pencil (nx, ny, nz) IFFT along Z
    A = np.fft.ifft(A_hat, axis=-1)
    end_time = time.time()

    # Check if the result is close to the expected output
    assert np.allclose(A, A_ref), "Input matrix has been modified."

    return end_time - start_time

if __name__ == '__main__':
    s = identity_test(nx=16, ny=16, nz=16)
    print(f"3D identity test took {s} [s]")
