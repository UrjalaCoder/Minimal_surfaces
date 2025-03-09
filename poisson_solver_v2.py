from zero_form import ZeroForm
import numpy as np

def poisson_solve(Phi_form: ZeroForm) -> np.array:
    """
    Solves Poisson's equation Δu = Phi using FFT in a periodic domain.

    Args:
        Phi_form (ZeroForm): Object containing the scalar field (Nx, Ny, Nz)
                             and the grid spacing `h`.

    Returns:
        np.array: Solution to Poisson's equation.
    """
    Phi = Phi_form.scalar_field
    h = Phi_form.h

    Nx, Ny, Nz = Phi.shape

    # Compute Fourier transform of Phi
    Phi_fft = np.fft.fftn(Phi)
    
    # Compute Fourier-space frequencies
    kx = np.fft.fftfreq(Nx, d=h)
    ky = np.fft.fftfreq(Ny, d=h)
    kz = np.fft.fftfreq(Nz, d=h)

    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    
    # Compute the Laplacian in Fourier space
    laplacian_fourier = 4 / (h ** 2) * (np.sin(np.pi * KX / Nx) ** 2 +
                                        np.sin(np.pi * KY / Ny) ** 2 +
                                        np.sin(np.pi * KZ / Nz) ** 2)

    # Regularize the zero frequency component
    laplacian_fourier[0, 0, 0] = 1  # Prevent division by zero

    # Solve Poisson's equation in Fourier space
    result_fft = Phi_fft / (-laplacian_fourier + 1E-9)

    # Enforce zero mean solution
    # result_fft[0, 0, 0] = 0

    # Transform back into real space
    result = np.fft.ifftn(result_fft).real
    return result
