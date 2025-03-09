from zero_form import ZeroForm
import numpy as np

def poisson_solve(Phi_form: ZeroForm) -> np.array:
    Phi = Phi_form.scalar_field
    h = Phi_form.h

    Nx, Ny, Nz = Phi.shape
    # Mapping into Fourier space:
    Phi_fft = np.fft.fftn(Phi)
    
    # Calculate the necessary frequencies for each dimension:
    K = np.asarray([[[[k_xi / Nx, k_yi / Ny, k_zi / Nz] for k_xi in range(Nx)] for k_yi in range(Ny)] for k_zi in range(Nz)])

    # Divisor with finite difference taken into account:
    sin_correction = np.sum(np.sin(np.pi * K) ** 2, axis = -1) * 4 / (h ** 2)
    sin_correction[0, 0, 0] = sin_correction[0, 0, 0] + 1E-9
    result = Phi_fft / (-sin_correction)

    # Transform back into original space
    result = np.fft.ifftn(result)
    return result
