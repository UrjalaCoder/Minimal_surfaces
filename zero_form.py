import numpy as np

class ZeroForm:

    def __init__(self, mesh_indices:np.array, evaluation: callable = None, L = 1.0, scalar_field = None):
        """
        The argument mesh_indices should be an (N, N, N, 3) array which contains the indices for each point of the mesh for each of the 3 cartesian directions.
        """
        assert len(mesh_indices.shape) == 4
        assert mesh_indices.shape[-1] == 3                  # 3-dimensional space of points, index space
        assert L >= 0

        self.Nx, self.Ny, self.Nz, _ = mesh_indices.shape   # The number of grid points in each cartesian coordinate direction.
        self.h = L / self.Nx                                # Assume regular grid in all directions
        self.mesh = mesh_indices * self.h
        
        # Either use the already provided scalar field, or create from evaluation function.
        # If both fail, set to identically 0 field.
        if scalar_field is None:
            self.scalar_field = evaluation(self.mesh) if evaluation is not None else np.zeros_like(self.mesh)
        else:
            self.scalar_field = scalar_field
    
    # Get the value of the scalar field at specific locations, either by index or by position.
    def __call__(self, positions: np.array = None, position_indices: np.array = None):
        # Assert that the argument is of shape(M, 3)
        assert len(positions.shape) == 2 and positions.shape[1] == 3

        if position_indices is None and positions is not None:
            # Floor to integers
            position_indices = np.round(positions / self.h).astype(int)
        
        # Obtain vector field values
        return self.scalar_field[position_indices[:, 0], position_indices[:, 1], position_indices[:, 2]]
    
    def differentiation(self):
        """Compute the differentiation (gradient) of a scalar field using the midpoint rule on a toroidal grid."""
        grad = np.zeros_like(self.mesh)
        phi = self.scalar_field
        grad[..., 0] = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2 * self.h)  # x-direction
        grad[..., 1] = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2 * self.h)  # y-direction
        grad[..., 2] = (np.roll(phi, -1, axis=2) - np.roll(phi, 1, axis=2)) / (2 * self.h)  # z-direction
        return grad

