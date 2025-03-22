import numpy as np

class OneForm:
    def __init__(self, mesh_indices:np.array, evaluation: callable = None, L = 1.0, vector_field = None):
        """
        The argument mesh_indices should be an (N, N, N, 3) array which contains the indices for each point of the mesh for each of the 3 cartesian directions.
        """
        assert len(mesh_indices.shape) == 4
        assert mesh_indices.shape[-1] == 3                  # 3-dimensional space of points, index space
        assert L >= 0

        self.Nx, self.Ny, self.Nz, _ = mesh_indices.shape   # The number of grid points in each cartesian coordinate direction.
        self.h = L / self.Nx                                # Assume regular grid in all directions
        self.mesh = mesh_indices * self.h
        
        # Either use the already provided vector field, or create from evaluation function.
        # If both fail, set to identically 0 field.
        if vector_field is None:
            self.vector_field = evaluation(self.mesh) if evaluation is not None else np.zeros_like(self.mesh)
        else:
            self.vector_field = vector_field
        
        self.mass = self._compute_mass()
        self.l2_norm = np.sqrt(self * self)
    
    def vertex_norms(self):
        return np.linalg.norm(self.vector_field, axis = -1)

    # Get the value of the 1-form at specific locations, either by index or by position.
    def __call__(self, positions: np.array = None, position_indices: np.array = None):
        # Assert that the argument is of shape(M, 3)
        assert len(positions.shape) == 2 and positions.shape[1] == 3

        if position_indices is None and positions is not None:
            # Floor to integers
            position_indices = np.round(positions / self.h).astype(int)
        
        # Obtain vector field values
        return self.vector_field[position_indices[:, 0], position_indices[:, 1], position_indices[:, 2]]
    
    def _compute_mass(self):
        norms_2 = self.vertex_norms()
        return np.sum(norms_2) * (self.h ** 3)
    
    def divergence(self):
        """Compute the divergence of a 3D vector field using central differences."""
        grad_phi = self.vector_field
        dx = dy = dz = self.h
        div_x = (np.roll(grad_phi[..., 0], -1, axis=0) - np.roll(grad_phi[..., 0], 1, axis=0)) / (2 * dx)
        div_y = (np.roll(grad_phi[..., 1], -1, axis=1) - np.roll(grad_phi[..., 1], 1, axis=1)) / (2 * dy)
        div_z = (np.roll(grad_phi[..., 2], -1, axis=2) - np.roll(grad_phi[..., 2], 1, axis=2)) / (2 * dz)
        return div_x + div_y + div_z


    def __mul__(self, b):
        assert b.h == self.h
        multiplication_field = self.vector_field * b.vector_field
        return np.sum(multiplication_field * (self.h ** 3))


