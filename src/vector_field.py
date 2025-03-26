import numpy as np
from scipy.spatial import cKDTree
from .grid import Grid

class VectorField:
    def __init__(self, grid: Grid, vectors=None):
        """
        Create a vector field associated with a Grid object on the ambient space R^3

        Parameters:
        - grid: A Grid instance
        - vectors: optional initial field, shape (N^3, 3) or (Nx, Ny, Nz, 3)
        """
        self.grid = grid
        self.shape = grid.res
        self.positions = grid.get_flat_positions()

        if vectors is None:
            # Start with zero field
            self.vectors_flat = np.zeros_like(self.grid.get_flat_positions())
        elif vectors.shape == self.grid.get_flat_positions().shape:
            self.vectors_flat = vectors
        elif vectors.shape == self.grid.get_grid_positions().shape:
            self.vectors_flat = vectors.reshape(-1, 3)
        else:
            raise ValueError("Shape mismatch between vector field and grid")
        
        self._build_kdtree()
    
    def _build_kdtree(self):
        self._kdtree = cKDTree(self.positions)

    def set_vectors_from_function(self, func):
        """
        Populate the vector field using a function: R^3 → R^3
        """
        positions = self.grid.get_flat_positions()
        self.vectors_flat = np.array([func(p) for p in positions])

    def apply_mask(self, mask_func):
        """
        Zero out vectors outside a region. mask_func: R^3 → bool
        """
        positions = self.grid.get_flat_positions()
        mask = np.array([mask_func(p) for p in positions])
        self.vectors_flat[~mask] = 0

    def get_flat(self):
        return self.vectors_flat

    def get_grid(self):
        return self.vectors_flat.reshape(*self.shape, 3)

    def get_magnitude(self):
        return np.linalg.norm(self.vectors_flat, axis=1)

    def get_normalized(self):
        norms = self.get_magnitude()
        norms[norms == 0] = 1  # Avoid division by zero
        return self.vectors_flat / norms[:, None]

    def __len__(self):
        return self.vectors_flat.shape[0]

    def __repr__(self):
        return f"VectorField3D(resolution={self.shape}, length={len(self)})"
    
    def __call__(self, point):
        """
        Return the vector value at a given 3D point using nearest-neighbor.
        Useful for StreamLines or external probing.
        """
        _, idx = self._kdtree.query(point)
        return self.vectors_flat[idx]
