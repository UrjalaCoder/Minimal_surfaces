import numpy as np

class Grid:
    def __init__(self, resolution=32, bounds=[[0, 1], [0, 1], [0, 1]]):
        """
        Create a 3D grid of integer indices and associated spatial coordinates.

        Parameters:
        - resolution: int or list [Nx, Ny, Nz]
        - bounds: list of 3 [min, max] intervals
        """
        if isinstance(resolution, int):
            self.res = [resolution, resolution, resolution]
        else:
            self.res = list(resolution)
        self.bounds = [list(b) for b in bounds]

        self._generate_indices()
        self._compute_scale_offset()
        self._compute_positions()

    def _generate_indices(self):
        nx, ny, nz = self.res
        i, j, k = np.meshgrid(
            np.arange(nx), np.arange(ny), np.arange(nz),
            indexing="ij"
        )
        self.indices_grid = np.stack([i, j, k], axis=-1)      # Shape (Nx, Ny, Nz, 3)
        self.indices_flat = self.indices_grid.reshape(-1, 3)  # Shape (Nx*Ny*Nz, 3)

    def _compute_scale_offset(self):
        self.scale = np.array([
            (self.bounds[0][1] - self.bounds[0][0]) / self.res[0],
            (self.bounds[1][1] - self.bounds[1][0]) / self.res[1],
            (self.bounds[2][1] - self.bounds[2][0]) / self.res[2],
        ])
        self.offset = np.array([
            self.bounds[0][0],
            self.bounds[1][0],
            self.bounds[2][0],
        ])

    def _compute_positions(self):
        self.positions_flat = self.indices_flat * self.scale + self.offset
        self.positions_grid = self.positions_flat.reshape(
            *self.res, 3
        )  # Shape (Nx, Ny, Nz, 3)

    def get_flat_indices(self):
        return self.indices_flat

    def get_grid_indices(self):
        return self.indices_grid

    def get_flat_positions(self):
        return self.positions_flat

    def get_grid_positions(self):
        return self.positions_grid

    def index_to_position(self, idx):
        """
        Convert a single integer index [i, j, k] to spatial coordinates.
        """
        return np.array(idx) * self.scale + self.offset

    def position_to_index(self, pos):
        """
        Map a point in space to its corresponding grid index (may need rounding).
        """
        idx = (np.array(pos) - self.offset) / self.scale
        return np.floor(idx).astype(int)

    def __repr__(self):
        return f"Grid3D(resolution={self.res}, bounds={self.bounds})"

if __name__ == '__main__':
    # Test grid initialization
    grid = Grid(resolution=32, bounds=[[-1, 1], [-1, 1], [-1, 1]])