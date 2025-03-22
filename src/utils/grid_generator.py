import numpy as np

def create_index_grid(resolution=32, bounds=((0, 1), (0, 1), (0, 1))):
    """Generate a grid of integer indices and compute the scaling factor."""
    nx, ny, nz = resolution, resolution, resolution
    scale = np.array([
        (bounds[0][1] - bounds[0][0]) / nx,
        (bounds[1][1] - bounds[1][0]) / ny,
        (bounds[2][1] - bounds[2][0]) / nz,
    ])
    offset = np.array([bounds[0][0], bounds[1][0], bounds[2][0]])

    # Create index grid
    i, j, k = np.meshgrid(
        np.arange(nx), np.arange(ny), np.arange(nz),
        indexing="ij"
    )
    indices = np.stack([i, j, k], axis=-1).reshape(-1, 3)

    return indices, scale, offset

if __name__ == '__main__':
    grid = create_index_grid(resolution=10)
    indices, scale, offset = grid
