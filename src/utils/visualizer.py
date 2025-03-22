import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

from one_form import OneForm

def visualize_1form_with_colors(ax, grid, one_form: OneForm, 
                                length=0.1, normalize=True, 
                                stride=2, cmap='viridis'):
    """
    Visualizes a 1-form as a vector field in 3D with colors based on magnitude.
    
    Parameters:
    - ax: matplotlib 3D axis object (if None, creates a new one).
    - grid: numpy array of shape (N, N, N, 3), containing (X, Y, Z) coordinates.
    - one_form: numpy array of shape (N, N, N, 3), containing (F1, F2, F3) components.
    - length: float, scaling factor for vector lengths.
    - normalize: bool, whether to normalize vectors.
    - stride: int, downsampling step to avoid clutter.
    - cmap: str, colormap for magnitude-based coloring.

    Returns:
    - ax: the matplotlib 3D axis object with the plotted vectors.
    """

    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

    one_form_field = one_form.vector_field.copy()
    X, Y, Z = grid[..., 0], grid[..., 1], grid[..., 2]  # Extract grid coordinates
    F1, F2, F3 = one_form_field[..., 0], one_form_field[..., 1], one_form_field[..., 2]  # Extract 1-form components

    # Compute magnitude of vector field
    magnitude = np.sqrt(F1**2 + F2**2 + F3**2)

    # Normalize vectors if requested
    if normalize:
        nonzero_mask = magnitude > 1e-10  # Avoid division by zero
        F1[nonzero_mask] /= magnitude[nonzero_mask]
        F2[nonzero_mask] /= magnitude[nonzero_mask]
        F3[nonzero_mask] /= magnitude[nonzero_mask]

    # Normalize magnitudes for colormap scaling
    norm = plt.Normalize(vmin=magnitude.min(), vmax=magnitude.max())
    colors = cm.get_cmap(cmap)(norm(magnitude))  # RGBA colors

    # Downsample grid to avoid clutter
    Xs, Ys, Zs = X[::stride, ::stride, ::stride], Y[::stride, ::stride, ::stride], Z[::stride, ::stride, ::stride]
    U, V, W = F1[::stride, ::stride, ::stride], F2[::stride, ::stride, ::stride], F3[::stride, ::stride, ::stride]
    C = colors[::stride, ::stride, ::stride]

    # Create colored quivers
    for i in range(Xs.shape[0]):
        for j in range(Xs.shape[1]):
            for k in range(Xs.shape[2]):
                ax.quiver(Xs[i, j, k], Ys[i, j, k], Zs[i, j, k], 
                          U[i, j, k], V[i, j, k], W[i, j, k], 
                          length=length, normalize=normalize, color=C[i, j, k])

    # Add colorbar only if ax is a new figure
    if ax.get_figure() is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.75, pad=0.1)
        cbar.set_label("Vector Magnitude")

    return ax  # Return the axis so it can be used in a notebook
