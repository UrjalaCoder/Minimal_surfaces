import numpy as np
from scipy.spatial import cKDTree
from .grid import Grid

class DifferentialForm:
    def __init__(self, grid: Grid, form_degree: int = 0, evaluation: callable = None, field=None):
        """
        Create a differential form of specified degree on a grid.
        
        Parameters:
        - grid: Grid instance containing the mesh information
        - form_degree: 0 for scalar fields, 1 for vector fields
        - evaluation: function to evaluate the form at points
        - field: optional initial field values
        """
        assert isinstance(grid, Grid), "grid must be an instance of Grid"
        assert form_degree in [0, 1], "Only 0-forms (scalar) and 1-forms (vector) are supported"

        self.grid = grid
        self.form_degree = form_degree
        self.h = grid.scale[0]  # Use grid's scale for step size
        
        # Initialize field based on degree
        if field is None:
            if form_degree == 0:
                self.field = evaluation(grid.get_grid_positions()) if evaluation is not None else np.zeros(grid.res)
            else:  # form_degree == 1
                self.field = evaluation(grid.get_grid_positions()) if evaluation is not None else np.zeros((*grid.res, 3))
        else:
            self.field = field

        # Precompute mass and L2 norm for 1-forms
        if form_degree == 1:
            self.mass = self._compute_mass()
            self.l2_norm = np.sqrt(self * self)
        else:
            self.mass = self.mass()
            self.l2_norm = self.l2_norm()

        # Build KDTree for nearest neighbor queries
        self._build_kdtree()

    def _build_kdtree(self):
        """Build KDTree for efficient nearest neighbor queries."""
        positions = self.grid.get_flat_positions()
        self._kdtree = cKDTree(positions)

    def _compute_mass(self):
        """Compute the mass norm of a 1-form."""
        if self.form_degree != 1:
            raise ValueError("Mass norm computation is only defined for 1-forms")
        norms_2 = self.vertex_norms()
        return np.sum(norms_2) * (self.h ** 3)

    def vertex_norms(self):
        """Compute the norm of the form at each vertex."""
        if self.form_degree == 0:
            return np.abs(self.field)
        return np.linalg.norm(self.field, axis=-1)

    def __call__(self, positions: np.array = None, position_indices: np.array = None):
        """
        Evaluate the form at given positions.
        
        Parameters:
        - positions: (M, 3) array of points
        - position_indices: (M, 3) array of grid indices
        """
        if positions is not None:
            assert len(positions.shape) == 2 and positions.shape[1] == 3
            if position_indices is None:
                position_indices = self.grid.position_to_index(positions)
        
        if self.form_degree == 0:
            return self.field[position_indices[:, 0], position_indices[:, 1], position_indices[:, 2]]
        else:
            return self.field[position_indices[:, 0], position_indices[:, 1], position_indices[:, 2], :]

    def evaluate_at_point(self, point: np.array) -> np.array:
        """
        Evaluate the form at a single point.
        This method is specifically designed for visualization compatibility.
        
        Parameters:
        - point: (3,) array representing a point in space
        
        Returns:
        - For 0-forms: scalar value
        - For 1-forms: (3,) array representing the vector
        """
        if self.form_degree == 0:
            return self.__call__(point.reshape(1, 3))[0]
        else:
            return self.__call__(point.reshape(1, 3))[0]

    def differentiation(self):
        """Compute the exterior derivative of the form."""
        if self.form_degree == 0:
            # Gradient of scalar field (0-form → 1-form)
            grad = np.zeros((*self.grid.res, 3))
            phi = self.field
            grad[..., 0] = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2 * self.h)
            grad[..., 1] = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2 * self.h)
            grad[..., 2] = (np.roll(phi, -1, axis=2) - np.roll(phi, 1, axis=2)) / (2 * self.h)
            return DifferentialForm(self.grid, form_degree=1, field=grad)
        else:
            # Divergence of vector field (1-form → 0-form)
            div = np.zeros(self.grid.res)
            for i in range(3):
                div += (np.roll(self.field[..., i], -1, axis=i) - 
                       np.roll(self.field[..., i], 1, axis=i)) / (2 * self.h)
            return DifferentialForm(self.grid, form_degree=0, field=div)

    def divergence(self):
        """Compute the divergence of a 1-form using central differences."""
        if self.form_degree != 1:
            raise ValueError("Divergence is only defined for 1-forms")
        
        grad_phi = self.field
        dx = dy = dz = self.h
        div_x = (np.roll(grad_phi[..., 0], -1, axis=0) - np.roll(grad_phi[..., 0], 1, axis=0)) / (2 * dx)
        div_y = (np.roll(grad_phi[..., 1], -1, axis=1) - np.roll(grad_phi[..., 1], 1, axis=1)) / (2 * dy)
        div_z = (np.roll(grad_phi[..., 2], -1, axis=2) - np.roll(grad_phi[..., 2], 1, axis=2)) / (2 * dz)
        return div_x + div_y + div_z

    def codivergence(self):
        """Compute the codivergence (δ) of a 1-form."""
        if self.form_degree != 1:
            raise ValueError("Codivergence is only defined for 1-forms")
        
        # Step 1: Compute Hodge star (convert to 2-form)
        X_dual = np.zeros_like(self.field)
        X_dual[..., 0] = self.field[..., 1]  # ⋆(dx) = dy∧dz
        X_dual[..., 1] = -self.field[..., 0]  # ⋆(dy) = -dx∧dz
        X_dual[..., 2] = self.field[..., 2]  # ⋆(dz) = dx∧dy

        # Step 2: Compute exterior derivative d(⋆X)
        dX_dual = np.zeros_like(X_dual)
        for i in range(3):
            dX_dual[..., i] = (np.roll(X_dual[..., i], -1, axis=i) - X_dual[..., i]) / self.h

        # Step 3: Compute final Hodge star to obtain codivergence
        codiv = dX_dual[..., 0] + dX_dual[..., 1] + dX_dual[..., 2]
        return DifferentialForm(self.grid, form_degree=0, field=codiv)

    def get_magnitude(self):
        """Compute the magnitude of the form."""
        if self.form_degree == 0:
            return np.abs(self.field)
        else:
            return np.linalg.norm(self.field, axis=-1)

    def get_normalized(self):
        """Return a normalized version of the form."""
        if self.form_degree == 0:
            raise ValueError("Normalization is only defined for 1-forms")
        norms = self.get_magnitude()
        norms[norms == 0] = 1  # Avoid division by zero
        normalized_field = self.field / norms[..., None]
        return DifferentialForm(self.grid, form_degree=1, field=normalized_field)

    def mass(self):
        """Compute the mass norm of the form."""
        if self.form_degree == 0:
            return np.sum(np.abs(self.field)) * (self.h ** 3)
        else:
            return np.sum(np.linalg.norm(self.field, axis=-1)) * (self.h ** 3)

    def l2_norm(self):
        """Compute the L2 norm of the form."""
        if self.form_degree == 0:
            return np.sqrt(np.sum(self.field ** 2) * (self.h ** 3))
        else:
            return np.sqrt(np.sum(self.field ** 2) * (self.h ** 3))

    def __mul__(self, other):
        """Compute the inner product of two forms."""
        if self.form_degree != other.form_degree:
            raise ValueError("Can only multiply forms of the same degree")
        if self.grid != other.grid:
            raise ValueError("Forms must be defined on the same grid")
        
        if self.form_degree == 0:
            return np.sum(self.field * other.field) * (self.h ** 3)
        else:
            return np.sum(np.sum(self.field * other.field, axis=-1)) * (self.h ** 3)

    def apply_mask(self, mask_func):
        """Zero out the form outside a region defined by mask_func."""
        positions = self.grid.get_flat_positions()
        mask = np.array([mask_func(p) for p in positions])
        mask = mask.reshape(self.grid.res)
        
        if self.form_degree == 0:
            self.field[~mask] = 0
        else:
            self.field[~mask, :] = 0

    def __repr__(self):
        return f"DifferentialForm(degree={self.form_degree}, grid={self.grid})" 