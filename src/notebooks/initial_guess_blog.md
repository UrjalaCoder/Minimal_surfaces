# Initializing η₀: The First Step in Minimal Surface Computation

In our minimal surface solver, the initialization of η₀ is a crucial first step. This 1-form must satisfy two key conditions:

1. Boundary condition: dη₀ = δ_Γ
2. Cohomology constraints: ∫_M dx_i ∧ ⋆η₀ = A_i

Let's break down how we implement this initialization step.

## 1. Understanding the Requirements

### 1.1 Boundary Condition

The boundary condition dη₀ = δ_Γ means that the exterior derivative of η₀ must equal the Dirac delta form of the boundary curve Γ. In our discrete setting, this translates to:

```python
def _enforce_boundary_condition(self, eta: DifferentialForm) -> bool:
    """
    Check if dη₀ = δ_Γ is satisfied.
    """
    d_eta = eta.differentiation()
    # Compare with δ_Γ at grid points
    return np.allclose(d_eta.field, self.delta_gamma.field)
```

### 1.2 Cohomology Constraints

The cohomology constraints ensure that our solution corresponds to a surface in ℝ³ rather than wrapping around the torus. We need to compute the projected area vector A first:

```python
def _compute_projected_area(self) -> np.ndarray:
    """
    Compute A = 1/2 ∮_Γ γ × dγ
    """
    # Compute tangent vectors using finite differences
    gamma = self.boundary_curve
    dgamma = np.roll(gamma, -1, axis=0) - np.roll(gamma, 1, axis=0)
    
    # Compute cross products and integrate
    A = np.zeros(3)
    for i in range(3):
        # Project onto coordinate planes
        j, k = (i+1)%3, (i+2)%3
        A[i] = 0.5 * np.sum(gamma[:,j] * dgamma[:,k] - gamma[:,k] * dgamma[:,j])
    
    return A
```

## 2. Constructing η₀

### 2.1 Initial Guess

We start with a simple initial guess that satisfies the boundary condition. One approach is to use a vector field that points towards the boundary curve:

```python
def _create_initial_guess(self) -> DifferentialForm:
    """
    Create an initial guess for η₀ that approximately satisfies dη₀ = δ_Γ.
    """
    # Get grid positions
    positions = self.grid.get_grid_positions()
    
    # Initialize field
    field = np.zeros((*self.grid.res, 3))
    
    # For each grid point, compute contribution from boundary curve
    for i in range(self.grid.res[0]):
        for j in range(self.grid.res[1]):
            for k in range(self.grid.res[2]):
                point = positions[i,j,k]
                # Compute vector field at this point
                field[i,j,k] = self._compute_vector_at_point(point)
    
    return DifferentialForm(self.grid, form_degree=1, field=field)

def _compute_vector_at_point(self, point: np.ndarray) -> np.ndarray:
    """
    Compute the vector field at a point based on boundary curve.
    """
    # Find closest point on boundary curve
    distances = np.linalg.norm(self.boundary_curve - point, axis=1)
    closest_idx = np.argmin(distances)
    closest_point = self.boundary_curve[closest_idx]
    
    # Compute direction vector
    direction = closest_point - point
    if np.linalg.norm(direction) > 0:
        direction = direction / np.linalg.norm(direction)
    
    return direction
```

### 2.2 Enforcing Cohomology Constraints

After creating the initial guess, we need to adjust it to satisfy the cohomology constraints:

```python
def _enforce_cohomology_constraints(self, eta: DifferentialForm) -> DifferentialForm:
    """
    Adjust η₀ to satisfy ∫_M dx_i ∧ ⋆η₀ = A_i.
    """
    # Compute current projected areas
    current_A = np.zeros(3)
    for i in range(3):
        # Create basis 1-form dx_i
        dx_i = np.zeros((*self.grid.res, 3))
        dx_i[..., i] = 1
        dx_i_form = DifferentialForm(self.grid, form_degree=1, field=dx_i)
        
        # Compute integral ∫_M dx_i ∧ ⋆η₀
        current_A[i] = (dx_i_form * eta.star()).field.sum() * (self.grid.scale[0] ** 3)
    
    # Compute correction needed
    correction = self.A - current_A
    
    # Apply correction
    corrected_field = eta.field.copy()
    for i in range(3):
        corrected_field[..., i] += correction[i] / (self.grid.res[0] * self.grid.res[1] * self.grid.res[2])
    
    return DifferentialForm(self.grid, form_degree=1, field=corrected_field)
```

## 3. Putting It All Together

The complete initialization process looks like this:

```python
def _initialize_eta_0(self) -> DifferentialForm:
    """
    Initialize η₀ that satisfies both boundary and cohomology constraints.
    """
    # Step 1: Create initial guess
    eta = self._create_initial_guess()
    
    # Step 2: Enforce cohomology constraints
    eta = self._enforce_cohomology_constraints(eta)
    
    # Step 3: Verify constraints
    assert self._enforce_boundary_condition(eta), "Boundary condition not satisfied"
    
    return eta
```

## 4. Numerical Considerations

Several important aspects need attention:

1. **Discretization Effects**:
   - The boundary condition dη₀ = δ_Γ is only approximately satisfied
   - We need to choose appropriate grid resolution
   - The initial guess quality affects convergence

2. **Performance**:
   - The initial guess computation is O(N³) where N is grid resolution
   - We can optimize by using spatial data structures
   - Parallelization is possible for the vector field computation

3. **Stability**:
   - The correction for cohomology constraints must be carefully scaled
   - We need to ensure the vector field remains well-defined
   - Periodic boundary conditions must be handled correctly

## Next Steps

In the next post, we'll look at:
1. How this initial guess affects the ADMM convergence
2. Methods to improve the initial guess quality
3. Visualization of the initial vector field
4. Performance optimizations

Stay tuned for more implementation details! 