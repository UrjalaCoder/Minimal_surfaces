import numpy as np

def ellipse_function(t, a=1, b=1, base_point=np.array([0, 0, 0])):
    z = base_point[-1]
    return base_point + np.array([a * np.cos(2 * np.pi * t), b * np.sin(2 * np.pi * t), 0])

def generate_periodic_helicoid_curve(t, num_turns=3, radius=1.0, domain_height=1.0):
    """
    Generates a periodic helicoid curve that wraps around a periodic domain.
    
    Parameters:
        num_points (int): Number of points in the curve.
        num_turns (int): Number of full turns.
        radius (float): Radius of the helicoid.
        domain_height (float): Height of the periodic domain (default: 1.0).
    
    Returns:
        np.ndarray: A (num_points, 3) array of (x, y, z) coordinates.
    """
    x = radius * np.cos(2 * np.pi * t * num_turns)
    y = radius * np.sin(2 * np.pi * t * num_turns)
    
    # Enforce periodicity: adjust pitch so that z wraps from 0 to domain_height
    pitch = domain_height
    z = pitch * t  # Ensures z ends at 1 when it started at 0
    
    return np.array([x, y, z])
