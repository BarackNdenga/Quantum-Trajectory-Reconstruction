import numpy as np
from typing import Callable

def free_particle(x: np.ndarray) -> np.ndarray:
    """
    Free particle potential (V=0).
    
    Args:
        x: Position grid.
        
    Returns:
        Zero potential array.
    """
    return np.zeros_like(x)

def harmonic_oscillator(x: np.ndarray, omega: float = 1.0, m: float = 1.0) -> np.ndarray:
    """
    Harmonic oscillator potential: V(x) = 0.5 * m * omega^2 * x^2.
    
    Args:
        x: Position grid.
        omega: Angular frequency.
        m: Mass of the particle.
        
    Returns:
        Potential energy array.
    """
    return 0.5 * m * (omega**2) * (x**2)

def get_potential(name: str, **kwargs) -> Callable[[np.ndarray], np.ndarray]:
    """
    Factory function for potentials.
    """
    if name == "free":
        return free_particle
    elif name == "harmonic":
        return lambda x: harmonic_oscillator(x, **kwargs)
    else:
        raise ValueError(f"Unknown potential: {name}")
