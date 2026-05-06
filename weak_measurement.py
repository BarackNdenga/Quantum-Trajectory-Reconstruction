import numpy as np
from typing import Optional

class WeakMeasurement:
    """
    Simulates a weak measurement of the position operator.
    The measurement outcome M_t is modeled as: M_t = <x> + noise.
    """
    
    def __init__(self, noise_std: float, seed: Optional[int] = None):
        self.noise_std = noise_std
        if seed is not None:
            np.random.seed(seed)
            
    def measure(self, true_position: float) -> float:
        """
        Returns a noisy measurement of the position.
        """
        noise = np.random.normal(0, self.noise_std)
        return true_position + noise

    def get_likelihood(self, particle_positions: np.ndarray, measurement: float) -> np.ndarray:
        """
        Calculates the likelihood of a measurement given particle positions.
        Assuming Gaussian noise: P(M|x) ~ exp(-(M-x)^2 / (2 * sigma^2))
        """
        return np.exp(-0.5 * ((measurement - particle_positions) / self.noise_std)**2)
