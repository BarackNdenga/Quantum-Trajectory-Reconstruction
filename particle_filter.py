import numpy as np
from typing import Optional, Callable

class ParticleFilter:
    """
    Sequential Monte Carlo (Particle Filter) for quantum trajectory reconstruction.
    """
    
    def __init__(
        self,
        num_particles: int,
        x_range: tuple[float, float],
        process_noise: float,
        seed: Optional[int] = None
    ):
        if seed is not None:
            np.random.seed(seed)
            
        self.N = num_particles
        self.particles = np.random.uniform(x_range[0], x_range[1], size=num_particles)
        self.weights = np.ones(num_particles) / num_particles
        self.process_noise = process_noise
        
    def predict(self, drift_func: Optional[Callable[[np.ndarray], np.ndarray]] = None, dt: float = 0.01):
        """
        Moves particles based on system dynamics (drift) and process noise.
        """
        if drift_func:
            # Simplistic drift (e.g., classical velocity or expected quantum drift)
            self.particles += drift_func(self.particles) * dt
            
        # Add diffusion/process noise to represent uncertainty in evolution
        self.particles += np.random.normal(0, self.process_noise, size=self.N)

    def update(self, likelihoods: np.ndarray):
        """
        Updates particle weights based on the measurement likelihood.
        """
        self.weights *= likelihoods
        self.weights += 1e-300  # Avoid division by zero
        self.weights /= np.sum(self.weights)
        
        # Effective sample size check for resampling
        if 1.0 / np.sum(self.weights**2) < self.N / 2:
            self.resample()

    def resample(self):
        """
        Systematic resampling of particles.
        """
        indices = np.random.choice(np.arange(self.N), size=self.N, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.N) / self.N

    def estimate(self) -> float:
        """
        Returns the weighted mean of the particles as the current estimate.
        """
        return float(np.sum(self.particles * self.weights))

    def get_variance(self) -> float:
        """
        Returns the weighted variance of the particles.
        """
        mean = self.estimate()
        return float(np.sum(self.weights * (self.particles - mean)**2))
