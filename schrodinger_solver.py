import numpy as np
from typing import Callable, Optional

class SchrodingerSolver:
    """
    1D Time-Dependent Schrodinger Equation solver using the Split-Operator FFT method.
    
    This method evolves the wavefunction psi(x, t) by alternating between 
    position space (potential operator) and momentum space (kinetic operator).
    """
    
    def __init__(
        self,
        x_min: float,
        x_max: float,
        N: int,
        dt: float,
        m: float = 1.0,
        hbar: float = 1.0
    ):
        self.x = np.linspace(x_min, x_max, N)
        self.dx = self.x[1] - self.x[0]
        self.N = N
        self.dt = dt
        self.m = m
        self.hbar = hbar
        
        # Momentum grid
        self.k = 2 * np.pi * np.fft.fftfreq(N, d=self.dx)
        
        # Precompute kinetic operator in momentum space
        self.exp_K = np.exp(-1j * self.hbar * (self.k**2) * (self.dt / (2 * self.m)))
        
        self.psi = np.zeros(N, dtype=complex)
        self.V = np.zeros(N)

    def set_initial_state(self, x0: float, p0: float, sigma: float):
        """
        Sets a Gaussian wavepacket as the initial state.
        psi(x) = (2*pi*sigma^2)^(-1/4) * exp(-(x-x0)^2 / (4*sigma^2)) * exp(i*p0*x/hbar)
        """
        norm = (2 * np.pi * sigma**2)**(-0.25)
        self.psi = norm * np.exp(-((self.x - x0)**2) / (4 * sigma**2)) * np.exp(1j * p0 * self.x / self.hbar)
        self.normalize()

    def set_potential(self, potential_func: Callable[[np.ndarray], np.ndarray]):
        """Sets the potential energy array."""
        self.V = potential_func(self.x)
        # Precompute potential operator in position space
        self.exp_V = np.exp(-1j * self.V * self.dt / self.hbar)

    def normalize(self):
        """Ensures the wavefunction is normalized to 1."""
        norm = np.sqrt(np.sum(np.abs(self.psi)**2) * self.dx)
        self.psi /= norm

    def step(self):
        """
        Performs a single time step using the split-operator method.
        exp(-iHdt) approx exp(-iVdt/2) * exp(-iKdt) * exp(-iVdt/2)
        """
        # Half-step potential
        self.psi *= np.exp(-1j * self.V * (self.dt / 2.0) / self.hbar)
        
        # Full-step kinetic (via FFT)
        phi_k = np.fft.fft(self.psi)
        phi_k *= self.exp_K
        self.psi = np.fft.ifft(phi_k)
        
        # Half-step potential
        self.psi *= np.exp(-1j * self.V * (self.dt / 2.0) / self.hbar)
        
        self.normalize()

    def get_probability_density(self) -> np.ndarray:
        return np.abs(self.psi)**2

    def get_expectation_x(self) -> float:
        return float(np.sum(self.x * self.get_probability_density()) * self.dx)
