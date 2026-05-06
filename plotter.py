import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List

class QuantumPlotter:
    """
    Handles visualization of quantum trajectories and wavefunctions.
    """
    
    @staticmethod
    def plot_trajectories(df: pd.DataFrame, title: str = "Quantum Trajectory Reconstruction"):
        plt.figure(figsize=(12, 6))
        
        plt.plot(df['time'], df['true_pos'], label='True Expectation Value <x>', color='blue', linewidth=2)
        plt.scatter(df['time'], df['measurement'], label='Weak Measurements', color='gray', alpha=0.3, s=10)
        plt.plot(df['time'], df['inferred_pos'], label='Inferred Trajectory (PF)', color='red', linestyle='--')
        
        # Plot uncertainty band (1 sigma)
        std = np.sqrt(df['variance'])
        plt.fill_between(df['time'], df['inferred_pos'] - std, df['inferred_pos'] + std, color='red', alpha=0.2)
        
        plt.xlabel("Time")
        plt.ylabel("Position")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('trajectories.png')
        plt.show()

    @staticmethod
    def plot_probability_evolution(x_grid: np.ndarray, psi_history: List[np.ndarray], dt: float, save_every: int):
        plt.figure(figsize=(10, 6))
        
        history = np.array(psi_history)
        time_axis = np.arange(len(history)) * dt * save_every
        
        plt.imshow(
            history.T, 
            extent=[time_axis[0], time_axis[-1], x_grid[0], x_grid[-1]], 
            aspect='auto', 
            origin='lower',
            cmap='viridis'
        )
        plt.colorbar(label='|ψ(x,t)|²')
        plt.xlabel("Time")
        plt.ylabel("Position")
        plt.title("Wavefunction Probability Density Evolution")
        plt.tight_layout()
        plt.savefig('evolution.png')
        plt.show()
