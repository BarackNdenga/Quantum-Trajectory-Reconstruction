import numpy as np
import pandas as pd
from typing import Dict, Any, List
from core.schrodinger_solver import SchrodingerSolver
from core.potentials import get_potential
from measurement.weak_measurement import WeakMeasurement
from inference.particle_filter import ParticleFilter

class QuantumExperiment:
    """
    Orchestrates the quantum simulation and trajectory reconstruction.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.solver = SchrodingerSolver(
            x_min=config['x_min'],
            x_max=config['x_max'],
            N=config['N_grid'],
            dt=config['dt']
        )
        
        # Setup Potential
        pot_func = get_potential(config['potential_type'], **config.get('potential_params', {}))
        self.solver.set_potential(pot_func)
        
        # Setup Initial State
        self.solver.set_initial_state(
            x0=config['x0'],
            p0=config['p0'],
            sigma=config['sigma']
        )
        
        # Setup Measurement and Inference
        self.measurer = WeakMeasurement(noise_std=config['meas_noise'], seed=config.get('seed'))
        self.pf = ParticleFilter(
            num_particles=config['num_particles'],
            x_range=(config['x_min'], config['x_max']),
            process_noise=config['process_noise'],
            seed=config.get('seed')
        )
        
        self.results: List[Dict[str, float]] = []
        self.psi_history: List[np.ndarray] = []

    def run(self, steps: int):
        """
        Runs the simulation loop.
        """
        for i in range(steps):
            t = i * self.config['dt']
            
            # 1. Quantum Evolution
            self.solver.step()
            true_pos = self.solver.get_expectation_x()
            
            # 2. Weak Measurement
            measurement = self.measurer.measure(true_pos)
            
            # 3. Bayesian Inference (Particle Filter)
            # In this simple model, drift is unknown to the filter (or zero)
            self.pf.predict(dt=self.config['dt'])
            likelihoods = self.measurer.get_likelihood(self.pf.particles, measurement)
            self.pf.update(likelihoods)
            inferred_pos = self.pf.estimate()
            
            # 4. Store Results
            self.results.append({
                'time': t,
                'true_pos': true_pos,
                'measurement': measurement,
                'inferred_pos': inferred_pos,
                'variance': self.pf.get_variance()
            })
            
            if i % self.config.get('save_psi_every', 10) == 0:
                self.psi_history.append(self.solver.get_probability_density().copy())

    def get_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)

    def save_data(self, path: str):
        df = self.get_dataframe()
        df.to_csv(path, index=False)
