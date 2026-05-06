import argparse
import numpy as np
from simulation.experiment import QuantumExperiment
from visualization.plotter import QuantumPlotter

def main():
    parser = argparse.ArgumentParser(description="Quantum Trajectory Reconstruction Prototype")
    parser.add_argument("--potential", type=str, default="harmonic", choices=["free", "harmonic"])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--noise", type=float, default=0.5, help="Measurement noise level")
    parser.add_argument("--particles", type=int, default=1000, help="Number of particles for PF")
    parser.add_argument("--neural", action="store_true", help="Enable bonus neural network inference")
    args = parser.parse_args()

    # Simulation Configuration
    config = {
        'x_min': -10.0,
        'x_max': 10.0,
        'N_grid': 512,
        'dt': 0.02,
        'potential_type': args.potential,
        'potential_params': {'omega': 1.0} if args.potential == "harmonic" else {},
        'x0': -2.0,
        'p0': 2.0,
        'sigma': 0.5,
        'meas_noise': args.noise,
        'num_particles': args.particles,
        'process_noise': 0.05,
        'seed': 42,
        'save_psi_every': 10
    }

    print(f"Starting Quantum Simulation: {args.potential} potential, {args.steps} steps...")
    
    # Initialize and Run Experiment
    experiment = QuantumExperiment(config)
    experiment.run(args.steps)
    
    print("Simulation complete. Generating plots...")
    
    # Get results
    df = experiment.get_dataframe()
    experiment.save_data("data/simulation_results.csv")
    
    # Bonus: Neural Inference
    if args.neural:
        print("\nTraining Neural Inference Engine...")
        from inference.neural_inference import NeuralInferenceEngine
        engine = NeuralInferenceEngine()
        engine.train_on_simulation(df['measurement'].values, df['true_pos'].values)
        df['neural_pos'] = engine.predict(df['measurement'].values)
        print("Neural inference complete.")

    # Plotting
    plotter = QuantumPlotter()
    
    # Custom plotting to include neural if requested
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(df['time'], df['true_pos'], label='True Expectation Value <x>', color='blue', linewidth=2)
    plt.scatter(df['time'], df['measurement'], label='Weak Measurements', color='gray', alpha=0.3, s=10)
    plt.plot(df['time'], df['inferred_pos'], label='Inferred Trajectory (PF)', color='red', linestyle='--')
    if 'neural_pos' in df.columns:
        plt.plot(df['time'], df['neural_pos'], label='Neural Reconstruction (GRU)', color='green', linestyle='-.')
    
    std = np.sqrt(df['variance'])
    plt.fill_between(df['time'], df['inferred_pos'] - std, df['inferred_pos'] + std, color='red', alpha=0.2)
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.title(f"Quantum Trajectory ({args.potential} potential)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('trajectories.png')
    plt.show()
    plotter.plot_probability_evolution(
        experiment.solver.x, 
        experiment.psi_history, 
        config['dt'], 
        config['save_psi_every']
    )
    
    print("\n--- Summary ---")
    print(f"Data saved to data/simulation_results.csv")
    print(f"Plots saved as 'trajectories.png' and 'evolution.png'")
    print("Quantum OS Powered by © Stillmind Hub.")

if __name__ == "__main__":
    main()
