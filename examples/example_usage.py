#!/usr/bin/env python3
"""
Example usage of the coupled RL circuits simulation

This script demonstrates how to use the coupled RL circuits system
with different configurations and analysis options.
"""

import jax.numpy as jnp
import diffrax
import numpy as np

from coupled_rl_circuits import CoupledRLCircuitsPID, create_example_coupled_circuits
from rlcircuitpid import RLCircuitPID
from coupled_plotting import (
    prepare_coupled_post, plot_coupled_results, plot_region_analysis,
    analyze_coupling_effects, create_coupling_comparison_plot
)
from pid_controller import create_adaptive_pid_controller


def example_1_basic_coupled_system():
    """Example 1: Basic 3-circuit coupled system with default settings"""
    
    print("=== Example 1: Basic Coupled System ===")
    
    # Create a basic 3-circuit system
    coupled_system = create_example_coupled_circuits(n_circuits=3, coupling_strength=0.08)
    coupled_system.print_configuration()
    
    # Simulation parameters
    t0, t1, dt = 0.0, 5.0, 0.001
    y0 = coupled_system.get_initial_conditions()
    
    # Run simulation
    print("Running simulation...")
    solver = diffrax.Dopri5()
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(coupled_system.vector_field),
        solver,
        t0=t0, t1=t1, dt0=dt, y0=y0,
        stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-4),
        saveat=diffrax.SaveAt(ts=jnp.arange(t0, t1, dt)),
    )
    
    # Post-process and analyze
    t, results = prepare_coupled_post(sol, coupled_system)
    analyze_coupling_effects(coupled_system, t, results)
    plot_coupled_results(sol, coupled_system, t, results, 
                        show_coupling_analysis=True, show_individual_details=True)
    
    return coupled_system, sol, t, results


def example_2_custom_circuits():
    """Example 2: Custom circuit configurations with specific PID tuning"""
    
    print("\n=== Example 2: Custom Circuit Configurations ===")
    
    # Create custom circuits directly
    circuits = []
    
    # Circuit 1: High-performance, low inductance
    circuit1 = RLCircuitPID(
        R=0.8,
        L=0.05,
        temperature=20.0,
        circuit_id="high_performance",
        pid_controller=create_adaptive_pid_controller(
            Kp_low=25.0, Ki_low=18.0, Kd_low=0.12,
            Kp_medium=30.0, Ki_medium=22.0, Kd_medium=0.08,
            Kp_high=35.0, Ki_high=25.0, Kd_high=0.05,
            low_threshold=50.0, high_threshold=300.0
        )
    )
    circuits.append(circuit1)
    
    # Circuit 2: Standard configuration
    circuit2 = RLCircuitPID(
        R=1.2,
        L=0.1,
        temperature=25.0,
        circuit_id="standard",
        pid_controller=create_adaptive_pid_controller()  # Default PID
    )
    circuits.append(circuit2)
    
    # Circuit 3: Heavy-duty, high inductance
    circuit3 = RLCircuitPID(
        R=1.8,
        L=0.15,
        temperature=35.0,
        circuit_id="heavy_duty",
        pid_controller=create_adaptive_pid_controller(
            Kp_low=8.0, Ki_low=4.0, Kd_low=0.15,
            Kp_medium=12.0, Ki_medium=6.0, Kd_medium=0.10,
            Kp_high=15.0, Ki_high=8.0, Kd_high=0.08,
            low_threshold=80.0, high_threshold=600.0
        )
    )
    circuits.append(circuit3)
    
    # Create asymmetric coupling matrix
    mutual_inductances = np.array([
        [0.00, 0.06, 0.03],  # Circuit 1 couples strongly with 2, weakly with 3
        [0.06, 0.00, 0.08],  # Circuit 2 couples strongly with both
        [0.03, 0.08, 0.00]   # Circuit 3 couples weakly with 1, strongly with 2
    ])
    
    # Create coupled system
    coupled_system = CoupledRLCircuitsPID(circuits, mutual_inductances)
    coupled_system.print_configuration()
    
    # Run simulation
    t0, t1, dt = 0.0, 6.0, 0.001
    y0 = coupled_system.get_initial_conditions()
    
    solver = diffrax.Dopri5()
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(coupled_system.vector_field),
        solver,
        t0=t0, t1=t1, dt0=dt, y0=y0,
        stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-4),
        saveat=diffrax.SaveAt(ts=jnp.arange(t0, t1, dt)),
    )
    
    # Analysis
    t, results = prepare_coupled_post(sol, coupled_system)
    analyze_coupling_effects(coupled_system, t, results)
    plot_region_analysis(coupled_system, t, results)
    
    return coupled_system, sol, t, results


def example_3_coupling_strength_study():
    """Example 3: Study the effect of coupling strength"""
    
    print("\n=== Example 3: Coupling Strength Study ===")
    
    coupling_strengths = [0.0, 0.02, 0.05, 0.1, 0.2]
    results_list = []
    
    for coupling in coupling_strengths:
        print(f"\nSimulating with coupling strength: {coupling}")
        
        # Create system with specific coupling
        coupled_system = create_example_coupled_circuits(n_circuits=2, coupling_strength=coupling)
        
        # Quick simulation
        t0, t1, dt = 0.0, 4.0, 0.002
        y0 = coupled_system.get_initial_conditions()
        
        solver = diffrax.Dopri5()
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(coupled_system.vector_field),
            solver,
            t0=t0, t1=t1, dt0=dt, y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-4),
            saveat=diffrax.SaveAt(ts=jnp.arange(t0, t1, dt)),
        )
        
        t, results = prepare_coupled_post(sol, coupled_system)
        results_list.append((coupling, t, results))
        
        # Calculate performance metrics
        circuit1_error = results['circuit_1']['error']
        circuit2_error = results['circuit_2']['error']
        rms1 = float(jnp.sqrt(jnp.mean(circuit1_error**2)))
        rms2 = float(jnp.sqrt(jnp.mean(circuit2_error**2)))
        
        print(f"  Circuit 1 RMS error: {rms1:.4f} A")
        print(f"  Circuit 2 RMS error: {rms2:.4f} A")
    
    # Plot comparison for different coupling strengths
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    colors = plt.cm.viridis(np.linspace(0, 1, len(coupling_strengths)))
    
    for i, (coupling, t, results) in enumerate(results_list):
        color = colors[i]
        
        # Circuit 1 current
        axes[0, 0].plot(t, results['circuit_1']['current'], color=color, 
                       label=f'M = {coupling}', linewidth=1.5)
        
        # Circuit 2 current  
        axes[0, 1].plot(t, results['circuit_2']['current'], color=color,
                       label=f'M = {coupling}', linewidth=1.5)
        
        # Circuit 1 error
        axes[1, 0].plot(t, jnp.abs(results['circuit_1']['error']), color=color,
                       label=f'M = {coupling}', linewidth=1.5)
        
        # Circuit 2 error
        axes[1, 1].plot(t, jnp.abs(results['circuit_2']['error']), color=color,
                       label=f'M = {coupling}', linewidth=1.5)
    
    axes[0, 0].set_title('Circuit 1 Current')
    axes[0, 0].set_ylabel('Current (A)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Circuit 2 Current')
    axes[0, 1].set_ylabel('Current (A)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Circuit 1 |Error|')
    axes[1, 0].set_ylabel('|Error| (A)')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Circuit 2 |Error|')
    axes[1, 1].set_ylabel('|Error| (A)')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results_list


def example_4_uncoupled_vs_coupled():
    """Example 4: Compare uncoupled vs coupled behavior"""
    
    print("\n=== Example 4: Uncoupled vs Coupled Comparison ===")
    
    # Create two identical systems - one with coupling, one without
    coupled_system = create_example_coupled_circuits(n_circuits=2, coupling_strength=0.1)
    uncoupled_system = create_example_coupled_circuits(n_circuits=2, coupling_strength=0.0)
    
    # Same simulation parameters for both
    t0, t1, dt = 0.0, 5.0, 0.001
    
    print("Running coupled simulation...")
    y0_coupled = coupled_system.get_initial_conditions()
    solver = diffrax.Dopri5()
    sol_coupled = diffrax.diffeqsolve(
        diffrax.ODETerm(coupled_system.vector_field),
        solver, t0=t0, t1=t1, dt0=dt, y0=y0_coupled,
        stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-4),
        saveat=diffrax.SaveAt(ts=jnp.arange(t0, t1, dt)),
    )
    
    print("Running uncoupled simulation...")
    y0_uncoupled = uncoupled_system.get_initial_conditions()
    sol_uncoupled = diffrax.diffeqsolve(
        diffrax.ODETerm(uncoupled_system.vector_field),
        solver, t0=t0, t1=t1, dt0=dt, y0=y0_uncoupled,
        stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-4),
        saveat=diffrax.SaveAt(ts=jnp.arange(t0, t1, dt)),
    )
    
    # Post-process both results
    t_coupled, results_coupled = prepare_coupled_post(sol_coupled, coupled_system)
    t_uncoupled, results_uncoupled = prepare_coupled_post(sol_uncoupled, uncoupled_system)
    
    # Create comparison plot
    create_coupling_comparison_plot(
        results_uncoupled, results_coupled, t_coupled,
        ['circuit_1', 'circuit_2']
    )
    
    # Performance comparison
    print("\nPerformance Comparison:")
    for circuit_id in ['circuit_1', 'circuit_2']:
        uncoupled_rms = float(jnp.sqrt(jnp.mean(results_uncoupled[circuit_id]['error']**2)))
        coupled_rms = float(jnp.sqrt(jnp.mean(results_coupled[circuit_id]['error']**2)))
        
        print(f"  {circuit_id}:")
        print(f"    Uncoupled RMS error: {uncoupled_rms:.4f} A")
        print(f"    Coupled RMS error: {coupled_rms:.4f} A")
        print(f"    Relative change: {((coupled_rms - uncoupled_rms) / uncoupled_rms * 100):+.1f}%")
    
    return (coupled_system, sol_coupled, results_coupled), (uncoupled_system, sol_uncoupled, results_uncoupled)


def example_5_scalability_test():
    """Example 5: Test scalability with more circuits"""
    
    print("\n=== Example 5: Scalability Test ===")
    
    n_circuits_list = [2, 3, 5, 8]
    
    for n_circuits in n_circuits_list:
        print(f"\nTesting {n_circuits} circuits...")
        
        # Create system
        coupled_system = create_example_coupled_circuits(
            n_circuits=n_circuits, 
            coupling_strength=0.05
        )
        
        # Quick simulation
        t0, t1, dt = 0.0, 2.0, 0.005  # Shorter simulation for speed
        y0 = coupled_system.get_initial_conditions()
        
        print(f"  State vector size: {len(y0)}")
        print(f"  Mutual inductance matrix: {coupled_system.M.shape}")
        
        import time
        start_time = time.time()
        
        solver = diffrax.Dopri5()
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(coupled_system.vector_field),
            solver, t0=t0, t1=t1, dt0=dt, y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-4),
            saveat=diffrax.SaveAt(ts=jnp.arange(t0, t1, dt)),
        )
        
        simulation_time = time.time() - start_time
        print(f"  Simulation time: {simulation_time:.2f} seconds")
        print(f"  Time steps: {len(sol.ts)}")
        
        # Basic analysis
        t, results = prepare_coupled_post(sol, coupled_system)
        
        # Calculate average RMS error across all circuits
        total_rms = 0.0
        for circuit_id in coupled_system.circuit_ids:
            error = results[circuit_id]['error']
            rms = float(jnp.sqrt(jnp.mean(error**2)))
            total_rms += rms
        
        avg_rms = total_rms / n_circuits
        print(f"  Average RMS error: {avg_rms:.4f} A")


def run_all_examples():
    """Run all examples in sequence"""
    
    print("Running all coupled RL circuit examples...\n")
    
    try:
        # Example 1: Basic system
        example_1_basic_coupled_system()
        
        # Example 2: Custom configurations
        example_2_custom_circuits()
        
        # Example 3: Coupling strength study
        example_3_coupling_strength_study()
        
        # Example 4: Coupled vs uncoupled
        example_4_uncoupled_vs_coupled()
        
        # Example 5: Scalability
        example_5_scalability_test()
        
        print("\n✓ All examples completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error in examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # You can run individual examples or all of them
    import sys
    
    if len(sys.argv) > 1:
        example_num = int(sys.argv[1])
        
        if example_num == 1:
            example_1_basic_coupled_system()
        elif example_num == 2:
            example_2_custom_circuits()
        elif example_num == 3:
            example_3_coupling_strength_study()
        elif example_num == 4:
            example_4_uncoupled_vs_coupled()
        elif example_num == 5:
            example_5_scalability_test()
        else:
            print("Available examples: 1, 2, 3, 4, 5")
    else:
        # Run all examples
        run_all_examples()
