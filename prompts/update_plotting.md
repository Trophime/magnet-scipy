# Task: Complete CLI Integration and Update Refactored Main Functions

Now that we have a clean strategy-based plotting system, we need to integrate it fully with the refactored CLI components and update the main functions to use the new architecture.

## Current State

- ✅ Plotting strategies implemented with clean separation of concerns
- ✅ Backward compatibility maintained for existing plotting calls
- ✅ CLI core components refactored (ArgumentParser, ConfigurationLoader, etc.)
- ✅ Simulation strategies implemented for different simulation types
- ⚠️ `main_refactor.py` and `coupled_main_refactor.py` still use old plotting integration
- ⚠️ CLI simulation components need updates to use new plotting system

## Key Integration Points

### 1. Update `cli_simulation.py` Components

- Replace `ResultProcessor` plotting logic with `EnhancedPlottingManager`
- Simplify `PlottingManager` class to use new system
- Remove redundant `AnalyticsManager` logic (now integrated into plotting)
- Update `FileManager` to use new results saving system

### 2. Complete `main_refactor.py` and `coupled_main_refactor.py`

- Integrate `EnhancedPlottingManager` into the workflow
- Remove separate plotting/analytics steps (now unified)
- Add plotting configuration from command-line arguments
- Ensure proper error handling with new plotting system

### 3. Add Advanced CLI Features

- `--plot-config` option for custom plotting configurations
- `--comparison-mode` for comparing multiple simulation runs
- `--benchmark-plotting` for performance analysis
- Enhanced `--show-analytics` with new analytics system

## Specific Requirements

### 1. Update the workflow in `run_single_circuit_workflow()`

```python
# Current complex workflow:
orchestrator = SimulationOrchestrator()
processor = ResultProcessor() 
plotter = PlottingManager()
analytics = AnalyticsManager()
file_manager = FileManager()

# Should become:
orchestrator = SimulationOrchestrator()
plotting_manager = EnhancedPlottingManager(output_options, plot_config)
# Everything else handled automatically
```

### 2. Integrate plotting configuration

- Add command-line arguments for plotting customization
- Create `PlotConfiguration` from CLI args
- Support different plotting profiles (publication, presentation, debug)

### 3. Simplify error handling

- Unified error handling across simulation and plotting
- Clear error messages when plotting strategies fail
- Graceful degradation for missing experimental data

### 4. Add new CLI capabilities

- Comparison mode for running multiple configurations
- Plotting performance benchmarking
- Export capabilities for different formats

## Expected Deliverables

1. **Updated `cli_simulation.py`** with simplified components using new plotting
2. **Complete `main_refactor.py`** with integrated plotting workflow  
3. **Complete `coupled_main_refactor.py`** with integrated plotting workflow
4. **Enhanced command-line argument parsing** for plotting configuration
5. **Integration examples** showing before/after workflow comparison

## Constraints

- Maintain all existing CLI functionality and backward compatibility
- Keep the clean separation of concerns established in the refactor
- Ensure the workflow remains simple and easy to follow
- Add comprehensive error handling and user feedback

## Focus Areas

- Clean integration between simulation and plotting strategies
- Unified workflow that eliminates redundant steps
- Enhanced user experience with better CLI options
- Performance optimization for the complete workflow
- Clear documentation of the new integrated workflow

## Expected Outcome

This integration should result in a much cleaner main function workflow that eliminates the complex orchestration of separate plotting, analytics, and file management components while providing enhanced functionality through the new plotting system.

The final workflow should be:

```python
def run_single_circuit_workflow(args) -> int:
    """Simplified workflow with integrated plotting"""
    try:
        # Load configuration
        circuit = ConfigurationLoader.load_single_circuit(args.config_file)
        
        # Run simulation
        orchestrator = SimulationOrchestrator()
        simulation_result = orchestrator.run_single_circuit_simulation(...)
        
        # Process and plot (unified operation)
        plotting_manager = EnhancedPlottingManager(output_options, plot_config)
        processed_results, analytics = plotting_manager.create_plots_from_simulation_result(
            simulation_result, circuit, args.config_file
        )
        
        # Done! No separate plotting/analytics/saving steps needed
        return 0
        
    except Exception as e:
        return handle_error(e)
```

This represents a significant simplification from the current multi-step orchestration while providing enhanced functionality.