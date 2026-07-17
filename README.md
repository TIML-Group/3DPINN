#  Dimension–Domain Co-Decomposition (3D) 

This repository contains various PDE benchmarks for 3D framework.
## Project Structure

```
src/
├── ac_3d.py                    # Allen-Cahn equation example
├── ADR/
│   ├── Oscillatory_Decay.py    # Oscillatory decay ADR example
│   └── Traveling_Wave.py       # Traveling wave ADR example
├── Burgers/
│   ├── burgers.py              # Viscous Burgers equation example
│   └── burgers_shock.mat       # Data for Burgers' equation
├── Poissons/
│   ├── poisson.py              # Sine-Product Poisson equation examples
│   └── synthetic_poisson.py    # Diagonal_Shock Poisson example
└── Waves/
    ├── wave.py                 # 1D Wave equation example
    └── wave_2d.py              # 2D Wave equation example

tools/
├── model.py                    # Models and architectures
├── utils.py                    # Utility functions 
└── vi.py                       # VI computations
```

## Overview

This project demonstrates the effectiveness and feasibility of 3D to solve various PDEs. Each example implements a different PDE problem using single expert or full 3D models. VI calculations are also included.

## Key Components

### 1. Model Architecture (`model.py`)
- **MLP**: Multi-layer Perceptron router
- **Expert Models**: Expert for specialized region processing
- **DomainMoE**: Mixture of Experts model for domain decomposition

### 2. Utility Functions (`utils.py`)
- **Sampling functions**: For generating interior, boundary, and initial condition points
- **Visualization tools**: For plotting solutions and errors
- **Inference functions**: For evaluating trained models

### 3. Analysis Tools (`vi.py`)
- **VI calculations**: For evaluating per-dimensional interpretability



## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- SciPy

## Usage

Each example can be run independently:

```bash
cd random
python poisson.py      
python wave.py        
python wave_2d.py     
python burgers.py     
python Oscillatory_Decay.py  
python Traveling_Wave.py    
python ac_3d.py      
python combine.py     
```

## Results

Each script will:
1. Train a PINN model to solve the specified PDE
2. Generate visualizations of the solution
3. Save results and error metrics

The models use a combination of Adam optimizer for initial training and L-BFGS for fine-tuning, with automatic differentiation to compute PDE residuals.


## License

This project is licensed under the MIT License. See the LICENSE file for details.
