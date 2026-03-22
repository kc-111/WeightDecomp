# BayesOptDTL

Bayesian Optimization framework for Design-Test-Learn (DTL) cycles using TabPFN v2 and Gaussian Processes.

## Features

- 🎯 **Flexible Objective Functions**: User-defined objective functions f(x) to maximize
- 🤖 **Multiple Surrogate Models**: TabPFN v2 (default) and Gaussian Processes
- 🔬 **Batch Experiments**: Propose batch_size=10 points simultaneously per cycle
- 📊 **Exploration & Exploitation**: Expected Improvement E[I(x)] where I(x) = max(f(x) - f* - xi, 0)
- 🧪 **Rich Benchmarks**: 7+ synthetic test functions (Ackley, Branin, Hartmann, etc.) - all set up for maximization
- 🔌 **Easy Integration**: Plug-and-play architecture for custom models
- ⬆️ **Always Maximizes**: Framework always maximizes f(x) - users negate if they need minimization

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd BayesOptDTL

# Install dependencies
pip install -e .

# Optional: Install TabPFN for TabPFN v2 support
pip install tabpfn
hf auth login  # Authenticate with HuggingFace
```

## Quick Start

### Basic Example

```python
from BayesOptDTL import DTLLoop, get_branin
from BayesOptDTL.models.gp_wrapper import GaussianProcessWrapper

# 1. Define objective
objective = get_branin(noise_std=0.1)

# 2. Create surrogate model
model = GaussianProcessWrapper()

# 3. Initialize DTL Loop
loop = DTLLoop(
    model=model,
    objective=objective,
    xi=0.1  # Exploration parameter
)

# 4. Run optimization
loop.initialize(n_initial=5)
for i in range(5):
    loop.run_cycle(batch_size=10)  # Sample 10 points per cycle

# 5. Get results
best_X, best_y = loop.get_best()
print(f"Best value: {best_y:.4f} at {best_X}")
```

### Custom Objective Function

```python
from BayesOptDTL import ObjectiveFunction, DTLLoop
import numpy as np

# Define your own function
def my_experiment(X):
    """Your custom objective function."""
    return np.sum(X**2, axis=1) + 0.1 * np.sin(10 * X[:, 0])

# Wrap it
objective = ObjectiveFunction(
    func=my_experiment,
    bounds=[(-5, 5), (-5, 5)],  # Search space
    noise_std=0.05,
    name="My Experiment"
)

# Use it in optimization
model = GaussianProcessWrapper()
loop = DTLLoop(model=model, objective=objective, xi=0.1)
loop.initialize(n_initial=5)
loop.run_cycle(batch_size=10)
```

### Using TabPFN v2

```python
from BayesOptDTL import TabPFNWrapper, get_ackley, DTLLoop

objective = get_ackley(dim=3)
model = TabPFNWrapper(device="cpu")  # Uses TabPFN v2 by default

loop = DTLLoop(model=model, objective=objective, xi=0.1)
loop.initialize(n_initial=10)

for i in range(5):
    loop.run_cycle(batch_size=10)
```

## Repository Structure

```
BayesOptDTL/
├── src/
│   └── BayesOptDTL/              # Main package
│       ├── __init__.py           # Package exports
│       ├── models/               # Surrogate models
│       │   ├── base.py          # BaseModelWrapper (abstract interface)
│       │   ├── tabpfn_wrapper.py # TabPFN v2 wrapper
│       │   └── gp_wrapper.py    # Gaussian Process wrapper
│       ├── acquisition/          # Acquisition functions
│       │   ├── ei.py            # Expected Improvement, UCB
│       │   └── multi_objective.py # Multi-objective utilities
│       ├── designers/            # Optimization strategies
│       │   └── acquisition_optimizer.py # Batch and single-point optimization
│       ├── benchmarks/          # Test functions
│       │   └── objectives.py   # Benchmark functions (Ackley, Branin, etc.)
│       └── loops/               # DTL orchestration
│           └── dtl_loop.py     # Main DTL loop
├── examples/                    # Example scripts
│   ├── quick_test.py           # Quick GP test
│   ├── test_gp_visualization.py # Comprehensive GP tests
│   ├── test_tabpfn.py          # TabPFN v2 tests
│   └── compare_gp_vs_tabpfn.py # Comparison script
├── tests/                       # Unit tests
├── notebooks/                   # Jupyter notebooks
├── pyproject.toml              # Package configuration
├── README.md                   # This file
└── TABPFN_SETUP.md            # TabPFN setup guide
```

## Available Benchmark Functions

- **Branin** (2D): Multi-modal function with 3 global minima
- **Ackley** (N-D): Highly multi-modal, tests global optimization
- **Hartmann3D** (3D): 4 local minima, 1 global minimum
- **Rosenbrock** (N-D): Narrow valley, tests local search
- **Sphere** (N-D): Simple convex function
- **Rastrigin** (N-D): Highly multi-modal with many local minima
- **Michalewicz** (N-D): Steep valleys and ridges

See `src/BayesOptDTL/benchmarks/objectives.py` for details.

## Examples

Run the examples:

```bash
# Activate virtual environment
source venv/bin/activate

# Quick test with GP
python examples/quick_test.py

# Full GP test with multiple benchmarks
python examples/test_gp_visualization.py

# Test with TabPFN v2 (requires: pip install tabpfn && hf auth login)
python examples/test_tabpfn.py

# Compare GP vs TabPFN v2
python examples/compare_gp_vs_tabpfn.py
```

See `examples/README.md` for detailed instructions.

## Creating Custom Models

Inherit from `BaseModelWrapper`:

```python
from BayesOptDTL.models.base import BaseModelWrapper
import numpy as np

class MyCustomModel(BaseModelWrapper):
    def fit(self, X, y):
        # Your fitting logic
        self._X_train = X
        self._y_train = y
    
    def predict_dist(self, X):
        # Return (mean, std)
        mean = ...  # Your predictions
        std = ...   # Your uncertainties
        return mean, std
```

## Citation

If you use this framework, please cite:

```bibtex
@software{bayesoptdtl2025,
  title={BayesOptDTL: Bayesian Optimization for Design-Test-Learn Cycles},
  author={Pak Lun Kevin Cheung},
  year={2025},
  url={https://github.com/your-repo}
}
```

## License

See LICENSE file.
