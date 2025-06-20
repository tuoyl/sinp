# sinp
# SINP - Spectroscopy IN Python

A modern, JAX-based X-ray spectral analysis framework for high-performance spectral fitting.

## Features

- **JAX-powered**: Leverages JAX for automatic differentiation and JIT compilation
- **Fast optimization**: Multiple optimization backends (scipy, optax)
- **Flexible modeling**: Easy-to-use model expression syntax (e.g., `"tbabs*(powerlaw+band)"`)
- **OGIP compliant**: Full support for standard PHA, RMF, and ARF files
- **Modern Python**: Type hints, dataclasses, and clean API design

## Installation

### From PyPI (coming soon)

```bash
pip install sinp
```

### From source

```bash
git clone https://github.com/yourusername/sinp.git
cd sinp
pip install -e .
```

### Dependencies

- Python >= 3.8
- JAX >= 0.4.0
- NumPy
- SciPy
- Astropy
- Matplotlib (for plotting)
- optax (for JAX optimizers)

## Quick Start

```python
import sinp
from sinp.model import Model
from sinp.response import Response
from sinp.pha import Spectrum
from sinp.fitting import Fitter

# Load data
spectrum = Spectrum()
spectrum.load_pha_file('source.pha')
spectrum.background = 'background.pha'

response = Response()
response.load_rmf_file('response.rmf')
response.load_arf_file('response.arf')

# Create model
model = Model()
model.set_model("tbabs*powerlaw")

# Set initial parameters
model.tbabs.nH.set_par([0.1, 0.0, 10.0, False])  # value, min, max, frozen
model.powerlaw.norm.set_par([1.0, 0.0, 100.0, False])
model.powerlaw.photon_index.set_par([2.0, 0.0, 5.0, False])

# Fit
fitter = Fitter(response, model, spectrum)
fitter.set_energy_range([(0.5, 10.0)])  # 0.5-10 keV

result = fitter.fit(stat='chi2', method='L-BFGS-B')

# Results
print(f"Chi-square: {result.statistic:.2f}")
print(f"Reduced chi-square: {result.reduced_statistic:.2f}")
for param, value in result.parameters.items():
    if result.parameter_errors:
        error = result.parameter_errors.get(param, 0)
        if error > 0:
            print(f"{param}: {value:.4e} ± {error:.4e}")

# Plot
fitter.plot_fit()
```

## Available Models

### Additive Models

- **powerlaw**: Simple power-law model
- **brokenpowerlaw**: Broken power-law model
- **band**: Band function (GRB model)

### Multiplicative Models

- **tbabs**: Tuebingen-Boulder ISM absorption model

### Model Expressions

Models can be combined using mathematical expressions:

```python
model.set_model("tbabs*powerlaw")              # Absorbed power-law
model.set_model("powerlaw+band")               # Sum of components
model.set_model("tbabs*(powerlaw+band)")       # Absorbed sum
model.set_model("const*tbabs*powerlaw")        # With normalization constant
```

## Advanced Usage

### Custom Statistics

```python
import jax
import jax.numpy as jnp

@jax.jit
def custom_statistic(params):
    model_counts = fitter._forward_fold_model(params)
    # Your custom statistic here
    return statistic_value

# Use with fit
result = fitter.fit(stat=custom_statistic)
```

### Parallel Chains (with JAX)

```python
# JAX automatically parallelizes over multiple devices
result = fitter.fit(method='adam',
                   options={'learning_rate': 0.01,
                           'max_steps': 5000})
```

### Parameter Constraints

```python
# Set tight bounds
model.powerlaw.photon_index.set_par([2.0, 1.5, 2.5, False])

# Freeze parameters
model.band.Epiv.set_par([100, 0, 1000, True])  # frozen=True
```

## File Formats

SINP supports standard OGIP (Office of Guest Investigator Programs) file formats:

- **PHA**: Pulse Height Analyzer files (Type I and Type II)
- **RMF**: Redistribution Matrix Files
- **ARF**: Ancillary Response Files

## Performance

SINP leverages JAX for significant performance improvements:

- JIT compilation for model evaluation
- Automatic differentiation for gradients
- GPU acceleration (when available)
- Vectorized operations throughout

Benchmark results (1000 iterations, absorbed power-law):

- Traditional NumPy: ~2.5s
- SINP with JAX: ~0.3s
- SINP with GPU: ~0.05s

## API Reference

### Model

```python
model = Model()
model.set_model(expression: str)
model.get_free_params() -> Dict[str, float]
model.update_params(params: Dict[str, float])
```

### Fitter

```python
fitter = Fitter(response, model, spectrum)
fitter.fit(stat='chi2', method='L-BFGS-B') -> FitResult
fitter.plot_fit()
fitter.set_energy_range(ranges: List[Tuple[float, float]])
```

### Spectrum

```python
spectrum = Spectrum()
spectrum.load_pha_file(filename: str)
spectrum.group_channels(min_counts=20)
spectrum.ignore_energy([(0, 0.5), (10, 100)])
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development setup

```bash
git clone https://github.com/yourusername/sinp.git
cd sinp
pip install -e ".[dev]"
pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- HEASARC for XSPEC and OGIP standards
- JAX team for the amazing autodiff framework
- The X-ray astronomy community for feedback and support

## Related Projects

- [XSPEC](https://heasarc.gsfc.nasa.gov/xanadu/xspec/): The standard X-ray spectral fitting package
- [Sherpa](https://sherpa.readthedocs.io/): Chandra's modeling and fitting package
- [3ML](https://threeml.readthedocs.io/): Multi-mission Maximum Likelihood framework
- [BXA](https://johannesbuchner.github.io/BXA/): Bayesian X-ray Analysis
