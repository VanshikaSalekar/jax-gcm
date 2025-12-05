# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

JAX-GCM is a JAX-based General Circulation Model that combines the Dinosaur dynamical core with JAX implementations of atmospheric physics parameterizations. It provides a fully differentiable climate model suitable for ML-enhanced weather and climate modeling.

## Key Components

### Architecture
- **Dynamical Core**: Uses [Dinosaur](https://github.com/neuralgcm/dinosaur) for atmospheric dynamics
- **Physics Packages**:
  - **SPEEDY Physics**: Complete JAX implementation of SPEEDY atmospheric physics
  - **ICON Physics**: New JAX implementation of ICON parameterizations (active development)
  - **Held-Suarez**: Simplified physics for testing
- **Configuration**: Hydra-based configuration system with YAML files

### Project Structure
```
jcm/                      # Main package (JAX Climate Model)
├── physics/              # Physics implementations
│   ├── speedy/          # SPEEDY physics modules
│   ├── icon/            # ICON physics (radiation, convection, clouds, etc.)
│   └── held_suarez/     # Simplified physics
├── model.py             # Main model class
├── physics_interface.py # Physics abstraction layer
├── geometry.py          # Grid and coordinate handling
├── date.py              # Time/date management
└── forcing.py           # Surface boundary conditions
```

## Development Commands

### Installation
```bash
pip install -e .  # Development mode
```

### Testing
```bash
# Run all tests
pytest

# Run specific module tests
pytest jcm/physics/icon/ -v

# Run with coverage (90% minimum required)
pytest --cov=jcm --cov-fail-under=90

# Skip slow tests
pytest -m "not slow"

# Run single test
pytest path/to/test_file.py::test_function_name -v
```

### Linting
```bash
# Uses Ruff (configured for Python 3.11)
ruff --format=github --target-version=py311 .
```

### Documentation
```bash
cd docs && make html
```

## JAX Development Patterns

### Critical JAX Rules
1. **No Python control flow on JAX arrays** - Use JAX alternatives:
   - `if/else` → `jnp.where` or `lax.cond`
   - `for` loops → `lax.scan` or vectorized operations
   - `while` loops → `lax.while_loop`

2. **Static shapes required** - All array shapes must be known at compile time

3. **Pure functions only** - No side effects or stateful operations

4. **Vectorization pattern** - Use `vmap` for spatial operations:
   ```python
   # Centralized vectorization in physics modules
   tendency_fn = jax.vmap(compute_tendency, in_axes=(0, None))
   ```

### Common Conversions
See `JAX_CONVERSION_PATTERNS.md` and `JAX_gotchas.md` for detailed patterns:
- Replace `np.maximum(0, x)` with `jax.nn.relu(x)`
- Replace boolean indexing with `jnp.where`
- Use `lax.scan` for accumulation over sequences
- Use `lax.cond` with lambda functions to avoid eager evaluation
- Implement custom gradients with `jax.custom_vjp`

## Testing Best Practices

1. **Use real objects instead of mocks**:
   - Create `DateData` with: `jcm.date.DateData(...)`
   - Create geometry with: `jcm.geometry.Geometry.from_grid_shape(...)`
   - These objects are JAX-compatible and easy to instantiate

2. **Test JAX transformations**:
   - Test `jax.jit` compilation
   - Test `jax.grad` for differentiability
   - Test `jax.vmap` for vectorization

3. **Coverage requirements**: Maintain >90% test coverage

## Physics Implementation Guidelines

### Physics Interface
All physics modules implement this protocol:
```python
class ExamplePhysics(Physics):
    def compute_tendencies(
        self,
        state: PhysicsState,
        forcing: ForcingData,
        geometry: Geometry,
        date: DateData,
    ) -> Tuple[PhysicsTendency, PhysicsData]:
       ...
```

Each major physics term typically has a method like:
```python
@jit
def _apply_radiation(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    forcing: ForcingData,
    geometry: Geometry
) -> tuple[PhysicsTendency, PhysicsData]:
   ...
```

### ICON Physics Structure
The ICON physics package (`jcm/physics/icon/`) includes:
- **Radiation**: Shortwave and longwave schemes
- **Convection**: Deep and shallow convection (Tiedtke-Nordeng scheme)
- **Clouds**: Cloud microphysics and cover
- **Vertical Diffusion**: Turbulent mixing
- **Surface**: Land-atmosphere interactions
- **Gravity Waves**: Orographic and non-orographic
- **Chemistry**: Chemical tracers (if enabled)
- **Aerosol**: Aerosol interactions (if enabled)

Each module follows:
1. Modular design with clear interfaces
2. Full JAX compatibility (autodiff, JIT, vmap)
3. Comprehensive test coverage (33+ test files in physics)
4. Detailed documentation

### Unit Conversions
See `jcm/physics/icon/UNIT_CONVERSIONS.md` for details on converting between:
- Physics interface units (normalized surface pressure, geopotential)
- ICON expected units (Pa, meters, kg/m³)

Key conversions handled automatically:
- Surface pressure: `normalized * p0` (p0 = 100000 Pa)
- Pressure levels: `sigma * surface_pressure_pa`
- Height: `geopotential / g`
- Air density: `pressure / (Rd * temperature)`

### State Management
- `PhysicsState`: Immutable atmospheric state (u, v, T, q, φ, ps)
- `PhysicsTendency`: Time derivatives of state variables
- Tree-structured data using `tree_math` for arithmetic operations

## Model Interface

### Constructor vs Run Method

**Model Configuration (Constructor)**: Physics and geometry settings are specified when creating the model:
```python
from jcm.model import Model
from jcm.physics.speedy import SpeedyPhysics
from jcm.physics.icon import IconPhysics

# Configure model with physics and geometry
model = Model(
    time_step=30.0,              # Model timestep in minutes
    layers=8,                    # Vertical layers
    horizontal_resolution=31,    # Spectral resolution
    physics=SpeedyPhysics(),     # Physics package
    use_hybrid_coords=False      # Coordinate system (auto-detected from physics)
)
```

**Simulation Parameters (Run Method)**: Run-specific settings are passed to `model.run()`:
```python
# Run simulation with specific parameters
predictions = model.run(
    initial_state=None,                    # Optional initial state
    forcing=None,                          # Boundary conditions (default aquaplanet)
    save_interval=10.0,                    # Save interval in days
    total_time=120.0,                      # Total simulation time in days
    start_date=Timestamp.from_datetime(datetime(2000, 1, 1))
)
```

### Coordinate System Auto-Detection
The model automatically detects coordinate systems:
- **ICON Physics** → Hybrid sigma-pressure coordinates
- **SPEEDY Physics** → Pure sigma coordinates
- Override with `use_hybrid_coords=True/False`

### Resume Capability
Use `model.resume()` to continue from where `run()` left off:
```python
# Continue simulation
more_predictions = model.resume(
    save_interval=10.0,
    total_time=60.0  # Additional 60 days
)
```

## Dependencies and Requirements

- **Python**: 3.11+ (required for jax-solar)
- **Core**: JAX, Dinosaur, tree-math
- **Configuration**: hydra-core
- **I/O**: xarray, netCDF4
- **Testing**: pytest, pytest-cov
- **Documentation**: Sphinx with Furo theme

## Validation Framework

The `validation/` directory contains comprehensive validation tests:

### Emergent Properties Validation
Tests fundamental atmospheric phenomena (ITCZ, Hadley cells, energy balance):
```bash
# Run comprehensive validation
python validation/emergent_climate_validation.py --duration 30 --output results/emergent_validation.json

# Quick test
python validation/test_emergent_climate.py --quick
```

### Test Cases
- Tropical convection (0°N, 180°E)
- Mid-latitude winter (50°N, 0°E)
- Arctic polar (85°N, 0°E)
- Subtropical clear (30°N, 30°W)

See `validation/README.md` for detailed validation framework documentation.
