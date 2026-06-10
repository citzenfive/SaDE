# Self-Adaptive Differential Evolution (SaDE)

A Python implementation of **Self-Adaptive Differential Evolution (SaDE)** for bounded, single-objective numerical optimization.

The implementation combines multiple Differential Evolution mutation strategies and updates their selection probabilities according to their observed success during the optimization. It also supports Sobol initialization, multiprocessing-based parallel evaluation, reproducible random number generation, automatic result saving, early stopping, and configurable boundary handling.

> This implementation is inspired by SaDE and related adaptive Differential Evolution methods. It is not intended to be a line-by-line reproduction of a single reference algorithm.

---

## Features

- Single-objective minimization
- Multiple Differential Evolution strategies
- Adaptive strategy probabilities
- Adaptive crossover-rate memory
- Sobol low-discrepancy initialization
- Parallel fitness evaluation through a custom `map` function
- Reproducible runs using a single random seed
- Automatic creation of output directories
- Reflection or clipping for box-constraint handling
- Early stopping based on:
  - target fitness;
  - fitness standard deviation;
  - generations without improvement
- Hall of Fame with the best individuals found
- Periodic logging and population snapshots
- Compatible with scalar or one-element tuple fitness values
- Avoids duplicate evaluation of the initial population

---

## Supported Differential Evolution Strategies

The strategy pool may contain the following operators:

- `DE/rand/1/bin`
- `DE/rand/2/bin`
- `DE/best/1/bin`
- `DE/rand-to-best/2/bin`
- `DE/current-to-rand/1/bin`
- `DE/current-to-pbest/1/bin`

Strategies that require more individuals than the configured population size are automatically removed from the active strategy pool.

---

## Repository Structure

A typical repository layout is:

```text
.
├── SaDE.py
├── mut_cross.py
├── main.py
├── run_benchmark.py
├── requirements.txt
├── README.md
└── tests/
```

### File descriptions

- `SaDE.py`: main optimizer implementation.
- `mut_cross.py`: mutation and binomial crossover strategies.
- `main.py`: command-line entry point.
- `run_benchmark.py`: objective or benchmark function used by the optimizer.
- `requirements.txt`: Python dependencies.
- `tests/`: output directories, tests, or example experiments.

---

## Requirements

- Python 3.10 or newer
- NumPy
- SciPy
- DEAP

The objective function may require additional packages depending on the model being optimized.

---

## Installation

Clone the repository:

```bash
git clone git@github.com:citzenfive/SaDE.git
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
python -m pip install --upgrade pip
python -m pip install numpy scipy deap
```

Alternatively, create a `requirements.txt` file containing:

```text
numpy
scipy
deap
```

Then run:

```bash
python -m pip install -r requirements.txt
```

---

## Objective Function

The optimizer minimizes the value returned by the evaluation function.

The evaluation function must:

1. receive one individual;
2. be defined at module level when multiprocessing is used;
3. return either a scalar or a one-element tuple;
4. avoid modifying the individual unless that behavior is intentional.

Example:

```python
def sphere(individual):
    value = sum(parameter**2 for parameter in individual)
    return (value,)
```

A scalar is also accepted:

```python
def sphere(individual):
    return sum(parameter**2 for parameter in individual)
```

For multiprocessing, avoid nested or local functions:

```python
# Correct: defined at module level
def objective_function(individual):
    return (sum(x**2 for x in individual),)
```

---

## Basic Sequential Usage

```python
from SaDE import SaDE


def sphere(individual):
    return (sum(parameter**2 for parameter in individual),)


bounds = [(-5.0, 5.0)] * 6

optimizer = SaDE(
    EVALUATION_FUNCTION=sphere,
    NUM_PAR=6,
    BOUNDS=bounds,
    POP_SIZE=150,
    MAX_GEN=1000,
    SAVE_PATH="results/sphere/",
    SEED=1234,
)

optimizer.print_config()
best = optimizer.run_SaDE()

print("Best parameters:", list(best))
print("Best fitness:", best.fitness.values[0])
```

---

## Parallel Usage

Fitness evaluation can be distributed across multiple processes by passing `pool.map` through `PARALLEL_MAP_FUNCTION`.

```python
import multiprocessing

from SaDE import SaDE
from run_benchmark import neut_bac_model


def main() -> None:
    bounds = [
        (1.0e-5, 1.0),
        (1.0e-5, 1.0),
        (1.0e-5, 1.0),
        (1.0e-5, 1.0),
        (1.0e-5, 1.0),
        (1.0e-5, 1.0),
    ]

    with multiprocessing.Pool(processes=8) as pool:
        optimizer = SaDE(
            EVALUATION_FUNCTION=neut_bac_model,
            PARALLEL=True,
            PARALLEL_MAP_FUNCTION=pool.map,
            NUM_PAR=6,
            BOUNDS=bounds,
            POP_SIZE=256,
            MAX_GEN=2500,
            SAVE_PATH="tests/neut_bact/",
            SEED=1234,
        )

        optimizer.print_config()
        optimizer.save_config()
        best = optimizer.run_SaDE()

    print("Best parameters:", list(best))
    print("Best fitness:", best.fitness.values[0])


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
```

The `if __name__ == "__main__":` guard is required for safe multiprocessing behavior, especially on operating systems that use the `spawn` start method.

`multiprocessing.freeze_support()` is mainly useful on Windows or when generating a standalone executable. It is harmless in regular Python execution and can be omitted on Linux-only projects.

### Important parallelism note

Setting:

```python
PARALLEL=True
```

records that the run is parallel, but the actual parallel execution is enabled by:

```python
PARALLEL_MAP_FUNCTION=pool.map
```

Without a custom parallel map function, DEAP uses the standard sequential `map`.

---

## Command-Line Interface

Run the optimizer with:

```bash
python main.py \
    --n_processes 8 \
    --save_path tests/neut_bact/ \
    --population_size 256 \
    --max-gen 2500
```

Run with a fixed seed:

```bash
python main.py \
    --n_processes 8 \
    --save_path tests/neut_bact/ \
    --population_size 256 \
    --max-gen 2500 \
    --seed 1234
```

Show the available options:

```bash
python main.py --help
```

---

## Main Configuration Parameters

| Parameter | Type | Default | Description |
|---|---:|---:|---|
| `EVALUATION_FUNCTION` | callable | required | Objective function to minimize. |
| `const_LP` | `int` | `50` | Learning period used to update strategy probabilities and crossover memory. |
| `POP_SIZE` | `int` | `150` | Number of individuals in the population. |
| `MAX_GEN` | `int` | `2500` | Maximum number of generations. |
| `PARALLEL` | `bool` | `False` | Records whether parallel execution is intended. |
| `SEED` | `int \| None` | `None` | Random seed. A new seed is generated when omitted. |
| `SAVE_PATH` | `str` | `"."` | Output directory. It is created automatically. |
| `PARALLEL_MAP_FUNCTION` | callable or `None` | `None` | Map-compatible function such as `pool.map`. |
| `BOUNDS` | sequence | automatic | Lower and upper bounds for each parameter. |
| `NUM_PAR` | `int` | `0` | Number of optimization parameters. Must be greater than zero. |
| `HOF_SIZE` | `int \| None` | `min(100, POP_SIZE)` | Maximum number of individuals in the Hall of Fame. |
| `SAVE_INTERVAL` | `int` | `10` | Interval between saved population snapshots. |
| `LOG_INTERVAL` | `int` | `10` | Interval between console log messages. |
| `ITERATIVE_SAVE` | `bool` | `True` | Enables generation snapshot files. |
| `PATIENCE` | `int \| None` | `15% of MAX_GEN` | Number of generations without improvement before stopping. |
| `FITNESS_TOL` | `float` | `1e-7` | Stops when the best fitness is less than or equal to this value. |
| `FITNESS_STD_TOL` | `float \| None` | `1e-5` | Stops when the population fitness standard deviation is below this value. |
| `P_BEST` | `float` | `0.1` | Fraction of top individuals considered by the p-best strategy. |
| `BOUNDARY_METHOD` | `str` | `"reflect"` | Boundary repair method: `"reflect"` or `"clip"`. |
| `PRINT_INITIAL_POP` | `bool` | `False` | Prints every initial individual when enabled. |

### Disabling standard-deviation stopping

Set:

```python
FITNESS_STD_TOL=None
```

### Disabling iterative generation files

Set:

```python
ITERATIVE_SAVE=False
```

---

## Initialization

The initial population is generated with a scrambled Sobol sequence and scaled to the user-defined bounds.

When the population size is not a power of two, the implementation generates the next valid Sobol base-2 sample and keeps only the requested number of individuals.

For example, when:

```python
POP_SIZE=150
```

the optimizer still creates and evaluates exactly 150 initial individuals.

The initial population is evaluated only once. Each optimization generation then evaluates one trial vector per target individual.

For a complete run without early stopping, the expected number of fitness evaluations is:

```text
POP_SIZE + POP_SIZE * MAX_GEN
```

or equivalently:

```text
POP_SIZE * (MAX_GEN + 1)
```

For `POP_SIZE=256` and `MAX_GEN=2500`, this corresponds to:

```text
640,256 fitness evaluations
```

---

## Strategy Adaptation

Each individual selects one mutation strategy according to the current strategy-probability vector.

After every learning period:

1. successful and unsuccessful applications are counted;
2. success rates are computed;
3. strategy probabilities are updated and normalized;
4. a minimum probability prevents a strategy from disappearing completely;
5. the crossover-rate memory is updated using the median of successful crossover rates.

Equal-fitness replacements are accepted, but only strict improvements are counted as successful adaptations.

---

## Control Parameters

### Scale factor

The mutation scale factor is sampled from a normal distribution centered at `0.5` with standard deviation `0.3`.

Non-positive values are rejected, and very large values are capped.

### Crossover rate

The crossover rate is sampled from a normal distribution centered at the current crossover-rate memory:

```text
CR ~ N(CRm, 0.1)
```

The sampled value is clipped to the interval `[0, 1]`.

---

## Boundary Handling

Two box-constraint repair methods are supported.

### Reflection

```python
BOUNDARY_METHOD="reflect"
```

Values outside the valid interval are reflected into the feasible region. This method generally preserves more information about the mutation step.

### Clipping

```python
BOUNDARY_METHOD="clip"
```

Values outside the valid interval are replaced by the nearest bound.

---

## Reproducibility

When a seed is provided:

```python
SEED=1234
```

the same NumPy random generator is used for:

- Sobol initialization;
- mutation-strategy selection;
- scale-factor sampling;
- crossover-rate sampling;
- individual selection;
- binomial crossover;
- p-best selection.

When `SEED=None`, a new seed is generated from system entropy. The actual seed is stored in the configuration file, allowing the run to be repeated later.

Reproducibility also requires the objective function to be deterministic. Sources of nondeterminism inside an external simulation, multithreaded numerical library, GPU kernel, or file-based workflow may still produce different results.

---

## Output Files

The output directory is created automatically, including missing parent directories.

A typical output structure is:

```text
tests/neut_bact/
├── BEST_OF_ALL.dat
├── HOF.dat
├── configurations.dat
└── gens/
    ├── gen_0.dat
    ├── gen_10.dat
    ├── gen_20.dat
    └── ...
```

### `configurations.dat`

Contains the main optimizer settings, including:

- population size;
- maximum generations;
- bounds;
- seed;
- active strategies;
- learning period;
- stopping tolerances;
- boundary method.

### `BEST_OF_ALL.dat`

Contains:

- best fitness;
- optimized parameter vector;
- last completed generation;
- total execution time.

### `HOF.dat`

Contains the best individuals stored in the Hall of Fame.

### `gens/gen_<generation>.dat`

Contains a snapshot of the complete population and the best individual at a saved generation.

Because complete population snapshots can be large, increase `SAVE_INTERVAL` or set `ITERATIVE_SAVE=False` for long runs.

---

## Programmatic Results

`run_SaDE()` returns the best DEAP individual:

```python
best = optimizer.run_SaDE()

parameters = list(best)
fitness = best.fitness.values[0]
```

The same individual is also stored in:

```python
optimizer.BEST
```

The Hall of Fame is available through:

```python
optimizer.HOF
```

The configuration dictionary is available through:

```python
config = optimizer.get_configs()
```

The alias below is also supported:

```python
best = optimizer.run()
```

---

## Performance Considerations

The objective function is usually the main computational cost.

Recommended practices:

- use parallel evaluation when each fitness calculation is expensive;
- avoid creating more worker processes than available CPU cores;
- account for external libraries that already use internal threads;
- avoid writing files from every worker to the same location;
- use a separate temporary directory for each evaluation when external programs generate files;
- reduce generation snapshot frequency for large populations;
- use a reasonable Hall of Fame size;
- keep the objective function serializable for multiprocessing.

### Avoiding CPU oversubscription

When each worker launches a simulation that also uses OpenMP, BLAS, or another threaded library, the total thread count may become much larger than the available number of CPU cores.

For example, eight worker processes running eight OpenMP threads each may create 64 active threads.

Depending on the application, set environment variables such as:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

before starting a process-based parallel run.

---

## Error Handling and Validation

The optimizer validates:

- positive learning period;
- positive number of generations;
- valid save and log intervals;
- minimum population size;
- number of bounds matching `NUM_PAR`;
- finite lower and upper bounds;
- lower bounds strictly smaller than upper bounds;
- valid p-best fraction;
- valid boundary method;
- non-empty fitness values.

The optimization is configured for minimization through DEAP weights:

```python
weights=(-1.0,)
```

---

## Troubleshooting

### `AttributeError: Can't pickle local object`

The objective function is probably nested inside another function or defined as a lambda.

Move it to module scope:

```python
def objective_function(individual):
    return (sum(x**2 for x in individual),)
```

### Recursive process creation

Ensure that process creation is protected by:

```python
if __name__ == "__main__":
    main()
```

### The run is not parallel

Check that a parallel map function was provided:

```python
PARALLEL_MAP_FUNCTION=pool.map
```

`PARALLEL=True` alone does not replace the sequential map function.

### The output directory does not exist

The optimizer creates `SAVE_PATH` and `SAVE_PATH/gens` automatically. Verify that the current user has permission to write to the selected parent directory.

### Different results with the same seed

Check whether the objective function or external simulator uses:

- another uncontrolled random generator;
- multithreaded reductions;
- nondeterministic GPU operations;
- shared temporary files;
- race conditions.

### Too many output files

Increase:

```python
SAVE_INTERVAL=100
```

or disable iterative saving:

```python
ITERATIVE_SAVE=False
```

### Premature stopping due to population uniformity

Disable this stopping rule:

```python
FITNESS_STD_TOL=None
```

---

## Testing

A simple smoke test can use the Sphere function:

```python
from SaDE import SaDE


def sphere(individual):
    return (sum(x**2 for x in individual),)


def test_sphere() -> None:
    optimizer = SaDE(
        EVALUATION_FUNCTION=sphere,
        NUM_PAR=4,
        BOUNDS=[(-5.0, 5.0)] * 4,
        POP_SIZE=32,
        MAX_GEN=100,
        SEED=1234,
        SAVE_PATH="tests/output/sphere/",
        ITERATIVE_SAVE=False,
        FITNESS_STD_TOL=None,
    )

    best = optimizer.run_SaDE()

    assert len(best) == 4
    assert best.fitness.values[0] >= 0.0
```

Run it directly or integrate it with `pytest`.

---

## Known Scope and Limitations

The current implementation is designed for:

- continuous parameters;
- finite box bounds;
- single-objective minimization;
- generational Differential Evolution;
- synchronous batch evaluation.

The current implementation does not directly provide:

- multi-objective optimization;
- nonlinear constraint handling;
- mixed continuous and discrete variables;
- asynchronous evaluation;
- checkpoint-based restart from a saved population;
- distributed-memory MPI orchestration.

These features can be added on top of the current architecture.

---

## Citation and Algorithm Reference

When using this repository in academic work, cite the repository and the original SaDE reference:

> A. K. Qin, V. L. Huang, and P. N. Suganthan, “Differential Evolution Algorithm With Strategy Adaptation for Global Numerical Optimization,” *IEEE Transactions on Evolutionary Computation*, vol. 13, no. 2, pp. 398–417, 2009. DOI: 10.1109/TEVC.2008.927706.

```bibtex
@software{sade_python,
  author  = {João Víctor Costa de Oliveira},
  title   = {SaDE},
  year    = {2026},
  url     = {https://github.com/citzenfive/SaDE}
}
```

Replace the placeholder URL with the final repository address.

---

## Author

**João Víctor Costa de Oliveira**

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.