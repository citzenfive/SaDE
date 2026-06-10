'''
How to use: 
    python main.py \
        --n_processes 8 \
        --save_path tests/neut_bact/ \
        --population_size 256 \
        --max-gen 2500 \
        --seed 1234
'''


import argparse
import multiprocessing
from pathlib import Path

from SaDE import SaDE
from run_benchmark import neut_bac_model


NUM_PARAMETERS = 6

BOUNDS = [
    (1.0e-5, 1.0),
    (1.0e-5, 1.0),
    (1.0e-5, 1.0),
    (1.0e-5, 1.0),
    (1.0e-5, 1.0),
    (1.0e-5, 1.0),
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SaDE optimization in parallel."
    )

    parser.add_argument(
        "--n_processes",
        "--n-processes",
        dest="n_processes",
        type=int,
        required=True,
        help="Number of parallel worker processes.",
    )

    parser.add_argument(
        "--save_path",
        "--save-path",
        dest="save_path",
        type=Path,
        required=True,
        help="Directory where the optimization results will be saved.",
    )

    parser.add_argument(
        "--population_size",
        "--population-size",
        dest="population_size",
        type=int,
        default=150,
        help="Population size. Default: 150.",
    )

    parser.add_argument(
        "--max-gen",
        "--max_gen",
        dest="max_gen",
        type=int,
        default=2500,
        help="Maximum number of generations. Default: 2500.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )

    parser.add_argument(
        "--learning-period",
        "--learning_period",
        dest="learning_period",
        type=int,
        default=50,
        help="SaDE learning period. Default: 50.",
    )

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    if args.n_processes <= 0:
        raise ValueError("--n_processes must be greater than zero.")

    if args.population_size <= 0:
        raise ValueError("--population_size must be greater than zero.")

    if args.max_gen <= 0:
        raise ValueError("--max-gen must be greater than zero.")

    if args.learning_period <= 0:
        raise ValueError("--learning-period must be greater than zero.")


def main() -> None:
    args = parse_arguments()
    validate_arguments(args)

    args.save_path.mkdir(parents=True, exist_ok=True)

    with multiprocessing.Pool(processes=args.n_processes) as pool:
        optimizer = SaDE(
            EVALUATION_FUNCTION=neut_bac_model,
            const_LP=args.learning_period,
            POP_SIZE=args.population_size,
            MAX_GEN=args.max_gen,
            PARALLEL=True,
            SEED=args.seed,
            SAVE_PATH=str(args.save_path),
            PARALLEL_MAP_FUNCTION=pool.map,
            BOUNDS=BOUNDS,
            NUM_PAR=NUM_PARAMETERS,
        )

        optimizer.print_config()
        optimizer.save_config()

        best = optimizer.run_SaDE()

    print("\nOptimization finished.")
    print(f"Best individual: {best}")
    print(f"Best fitness: {best.fitness.values[0]:.8e}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()


