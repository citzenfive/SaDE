"""Self-adaptive Differential Evolution (SaDE), optimized implementation."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from deap import base, creator, tools
from numpy.random import Generator, Philox, SeedSequence
from scipy.stats import qmc

from mut_cross import (
    de_best_1_bin,
    de_current_to_pbest_1,
    de_current_to_rand_1,
    de_rand_1_bin,
    de_rand_2_bin,
    de_rand_to_best_2_bin,
)


class SaDE:
    """Single-objective minimization with self-adaptive DE strategies.

    The original public argument names and ``run_SaDE`` method are retained for
    compatibility with existing scripts.
    """

    def __init__(
        self,
        EVALUATION_FUNCTION: Callable,
        const_LP: int = 50,
        POP_SIZE: int = 150,
        MAX_GEN: int = 2500,
        PARALLEL: bool = False,
        SEED: int | None = None,
        SAVE_PATH: str = "",
        PARALLEL_MAP_FUNCTION: Callable | None = None,
        BOUNDS: Sequence[tuple[float, float]] | None = None,
        NUM_PAR: int = 0,
        HOF_SIZE: int | None = None,
        SAVE_INTERVAL: int = 10,
        LOG_INTERVAL: int = 10,
        ITERATIVE_SAVE: bool = True,
        PATIENCE: int | None = None,
        FITNESS_TOL: float = 1.0e-7,
        FITNESS_STD_TOL: float | None = 1.0e-6,
        P_BEST: float = 0.1,
        BOUNDARY_METHOD: str = "reflect",
        PRINT_INITIAL_POP: bool = False,
    ) -> None:
        self._validate_basic_arguments(
            const_LP, POP_SIZE, MAX_GEN, NUM_PAR, SAVE_INTERVAL, LOG_INTERVAL
        )

        self.TOOLBOX = base.Toolbox()
        self.LP = int(const_LP)
        self.POP_SIZE = int(POP_SIZE)
        self.MAX_GEN = int(MAX_GEN)
        self.NUM_PAR = int(NUM_PAR)
        self.PARALLEL = bool(PARALLEL)
        self.ITERATIVE_SAVE = bool(ITERATIVE_SAVE)
        self.SAVE_INTERVAL = int(SAVE_INTERVAL)
        self.LOG_INTERVAL = int(LOG_INTERVAL)
        self.PRINT_INITIAL_POP = bool(PRINT_INITIAL_POP)
        self.P_BEST = float(P_BEST)
        self.BOUNDARY_METHOD = BOUNDARY_METHOD.lower()
        self.FITNESS_TOL = float(FITNESS_TOL)
        self.FITNESS_STD_TOL = FITNESS_STD_TOL
        self.PATIENCE = (
            max(1, int(0.15 * self.MAX_GEN))
            if PATIENCE is None
            else max(1, int(PATIENCE))
        )

        if not 0.0 < self.P_BEST <= 1.0:
            raise ValueError("P_BEST must be in the interval (0, 1].")
        if self.BOUNDARY_METHOD not in {"reflect", "clip"}:
            raise ValueError("BOUNDARY_METHOD must be 'reflect' or 'clip'.")

        self.EVALUATION_FUNCTION = EVALUATION_FUNCTION
        self.config_seed = (
            int(SEED) if SEED is not None else int(SeedSequence().entropy)
        )
        self.prng: Generator = Generator(Philox(seed=self.config_seed))

        self.BOUNDS = self._validate_bounds(BOUNDS)
        bounds_array = np.asarray(self.BOUNDS, dtype=float)
        self.l_bounds = bounds_array[:, 0]
        self.u_bounds = bounds_array[:, 1]
        self._bound_span = self.u_bounds - self.l_bounds

        self.SAVE_PATH = Path(SAVE_PATH) if SAVE_PATH else Path(".")
        self.SAVE_PATH_GEN = self.SAVE_PATH / "gens"
        self.SAVE_PATH.mkdir(parents=True, exist_ok=True)
        self.SAVE_PATH_GEN.mkdir(parents=True, exist_ok=True)

        if PARALLEL_MAP_FUNCTION is not None:
            self.TOOLBOX.register("map", PARALLEL_MAP_FUNCTION)

        self._configure_deap_creator()
        self.CONFIGURED_CREATOR = creator
        self.TOOLBOX.register("evaluate", self.EVALUATION_FUNCTION)
        self.TOOLBOX.register("select_best", tools.selBest)

        self.strategy_pool = self._build_strategy_pool()
        self.strategy_names = [strategy.__name__ for strategy in self.strategy_pool]
        self.num_strategies = len(self.strategy_pool)
        self.str_prob = np.full(self.num_strategies, 1.0 / self.num_strategies)
        self.success_counter = np.zeros(self.num_strategies, dtype=np.int64)
        self.failure_counter = np.zeros(self.num_strategies, dtype=np.int64)
        self.cr_memory: list[float] = []
        self.crm = 0.5

        requested_hof = min(100, self.POP_SIZE) if HOF_SIZE is None else HOF_SIZE
        self.HOF_SIZE = max(1, int(requested_hof))
        self.HOF = tools.HallOfFame(self.HOF_SIZE)

        self.stats = tools.Statistics(
            key=lambda individual: individual.fitness.values[0]
        )
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)

        self.START_TIME = 0.0
        self.END_TIME = 0.0
        self.BEST = None
        self.INITIAL_POP = self.__gen_initial_pop()

    @staticmethod
    def _validate_basic_arguments(
        learning_period: int,
        population_size: int,
        max_generations: int,
        num_parameters: int,
        save_interval: int,
        log_interval: int,
    ) -> None:
        if num_parameters <= 0:
            raise ValueError("NUM_PAR must be greater than zero.")
        if population_size < 4:
            raise ValueError("POP_SIZE must be at least 4 for Differential Evolution.")
        if learning_period <= 0:
            raise ValueError("const_LP must be greater than zero.")
        if max_generations <= 0:
            raise ValueError("MAX_GEN must be greater than zero.")
        if save_interval <= 0 or log_interval <= 0:
            raise ValueError("SAVE_INTERVAL and LOG_INTERVAL must be positive.")

    def _validate_bounds(
        self, bounds: Sequence[tuple[float, float]] | None
    ) -> list[tuple[float, float]]:
        if bounds is None or len(bounds) == 0:
            bounds = [(1.0e-5, 9.8e-1)] * self.NUM_PAR

        if len(bounds) != self.NUM_PAR:
            raise ValueError(
                f"Expected {self.NUM_PAR} bounds, but received {len(bounds)}."
            )

        validated: list[tuple[float, float]] = []
        for index, pair in enumerate(bounds):
            if len(pair) != 2:
                raise ValueError(f"Bound {index} must contain (lower, upper).")
            lower, upper = map(float, pair)
            if not np.isfinite(lower) or not np.isfinite(upper):
                raise ValueError(f"Bound {index} must be finite.")
            if lower >= upper:
                raise ValueError(
                    f"Invalid bound {index}: lower={lower} must be < upper={upper}."
                )
            validated.append((lower, upper))
        return validated

    @staticmethod
    def _configure_deap_creator() -> None:
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        elif tuple(creator.FitnessMin.weights) != (-1.0,):
            raise RuntimeError(
                "DEAP creator.FitnessMin already exists with incompatible weights."
            )

        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)

    def _build_strategy_pool(self) -> list[Callable]:
        # Strategies are filtered for small populations instead of failing later.
        candidates = [
            (de_rand_1_bin, 4),
            (de_rand_to_best_2_bin, 7),
            (de_rand_2_bin, 6),
            (de_current_to_rand_1, 4),
            (de_current_to_pbest_1, 4),
            (de_best_1_bin, 4),
        ]
        strategies = [
            strategy
            for strategy, minimum_size in candidates
            if self.POP_SIZE >= minimum_size
        ]
        if not strategies:
            raise ValueError("Population is too small for the available strategies.")
        return strategies

    def _sample_scale_factor(self) -> float:
        # SaDE uses N(0.5, 0.3); reject non-positive values and cap extremes.
        for _ in range(16):
            value = float(self.prng.normal(0.5, 0.3))
            if value > 0.0:
                return min(value, 2.0)
        return 0.5

    def _repair_bounds(self, individual) -> None:
        values = np.asarray(individual, dtype=float)
        if self.BOUNDARY_METHOD == "clip":
            repaired = np.clip(values, self.l_bounds, self.u_bounds)
        else:
            # Repeated reflection also handles values farther than one span away.
            shifted = np.mod(values - self.l_bounds, 2.0 * self._bound_span)
            repaired = (
                self.l_bounds + self._bound_span - np.abs(shifted - self._bound_span)
            )
        individual[:] = repaired.tolist()

    @staticmethod
    def _normalize_fitness(fitness) -> tuple[float, ...]:
        if np.isscalar(fitness):
            return (float(fitness),)
        normalized = tuple(float(value) for value in fitness)
        if not normalized:
            raise ValueError("The evaluation function returned an empty fitness.")
        return normalized

    def _evaluate_population(self, population: Sequence) -> None:
        fitnesses = self.TOOLBOX.map(self.TOOLBOX.evaluate, population)
        for individual, fitness in zip(population, fitnesses):
            individual.fitness.values = self._normalize_fitness(fitness)

    def run_SaDE(self):
        self.START_TIME = time.perf_counter()
        current_population = list(self.INITIAL_POP)
        self.HOF.update(current_population)
        no_improvement_generations = 0
        last_generation = -1

        for generation in range(self.MAX_GEN):
            last_generation = generation
            generation_start = time.perf_counter()

            current_best = tools.selBest(current_population, 1)[0]
            previous_best_fitness = current_best.fitness.values[0]

            ranked_population = sorted(
                current_population,
                key=lambda individual: individual.fitness.values[0],
            )
            pbest_count = min(
                self.POP_SIZE,
                max(2, int(math.ceil(self.P_BEST * self.POP_SIZE))),
            )
            pbest_pool = ranked_population[:pbest_count]

            scale_factors = [self._sample_scale_factor() for _ in range(self.POP_SIZE)]
            crossover_rates = np.clip(
                self.prng.normal(self.crm, 0.1, size=self.POP_SIZE), 0.0, 1.0
            )
            strategy_indices = self.prng.choice(
                self.num_strategies,
                size=self.POP_SIZE,
                p=self.str_prob,
            )

            trial_vectors = []
            for index, target in enumerate(current_population):
                strategy_index = int(strategy_indices[index])
                strategy = self.strategy_pool[strategy_index]
                arguments = {
                    "ind": target,
                    "population": current_population,
                    "f": scale_factors[index],
                    "cr": float(crossover_rates[index]),
                    "creator": self.CONFIGURED_CREATOR,
                    "rng": self.prng,
                    "pbest_pool": pbest_pool,
                    "p": self.P_BEST,
                }
                if strategy in {de_rand_to_best_2_bin, de_best_1_bin}:
                    arguments["best"] = current_best

                trial = strategy(**arguments)
                if trial is None:
                    # This should only occur for unusual small-population edge cases.
                    trial = self.CONFIGURED_CREATOR.Individual(target)
                self._repair_bounds(trial)
                trial_vectors.append(trial)

            evaluation_start = time.perf_counter()
            self._evaluate_population(trial_vectors)
            evaluation_time = time.perf_counter() - evaluation_start

            next_population = []
            for index, (target, trial) in enumerate(
                zip(current_population, trial_vectors)
            ):
                strategy_index = int(strategy_indices[index])
                trial_fitness = trial.fitness.values[0]
                target_fitness = target.fitness.values[0]

                if trial_fitness <= target_fitness:
                    next_population.append(trial)
                else:
                    next_population.append(target)

                # Equal-fitness replacements are accepted, but only strict
                # improvements teach the adaptive probability model.
                if trial_fitness < target_fitness:
                    self.success_counter[strategy_index] += 1
                    self.cr_memory.append(float(crossover_rates[index]))
                else:
                    self.failure_counter[strategy_index] += 1

            current_population = next_population
            self.HOF.update(current_population)

            if (generation + 1) % self.LP == 0:
                self.__update_strategy_probabilities()
                self.success_counter.fill(0)
                self.failure_counter.fill(0)
                if self.cr_memory:
                    self.crm = float(np.median(self.cr_memory))
                self.cr_memory.clear()

            generation_stats = self.stats.compile(current_population)
            new_best = tools.selBest(current_population, 1)[0]
            new_best_fitness = new_best.fitness.values[0]

            improvement_threshold = 1.0e-14 * max(1.0, abs(previous_best_fitness))
            if new_best_fitness < previous_best_fitness - improvement_threshold:
                no_improvement_generations = 0
            else:
                no_improvement_generations += 1

            generation_time = time.perf_counter() - generation_start

            if generation % self.LOG_INTERVAL == 0:
                probabilities = ", ".join(
                    f"{name}={probability:.3f}"
                    for name, probability in zip(self.strategy_names, self.str_prob)
                )
                print(
                    f"Gen {generation}: best={new_best_fitness:.8e} | "
                    f"std={generation_stats['std']:.3e} | "
                    f"no_improvement={no_improvement_generations}/{self.PATIENCE} | "
                    f"time={generation_time:.3f}s | eval={evaluation_time:.3f}s | "
                    f"crm={self.crm:.3f} | {probabilities}"
                )

            if self.ITERATIVE_SAVE and generation % self.SAVE_INTERVAL == 0:
                self.__save_gen(current_population, generation, generation_time)

            stop_reason = None
            if no_improvement_generations >= self.PATIENCE:
                stop_reason = "stagnation"
            elif new_best_fitness <= self.FITNESS_TOL:
                stop_reason = "fitness tolerance"
            elif (
                self.FITNESS_STD_TOL is not None
                and generation_stats["std"] <= self.FITNESS_STD_TOL
            ):
                stop_reason = "population fitness standard deviation"

            if stop_reason is not None:
                print(f"Stopping at generation {generation}: {stop_reason}.")
                if self.ITERATIVE_SAVE:
                    self.__save_gen(current_population, generation, generation_time)
                break

        self.BEST = tools.selBest(current_population, 1)[0]
        self.HOF.update(current_population)
        self.END_TIME = time.perf_counter()
        self.save_results(last_generation)

        print(f"Duration = {self.END_TIME - self.START_TIME:.6f} seconds")
        return self.BEST

    # Convenient alias without breaking existing code.
    run = run_SaDE

    def print_config(self) -> None:
        print("\nSaDE implementation by João Víctor Costa de Oliveira")
        for key, value in self.get_configs().items():
            print(f"  {key}: {value}")
        print()

    def save_config(self) -> None:
        config_path = self.SAVE_PATH / "configurations.dat"
        with config_path.open("w", encoding="utf-8") as config_file:
            config_file.write("SaDE implementation by João Víctor Costa de Oliveira\n")
            for key, value in self.get_configs().items():
                config_file.write(f"{key}: {value}\n")

    def save_results(self, last_generation: int | None = None) -> None:
        self.save_config()

        best_path = self.SAVE_PATH / "BEST_OF_ALL.dat"
        with best_path.open("w", encoding="utf-8") as best_file:
            best_file.write(f"Optimization for {self.NUM_PAR} parameters\n")
            best_file.write(f"Fitness value: {self.BEST.fitness.values[0]}\n")
            best_file.write(f"Parameters optimized: {list(self.BEST)}\n")
            if last_generation is not None:
                best_file.write(f"Last generation: {last_generation}\n")
            best_file.write(f"Spent time: {self.END_TIME - self.START_TIME} seconds\n")

        hof_path = self.SAVE_PATH / "HOF.dat"
        hof_lines = [f"Hall of Fame with {len(self.HOF)} individuals\n"]
        for position, individual in enumerate(self.HOF):
            hof_lines.append(
                f"Individual {position}:\n"
                f"  Gene: {list(individual)}\n"
                f"  Fitness: {individual.fitness.values[0]}\n"
            )
        hof_path.write_text("".join(hof_lines), encoding="utf-8")

    def get_configs(self) -> dict:
        return {
            "learning_period": self.LP,
            "population_size": self.POP_SIZE,
            "max_generations": self.MAX_GEN,
            "number_of_parameters": self.NUM_PAR,
            "lower_bounds": self.l_bounds.tolist(),
            "upper_bounds": self.u_bounds.tolist(),
            "seed": self.config_seed,
            "parallel": self.PARALLEL,
            "crm": self.crm,
            "hall_of_fame_size": self.HOF_SIZE,
            "strategies": self.strategy_names,
            "save_interval": self.SAVE_INTERVAL,
            "log_interval": self.LOG_INTERVAL,
            "patience": self.PATIENCE,
            "fitness_tolerance": self.FITNESS_TOL,
            "fitness_std_tolerance": self.FITNESS_STD_TOL,
            "boundary_method": self.BOUNDARY_METHOD,
        }

    def __update_strategy_probabilities(self) -> None:
        attempts = self.success_counter + self.failure_counter
        success_rates = np.divide(
            self.success_counter,
            attempts,
            out=np.zeros(self.num_strategies, dtype=float),
            where=attempts > 0,
        )

        if not np.any(success_rates):
            self.str_prob.fill(1.0 / self.num_strategies)
            return

        raw_probabilities = success_rates / success_rates.sum()
        minimum_probability = min(0.05, 0.99 / self.num_strategies)
        corrected = np.maximum(raw_probabilities, minimum_probability)
        self.str_prob = corrected / corrected.sum()

    def __gen_initial_pop(self) -> list:
        try:
            sampler = qmc.Sobol(d=self.NUM_PAR, scramble=True, rng=self.prng)
        except TypeError:
            # Compatibility with SciPy versions that still use ``seed``.
            sampler = qmc.Sobol(d=self.NUM_PAR, scramble=True, seed=self.config_seed)

        power = int(math.ceil(math.log2(self.POP_SIZE)))
        unit_sample = sampler.random_base2(m=power)[: self.POP_SIZE]
        scaled_sample = qmc.scale(unit_sample, self.l_bounds, self.u_bounds)

        population = [
            self.CONFIGURED_CREATOR.Individual(sample.tolist())
            for sample in scaled_sample
        ]
        self._evaluate_population(population)

        if self.PRINT_INITIAL_POP:
            self.__print_gen(population)
        else:
            initial_best = tools.selBest(population, 1)[0]
            print(
                "Initial population evaluated: "
                f"size={len(population)}, "
                f"best={initial_best.fitness.values[0]:.8e}"
            )
        return population

    def __gen_new_rnd_pop(self, size: int) -> list:
        samples = self.prng.uniform(
            self.l_bounds, self.u_bounds, size=(int(size), self.NUM_PAR)
        )
        population = [
            self.CONFIGURED_CREATOR.Individual(sample.tolist()) for sample in samples
        ]
        self._evaluate_population(population)
        return population

    @staticmethod
    def __print_gen(population: Iterable) -> None:
        for position, individual in enumerate(population):
            print(
                f"Ind {position}: fitness={individual.fitness.values} "
                f"gene={list(individual)}"
            )

    def __save_gen(self, population, generation: int, elapsed: float) -> None:
        best_generation = tools.selBest(population, 1)[0]
        lines = [
            f"Best individual:\n",
            f"  Gene: {list(best_generation)}\n",
            f"  Fitness: {best_generation.fitness.values[0]}\n",
            f"Generation: {generation}\n",
            f"Spent time: {elapsed} seconds\n\n",
        ]
        for position, individual in enumerate(population):
            lines.append(
                f"Individual {position}:\n"
                f"  Gene: {list(individual)}\n"
                f"  Fitness: {individual.fitness.values[0]}\n"
            )

        output_path = self.SAVE_PATH_GEN / f"gen_{generation}.dat"
        output_path.write_text("".join(lines), encoding="utf-8")
