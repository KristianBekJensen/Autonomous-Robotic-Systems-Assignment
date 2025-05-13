# evolution_nn.py

import os
import sys
import numpy as np
import pygame
import graph
from matplotlib import pyplot as plt

from leap_ec import Representation, test_env_var
from leap_ec import ops, probe
from leap_ec.algorithm import generational_ea
from leap_ec.util import wrap_curry
from leap_ec.ops import UniformCrossover, tournament_selection, clone

from Custom_Individual import Custom_Individual
from graph import PopulationMetricsPlotProbe
from maze_solver_nn import MazeSolver
from trajectory_recorder import save_pop, load_pop

from config_nn import num_sensors, input_size, hidden_size, output_size, genome_length, pop_size, generations
class Evolution_nn():
    def __init__(self):
        self.gen = 0
        self.save = False
        pygame.init()
    @wrap_curry
    def MyMethodAsync(self, pop):
        #while True:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_1]:
                self.save = not self.save
                print("saving next gen")
            return pop

    @wrap_curry
    def gen_tick(self, pop):
        self.gen += 1
        return pop

    @wrap_curry
    def save_gen(self, pop, filename, interval=1):
        if self.gen % interval == 0:
            filename += str(self.gen) + ".pkl"
            save_pop(pop, filename + ".pkl")
            plt.savefig(filename + ".png")
        return pop

    def run(self):
        
        # Training parameters:
        # defined in config_nn.py

        # ───────────────────────────
        # Create the problem
        # ───────────────────────────
        problem = MazeSolver(
            maximize=False,
            visualization=False,
            num_sensors=num_sensors,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size
        )
        # if true load (sample) from previous population ; if false, random
        load_old_pop = False

        # ───────────────────────────
        # Initializer: Load from previous population
        # ───────────────────────────
        def init_genome_from_file():
            """
            Load genomes from previous final population file.
            If fewer than pop_size individuals are present, sample with replacement.
            """
            filename = "final_nn_pop.pkl"
            prev_pop = load_pop(filename)

            print(prev_pop)

            if len(prev_pop) < pop_size:
                chosen = np.random.choice(prev_pop, size=pop_size, replace=True)
            else:
                chosen = np.random.choice(prev_pop, size=pop_size, replace=False)

            # Return one genome at a time (LEAP will call this repeatedly)
            for ind in chosen:
                yield ind.genome


        # Create an iterator of genomes
        genome_iterator = init_genome_from_file()

        def init_genome():
            if load_old_pop:
                return next(genome_iterator)
            else:
                #return np.random.uniform(-1.0, 1.0, size=genome_length)
                return  np.random.normal(0.0, 0.1, size=genome_length)


        # ───────────────────────────
        # A float‐vector initializer
        # ───────────────────────────
        """ def init_genome():
            
            Return *one* genome (a NumPy array of length genome_length),
            uniformly sampled from [-1,1].  LEAP will call this pop_size times.
            
            return np.random.uniform(-1.0, 1.0, size=genome_length) 
        """

        # ───────────────────────────
        # A Gaussian‐mutation operator
        # ───────────────────────────

        @wrap_curry
        def mutate_gaussian(pop, sigma: float = 0.1, frac_genes: float = 0.05):
            """
            Generator‐style operator that Gaussian‐mutates an Individual's genome.
            - pop is a generator of Individual objects.
            - sigma is the stddev of N(0,σ²).
            - frac_genes is the fraction of genes to mutate per individual.
            """
            import itertools

            # Peek at the first individual to get genome length
            pop1, pop2 = itertools.tee(pop, 2)
            first = next(pop1, None)
            if first is None:
                # empty population: just return the second stream
                yield from pop2
                return

            n = first.genome.shape[0]
            k = max(1, int(frac_genes * n))

            # Now actually mutate each individual from pop2
            for ind in pop2:
                # clone the entire Individual (including genome)
                child = ind.clone()

                # pick k distinct gene indices
                idxs = np.random.choice(n, size=k, replace=False)
                # add Gaussian noise
                child.genome[idxs] += np.random.normal(0, sigma, size=k)

                # invalidate any old fitness
                child.fitness = None

                yield child


        # ───────────────────────────
        # Checkpoint helper
        # ───────────────────────────
        


        # ───────────────────────────
        # Live plotting setup
        # ───────────────────────────
        plt.ion()
        fig, axes  = plt.subplots(1, 3, figsize=(14,5))
        p1 = PopulationMetricsPlotProbe(
            metrics=[ lambda pop: probe.best_of_gen(pop).fitness ],
            xlim=(0, generations),
            title="Best-of-Generation Fitness",
            ax=axes[0]
        )
        p2 = PopulationMetricsPlotProbe(
            metrics=[probe.pairwise_squared_distance_metric],
            xlim=(0, generations),
            title="Population Diversity",
            ax=axes[1]
        )
        p3 = PopulationMetricsPlotProbe(
            metrics=[ lambda pop: graph.best_avg_sigma(pop).avg_sigma ],
            xlim=(0, generations),
            title="Best-of-Generation Avg_Sigma",
            ax=axes[2])
        viz_probes = [p1, p2, p3]


        plt.show(block=False)

        # ───────────────────────────
        # Run the EA
        # ───────────────────────────
        final_pop = generational_ea(
            max_generations=generations,
            pop_size=pop_size,
            problem=problem,
            representation=Representation(
                # each call returns ONE genome of length genome_length
                initialize=init_genome,
                individual_cls=Custom_Individual
            ),
            pipeline=[
                tournament_selection(k=3),
                clone,
                # Uniform crossover works on real arrays
                UniformCrossover(p_swap=0.5),
                # Gaussian mutation
                mutate_gaussian(sigma=0.3, frac_genes=0.2),
                ops.evaluate,               # calls MazeSolver.evaluate()
                ops.pool(size=pop_size),    # keep best pop_size
                self.gen_tick(),
                self.save_gen(filename="current_nn_pop", interval=10),
                probe.FitnessStatsCSVProbe(stream=sys.stdout),
                *viz_probes
            ]
        )

        # ───────────────────────────
        # Finalize plots
        # ───────────────────────────
        """ plt.ioff()
        if os.environ.get(test_env_var, "") != "":
            plt.show()
        plt.close(fig) """


        # ───────────────────────────
        # Save & report
        # ───────────────────────────
        os.makedirs("populations", exist_ok=True)

        final_fname = "final_nn_pop.pkl"
        self.save_gen(final_pop, final_fname)

        best = min(final_pop, key=lambda ind: ind.fitness)
        print("Training complete.")
        print(f"  Best fitness: {best.fitness:.3f}")
        print(f"  Saved population to populations/{final_fname!r}")


x = Evolution_nn()
x.MyMethodAsync()
x.run()