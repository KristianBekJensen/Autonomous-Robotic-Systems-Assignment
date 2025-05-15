import os
import sys
import numpy as np
import graph
import itertools
import config as conf
from matplotlib import pyplot as plt
from leap_ec.distrib import synchronous
from leap_ec import Representation
from leap_ec import ops, probe
from leap_ec.algorithm import generational_ea
from leap_ec.util import wrap_curry
from leap_ec.ops import UniformCrossover, tournament_selection, clone
from distributed import Client
from Custom_Individual import Custom_Individual
from graph import PopulationMetricsPlotProbe
from maze_solver import MazeSolver
from trajectory_recorder import save_pop, load_pop

class Evolution():
    """ a class for setting up and running neuroevolution """

    def __init__(self):
        self.gen = 0
    
    # continue from previously trained population
    def init_genome_from_file(self):
        """ load genomes from a saved population file """
        prev_pop = load_pop(conf.start_pop_filename)
        if len(prev_pop) < conf.pop_size:
            # sample with replacement if prev_pop is fewer than population size
            chosen = np.random.choice(prev_pop, size=conf.pop_size, replace=True)
        else:
            chosen = np.random.choice(prev_pop, size=conf.pop_size, replace=False)
        # return one genome at a time
        for ind in chosen:
            yield ind.genome
    
    def init_genome(self):
        """ if there is no previous population then initialize randonly from a normal distibution """
        if conf.start_pop_filename:
            return next(self.init_genome_from_file())
        else:
            return  np.random.normal(0.0, 0.1, size=conf.genome_length)
    
    @wrap_curry
    def grouped_evaluate(self, population, client, max_individuals_per_chunk=4):
        """ evaluate the population in groups to speed up training process """
        if max_individuals_per_chunk is None:
            max_individuals_per_chunk = len(population)

        def chunks(lst, n):
            """ helper function to yield chunks from a list"""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        evaluated_pop = []
        for chunk in chunks(population, max_individuals_per_chunk):
            evaluated_pop.extend( synchronous.eval_population(chunk, client))

        return evaluated_pop

    @wrap_curry
    def gen_tick(self, pop):
        """ increment generatoin counter after each run """
        self.gen += 1
        return pop

    @wrap_curry
    def save_gen(self, pop, filename, interval=1):
        """ checkpoint for saving population and plotting """
        if self.gen % interval == 0:
            filename += "_gen" + str(self.gen)
            save_pop(pop, filename + ".pkl")
            plt.savefig(filename + ".png")
        return pop

    def run(self):
        """ create evolutionary algorithm components and run it. training params
        are set in config.py """

        # create the problem        
        problem = MazeSolver(
            maximize=False,
            visualization=False,
            num_sensors=conf.num_sensors,
            input_size=conf.input_size,
            hidden_size=conf.hidden_size,
            output_size=conf.output_size,
            max_steps=conf.max_steps,
            close_controller=conf.close_controller,
            controller_type=conf.controller,
            fitness_func=conf.fitness_func,
            random = self.gen if conf.random_map else 44
        )

        # define a gaussian mutation to use in the pipeline
        @wrap_curry
        def mutate_gaussian(pop, sigma=0.1, frac_genes=0.05):
            """ mutates each robot's genome by adding gaussian noise to a fraction
            of genes """           
            # create two iterators
            pop1, pop2 = itertools.tee(pop, 2)
            first = next(pop1, None)
            if first is None:
                # if the population is empty yield nothing new
                yield from pop2
                return

            # genome length and number of genes to mutate
            n = first.genome.shape[0]
            k = max(1, int(frac_genes * n))

            # now actually mutate each individual from pop2
            for ind in pop2:
                child = ind.clone()

                # randomly choose k distinct genes to change
                idxs = np.random.choice(n, size=k, replace=False)
                # add gaussian noise to those genes
                child.genome[idxs] += np.random.normal(0, sigma, size=k)

                # clear any old fitness
                child.fitness = None

                yield child


        # plotting setup
        plt.ion()
        fig, axes = plt.subplots(1, 3, figsize=(14,5))
        p1 = PopulationMetricsPlotProbe(
            metrics=[lambda pop: probe.best_of_gen(pop).fitness],
            xlim=(0, self.gen),
            title="Best-of-Generation Fitness",
            ax=axes[0]
        )
        p2 = PopulationMetricsPlotProbe(
            metrics=[probe.pairwise_squared_distance_metric],
            xlim=(0, self.gen),
            title="Population Diversity",
            ax=axes[1]
        )
        p3 = PopulationMetricsPlotProbe(
            metrics=[lambda pop: graph.best_avg_sigma(pop).avg_sigma],
            xlim=(0, self.gen),
            title="Best-of-Generation Avg_Sigma",
            ax=axes[2])
        viz_probes = [p1, p2, p3]


        plt.show(block=False)
        if __name__ == '__main__':
            
            # dask client for parallel runs to speed up training
            client = Client()
            
            # run the EA
            final_pop = generational_ea(
                max_generations=conf.generations,
                pop_size=conf.pop_size,
                problem=problem,
                representation=Representation(initialize=self.init_genome, individual_cls=Custom_Individual),
                pipeline=[
                    tournament_selection(k=10), # select parents from tournament
                    clone, # clone parents before mutating
                    UniformCrossover(p_swap=0.5), # combine genomes with uniform crossover
                    mutate_gaussian(sigma=0.4, frac_genes=0.2), # mutate 20% of genes with gaussian noise
                    ops.pool(size=conf.pop_size), # reduce back to pop_size
                    self.grouped_evaluate(client=client, max_individuals_per_chunk=4), # calls MazeSolver.evaluate()
                    self.gen_tick(), # increase generation number by one
                    self.save_gen(filename=conf.save_as, interval=10),
                    probe.FitnessStatsCSVProbe(stream=sys.stdout),
                    *viz_probes
                ]
            )

            # save & report
            os.makedirs("populations", exist_ok=True)

            final_fname = "final_nn_pop.pkl"
            self.save_gen(final_pop, final_fname)

            # find and print the best
            best = min(final_pop, key=lambda ind: ind.fitness)
            print("Training complete.")
            print(f"  Best fitness: {best.fitness:.3f}")
            print(f"  Saved population to populations/{final_fname!r}")


ea = Evolution()
ea.run()