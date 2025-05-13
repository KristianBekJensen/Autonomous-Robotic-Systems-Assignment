from leap_ec.individual import Individual


class Custom_Individual(Individual):
    def __init__(self, genome, decoder=..., problem=None):
        
        super().__init__(genome, decoder, problem)

    # overides
    def evaluate(self):
        """ determine this individual's fitness

        This is done by outsourcing the fitness evaluation to the associated
        `Problem` object since it "knows" what is good or bad for a given
        phenome.


        :see also: ScalarProblem.worse_than

        :return: the calculated fitness
        """
        self.fitness, self.avg_sigma = self.evaluate_imp()
        return self.fitness