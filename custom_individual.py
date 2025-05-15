from leap_ec.individual import Individual


class Custom_Individual(Individual):
    def __init__(self, genome, decoder=..., problem=None):
        
        super().__init__(genome, decoder, problem)

    # overrides
    def evaluate(self):
        """ overrides to save both fitness and sigma on indivisual """
        self.fitness, self.avg_sigma = self.evaluate_imp()
        return self.fitness