from leap_ec import ops, probe
from leap_ec.global_vars import context
from matplotlib import pyplot as plt

class PopulationMetricsPlotProbe(probe.PopulationMetricsPlotProbe):

    def __init__(self, ax=None,
                 metrics=None,
                 xlim=None, ylim=None, modulo=1,
                 title='Population Metrics',
                 x_axis_value=None, context=context):
        """

        FIXME s/modulo/step/

        :param ax: matplotlib ax
        :param metrics: ???
        :param xlim: x axis bounds
        :param ylim: y axis bounds
        :param modulo: update interval
        :param title: for the plot
        :param x_axis_value: ???
        :param context: for accessing current generation
        """

        if ax is None:
            _, ax = plt.subplots()

        self.metrics = metrics
        self.modulo = modulo
        # x-axis defaults to generation
        if x_axis_value is None:
            x_axis_value = lambda: context['leap']['generation']
        self.x_axis_value = x_axis_value
        self.context = context

        # Set axis limits, and some variables we'll use for real-time scaling
        self.xlim = xlim
        self.ylim = ylim
        self.ax = ax
        ax.set_title(title)

        self.reset()
  
def best_avg_sigma(pop):
    return min(pop, key=lambda x: x.avg_sigma)