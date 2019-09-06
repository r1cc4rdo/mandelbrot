""" Mandelbrot set visualization (2019) Riccardo Gherardi

A vanilla python/numpy implementation of a Mandelbrot set visualizer (on github, here: [1]).
The code has been liberally inspired by an implementation from Jean-François Puget [2].
Find there also more information on the Mandelbrot set and the colorization procedure.

[1] https://github.com/r1cc4rdo/mandelbrot
[2] https://www.ibm.com/developerworks/community/blogs/jfp/entry/My_Christmas_Gift?lang=en
"""
import numpy as np
from matplotlib import colors, pyplot as plot


def complex_grid(x_min, x_max, y_min, y_max, res=1024):
    xx = np.linspace(x_min, x_max, num=res)
    yy = np.linspace(y_min, y_max, num=res)
    x, y = np.meshgrid(xx, yy)
    return x + 1j * y


def mandelbrot(coordinates, max_iterations=256):
    horizon = 2.0 ** 40
    log_horizon = np.log(np.log(horizon)) / np.log(2)

    values = coordinates.copy()
    iteration_count = np.zeros_like(values, np.float)
    for iteration in range(max_iterations):
        nya = iteration_count == 0  # not yet assigned
        values[nya] = values[nya] ** 2 + coordinates[nya]

        abs_values = np.abs(values)
        jd = np.logical_and(nya, abs_values > horizon)  # just diverged
        iteration_count[jd] = iteration - np.log(np.log(abs_values[jd])) / np.log(2) + log_horizon

    return iteration_count


if __name__ == '__main__':

    region = 'full'  # region of the complex plane to display

    pois = {  # points of interest
        'full': (-2.0, 0.5, -1.25, 1.25),
        'valley': (-0.8, -0.7, 0.0, 0.1),
        'sea_horses': (-0.755, -0.745, 0.06, 0.07),
        'sea_horse': (-0.75, -0.747, 0.063, 0.066),
        'sea_horse_tail': (-0.749, -0.748, 0.065, 0.066),
        'black_dot': (-0.74877, -0.74872, 0.06505, 0.06510)}

    image = mandelbrot(complex_grid(*pois[region], res=512))
    plot.imshow(image, cmap='gnuplot2', norm=colors.PowerNorm(0.3))
    plot.xticks(np.linspace(0, image.shape[0], num=6), np.round(np.linspace(*pois[region][:2], num=6), 4))
    plot.yticks(np.linspace(0, image.shape[1], num=11), np.round(np.linspace(*pois[region][2:], num=11), 4))
    plot.show()
