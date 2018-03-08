"""
A test validating mina-weld agains Zheng Fan's MATLAB implementation from the
NDE Group repository.

Created by: Michal K. Kalkowski
20/02/2018

License: MIT
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import mina.mina_model as mina

# Create weld model
weld_parameters = dict([('remelt_h', 0.362),
                        ('remelt_v', 0.173),
                        ('theta_b', np.deg2rad(20.5)),
                        ('theta_c', np.deg2rad(5.63)),
                        ('order', 'left_to_right'),
                        ('number_of_layers', 30),
                        ('number_of_passes', np.array([1]*5 + 9*[2] + 14*[3] +
                                                      [4]*2)),
                        ('electrode_diameter', np.array([1.2]*30)),
                        ('a', 60),
                        ('b', 4),
                        ('c', 40)])
weld = mina.MINA_weld(weld_parameters)
weld.define_grid_size(1., use_centroids=True)
weld.solve()
#weld.fill_missing()



