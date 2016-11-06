#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check the fundamental `martin_eqn` by recreating the example in
Martin et al (1998)[1]_, then do some reverse engineering to check
other solvers.


References
----------
.. [1] Martin JC, Milliken DL, Cobb JE, McFadden KL, Coggan AR. Validation of
       a Mathematical Model for Road Cycling Power. Journal of Applied
       Biomechanics 14: 276--291, 1998.

"""
import numpy as np

from rouleur import CyclingParams


MARTIN1998_EXAMPLE = {
    "mass_total": 90,
    "rider_velocity": 471.8 / 56.42,
    "road_gradient": 0.003,

    "air_density": 1.2234,
    "CdA": 0.2565,
    "Fw": 0.0044,

    "wind_velocity": 2.94,
    "wind_direction": 310,
    "travel_direction": 340,

    "initial_rider_velocity": 8.28,
    "final_rider_velocity": 8.45,
    "delta_time": 56.42
}

pars = CyclingParams(**MARTIN1998_EXAMPLE)
pars['power_output'] = pars.solve_for.power_output()
assert pars.power_output.round() == 213

assert np.isclose(
    round(pars.solve_for.rider_velocity(), 1),
    pars.rider_velocity.round(1))

# Generate some speed and expected power values,
# then use these to reverse-engineer the input
# CdA (0.2565).
pars['rider_velocity'] = pars['rider_velocity'] * np.arange(1, 2, 0.1)
pars.calculate_air_velocity(inplace=True)
pars['power_output'] = pars.solve_for.power_output()

assert np.isclose(
    round(pars.solve_for.CdA(), 2),
    pars.CdA.round(2))
