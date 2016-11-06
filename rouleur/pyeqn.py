#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A pure-python version of Martin et al.[1]_'s equation for modelling cycling
power output.

Vectorised math operations are used, so inputs can be any combination of
(broadcastable) scalars and arrays.

The function can optionally return a `breakdown` of power components that
contribute to the total; i.e. the contributions of air resistance, rolling
resistance...etc.

References
----------
.. [1] Martin JC, Milliken DL, Cobb JE, McFadden KL, Coggan AR. Validation of
       a Mathematical Model for Road Cycling Power. Journal of Applied
       Biomechanics 14: 276--291, 1998.

"""
from numpy import sin, cos, arctan


def martin_eqn(*,
               power_output,
               rider_velocity,
               air_velocity,
               air_density,
               CdA,
               Fw,
               Crr,
               mass_total,
               road_gradient,
               chain_efficiency_factor,
               g,
               wind_velocity,
               wind_direction,
               travel_direction,
               initial_rider_velocity,
               final_rider_velocity,
               delta_time,
               wheel_inertia,
               tire_radius,

               breakdown=False):

    air_resistance = (
        air_velocity**2
        * rider_velocity
        * 0.5
        * air_density
        * (CdA + Fw)
    )

    rolling_resistance = (
        rider_velocity
        * Crr
        * mass_total
        * g
        * cos(arctan(road_gradient))
    )

    wheel_friction = (
        rider_velocity
        * (91 + 8.7 * rider_velocity)
        * 0.001
    )

    potential_energy_changes = (
        rider_velocity
        * mass_total
        * g
        * sin(arctan(road_gradient))
    )

    kinetic_energy_changes = (
        0.5
        * (mass_total + wheel_inertia / tire_radius**2)
        * (final_rider_velocity**2 - initial_rider_velocity**2)
        / delta_time
    )

    if breakdown:
        return {'air_resistance': air_resistance,
                'rolling_resistance': rolling_resistance,
                'wheel_friction': wheel_friction,
                'potential_energy_changes': potential_energy_changes,
                'kinetic_energy_changes': kinetic_energy_changes}

    out = (
        air_resistance
        + rolling_resistance
        + wheel_friction
        + potential_energy_changes
        + kinetic_energy_changes

    ) / chain_efficiency_factor

    return out
