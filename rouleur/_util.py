#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Miscellaneous utility functions.

"""
from functools import wraps
import numbers
from warnings import warn

import numpy as np
from scipy import constants

import rouleur._exceptions as exc


GAS_CONST_VAPOUR = 461.495  # constant for water vapor, J/(kg*degK)
GAS_CONST_DRY = 287.05      # specific gas constant, J/(kg*degK) for dry air


class DefaultFill:
    """A descriptor for the `default_fill` attribute of a SlottedDict."""

    def __init__(self, value=None):
        self._check(value)
        self.val = value

    def __get__(self, instance, owner):
        return self.val

    def __set__(self, instance, value):
        self._check(value)

        old_fill, new_fill = self.val, value

        # Propagate changes in the instance.
        for key, item in instance.items():
            if item is old_fill:
                instance[key] = new_fill

        self.val = new_fill

    def __delete__(self, instance):
        raise exc.DefaultFillError("can't delete default fill attribute")

    def __repr__(self):
        return repr(self.val)

    def __str__(self):
        return str(self.val)

    def _check(self, value):
        if isinstance(value, np.ndarray) and onedim(value):
            return
        elif is_scalar(value):
            return
        else:
            raise exc.DefaultFillError('expected a scalar value')


def copy_self(func):
    """Decorator for copying self of a bound method."""
    @wraps(func)
    def wrapper(instance, *args, **kwargs):
        return func(instance.copy(), *args, **kwargs)
    return wrapper


def solver_method(func):
    """Decorator for bound methods of the ``Solvers`` class."""
    @wraps(func)
    def wrapper(solver, *args, **kwargs):
        out = func(solver.params_instance.copy(), *args, **kwargs)
        # Return scalar values if appropriate.
        if isinstance(out, np.ndarray) and len(out) == 1:
            return float(out)
        else:
            return out

    return wrapper


def is_scalar(x):
    """
    >>> is_scalar(1)
    True
    >>> is_scalar(1.1)
    True
    >>> is_scalar([1, 2, 3])
    False
    >>> is_scalar((1, 2, 3))
    False
    >>> is_scalar({'a': 1})
    False
    >>> is_scalar({1, 2, 3})
    False
    >>> is_scalar('spam')
    True
    """
    try:
        len(x)
    except TypeError:
        return True
    else:
        return isinstance(x, (str, bytes))


def process_new_par(value, key=None):
    """Check and prepare incoming model parameters

    Parameters
    ----------
    value : number, scalar or array
        The parameter value to be checked.
    key : str, optional
        The name of the parameter; used for error messages.
    """
    if isinstance(value, numbers.Real):
        return array_len1(value)

    elif not is_scalar(value):
        # asarray will raise an error if the dtype can't be used,
        # e.g. an array of strings.
        out = np.asarray(value, dtype=np.float64)

        # arrays also need to be one dimensional.
        if not onedim(out):
            raise exc.TypeError(
                'array parameters should be one dimensional; %s is not' % key)

        return out

    else:  # unrecognised type
        raise exc.TypeError(
            '%s ride parameters should be numeric (scalar or array)' % key)

def forward_integ_arg_check(start, Hz):
    """Default for both incoming arguments is None."""
    if start is None:
        raise exc.TypeError('starting velocity (start) argument is required')

    if Hz is None:
        raise exc.TypeError('sampling frequency (Hz) argument is required')

    if start == 0:
        raise exc.ZeroDivisionError('starting velocity cannot be zero')

    if start < 1:
        warn('starting_velocity should probably be > 1 m/s')


def array_len1(x):
    """ 42 --> np.array([ 42.]) """
    return np.array(x, dtype=np.float64, ndmin=1)


def onedim(x):
    """
    >>> x = np.array([1, 2, 3])
    >>> onedim(x)
    True
    >>> x = np.array([[1, 2, 3], [4, 5, 6]])
    >>> onedim(x)
    False
    """
    return len(x.shape) == 1


def print_value(value, ndigits=3):
    """Prepare a parameter value for printing, used in CyclingParams.__str__"""
    if len(value) == 1:
        return format(value[0], '.%df' % ndigits)
    else:
        return 'array(%d)' % len(value)


def calculate_air_density(temperature_C, mmHg, rel_humidity):
    """Calculate air density from other known weather conditions

    Parameters
    ----------
    temperature_C : scalar number
        Atmospheric temperature in degrees celsius.
    mmHg : scalar number
        Atmospheric temperature in Torr.
    rel_humidity : scalar number
        Relative humidity as a decimal fraction.

    Examples
    --------
        >>> calculate_air_density(22, 750, 0.5)
        1.1743250592193775

    NOTE: dew point for the above example would be 11.1 degC. Pressure is
          1 bar (1000 mb).

    References
    ----------
    https://wahiduddin.net/calc/density_altitude.htm
    """
    p_vapour = vapour_pressure(temperature_C) * rel_humidity
    p_dry = mmHg * constants.mmHg - p_vapour

    temperature_K = temperature_C + constants.zero_Celsius

    rho = ((p_dry / GAS_CONST_DRY / temperature_K)
           + (p_vapour / GAS_CONST_VAPOUR / temperature_K))

    return rho


def vapour_pressure(temperature_C):
    """Tetens' formula for calculating saturation pressure of water vapor,
    return value is in pascals.

    https://wahiduddin.net/calc/density_altitude.htm

    NOTE: 1 Pascal == 100 mb

        >>> vapour_pressure(22) / 100   # in mb
        26.43707387256724
    """
    return 100 * 6.1078 * 10**(
        (7.5 * temperature_C) / (237.3 + temperature_C))
