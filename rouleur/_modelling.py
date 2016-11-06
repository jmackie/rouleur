#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The business end of this package.

The entire API really revolves around the `CyclingParams` class, which is a
dictionary-like object with methods for the physical modelling of cycling.

Important references for this module are Martin et al. (1998)[1]_, Martin et
al. (2006)[2]_, Atkinson et al. (2007)[3]_, and Wells & Marwood (2016)[4]_.


References
----------

.. [1] Martin JC, Milliken DL, Cobb JE, McFadden KL, Coggan AR. Validation of
       a Mathematical Model for Road Cycling Power. Journal of Applied
       Biomechanics 14: 276--291, 1998.
.. [2] Martin JC, Gardner AS, Barras M, Martin DT. Modeling sprint cycling
       using field-derived parameters and forward integration. Med Sci Sports
       Exerc 38: 592--597, 2006.
.. [3] Atkinson G, Peacock O, Passfield L. Variable versus constant power
       strategies during cycling time-trials: Prediction of time savings using
       an up-to-date mathematical model. Journal of Sports Sciences 25:
       1001--1009, 2007.
.. [4] Wells MS, Marwood S. Effects of power variation on cycle performance
       during simulated hilly time-trials. European Journal of Sport Science
       16: 912--918, 2016.

"""
from collections.abc import Mapping
import copy
from itertools import cycle
from warnings import warn

import numpy as np
from numpy.linalg import lstsq
from scipy import constants
from scipy.optimize import minimize_scalar

import rouleur._util as util
import rouleur._exceptions as exc
from rouleur.pyeqn import martin_eqn
from rouleur.ceqn import martin_eqn_compiled


NA = np.nan


NA_ARRAY = np.array(NA, dtype=np.float64, ndmin=1)   # array([nan])


MODEL_DEFAULTS = (
    # (name, default, unit)

    ('power_output', None, 'watts'),

    ('rider_velocity', None, 'm/s'),
    ('air_velocity', None, 'm/s'),

    ('air_density', 1.225, 'kg/m^3'),   # rho
    ('CdA', 0.258, 'au'),               # effective frontal area
    ('Fw', 0.0044, 'm^2'),              # spoke drag

    ('Crr', 0.0032, 'au'),              # coefficient of rolling resistance

    ('mass_total', 80, 'kg'),           # bike + rider
    ('road_gradient', 0, 'fraction'),   # rise/run

    ('chain_efficiency_factor', 0.976, 'au'),

    ('g', constants.g, 'm/s^2'),

    # Parameters for resolving air velocity
    # -------------------------------------
    ('wind_velocity', None, 'm/s'),
    ('wind_direction', None, 'degrees'),
    ('travel_direction', None, 'degrees'),

    # Parameters for quantifying power related
    # to changes in kinetic energy (KE)
    # ----------------------------------------
    ('initial_rider_velocity', None, 'm/s'),
    ('final_rider_velocity', None, 'm/s'),
    ('delta_time', None, 'seconds'),
    ('wheel_inertia', 0.14, 'kg/m^2'),
    ('tire_radius', 0.311, 'm'),
)


KE_PARAMS = ('initial_rider_velocity', 'final_rider_velocity', 'delta_time')


class SlottedDict(Mapping):

    default_fill = util.DefaultFill()

    def __init__(self, keys, *, default_fill=None, **filling):
        """Initialise a new slotted dictionary.

        The behaviour of this dict-like object is such that item existence is
        assured, if only as 'missing' values (or specifically, the
        `default_fill` value).

        Parameters
        ----------
        keys : list of strings or a single (space- or comma-delimited) string
            This argument defines the 'key slots' for the dictionary. It may
            either be a list of strings, or a single space-delimited string ala
            namedtuple.
        default_fill : scalar, optional
            Any missing (or subsequently deleted) items will assume this value.
            Should probably be None, float('nan') or similar.
        **filling : misc.
            Keyword arguments with which to fill the dictionary initially.
            Should be a subset of those keywords given in `slots`. Any keywords
            provided that are not recognised (i.e. *not* in `slots`) will be
            discarded with a warning.

        Examples
        --------

        >>> sd = SlottedDict('a b c', a=4, b=2)
        >>> sd
        SlottedDict(keys='a b c', default_fill=None, a=4, b=2)

        >>> sd['b'] = 42
        >>> sd
        SlottedDict(keys='a b c', default_fill=None, a=4, b=42)

        >>> sd['d'] = 42
        Traceback (most recent call last):
            ...
        BadAssignmentError: d is not a valid key for this instance

        >>> sd.default_fill = 'missing'
        >>> sd
        SlottedDict(keys='a b c', default_fill='missing', a=4, b=42)
        >>> sd.c
        'missing'
        >>> del sd['b']
        >>> sd.b
        'missing'

        >>> sd['b'] = [5, 6, 7]
        >>> sd.b is sd.copy().b
        False

        Notes
        -----
        The dictionary at the core of any instance is accessible via the `core`
        attribute (readonly).

        The `default_fill` attribute can be modified, and changes will
        propagate.

        """
        try:
            keys = keys.strip().replace(',', ' ').split()
        except AttributeError:  # assume it's already a collection of strings
            pass

        self.__core = {}
        self.__keys = tuple(keys)  # copy!

        for key in keys:
            self.__core[key] = filling.pop(key, default_fill)

        if default_fill is not None:   # i.e. argument specified
            self.default_fill = default_fill

        if filling:   # any arguments unaccounted for?
            err = 'The following kwargs were ignored: {!r}'
            warn(err.format(filling), SyntaxWarning)

    @property
    def core(self):
        return self.__core

    @property
    def keys(self):
        return self.__keys

    # Abstract methods
    # ----------------
    def __getitem__(self, item):
        return self.__core[item]

    def __iter__(self):
        return iter(self.__core)   # delegate

    def __len__(self):
        return len(self.__core)    # delegate

    # Other methods
    # -------------
    def __setitem__(self, key, value):
        if key in self.__keys:
            self.__core[key] = value
        else:
            err = '%s is not a valid key for this instance'
            raise exc.KeyError(err % key)

    def __delitem__(self, item):
        """Rather than removing, set deleted item to the default value."""
        if item in self.__keys:
            self.__core[item] = self.default_fill
        else:
            raise SlottedDict

    def __getattr__(self, name):
        """Allow retrieving dictionary values as attributes."""
        try:
            return self.__core[name]

        except KeyError as e:
            raise exc.KeyError(e)

    def __repr__(self):
        fmt = '{}(keys={!r}, default_fill={!r}, {})'

        cls = type(self).__name__
        keys = ' '.join(self.__keys)
        default_fill = self.default_fill

        kw = '{}={!r}'
        kwds = ', '.join(kw.format(key, self[key])
                         for key in self.__keys   # for consistent ordering
                         if self[key] is not self.default_fill)
        return fmt.format(cls, keys, default_fill, kwds)

    def _copy_filling(self):
        """Used within copy methods."""
        filling = dict.fromkeys(self, self.default_fill)
        filling.update({key: copy.copy(value) for key, value in self.items()
                        if value is not self.default_fill})
        return filling

    def copy(self):
        keys = self.__keys
        filling = self._copy_filling
        cls = type(self)
        return cls(keys, default_fill=self.default_fill, **filling)


class CyclingParams(SlottedDict):
    """A dict-like object to facilitate the physical modelling of cycling.

    When a new instance of this class is created without any keyword
    arguments, an object resembling a dictionary is returned, populated with
    all variables (and sensible defaults) required for the physical modelling
    of cycling [1]_. Any arguments supplied to the constructor must be
    recognised model parameters (*see below*) and will be inserted in place of
    defaults.

    The following parameters are currently recognised:

    =========================   ==================   =========================
    parameter name              units                short description
    =========================   ==================   =========================
    power_output                Watts
    rider_velocity              m/s
    air_velocity                m/s
    air_density                 kg/m**3
    CdA                         a.u.                 Effective frontal area.
    Fw                          m**2                 Spoke drag.
    Crr                         a.u.                 Coefficient of rolling
                                                     resistance
    mass_total                  kg                   Total mass of bike and
                                                     rider.
    road_gradient               decimal fraction     e.g. 0.03 (3%)
    chain_efficiency_factor     decimal fraction     Very close to 1.
    g                           m/s**2               Standard gravity.
    wind_velocity               m/s
    wind_direction              degrees
    travel_direction            degrees
    initial_rider_velocity      m/s
    final_rider_velocity        m/s
    delta_time                  seconds              Time difference
                                                     between measurements of
                                                     `initial_rider_velocity`
                                                     and
                                                     `final_rider_velocity`.
    wheel_inertia               kg/m**2
    tire_radius                 metres
    =========================   ==================   =========================

    Notes
    -----
    **Something about viewing defaults...**

    References
    ----------
    .. [1] Martin JC, Milliken DL, Cobb JE, McFadden KL, Coggan AR. Validation
           of a Mathematical Model for Road Cycling Power. Journal of Applied
           Biomechanics 14: 276-291, 1998.
    """

    default_fill = util.DefaultFill(NA_ARRAY)

    def __init__(self, *, resolve_air_velocity=True, **kwargs):
        """Initialise cycling parameters for modelling.

        Parameters
        ----------
        resolve_air_velocity : bool, optional
            Attempt to resolve the ``air_velocity`` parameter from other
            supplied keyword arguments. Defaults to True.
        **kwargs : numeric scalars or arrays
            keyword arguments must be recognised cycling model parameters (see
            above). These parameters can be either scalars or array-like
            objects, but in any case must be numeric..

        Notes
        -----
        Beware of NaNs in the inputs. If any solve_for.* method returns NaN
        unexpectedly, that's probably why.

        If `rider_velocity` is updated at any point, you'll want to update
        `air_velocity`.
        """

        # Keep track of defaults that have been
        # incorporated into the current instance.
        self._defaults_in_use = []

        filling = {}

        for key, default, _ in MODEL_DEFAULTS:

            par = kwargs.pop(key, MissingParam(default))

            if isinstance(par, MissingParam):
                par = (self.default_fill if par.value is None
                       else util.array_len1(par.value))
                self._defaults_in_use.append(key)

            else:
                par = util.process_new_par(par, key)  # could complain

            filling[key] = par  # fine

        keys = [key for key, *ignore in MODEL_DEFAULTS]
        super().__init__(keys, **filling)

        self.solve_for = Solvers(self)

        if resolve_air_velocity:
            if is_nan(self['air_velocity']):
                self.calculate_air_velocity(inplace=True)

            if is_nan(self['air_velocity']):  # no luck? still using default
                self._defaults_in_use.append('air_velocity')

    @property
    def _wind_tan(self):
        """Calculate the tangent wind component."""
        wtan = self['wind_velocity'] * np.cos(np.radians(
            self['travel_direction'] - self['wind_direction']))

        # 0 implies windless conditions.
        return 0 if is_nan(wtan) else wtan

    def calculate_air_velocity(self, inplace=False):
        """Attempt to resolve the ``air_velocity`` parameter.

        Parameters
        ----------
        inplace : bool
            Should modifications be made directly to the current instance
            (True), or a modified copy returned (False)?
        """
        air_vel = self['rider_velocity'] + self._wind_tan
        if is_nan(air_vel):
            air_vel = self.default_fill

        if inplace:
            self['air_velocity'] = air_vel
        else:
            out = self.copy()
            out['air_velocity'] = air_vel
            return out

    def __setitem__(self, key, value):
        """Watch item setting.

        Setting needs to be watched so that:
            + the `_defaults_in_use` attribute can be updated.
            + parameters can be cast to ndarrays and checked for type etc.
        """
        try:
            idx = self._defaults_in_use.index(key)
        except ValueError:
            pass
        else:
            self._defaults_in_use.pop(idx)

        value = util.process_new_par(value, key)
        super().__setitem__(key, value)   # could raise an error

    def __delitem__(self, item):
        if item in self:
            self._defaults_in_use.append(item)
        super().__delitem__(item)

    def __repr__(self):
        fmt = 'CyclingParams({})'
        kw = '{}={!r}'
        kwds = ', '.join(kw.format(key, self[key])
                         for key in self.keys   # for consistent ordering
                         if key not in self._defaults_in_use)
        return fmt.format(kwds)

    @util.copy_self
    def _ignore_KE(pars, arbitrary_value=42):
        """Assign an arbitrary value, and thus nullify, those parameters
        related to kinetic energy changes."""
        for key in KE_PARAMS:
            pars[key] = arbitrary_value
        return pars

    def _missing(self, keys):
        """Pass to `any` and `all` tests for 'missing' parameters."""
        for value in (self.get(key) for key in keys):
            if util.is_scalar(value):
                yield value is None or not np.isfinite(value)
            else:
                yield not np.isfinite(value[0])  # for np.array([nan])

    def _gen_rows(self, nrow=None):
        """In effect: expand parameters (with recycling) to a table and
        iterate over the rows."""
        if nrow is None:
            nrow = max(len(value) for value in self.values())

        # itertools.cycle will produce scalars
        generator_dict = {key: cycle(value) for key, value in self.items()}
        for i in range(nrow):
            yield {key: next(it) for key, it in generator_dict.items()}

    def copy(self):
        cls = type(self)
        filling = self._copy_filling()
        return cls(resolve_air_velocity=False, **filling)

    @classmethod
    def defaults(cls):
        """Display modelling defaults."""
        return {key: value for key, value, _ in MODEL_DEFAULTS
                if value is not None}

    @property
    def supplied(self):
        """Display modelling parameters that have been supplied by the user."""
        return {key: value for key, value in self.items()
                if key not in self._defaults_in_use}

    @property
    def all_scalar(self):
        """Are all parameters of length 1?"""
        return all(len(value) == 1 for value in self.values())

    def __str__(self):
        """Pretty tabular representation of parameters."""
        row_fmt = '{{:<{0}}}{{:>{1}}}{{:<{2}}}'   # ew

        rows = [('Parameter', 'Value', 'Unit')]   # header
        rows.extend((key, util.print_value(self[key]), unit)
                    for key, _, unit in MODEL_DEFAULTS)

        colwidths = [max(len(col) for col in cols) + 3   # spacing
                     for cols in zip(*rows)]

        # Include an underline.
        rows.insert(1, tuple('-' * (width - 1) for width in colwidths))

        row_fmt = row_fmt.format(*colwidths)

        lines = (row_fmt.format(par, val, '  ' + unit)   # padding
                 for par, val, unit in rows)
        return '\n'.join(lines)


class Solvers:
    """The `solve_for` attribute of a `CyclingParams` instance."""
    def __init__(self, params_instance):
        self.params_instance = params_instance

    @util.solver_method
    def power_output(pars, breakdown=False):
        """Solve power output.

        Implements the equation first published in Martin et al. (1998)[1]_.

        Parameters
        ----------
        breakdown : bool, optional
            Return a dictionary describing the contribution of different
            resistances to total power output?

        References
        ----------
        .. [1] Martin JC, Milliken DL, Cobb JE, McFadden KL, Coggan AR.
               Validation of a Mathematical Model for Road Cycling Power.
               Journal of Applied Biomechanics 14: 276--291, 1998.
        """
        if all(pars._missing(KE_PARAMS)):
            pars = pars._ignore_KE()

        return martin_eqn(breakdown=breakdown, **pars.core)

    @util.solver_method
    def rider_velocity(pars, *, start=None, Hz=None):
        """Solve for rider velocity.

        There are two possible approaches to be taken here, depending on the
        nature of the parameters passed. If all parameters are "scalar" (i.e.
        length one), then a simple scalar minimisation is performed to find
        the ``rider_velocity`` that produces an estimate of power output
        closest to that actually observed. Alternatively, if any parameters
        have a length >1, a forward integration procedure is used[1]_.

        Parameters
        ----------
        start : scalar number, optional
            If using forward integration, this is the initial velocity
            (metres/second) with which to start that procedure.
        Hz : scalar number, optional
            If using forward integration, this is the sampling frequency of
            the data. Hence, passed data should be sampled consistently.

        References
        ----------
        .. [1] Martin JC, Gardner AS, Barras M, Martin DT. Modeling sprint
               cycling using field-derived parameters and forward integration.
               Med Sci Sports Exerc 38: 592--597, 2006.
        """
        pars = pars._ignore_KE()   # important!

        if pars.all_scalar:
            # Scalar minimisation approach
            # ----------------------------
            wtan = pars._wind_tan      # rm overhead from the optimisation
            pars = {key: float(value)  # simplify parameters to a dict
                    for key, value in pars.items()}

            solution = minimize_scalar(
                velocity_optimiser, args=(pars, wtan),
                # no negative values allowed
                method='bounded', bounds=(0, 50))

            if not solution.success:
                raise exc.RuntimeError('minimisation was unsuccessful')
            else:
                return solution.x

        else:
            # Forward integration approach
            # ----------------------------
            util.forward_integ_arg_check(start, Hz)

            nrow = max(len(value) for value in pars.values())
            velocities = np.zeros(nrow, dtype=np.float64)
            velocities[0] = start
            # HACK: but saves some repetition!
            wind_tan = CyclingParams._wind_tan.fget

            for i, row in enumerate(pars._gen_rows(nrow)):

                if i == (nrow - 1):   # zero-indexing
                    break

                vel = velocities[i]
                row.update(rider_velocity=vel,
                           air_velocity=vel + wind_tan(row))
                excess_power = row['power_output'] - martin_eqn_compiled(**row)
                accel = excess_power / row['mass_total'] / vel
                velocities[i + 1] = vel + accel / Hz

            return velocities

    @util.solver_method
    def CdA(pars):
        """Solve for effective frontal area (CdA).

        Uses a multiple linear regression approach to derive CdA[1]_, provided
        rider power output and velocity.

        References
        ----------
        .. [1] Martin JC, Gardner AS, Barras M, Martin DT. Modeling sprint
               cycling using field-derived parameters and forward integration.
               Med Sci Sports Exerc 38: 592--597, 2006.
        """

        # Assemble response (y) values; following the paper.
        P, E = pars['power_output'], pars['chain_efficiency_factor']

        pwrs = martin_eqn(breakdown=True, **pars.core)
        dPE = or_zero(pwrs['potential_energy_changes'])
        dKE = or_zero(pwrs['kinetic_energy_changes'])

        y = P * E - dPE - dKE

        x1 = (0.5                         # CdA
              * pars['air_density']
              * pars['air_velocity']**2
              * pars['rider_velocity'])
        x2 = (pars['rider_velocity']      # mu (global coefficient of friction)
              * pars['mass_total']
              * pars['g'])

        const = np.ones(len(x1))          # we are modelling an intercept?
        mx = np.column_stack([x1, x2, const])
        CdA, mu, intercept = lstsq(mx, y)[0]
        return CdA - or_zero(pars['Fw'])


class MissingParam:
    """Tag a default model parameter.

    Used to make it clear that a default parameter is being used in the
    CyclingParams.__init__ method, thus making argument checks cleaner.
    """
    def __init__(self, value):
        self.value = value


def velocity_optimiser(rider_velocity, pars: dict, wind_tan):
    """Function to be passed to ``scipy.optimize.minimize_scalar``.

    Returns the absolute difference between modelled power output and
    actual power output for any given `rider_velocity`. This function
    can thus be minimised to model velocity from power output.
    """
    pars.update(rider_velocity=rider_velocity,
                air_velocity=rider_velocity + wind_tan)
    epsilon = abs(martin_eqn_compiled(**pars) - pars['power_output'])
    return epsilon


def is_nan(x: np.ndarray):
    return np.isnan(x).all()    # can deal with array inputs


def or_zero(x):
    """Return 0 instead of NaN."""
    return 0 if is_nan(x) else x
