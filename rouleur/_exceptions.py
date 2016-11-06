#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exception classes for this module.

"""
class RouleurException(Exception):
    pass


class DefaultFillError(RouleurException):
    pass


class SlottedDictError(RouleurException):
    pass


class CyclingParamsError(RouleurException):
    pass


class SolverError(RouleurException):
    pass


# Multiple inheritance so that built-in and package-specific
# exceptions can be caught.

class KeyError(SlottedDictError, KeyError):
    pass


class TypeError(CyclingParamsError, TypeError):
    pass


class RuntimeError(SolverError, RuntimeError):
    pass


class ZeroDivisionError(SolverError, ZeroDivisionError):
    pass
