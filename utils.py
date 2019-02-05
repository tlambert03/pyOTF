#!/usr/bin/env python
# -*- coding: utf-8 -*-
# utils.py
# Utility functions for the pyOTF module
# Copyright (c) 2016, David Hoffman
import numpy as np
try:
    from pyfftw.interfaces.numpy_fft import (fftshift, ifftshift,
                                             fftn, ifftn)
    import pyfftw
    # Turn on the cache for optimum performance
    pyfftw.interfaces.cache.enable()
except ImportError:
    from numpy.fft import fftshift, ifftshift, fftn, ifftn
from scipy.signal import signaltools as sig

#from dphutils import slice_maker


def _calc_pad(oldnum, newnum):
    """ Calculate the proper padding for fft_pad

    We have three cases:
    old number even new number even
    >>> _calc_pad(10, 16)
    (3, 3)

    old number odd new number even
    >>> _calc_pad(11, 16)
    (2, 3)

    old number odd new number odd
    >>> _calc_pad(11, 17)
    (3, 3)

    old number even new number odd
    >>> _calc_pad(10, 17)
    (4, 3)

    same numbers
    >>> _calc_pad(17, 17)
    (0, 0)

    from larger to smaller.
    >>> _calc_pad(17, 10)
    (-4, -3)
    """
    # how much do we need to add?
    width = newnum - oldnum
    # calculate one side, smaller
    pad_s = width // 2
    # calculate the other, bigger
    pad_b = width - pad_s
    # if oldnum is odd and newnum is even
    # we want to pull things backward
    if oldnum % 2:
        pad1, pad2 = pad_s, pad_b
    else:
        pad1, pad2 = pad_b, pad_s
    return pad1, pad2


def _calc_crop(s1, s2):
    """Calc the cropping from the padding"""
    a1 = abs(s1) if s1 < 0 else None
    a2 = s2 if s2 < 0 else None
    return slice(a1, a2, None)


def padding_slices(oldshape, newshape):
    """This function takes the old shape and the new shape and calculates
    the required padding or cropping.newshape

    Can be used to generate the slices needed to undo fft_pad above"""
    # generate pad widths from new shape
    padding = tuple(_calc_pad(o, n) if n is not None else _calc_pad(o, o)
                    for o, n in zip(oldshape, newshape))
    # Make a crop list, if any of the padding is negative
    slices = tuple(_calc_crop(s1, s2) for s1, s2 in padding)
    # leave 0 pad width where it was cropped
    padding = [(max(s1, 0), max(s2, 0)) for s1, s2 in padding]
    return padding, slices


def fft_pad(array, newshape=None, mode='median', **kwargs):
    """Pad an array to prep it for fft"""
    # pull the old shape
    oldshape = array.shape
    if newshape is None:
        # update each dimension to a 5-smooth hamming number
        newshape = tuple(sig.fftpack.helper.next_fast_len(n) for n in oldshape)
    else:
        if isinstance(newshape, int):
            newshape = tuple(newshape for n in oldshape)
        else:
            newshape = tuple(newshape)
    # generate padding and slices
    padding, slices = padding_slices(oldshape, newshape)
    return np.pad(array[slices], padding, mode=mode, **kwargs)


def easy_fft(data, axes=None):
    """utility method that includes fft shifting"""
    return fftshift(
        fftn(
            ifftshift(
                data, axes=axes
            ), axes=axes
        ), axes=axes)


def easy_ifft(data, axes=None):
    """utility method that includes fft shifting"""
    return ifftshift(
        ifftn(
            fftshift(
                data, axes=axes
            ), axes=axes
        ), axes=axes)


def cart2pol(y, x):
    """utility function for converting from cartesian to polar"""
    theta = np.arctan2(y, x)
    rho = np.hypot(y, x)
    return rho, theta


class NumericProperty(property):
    """Define a property that must be numeric"""

    def __init__(self, fget=None, fset=None, fdel=None, doc=None,
                 attr=None, vartype=None):
        """A property that must be numeric.

        Parameters
        ----------
        fget : callable or None
            Callable function to get the parameter
            Must have signature fget(self)
        fset : callable or None
            Callable function to set the parameter
            Must have signature fset(self, value)
        fdel : callable or None
            Callable function to delete the parameter
            Must have signature fdel(self)
        doc : str
            Docstring for the parameter
        attr : str
            The name of the backing attribute.
        vartype : type
            The type to validate, defaults to `int`
        """
        if attr is not None and vartype is not None:
            self.attr = attr
            self.vartype = vartype

            def fget(obj):
                return getattr(obj, self.attr)

            def fset(obj, value):
                if not isinstance(value, self.vartype):
                    raise TypeError(
                        "{} must be an {}, var = {!r}".format(
                            self.attr, self.vartype, value)
                    )
                if value <= 0:
                    raise ValueError(
                        "{} must be larger than 0".format(self.attr)
                    )
                if getattr(obj, self.attr, None) != value:
                    setattr(obj, self.attr, value)
                    # call update code
                    obj._attribute_changed()
        super().__init__(fget, fset, fdel, doc)


def center_data(data):
    """Utility to center the data

    Parameters
    ----------
    data : ndarray
        Array of data points

    Returns
    -------
    centered_data : ndarray same shape as data
        data with max value at the central location of the array
    """
    # copy data
    centered_data = data.copy()
    # extract shape and max location
    data_shape = data.shape
    max_loc = np.unravel_index(data.argmax(), data_shape)
    # iterate through dimensions and roll data to the right place
    for i, (x0, nx) in enumerate(zip(max_loc, data_shape)):
        centered_data = np.roll(centered_data, nx // 2 - x0, i)
    return centered_data


def remove_bg(data, multiplier=1.5):
    """Utility that measures mode of data and subtracts a multiplier of it"""
    # should add bit for floats, that will find the mode using the hist
    # function bincounts with num bins specified
    mode = np.bincount(data.ravel()).argmax()
    return data - multiplier * mode


def psqrt(data):
    """Take the positive square root, negative values will be set to zero."""
    # make zero array
    sdata = np.zeros_like(data, float)
    # fill only sqrt of positive values
    sdata[data > 0] = np.sqrt(data[data > 0])
    return sdata


def prep_data_for_PR(data, xysize=None, multiplier=1.5):
    """A utility to prepare data for phase retrieval

    Will pad or crop to xysize and remove mode times multiplier

    Parameters
    ----------
    data : ndarray
        The PSF data to prepare for phase retrieval
    xysize : int
        Size to pad or crop `data` to along the y, x dimensions
    multiplier : float
        The amount to by which to multiply the mode before subtracting

    Returns
    -------
    prepped_data : ndarray
        The data that has been prepped for phase retrieval.
    """
    # pull shape
    nz, ny, nx = data.shape
    # remove background
    data_without_bg = remove_bg(data, multiplier)
    # figure out padding or cropping
    if xysize is None:
        xysize = max(ny, nx)
    if xysize == ny == nx:
        pad_data = data_without_bg
    elif xysize >= max(ny, nx):
        pad_data = fft_pad(data_without_bg, (nz, xysize, xysize),
                           mode="constant")
    else:
        raise NotImplementedError('removed dphutils, cannot crop')
        # if need to crop, crop and center and return
        #my_slice = slice_maker(((ny + 1) // 2, (nx + 1) // 2), xysize)
        #return center_data(data_without_bg)[[Ellipsis] + my_slice]
    # return centered data
    return center_data(pad_data)
