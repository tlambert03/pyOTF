#!/usr/bin/env python
# -*- coding: utf-8 -*-
# phaseretrieval.py
"""
Back focal plane (pupil) phase retrieval algorithm base on:
[(1) Hanser, B. M.; Gustafsson, M. G. L.; Agard, D. A.; Sedat, J. W.
Phase Retrieval for High-Numerical-Aperture Optical Systems.
Optics Letters 2003, 28 (10), 801.](dx.doi.org/10.1364/OL.28.000801)

Copyright (c) 2016, David Hoffman
"""
import copy
import numpy as np
try:
    from pyfftw.interfaces.numpy_fft import fftshift, ifftshift, fftn
    import pyfftw
    # Turn on the cache for optimum performance
    pyfftw.interfaces.cache.enable()
except ImportError:
    from numpy.fft import fftshift, ifftshift, fftn

from numpy.linalg import lstsq
from .utils import psqrt, fft_pad
from .otf import HanserPSF
from .zernike import zernike, noll2name
from skimage.restoration import unwrap_phase
from matplotlib import pyplot as plt

import logging

logger = logging.getLogger(__name__)


def retrieve_phase(data, params, max_iters=200, pupil_tol=1e-8,
                   mse_tol=1e-8, phase_only=False, mclass=HanserPSF):
    """Retrieve the phase across the objective's back pupil from an
    experimentally measured PSF.

    Follows: [Hanser, B. M.; Gustafsson, M. G. L.; Agard, D. A.;
    Sedat, J. W. Phase Retrieval for High-Numerical-Aperture Optical Systems.
    Optics Letters 2003, 28 (10), 801.](dx.doi.org/10.1364/OL.28.000801)

    Parameters
    ----------
    data : ndarray (3 dim)
        The experimentally measured PSF of a subdiffractive source
    params : dict
        Parameters to pass to HanserPSF, size and zsize will be automatically
        updated from data.shape
    max_iters : int
        The maximum number of iterations to run, default is 200
    pupil_tol : float
        the tolerance in percent change in change in pupil, default is 1e-8
    mse_tol : float
        the tolerance in percent change for the mean squared error between
        data and simulated data, default is 1e-8
    phase_only : bool
        True means only the phase of the back pupil is retrieved while the
        amplitude is not.

    Returns
    -------
    PR_result : PhaseRetrievalResult
        An object that contains the phase retrieval result
    """
    # make sure data is square
    assert data.shape[1] == data.shape[2], "Data is not square in x/y"
    assert data.ndim == 3, "Data doesn't have enough dims"
    # make sure the user hasn't screwed up the params
    params.update(dict(
        vec_corr="none",
        condition="none",
        zsize=data.shape[0],
        size=data.shape[-1]
    ))
    # assume that data prep has been handled outside function
    # The field magnitude is the square root of the intensity
    mag = psqrt(data)
    # generate a model from parameters
    model = mclass(**params)
    # generate coordinates
    model._gen_kr()
    # start a list for iteration
    mse = np.zeros(max_iters)
    mse_diff = np.zeros(max_iters)
    pupil_diff = np.zeros(max_iters)
    # generate a pupil to start with
    new_pupil = model._gen_pupil()
    # save it as a mask
    mask = new_pupil.real
    # iterate
    old_mse, old_pupil = None, None
    for i in range(max_iters):
        # generate new mse and add it to the list
        model._gen_psf(new_pupil)
        new_mse = _calc_mse(data, model.PSFi)
        mse[i] = new_mse
        if i > 0:
            # calculate the difference in mse to test for convergence
            mse_diff[i] = abs(old_mse - new_mse) / old_mse
            # calculate the difference in pupil
            pupil_diff[i] = (abs(old_pupil - new_pupil)**2).mean() / (abs(old_pupil)**2).mean()
        else:
            mse_diff[i] = np.nan
            pupil_diff[i] = np.nan
        # check tolerances, how much has the pupil changed, how much has the mse changed
        # and what's the absolute mse
        logger.info("Iteration {}, mse_diff = {:.2g}, pupil_diff = {:.2g}".format(i, mse_diff[i], pupil_diff[i]))
        if pupil_diff[i] < pupil_tol or mse_diff[i] < mse_tol or mse[i] < mse_tol:
            break
        # update old_mse
        old_mse = new_mse
        # retrieve new pupil
        old_pupil = new_pupil
        # keep phase
        phase = np.angle(model.PSFa.squeeze())
        # replace magnitude with experimentally measured mag
        new_psf = mag * np.exp(1j * phase)
        # generate the new pupils
        new_pupils = fftn(ifftshift(new_psf, axes=(1, 2)), axes=(1, 2))
        # undo defocus and take the mean
        new_pupils /= model._calc_defocus()
        new_pupil = new_pupils.mean(0) * mask
        # if phase only discard magnitude info
        if phase_only:
            new_pupil = np.exp(1j * np.angle(new_pupil)) * mask
    else:
        logger.warn("Reach max iterations without convergence")
    mse = mse[:i + 1]
    mse_diff = mse_diff[:i + 1]
    pupil_diff = pupil_diff[:i + 1]
    # shift mask
    mask = fftshift(mask)
    # shift phase then unwrap and mask
    phase = unwrap_phase(fftshift(np.angle(new_pupil))) * mask
    # shift magnitude
    magnitude = fftshift(abs(new_pupil)) * mask
    return PhaseRetrievalResult(magnitude, phase, mse, pupil_diff, mse_diff, model)


class PhaseRetrievalResult(object):
    """An object for holding the result of phase retrieval"""

    def __init__(self, mag, phase, mse, pupil_diff, mse_diff, model):
        """The results of retrieving a pupil function's phase and magnitude

        Paramters
        ---------
        mag : ndarray (n, n)
            Coefficients for the zernike decomposition of the magnitude
        phase : ndarray (n, n)
            Coefficients for the zernike decomposition of the phase
        mse : ndarray (m, )
            Mean squared error as a function of the number of iterations (m)
            performed
        pupil_diff : ndarray (m, )
            The relative change in the retrieved pupil function as a function
            of the number of iterations (m) performed
        mse_diff : ndarray (m, )
            The relative change in the mean squared error as a function of the
            number of iterations (m) performed
        model : HanserPSF object
            the model used to retrieve the pupil function
        """
        # update internals
        self.mag = mag
        self.phase = phase
        self.mse = mse
        self.pupil_diff = pupil_diff
        self.mse_diff = mse_diff
        self.model = model
        # calculate coordinate system
        model._gen_kr()
        r, theta = model._kr, model._phi
        self.r, self.theta = fftshift(r), fftshift(theta)
        # pull specific model parameters
        self.na, self.wl = model.na, model.wl

    def fit_to_zernikes(self, num_zerns):
        """Fits the data to a number of zernikes"""
        # normalize r so that 1 = diffraction limit
        r, theta = self.r, self.theta
        r = r / (self.na / self.wl)
        # generate zernikes
        zerns = zernike(r, theta, np.arange(1, num_zerns + 1))
        mag_coefs = _fit_to_zerns(self.mag, zerns, r)
        phase_coefs = _fit_to_zerns(self.phase, zerns, r)
        self.zd_result = ZernikeDecomposition(mag_coefs, phase_coefs, zerns)
        return self.zd_result

    def generate_psf(self, sphase=slice(4, None, None), size=None, zsize=None,
                     zrange=None):
        """Make a perfect PSF"""
        # make a copy of the internal model
        model = copy.copy(self.model)
        # update zsize or zrange
        if zsize is not None:
            model.zsize = zsize
        if zrange is not None:
            model.zrange = zrange
        # generate the PSF from the reconstructed phase
        if not hasattr(self, 'zd_result'):
            self.fit_to_zernikes(120)
        model._gen_psf(ifftshift(self.zd_result.complex_pupil(sphase=sphase)))
        # reshpae PSF if needed in x/y dimensions
        psf = model.PSFi
        nz, ny, nx = psf.shape
        assert ny == nx, "Something is very wrong"
        if size is not None:
            if nx < size:
                # if size is too small, pad it out.
                psf = fft_pad(psf, (nz, size, size), mode="constant")
            elif nx > size:
                # if size is too big, crop it
                lb = size // 2
                hb = size - lb
                myslice = slice(nx // 2 - lb, nx // 2 + hb)
                psf = psf[:, myslice, myslice]
        # return data
        return psf

    def plot(self, axs=None):
        """Plot the retrieved results"""
        if axs is None:
            fig, (ax_phase, ax_mag) = plt.subplots(1, 2, figsize=(12, 5))
        else:
            (ax_phase, ax_mag) = axs
            fig = ax_phase.get_figure()

        phase_img = ax_phase.matshow(self.phase, cmap="seismic", vmin=-np.pi, vmax=np.pi)
        plt.colorbar(phase_img, ax=ax_phase)
        mag_img = ax_mag.matshow(self.mag, cmap="inferno")
        plt.colorbar(mag_img, ax=ax_mag)
        ax_phase.set_title("Pupil Phase")
        ax_mag.set_title("Pupil Magnitude")
        fig.tight_layout()
        return fig, (ax_phase, ax_mag)

    def plot_convergence(self):
        """Diagnostic plots of the convergence criteria"""
        with np.errstate(invalid="ignore"):
            fig, axs = plt.subplots(3, 1, figsize=(6, 6), sharex=True)
            for ax, data in zip(axs, (self.mse, self.mse_diff, self.pupil_diff)):
                ax.semilogy(data)
            for ax, t in zip(axs, ("Mean Squared Error",
                                   "Relative Change in MSE",
                                   "Relative Change in Pupil")):
                ax.set_title(t)
            fig.tight_layout()
        return fig, axs

    @property
    def complex_pupil(self):
        """Return the complex pupil function"""
        return self.mag * np.exp(1j * self.phase)


def fake_zernike_decomp(mcoefs, pcoefs, model):
    assert mcoefs.size == pcoefs.size
    num_zerns = len(mcoefs)
    r, theta = fftshift(model._kr), fftshift(model._phi)
    zerns = zernike(r, theta, np.arange(1, num_zerns + 1))
    return ZernikeDecomposition(mcoefs, pcoefs, zerns)


class ZernikeDecomposition(object):
    """An object for holding the results of a zernike decomposition"""

    def __init__(self, mag_coefs, phase_coefs, zerns):
        """The results of decomposing a pupil function's phase and magnitude
        into zernike modes

        Paramters
        ---------
        mag_coefs : ndarray (m, )
            Coefficients for the zernike decomposition of the magnitude
        phase_coefs : ndarray (m, )
            Coefficients for the zernike decomposition of the phase
        zerns : ndarray (m, n, n)
            Actual zernike modes used in the decomposition

        """
        # verify inputs make sense
        assert mag_coefs.size == phase_coefs.size == zerns.shape[0]
        self.mcoefs = mag_coefs
        self.pcoefs = phase_coefs
        self.zerns = zerns

    def plot_named_coefs(self):
        """Plot the first 15 zernike mode coefficients

        These coefficients correspond to the classical abberations
        """
        # set up the subplot
        fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6, 6))
        # get the ordered names
        ordered_names = [noll2name[i + 1] for i in range(len(noll2name))]
        # make an x range for the bar plot
        x = np.arange(len(ordered_names)) + 1
        # pull the data
        data = self.pcoefs[:len(ordered_names)]
        # make the bar plot
        ax.bar(x, data, align="center", tick_label=ordered_names)
        # set up axes
        ax.axis("tight")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_ylabel("Phase Coefficient")
        fig.tight_layout()
        # return figure handles
        return fig, ax

    def plot_coefs(self):
        """Same as `plot_named_coefs` but for all coefs"""
        fig, axs = plt.subplots(2, 1, sharex=True)
        for ax, data in zip(axs, (self.mcoefs, self.pcoefs)):
            ax.bar(np.arange(data.size) + 1, data)
            ax.axis("tight")
        for ax, t in zip(axs, ("Magnitude Coefficients",
                               "Phase Coefficients")):
            ax.set_title(t)
        ax.set_xlabel("Noll's Number")
        fig.tight_layout()
        return fig, axs

    def _recon(self, coefs, s=Ellipsis):
        """reconstruct mag or phase, base function for dispatch"""
        return _recon_from_zerns(coefs[s], self.zerns[s])

    def phase(self, *args, **kwargs):
        """Reconstruct the phase from the specified slice"""
        return self._recon(self.pcoefs, *args, **kwargs)

    def mag(self, *args, **kwargs):
        """Reconstruct the magnitude from the specified slice"""
        return self._recon(self.mcoefs, *args, **kwargs)

    def complex_pupil(self, smag=Ellipsis, sphase=Ellipsis, *args, **kwargs):
        """Reconstruct the complex pupil from the specified slice"""
        mag = self.mag(*args, s=smag, **kwargs)
        phase = self.phase(*args, s=sphase, **kwargs)
        return mag * np.exp(1j * phase)


def _calc_mse(data1, data2):
    """utility to calculate mean square error"""
    return ((data1 - data2) ** 2).mean()


def _fit_to_zerns(data, zerns, r, **kwargs):
    """sub function that does the reshaping and the least squares

    Parameters
    ----------
    data : ndarray (n, n)
        phase or magnitude data to fit to zernikes
    zerns : ndarray (m, n, n)
        precalculated zernikes
    r : ndarray (n, n)
        radial coordinate in terms of diffraction limit
        where r = 1 is the diffraction limit

    Returns
    -------
    coefs : ndarray (m, )
        least squares coefficients of the fit of the zernikes to
        data
    """
    # find the points to fit
    valid_points = (r <= 1)
    data2fit = data[valid_points]
    zerns2fit = zerns[:, valid_points].T
    # fit the points
    coefs, _, _, _ = lstsq(zerns2fit, data2fit, **kwargs)
    # return the coefficients
    return coefs


def _recon_from_zerns(coefs, zerns):
    """Utility to reconstruct from coefs"""
    return (coefs[:, np.newaxis, np.newaxis] * zerns).sum(0)


if __name__ == "__main__":
    # phase retrieve a pupil
    import os
    import time
    from skimage.external import tifffile as tif
    from .utils import prep_data_for_PR
    # read in data from fixtures
    data = tif.imread(os.path.split(__file__)[0] + "/fixtures/psf_wl520nm_z300nm_x130nm_na0.85_n1.0.tif")
    # prep data
    data_prepped = prep_data_for_PR(data, 512)
    # set up model params
    params = dict(
        wl=520,
        na=0.85,
        ni=1.0,
        res=130,
        zres=300
    )
    # retrieve the phase
    pr_start = time.time()
    print("Starting phase retrieval")
    pr_result = retrieve_phase(data_prepped, params)
    print("It took {} seconds to retrieve the pupil function".format(
        time.time() - pr_start))
    # plot
    pr_result.plot()
    pr_result.plot_convergence()
    # fit to zernikes
    zd_start = time.time()
    print("Starting zernike decomposition")
    pr_result.fit_to_zernikes(120)
    print("It took {} seconds to fit 120 Zernikes".format(
        time.time() - zd_start))
    # plot
    pr_result.zd_result.plot_named_coefs()
    pr_result.zd_result.plot_coefs()
    # show
    plt.show()
