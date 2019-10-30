# -*- coding: utf-8 -*-
"""
author: Richard Bonye (github Boyne272)
Last updated on Mon Oct  7 10:32:24 2019
"""

import numpy as np
from scipy.signal import convolve2d


# laplacian array
LAP_8 = np.array([[1., 1., 1.],
                  [1., -8., 1.],
                  [1., 1., 1.]])
LAP_4 = np.array([[0., 1., 0.],
                  [1., -4., 1.],
                  [0., 1., 0.]])
LAP_H = np.array([[1., 2., 1.],
                  [0., 0., 0.],
                  [-1., -2., -1.]])
LAP_V = np.array([[-1., 0., 1.],
                  [-2., 0., 2.],
                  [-1., 0., 1.]])


LAP_HORIZONTAL = np.ones((3, 3))


def outline(mask, multi=True, diag=True):
    """
    Take the given integer mask and use a laplacian convolution
    to find the outlines.

    Parameters
    ----------

    mask : 2D array
        The mask to find edges of

    multi : bool (optional)
        If set to true a combination of horizontal and vertical laplacians
        will be used to reduce the chance of particular mask geometries
        tircking the convolution algorithm.

    diag : bool (optional)
        decides if diagonals are to be included in the laplacian

    """

    # create the laplacian
    lap = LAP_8 if diag else LAP_4

    # do the convolution to find the edges
    conv = convolve2d(mask, lap, mode='valid').astype(bool)

    if multi:
        conv2 = convolve2d(mask, LAP_H, mode='valid').astype(bool)
        conv3 = convolve2d(mask, LAP_V, mode='valid').astype(bool)
        conv = (conv + conv2 + conv3).astype(bool)

    # pad back boarders to have same shape as original image
    conv = np.pad(conv, 1, 'edge')

    return conv.astype(float)
