# -*- coding: utf-8 -*-
"""
author: Richard Bonye (github Boyne272)
Last updated on Tue Oct  1 13:44:41 2019
"""


import numpy as np
import matplotlib.pyplot as plt


def split_indexs(dims, splits, overlap=0):
    "Split an image with dimension dims by splits times in each dimension"

    assert len(dims) == 2, 'Must be 2d image dimensions'
    assert len(splits) == 2, 'Must be 2d image splits'
    assert all([isinstance(i, int) for i in (*dims, *splits)]), \
        'every entry must be an integer'

    nx, ny = splits
    x_low = np.linspace(0, dims[1], nx+1, dtype=int)[:-1] - overlap
    x_low[0] += overlap
    x_hig = np.linspace(0, dims[1], nx+1, dtype=int)[1:] + overlap
    x_hig[-1] -= overlap
    y_low = np.linspace(0, dims[0], ny+1, dtype=int)[:-1] - overlap
    y_low[0] += overlap
    y_hig = np.linspace(0, dims[0], ny+1, dtype=int)[1:] + overlap
    y_hig[-1] -= overlap
    indexs = np.zeros([int(nx*ny), 4], dtype=int)

    indexs[:, 0] = np.repeat(x_low, ny)
    indexs[:, 1] = np.repeat(x_hig, ny)
    indexs[:, 2] = np.hstack([y_low]*nx)
    indexs[:, 3] = np.hstack([y_hig]*nx)

    return indexs


def test_split_indexs():

    x, y = np.meshgrid(*[np.linspace(-1, 1, 1000)]*2)
    img = mandelbrot(x + y*1j)
    plt.imshow(img)

    split_grid = [3, 2]
    indexs = split_indexs(img.shape[:2], split_grid, 0)
    fig, axs = plt.subplots(*split_grid)

    print('Image shape:', img.shape[:2])
    print('Segmentation Indexs:\n', indexs, '\n Image Dimensions:')

    for (i,j,p,q), ax in zip(indexs, axs.ravel()):
        print(img[i:j, p:q].shape[:2])
        ax.imshow(img[i:j, p:q])


def mandelbrot(c, max_iter=80):
    n = np.zeros(c.shape)
    z = np.zeros_like(c)
    for i in range(max_iter):
        z = z*z + c
        msk = abs(z) > 2
        z[msk] = c[msk] = 0
        n[msk] = i
    return n


if __name__ == '__main__':
    test_split_indexs()
