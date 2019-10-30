# -*- coding: utf-8 -*-
"""
author: Richard Bonye (github Boyne272)
Last updated on Mon Oct  7 14:32:46 2019
"""

import numpy as np
from ..kmeans import SLIC
from .split_indexs import split_indexs


class mask_handler():
    '''
    A class to deal with the division of an image into regions for parallel
    processing, then recombining resultant masks from these regions into a
    single mask for the whole image.

    mask_handler(img, split_grid, overlap)

    Paramters
    ---------

    img : 3d numpy array
        The image to be split in format (x, y, 3 color channels) with
        values in float format between (0, 1).

    split_grid : tuple (len 2)
        The number of splits to do in x and y on the image, these do not
        have to be a whole division of the images shape

    overlap : int
        The number of pixels to overlap between the splits. An appropirate
        remerging of these regions prevents hard strait edges between the
        regions in the mask segmentation. The specific method for merging
        these is to look for common ids in the over lap region, identify
        them as the same and handle conflicts by voting in the local region
    '''

    def __init__(self, img, split_grid, overlap):

        # validate input
        assert img.shape[2] == 3, 'must have three channels'
        assert img.ndim == 3, 'must be an image with x, y, values'
        assert img.max() <= 1. and img.min() >= 0., \
            'rbg values should be a float between 0 and 1'

        # store fiven parameters
        self.img = img
        self.shape = self.img.shape[:2]
        self.overlap = overlap

        # determine the indexs to split on
        self.indexs = split_indexs(self.shape, split_grid, overlap=overlap)

        # create future attributes
        self.combo_mask = np.full(self.shape, np.nan)
        self.id_counter = 0


    def relabel_mask(self, mask):
        'Reassigns mask labels to be unique'
        count = mask.max()
        mask = mask + self.id_counter
        self.id_counter += count + 1
        return mask


    def match_masks(self, mask_a, mask_b):
        '''
        Given two masks with different segment ids this checks there alignment
        and where there is a 50% occurance of an id pair compared to the
        frequency of either ids idevidually, then consider these segments
        to be the same. returns a list of these id pairs.
        '''

        assert mask_a.size == mask_b.size, \
            'both masks must have same number of elements'

        # count the frequency of each id in both masks
        dict_a = get_freq_dict(mask_a.ravel())
        dict_b = get_freq_dict(mask_b.ravel())

        # count the frequency of unique pairs between the two
        pairs = get_freq_dict(zip(mask_a.ravel(), mask_b.ravel()))

        # for every id pair, compare frequency to that of each id seperately
        concurrent_ids = []
        for (a, b), freq in pairs.items():
            if freq > min(dict_a[a]//2, dict_b[b]//2):
                concurrent_ids.append((a, b))
        return concurrent_ids


    def conflict_resolve(self, mask_a, mask_b):
        '''
        Given two masks with the same segment IDs in them this will return
        a merged array where a majority voting algorithm has been used
        to identify the disagreeing segments
        '''

        assert mask_a.shape == mask_b.shape, \
            'both masks must have same dimension'

        # function efficently that finds the mode in a list
        mode = lambda lst: max(set(lst), key=lst.count)

        # function that selects a 3x3 grid from an array and returns the
        # elements as a list
        pick = lambda arr, i, j: arr[max(i-1, 0):i+2,
                                     max(j-1, 0):j+2].ravel().tolist()

        output = np.zeros_like(mask_a)
        for i in range(mask_a.shape[0]):
            for j in range(mask_a.shape[1]):

                # if the masks agree use the agreed segment ID
                if mask_a[i, j] == mask_b[i, j]:
                    output[i, j] = mask_a[i, j]

                # if the masks disagree use the majority vote between the
                # surrounding regions (ties are assigned randomly)
                else:
                    output[i, j] = mode(pick(mask_a, i, j) +
                                        pick(mask_b, i, j))

        return output


    def merge_masks(self, master_mask, sub_mask):
        '''
        Given a slice of the master mask and the sub mask this method
        combines them resolving conflicts (should they exist) on the
        overlapping regions.
        '''

        assert master_mask.shape == sub_mask.shape, \
            'both masks must have same dimension'

        # anywhere where master is nan set it to submask
        is_nan = np.isnan(master_mask)
        not_nan = np.logical_not(is_nan)
        master_mask[is_nan] = sub_mask[is_nan]

        # seperate the overlap regions whilst keeping their shape
        master_over = master_mask[not_nan].reshape(-1, self.overlap)
        sub_over = sub_mask[not_nan].reshape(-1, self.overlap)

        # identify common segments between the two masks
        concurrent_ids = self.match_masks(master_over, sub_over)

        # relabel these common segments to be the same in the combo and
        # master mask
        for a, b in concurrent_ids:
            self.combo_mask[self.combo_mask == a] = b
            master_mask[master_mask == a] = b

        # recalculate the overlap region with the relabelled segments
        master_over = master_mask[not_nan].reshape(-1, self.overlap)

        # resolve any conflics between these two to give the resulting mask
        master_mask[not_nan] = self.conflict_resolve(master_over,
                                                     sub_over).ravel()

        return master_mask


    def load_and_merge(self, paths):
        '''
        Load the masks in each of the paths given considering them to be
        the mask at the indexs calculated for the given image split and
        overlap. This then merges them together, handleing the overlap
        to create a whole image mask.
        '''

        assert len(paths) == len(self.indexs), \
            'must be an array path for each index'

        for path, (x0, x1, y0, y1) in zip(paths, self.indexs):

            sub_mask = np.load(path)
            assert sub_mask.shape == (x1-x0, y1-y0), \
                'mask does not have the expected index'

            sub_mask = self.relabel_mask(sub_mask)
            self.combo_mask[x0:x1, y0:y1] = self.merge_masks(
                self.combo_mask[x0:x1, y0:y1], sub_mask)


    def pass_and_merge(self, sub_masks):
        '''
        given an iterable of the masks for each of the indexs calculated
        for the given image split and overlap. This then merges them together,
        handleing the overlap to create a whole image mask.
        '''
        for sub_mask, (x0, x1, y0, y1) in zip(sub_masks, self.indexs):
            sub_mask = self.relabel_mask(sub_mask)
            self.combo_mask[x0:x1, y0:y1] = self.merge_masks(
                self.combo_mask[x0:x1, y0:y1], sub_mask)


    def get_subimages(self):
        '''
        From the image, split and overlap this returns the split images
        in a 4d array (img, x, y, rgb). Each sub image can then be used in
        the desired process (e.g. SLIC segmentation) and pass_and_merge will
        combine the resulting masks.
        '''
        return np.stack([self.img[x0:x1, y0:y1]
                         for (x0, x1, y0, y1) in self.indexs])



#%%


def get_freq_dict(lst):
    'Calculates the frequencies of each element in this list'
    freq_dict = {}
    for e in lst:
        freq_dict[e] = freq_dict.get(e, 0) + 1
    return freq_dict


def thread_func(img, iterate, bin_grid, save_mask='', **kwargs):
    'Given an image run SLIC on it and save the mask on the given path'

    # iterate SLIC on the given image
    slic = SLIC(img, bin_grid, **kwargs)
    slic.iterate(iterate)
    mask = slic.get_segmentation().astype(int)

    # save mask if wanted
    if save_mask:
        np.save(save_mask, mask)
        print('saved as ', save_mask)

    return mask


#%%

if __name__ == '__main__':
    # example of how to use this module

    import matplotlib.pyplot as plt
    from skimage.io import imread
    from outline import outline


    def file_names(prefix='mask', counter=0):
        'Generator for unique file names'
        while True:
            counter += 1
            yield prefix + '_%i.npy'%(counter-1)


    # load a sample image and check it can split the images correctly
    image = imread('large_white.jpg')[:2000, :2000]/255.
    obj = mask_handler(image, (2, 2), overlap=20)
    print(obj.get_subimages().shape)


#    # load some previously created masks with this image and settings
#    g = file_names()
#    mask_paths = [next(g) for _ in range(4)]
#    obj.load_and_merge(mask_paths)


    # calculate the masks with SLIC
    submasks = []
    for subimg in obj.get_subimages():
        submasks.append(thread_func(subimg, iterate=10,
                                    bin_grid=(10, 10)))
    obj.pass_and_merge(submasks)


    # plot the image and the stiched together mask to observe the handling
    # of the overlap region
    plt.imshow(image)
    rgba = np.dstack([outline(obj.combo_mask)]*4)
    rgba[:, :, 1:2] = 0.
    plt.imshow(rgba)
