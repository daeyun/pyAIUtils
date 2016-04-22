from __future__ import division

import numpy as np

# Python 2.7 doesn't have functools
import functools32 as functools


class VoxelGrid(object):
    def __init__(self, box, resolution=100):
        assert box.shape == (2, 3)
        self.bstart = box.min(axis=0)
        self.bend = box.max(axis=0)
        self.resolution = resolution
        self.voxel_size = (self.bend - self.bstart).max() / float(
                resolution - 1)
        self.shape = np.round(
                ((self.bend - self.bstart) / self.voxel_size + 1)).astype(
                np.int64)
        self.size = np.prod(self.shape)
        self.values = None

    @functools.lru_cache(maxsize=1)
    def xyz(self):
        xrange = np.arange(self.bstart[0], self.bend[0] + self.voxel_size / 2,
                           self.voxel_size)
        yrange = np.arange(self.bstart[1], self.bend[1] + self.voxel_size / 2,
                           self.voxel_size)
        zrange = np.arange(self.bstart[2], self.bend[2] + self.voxel_size / 2,
                           self.voxel_size)
        X, Y, Z = np.meshgrid(xrange, yrange, zrange)
        return np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T

    @functools.lru_cache(maxsize=1)
    def linear_index_grid(self):
        return np.ravel_multi_index(np.indices(self.shape), dims=self.shape,
                                    order='C')

    @functools.lru_cache(maxsize=1)
    def multi_index_grid(self):
        return np.indices(self.shape)

    def zeros_like(self):
        return np.zeros(self.shape)

    def linear_to_multi(self, ids):
        return np.unravel_index(ids, dims=self.shape, order='C')

    def project_to_image(self, cam, imshape):
        xy = np.round(cam.world_to_image(self.xyz()))
        inimage = np.logical_and((xy > 0).all(axis=1),
                                 (xy < imshape[:2][::-1]).all(axis=1))
        xy = xy[inimage, :].astype(np.int32)
        ids = np.arange(self.size)[inimage].astype(np.int32)
        return xy, ids

    def set_values(self, voxel_grid_values):
        assert np.all(voxel_grid_values.shape == self.shape)
        self.values = voxel_grid_values

    def projection_image(self, cam, imshape):
        if self.values is None:
            raise RuntimeError('voxel has no values.')
        imshape = imshape[:2]
        xy, ids = self.project_to_image(cam, imshape)
        im = np.zeros(imshape)
        im[xy[:, 1], xy[:, 0]] = self.values[self.linear_to_multi(ids)]
        return im
