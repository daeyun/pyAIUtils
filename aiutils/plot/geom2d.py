from __future__ import division

import matplotlib.pyplot as pt
import numpy as np
import matplotlib.cm as cm
from matplotlib.patches import Polygon

def draw_triangles(triangles, ax=None, facecolor='blue', alpha=1):
    """
    :param triangles: (n,3,2) 2d triangles.
    :return:
    """
    assert triangles.shape[1] == 3
    assert triangles.shape[2] == 2

    if ax is None:
        fig = pt.figure()
        ax = fig.gca()

    for i in range(triangles.shape[0]):
        pts = np.array(triangles[i, :, :])
        p = Polygon(pts, closed=True, facecolor=facecolor, alpha=alpha)
        ax.add_patch(p)

    ax.set_xlim([triangles[:, :, 0].min(), triangles[:, :, 0].max()])
    ax.set_ylim([triangles[:, :, 1].min(), triangles[:, :, 1].max()])

def pts(xy, ax=None, markersize=10):
    """
    Plot 2d points.
    """
    if ax is None:
        fig = pt.figure()
        ax = fig.gca()

    ax.scatter(xy[:, 0], xy[:, 1], marker='.', s=markersize)

def draw_depth(depth, ax=None, clim=None, nancolor='y', cmap='gray'):
    assert isinstance(depth, np.ma.core.MaskedArray)
    g = cm.get_cmap(cmap, 1024 * 2)
    g.set_bad(nancolor, 1.)

    if ax is None:
        fig = pt.figure()
        ax = fig.gca()

    mappable = ax.imshow(depth, cmap=g, interpolation='nearest', aspect='equal')
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)

    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = pt.colorbar(mappable, cax=cax)
    cb.ax.tick_params(labelsize=10)
    if clim is not None:
        cb.set_clim(clim)

def montage(images, gridwidth=None, empty_value=0):
    """
    Similar to http://www.mathworks.com/help/images/ref/montage.html
    :param images: (n, h, w) array or size-n list of (h, w) arrays.
    """
    if type(images) is not list:
        images = [images[i] for i in range(images.shape[0])]
    imtype = images[0].dtype

    if gridwidth is None:
        gridwidth = int(np.ceil(np.sqrt(len(images))))
    gridheight = int(np.ceil(len(images) / gridwidth))
    remaining = gridwidth * gridheight
    rows = []

    while remaining > 0:
        rowimgs = images[:gridwidth]
        images = images[gridwidth:]
        nblank = gridwidth - len(rowimgs)
        empty_block = np.zeros(rowimgs[0].shape)
        empty_block[:] = empty_value
        rowimgs.extend([empty_block] * nblank)
        remaining -= gridwidth
        row = np.hstack(rowimgs)
        rows.append(row)
    m = np.vstack(rows)
    return m.astype(imtype)
