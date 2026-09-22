#!/usr/bin/env python3
"""Reader for the binary fields written by tests::solution_writer.

Layout: four float32 slots holding int32 [Nx, Ny, Nz, tensor_dim], then the field with
offset = 4 + ((z*Ny + y)*Nx + x)*tensor_dim + t.  Slot t = 0 is psi, t = 1 is phi.
The grid is cell-centred on the unit cube: x_i = (i + 1/2)/N.
"""

import numpy as np


def read_field(path):
    raw = np.fromfile(path, dtype=np.float32)
    nx, ny, nz, td = raw[:4].view(np.int32)
    body = raw[4:].reshape(int(nz), int(ny), int(nx), int(td))
    return {"psi": body[..., 0], "phi": body[..., 1], "shape": (int(nx), int(ny), int(nz))}


def radial_profile(field, nbins=240, centre=(0.5, 0.5, 0.5), rmax=0.5):
    """Mean, min and max of `field` over spherical shells about the box centre."""
    nz, ny, nx = field.shape
    z = (np.arange(nz) + 0.5) / nz - centre[2]
    y = (np.arange(ny) + 0.5) / ny - centre[1]
    x = (np.arange(nx) + 0.5) / nx - centre[0]
    r = np.sqrt(z[:, None, None] ** 2 + y[None, :, None] ** 2 + x[None, None, :] ** 2).ravel()
    v = field.ravel()

    sel = r <= rmax
    r, v = r[sel], v[sel]
    edges = np.linspace(0.0, rmax, nbins + 1)
    idx = np.clip(np.digitize(r, edges) - 1, 0, nbins - 1)

    cnt = np.bincount(idx, minlength=nbins)
    tot = np.bincount(idx, weights=v, minlength=nbins)
    lo = np.full(nbins, np.inf)
    hi = np.full(nbins, -np.inf)
    np.minimum.at(lo, idx, v)
    np.maximum.at(hi, idx, v)

    ok = cnt > 0
    centres = 0.5 * (edges[1:] + edges[:-1])
    return centres[ok], (tot[ok] / cnt[ok]), lo[ok], hi[ok]


def zero_crossing(r, phi):
    """Radius where the spherically averaged phi changes sign, by linear interpolation."""
    s = np.where(np.sign(phi[:-1]) != np.sign(phi[1:]))[0]
    if len(s) == 0:
        return np.nan
    i = s[0]
    t = phi[i] / (phi[i] - phi[i + 1])
    return r[i] + t * (r[i + 1] - r[i])
