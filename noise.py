# pylint: disable=invalid-name
# SPDX-FileCopyrightText: 2017 Scott Shawcroft, written for Adafruit Industries
# SPDX-FileCopyrightText: Copyright (c) 2023 Tod Kurt
#
# SPDX-License-Identifier: MIT
"""
`noise`
================================================================================

Simplex noise (like Perlin) for CircuitPython.


* Author(s): Tod Kurt

Implementation Notes
--------------------

A basic port of https://weber.itn.liu.se/~stegu/simplexnoise/simplexnoise.pdf
Also see: https://github.com/stegu/perlin-noise/blob/master/src/simplexnoise1234.c

Optimised for CircuitPython/MicroPython by:

- F2, G2, and the x2/y2 offset are pre-computed constants (no math.sqrt at startup)
- ``floor()`` is inlined, avoiding two function calls per ``noise()`` invocation
- ``dot()`` is inlined via ``_gx``/``_gy`` tuples, avoiding per-call function and attribute lookups

Also works in desktop CPython.

"""


# imports

__version__ = "0.0.0+auto.0"
__repo__ = "https://github.com/todbot/CircuitPython_Noise.git"

try:
    from micropython import native
except ImportError:
    native = lambda f: f  # no-op on CircuitPython / CPython


class Grad:  # pylint: disable=too-few-public-methods
    """Holder for 3D grad. Used internally by noise()"""

    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


# Pre-computed constants (avoids math.sqrt on import, and repeated arithmetic in noise())
F2 = 0.3660254037844386
G2 = 0.21132486540518713
_G2x2m1 = 2.0 * G2 - 1.0  # == -0.5773..., used for x2/y2 corner offsets

# fmt: off
p = (  # 256 elements
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,
    142,8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,
    203,117,35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,
    74,165,71,134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,
    220,105,92,41,55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,
    132,187,208,89,18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,
    186,3,64,52,217,226,250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,
    59,227,47,16,58,17,182,189,28,42,223,183,170,213,119,248,152,2,44,154,163,
    70,221,153,101,155,167,43,172,9,129,22,39,253,19,98,108,110,79,113,224,232,
    178,185,112,104,218,246,97,228,251,34,242,193,238,210,144,12,191,179,162,
    241,81,51,145,235,249,14,239,107,49,192,214,31,181,199,106,157,184,84,204,
    176,115,121,50,45,127,4,150,254,138,236,205,93,222,114,67,29,24,72,243,141,
    128,195,78,66,215,61,156,180
)
grad3 = (
    Grad(1,1,0),Grad(-1,1,0),Grad(1,-1,0),Grad(-1,-1,0),Grad(1,0,1),
    Grad(-1,0,1),Grad(1,0,-1),Grad(-1,0,-1),Grad(0,1,1),Grad(0,-1,1),
    Grad(0,1,-1),Grad(0,-1,-1)
)
# x,y components of grad3 as plain tuples — faster than Grad attribute lookup
_gx = (1, -1,  1, -1, 1, -1,  1, -1, 0,  0,  0,  0)
_gy = (1,  1, -1, -1, 0,  0,  0,  0, 1, -1,  1, -1)
# fmt: on

perm = [0] * 512  # filled out in noise_init()
permMod12 = [0] * 512


def dot(grad, x, y):
    """Compute dot product of grad against x,y. Used internally by noise()"""
    return grad.x * x + grad.y * y


def noise_init():
    """Initialize permutation arrays. Done automatically on module load"""
    for i in range(512):
        perm[i] = p[i & 255]
        permMod12[i] = perm[i] % 12


@native
def noise(xin, yin=0):  # pylint: disable=too-many-locals
    """2D SimplexNoise

    :param float xin: x-location in 2D noise space
    :param float yin: y-location in 2D noise space

    :return float: noise value between -1 and 1."""

    # Local aliases for globals: MicroPython resolves globals via dict lookup on every access;
    # locals are indexed directly, so aliasing here pays off across the 17 uses below.
    f2 = F2
    g2 = G2
    g2m = _G2x2m1
    pm12 = permMod12
    pm = perm
    gx = _gx
    gy = _gy

    # Skew input space to find which simplex cell we're in.
    # Inline floor() via int() + fixup: avoids two math.floor function calls.
    s = (xin + yin) * f2
    xs = xin + s
    ys = yin + s
    i = int(xs)
    j = int(ys)
    if xs < i:
        i -= 1  # fix floor for negative values
    if ys < j:
        j -= 1

    # Unskew the cell origin back to (x,y) space
    t = (i + j) * g2
    x0 = xin - i + t
    y0 = yin - j + t

    # Determine which simplex triangle we're in
    i1 = j1 = 0
    if x0 > y0:
        i1 = 1  # lower triangle: step (1,0) then (1,1)
    else:
        j1 = 1  # upper triangle: step (0,1) then (1,1)

    # Offsets for the middle and last corners in (x,y) unskewed coords.
    # x2 = x0 - 1.0 + 2.0*G2  ==  x0 + g2m  (pre-computed, saves a multiply)
    x1 = x0 - i1 + g2
    y1 = y0 - j1 + g2
    x2 = x0 + g2m
    y2 = y0 + g2m

    # Gradient indices for the three corners
    ii = i & 255
    jj = j & 255
    gi0 = pm12[ii + pm[jj]]
    gi1 = pm12[ii + i1 + pm[jj + j1]]
    gi2 = pm12[ii + 1 + pm[jj + 1]]

    # Corner contributions.
    # dot() is inlined via gx/gy tuples: avoids 2-3 function calls and Grad attribute lookups.
    n0 = n1 = n2 = 0.0
    t0 = 0.5 - x0 * x0 - y0 * y0
    if t0 >= 0.0:
        t0 *= t0
        n0 = t0 * t0 * (gx[gi0] * x0 + gy[gi0] * y0)

    t1 = 0.5 - x1 * x1 - y1 * y1
    if t1 >= 0.0:
        t1 *= t1
        n1 = t1 * t1 * (gx[gi1] * x1 + gy[gi1] * y1)

    t2 = 0.5 - x2 * x2 - y2 * y2
    if t2 >= 0.0:
        t2 *= t2
        n2 = t2 * t2 * (gx[gi2] * x2 + gy[gi2] * y2)

    return 70.0 * (n0 + n1 + n2)


noise_init()
