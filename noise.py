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

No special requirements. Also works in desktop CPython.

"""


# imports

__version__ = "0.0.0+auto.0"
__repo__ = "https://github.com/todbot/CircuitPython_Noise.git"


from math import sqrt, floor


class Grad:  # pylint: disable=too-few-public-methods
    """Holder for 3D grad"""

    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


F2 = 0.5 * (sqrt(3.0) - 1.0)
G2 = (3.0 - sqrt(3.0)) / 6.0

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
# fmt: on

perm = [0] * 512  # filled out in noise_init()
permMod12 = [0] * 512


def dot(grad, x, y):
    """Compute dot product of grad against x,y"""
    return grad.x * x + grad.y * y


def noise_init():
    """Initialize permutation arrays. Done automatically on module load"""
    for i in range(512):
        perm[i] = p[i & 255]
        permMod12[i] = perm[i] % 12


def noise(xin, yin=0):  # pylint: disable=too-many-locals
    """2D SimplexNoise
    :param float xin x-location in 2D noise space
    :param float yin y-location in 2D noise space

    :return float noise value between -1 and 1
    """

    s = (xin + yin) * F2
    i = floor(xin + s)
    j = floor(yin + s)
    t = (i + j) * G2
    x0 = xin - (i - t)
    y0 = yin - (j - t)
    i1 = 0
    j1 = 1
    if x0 > y0:
        i1 = 1
        j1 = 0

    x1 = x0 - i1 + G2
    y1 = y0 - j1 + G2
    x2 = x0 - 1.0 + 2.0 * G2
    y2 = y0 - 1.0 + 2.0 * G2
    ii = i & 255
    jj = j & 255
    gi0 = permMod12[ii + perm[jj]]
    gi1 = permMod12[ii + i1 + perm[jj + j1]]
    gi2 = permMod12[ii + 1 + perm[jj + 1]]
    n0 = 0.0
    t0 = 0.5 - x0 * x0 - y0 * y0
    if t0 >= 0.0:
        t0 *= t0
        n0 = t0 * t0 * dot(grad3[gi0], x0, y0)
    n1 = 0.0
    t1 = 0.5 - x1 * x1 - y1 * y1
    if t1 >= 0.0:
        t1 *= t1
        n1 = t1 * t1 * dot(grad3[gi1], x1, y1)
    n2 = 0.0
    t2 = 0.5 - x2 * x2 - y2 * y2
    if t2 >= 0.0:
        t2 *= t2
        n2 = t2 * t2 * dot(grad3[gi2], x2, y2)

    return 70.0 * (n0 + n1 + n2)


noise_init()
