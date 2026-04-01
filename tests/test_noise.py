# SPDX-FileCopyrightText: Copyright (c) 2023 Tod Kurt
#
# SPDX-License-Identifier: MIT
"""Tests for the noise module."""

from math import floor, sqrt

import noise

# ---------------------------------------------------------------------------
# Reference implementation (original float-only version) used for comparison
# ---------------------------------------------------------------------------

_F2_REF = 0.5 * (sqrt(3.0) - 1.0)
_G2_REF = (3.0 - sqrt(3.0)) / 6.0

_p_ref = (
    151,
    160,
    137,
    91,
    90,
    15,
    131,
    13,
    201,
    95,
    96,
    53,
    194,
    233,
    7,
    225,
    140,
    36,
    103,
    30,
    69,
    142,
    8,
    99,
    37,
    240,
    21,
    10,
    23,
    190,
    6,
    148,
    247,
    120,
    234,
    75,
    0,
    26,
    197,
    62,
    94,
    252,
    219,
    203,
    117,
    35,
    11,
    32,
    57,
    177,
    33,
    88,
    237,
    149,
    56,
    87,
    174,
    20,
    125,
    136,
    171,
    168,
    68,
    175,
    74,
    165,
    71,
    134,
    139,
    48,
    27,
    166,
    77,
    146,
    158,
    231,
    83,
    111,
    229,
    122,
    60,
    211,
    133,
    230,
    220,
    105,
    92,
    41,
    55,
    46,
    245,
    40,
    244,
    102,
    143,
    54,
    65,
    25,
    63,
    161,
    1,
    216,
    80,
    73,
    209,
    76,
    132,
    187,
    208,
    89,
    18,
    169,
    200,
    196,
    135,
    130,
    116,
    188,
    159,
    86,
    164,
    100,
    109,
    198,
    173,
    186,
    3,
    64,
    52,
    217,
    226,
    250,
    124,
    123,
    5,
    202,
    38,
    147,
    118,
    126,
    255,
    82,
    85,
    212,
    207,
    206,
    59,
    227,
    47,
    16,
    58,
    17,
    182,
    189,
    28,
    42,
    223,
    183,
    170,
    213,
    119,
    248,
    152,
    2,
    44,
    154,
    163,
    70,
    221,
    153,
    101,
    155,
    167,
    43,
    172,
    9,
    129,
    22,
    39,
    253,
    19,
    98,
    108,
    110,
    79,
    113,
    224,
    232,
    178,
    185,
    112,
    104,
    218,
    246,
    97,
    228,
    251,
    34,
    242,
    193,
    238,
    210,
    144,
    12,
    191,
    179,
    162,
    241,
    81,
    51,
    145,
    235,
    249,
    14,
    239,
    107,
    49,
    192,
    214,
    31,
    181,
    199,
    106,
    157,
    184,
    84,
    204,
    176,
    115,
    121,
    50,
    45,
    127,
    4,
    150,
    254,
    138,
    236,
    205,
    93,
    222,
    114,
    67,
    29,
    24,
    72,
    243,
    141,
    128,
    195,
    78,
    66,
    215,
    61,
    156,
    180,
)


class _Grad:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


_grad3_ref = (
    _Grad(1, 1, 0),
    _Grad(-1, 1, 0),
    _Grad(1, -1, 0),
    _Grad(-1, -1, 0),
    _Grad(1, 0, 1),
    _Grad(-1, 0, 1),
    _Grad(1, 0, -1),
    _Grad(-1, 0, -1),
    _Grad(0, 1, 1),
    _Grad(0, -1, 1),
    _Grad(0, 1, -1),
    _Grad(0, -1, -1),
)
_perm_ref = [0] * 512
_pm12_ref = [0] * 512
for _i in range(512):
    _perm_ref[_i] = _p_ref[_i & 255]
    _pm12_ref[_i] = _perm_ref[_i] % 12


def _dot_ref(g, x, y):
    return g.x * x + g.y * y


def _noise_ref(xin, yin=0):
    s = (xin + yin) * _F2_REF
    i = floor(xin + s)
    j = floor(yin + s)
    t = (i + j) * _G2_REF
    x0 = xin - (i - t)
    y0 = yin - (j - t)
    i1 = 0
    j1 = 1
    if x0 > y0:
        i1 = 1
        j1 = 0
    x1 = x0 - i1 + _G2_REF
    y1 = y0 - j1 + _G2_REF
    x2 = x0 - 1.0 + 2.0 * _G2_REF
    y2 = y0 - 1.0 + 2.0 * _G2_REF
    ii = int(i) & 255
    jj = int(j) & 255
    gi0 = _pm12_ref[ii + _perm_ref[jj]]
    gi1 = _pm12_ref[ii + i1 + _perm_ref[jj + j1]]
    gi2 = _pm12_ref[ii + 1 + _perm_ref[jj + 1]]
    n0 = 0.0
    t0 = 0.5 - x0 * x0 - y0 * y0
    if t0 >= 0.0:
        t0 *= t0
        n0 = t0 * t0 * _dot_ref(_grad3_ref[gi0], x0, y0)
    n1 = 0.0
    t1 = 0.5 - x1 * x1 - y1 * y1
    if t1 >= 0.0:
        t1 *= t1
        n1 = t1 * t1 * _dot_ref(_grad3_ref[gi1], x1, y1)
    n2 = 0.0
    t2 = 0.5 - x2 * x2 - y2 * y2
    if t2 >= 0.0:
        t2 *= t2
        n2 = t2 * t2 * _dot_ref(_grad3_ref[gi2], x2, y2)
    return 70.0 * (n0 + n1 + n2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Maximum allowed difference from the reference float implementation.
# Fixed-point Q16.16 introduces at most a few ULPs of rounding per step.
_MAX_ERROR = 0.005


def test_output_range_1d():
    """1D noise (yin=0) always returns a value in [-1, 1]."""
    for i in range(500):
        val = noise.noise(i * 0.02)
        assert -1.0 <= val <= 1.0, f"Out of range at i={i}: {val}"


def test_output_range_2d():
    """2D noise always returns a value in [-1, 1]."""
    for xi in range(20):
        for yi in range(20):
            val = noise.noise(xi * 0.1, yi * 0.1)
            assert -1.0 <= val <= 1.0, f"Out of range at ({xi},{yi}): {val}"


def test_matches_reference_1d():
    """Fixed-point output is close to the original float implementation (1D sweep)."""
    for i in range(200):
        xin = i * 0.02
        ref = _noise_ref(xin)
        got = noise.noise(xin)
        assert (
            abs(got - ref) < _MAX_ERROR
        ), f"1D mismatch at xin={xin:.4f}: ref={ref:.6f} got={got:.6f}"


def test_matches_reference_2d():
    """Fixed-point output is close to the reference for a set of 2D coordinates."""
    points = [
        (0.1, 0.2),
        (0.5, 0.3),
        (1.3, 2.7),
        (3.7, 9.1),
        (0.0, 0.0),
        (10.0, 10.0),
        (0.99, 0.01),
    ]
    for xin, yin in points:
        ref = _noise_ref(xin, yin)
        got = noise.noise(xin, yin)
        assert (
            abs(got - ref) < _MAX_ERROR
        ), f"2D mismatch at ({xin},{yin}): ref={ref:.6f} got={got:.6f}"


def test_matches_reference_negative():
    """Fixed-point output is close to the reference for negative coordinates."""
    points = [(-0.3, 0.1), (-1.0, -1.0), (-0.5, 0.0), (0.0, -0.5)]
    for xin, yin in points:
        ref = _noise_ref(xin, yin)
        got = noise.noise(xin, yin)
        assert (
            abs(got - ref) < _MAX_ERROR
        ), f"Negative mismatch at ({xin},{yin}): ref={ref:.6f} got={got:.6f}"


def test_origin():
    """noise(0, 0) returns 0."""
    assert noise.noise(0, 0) == 0.0


def test_1d_is_2d_with_zero_y():
    """noise(x) and noise(x, 0) return identical results."""
    for i in range(50):
        xin = i * 0.07
        assert noise.noise(xin) == noise.noise(xin, 0)


def test_deterministic():
    """Same inputs always produce the same output."""
    for _ in range(5):
        assert noise.noise(1.23, 4.56) == noise.noise(1.23, 4.56)
