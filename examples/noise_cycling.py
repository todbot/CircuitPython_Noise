# pylint: disable=invalid-name,consider-using-f-string
# SPDX-FileCopyrightText: 2017 Scott Shawcroft, written for Adafruit Industries
# SPDX-FileCopyrightText: Copyright (c) 2023 Tod Kurt
#
# SPDX-License-Identifier: Unlicense

"""
Print an undulating terrain to the console with asterisks,
but that is guaranteed to repeat seemlessly since we are
spinning in a circle in 2D noise space.
"""

import time
from math import sin, cos, pi

from noise import noise

show_stats = False

# where in the noise field to be centered
x0, y0 = 0, 0
# and what distance from that center to move
r = 1

# stats
sumt, nmin, nmax = 0, 0, 0

theta = 0
i = 1
while True:
    x = r * sin(theta) + x0
    y = r * cos(theta) + y0
    theta = (theta + 0.02) % (pi * 2)

    t = time.monotonic()

    n = noise(x, y)

    sumt += time.monotonic() - t
    nmax = max(n, nmax)
    nmin = min(n, nmin)

    print(" " * int(max(n + 1, 0) * 40), "*")
    if show_stats:
        print(
            "n:%+3.3f (%+3.3f:%+3.3f) %+3.3f %0.4f" % (n, nmin, nmax, theta, sumt / i)
        )

    i += 1
    time.sleep(0.01)
