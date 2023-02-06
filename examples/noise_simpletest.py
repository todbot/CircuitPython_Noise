# SPDX-FileCopyrightText: 2017 Scott Shawcroft, written for Adafruit Industries
# SPDX-FileCopyrightText: Copyright (c) 2023 Tod Kurt
#
# SPDX-License-Identifier: Unlicense

"""
Print an undulating terrain to the console with asterisks.
"""

import time
from noise import noise

i = 0
while True:
    n = noise(0.02 * i)  # get a 1D noise value
    i += 1
    # print a random terrain with asterisks
    print(" " * int(max(n + 1, 0) * 40), "*")
    time.sleep(0.01)
