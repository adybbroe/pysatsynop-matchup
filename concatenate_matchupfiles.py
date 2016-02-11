#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 Adam.Dybbroe

# Author(s):

#   Adam.Dybbroe <a000680@c20671.ad.smhi.se>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Concatenate a set of matchup files and write one big
"""

#DATADIR = "/home/a000680/laptop/satellite_synop_matchup/pysatsynop-matchup/radvaldata"
DATADIR = "/home/a000680/laptop/satellite_synop_matchup/pysatsynop-matchup/data"

import os
from glob import glob


allfiles = glob(DATADIR + "/matchup_npp_*txt")
datalines = []
header = []
for fname in allfiles:
    name = os.path.basename(fname)
    with open(fname, 'r') as fpt:
        lines = fpt.readlines()
        datalines = datalines + lines[3:]
        if len(header) == 0:
            header = lines[0:3]

outputfile = "./matchup_synop_all_npp.txt"
#outputfile = "./matchup_radval_all_npp.txt"

with open(outputfile, 'w') as fpt:
    fpt.writelines(header + datalines)
