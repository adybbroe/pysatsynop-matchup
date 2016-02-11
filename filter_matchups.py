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

"""Filter/reduce the ascii files with synop-satellite matchups to contain only
selected stations

"""
import pandas as pd

#RADVAL_FILE = "./radval-stlist.txt"
RADVAL_FILE = "./radval-stlist-wmo.txt"

SYNOPFILE = "./matchup_synop_all_npp.txt"
OUTPUTFILE = "./matchup_synop_radval_npp.txt"


from pps_matchup import get_radvaldata

datapoints = get_radvaldata(RADVAL_FILE)
pdata = pd.DataFrame(datapoints)

ids = [x[1]['id'] for x in pdata.iterrows()]

with open(SYNOPFILE, "r") as fpt:
    lines = fpt.readlines()

outlines = []
for line in lines[3:]:
    sl_ = line.split()
    if sl_[2] in ids:
        outlines.append(line)

with open(OUTPUTFILE, "w") as fpt:
    fpt.writelines(lines[0:3] + outlines)
