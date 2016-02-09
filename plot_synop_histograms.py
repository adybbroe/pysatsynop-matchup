#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2015 Adam.Dybbroe

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

"""Plot a histogram of the cloud cover for selected Synop stations
"""

from analyse_results import (MatchupAnalysis, hist1d_plot,
                             sunz_filter, station_filter)
from glob import glob
import numpy as np


# --------------------------------------------------------
if __name__ == "__main__":

    #filenames = glob('./data/matchup_*txt')
    filenames = glob('./data/results_n*txt')

    manual_stations = ['02616', '02496', '02020']

    #this = MatchupAnalysis(mtype='point')
    this = MatchupAnalysis(mtype='synop')
    res = this.get_results(filenames)

    res = res[np.isfinite(res['obs'])]
    #res = geo_filter(res, outside=False)
    #res = geo_filter(res)

    res = sunz_filter(res, [0, 80])  # Daytime
    # res = sunz_filter(res, [90, 180])  # Night
    # res = sunz_filter(res, [80, 90])  # Twilight
    res = station_filter(res, manual_stations)

    hist1d_plot(res, 'obs', 'Observations')
