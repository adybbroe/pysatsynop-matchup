#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2014, 2015 Adam.Dybbroe

# Author(s):

#   Adam.Dybbroe <a000680@c14526.ad.smhi.se>

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

"""Read the AVHRR/VIIRS-Synop collocations and make some plots of the data
distribution
"""

from analyse_results import MatchupAnalysis
from datetime import datetime, timedelta
import numpy as np
from glob import glob

# --------------------------------------------------------
if __name__ == "__main__":

    import os
    # To get the plot_date to write labels in Amarican English
    os.environ['LC_ALL'] = 'en_US'

    #filenames = glob('./Results_dr_norrkoping_v2014/results*txt')
    #filenames = glob('./data/results_*txt')
    filenames = glob('./data/matchup_*txt')

    this = MatchupAnalysis(mtype='point')
    res = this.get_results(filenames)
    res = res[np.isfinite(res['sat'])]

    x = res.date + res.delta_t.apply(lambda d: timedelta(minutes=d))

    startdate = x.min()
    enddate = x.max()
    mydate = startdate
    delta_t = timedelta(minutes=60 * 24 * 10)  # 10 days
    datelist = []
    frequency = []
    while mydate < enddate:
        mydate = mydate + delta_t
        a = np.where(np.logical_and(x.astype(datetime).values < mydate,
                                    x.astype(datetime).values > mydate - delta_t),
                     1, 0)
        frequency.append(np.repeat(a, a).sum())
        datelist.append(mydate)

    import matplotlib.pyplot as plt

    ax = plt.subplot(111)
    ax.bar(datelist, frequency, width=10, color='green')
    ax.xaxis_date()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    plt.xticks(rotation='vertical')
    # plt.show()
    plt.savefig('./collocations_over_time_radval.png')
