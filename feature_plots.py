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

"""Plot some pps features as a function of cloudiness from PPS and synop
"""

from analyse_results import (MatchupAnalysis,
                             sunz_filter,
                             synop_validation)

import numpy as np
from glob import glob

# --------------------------------------------------------
if __name__ == "__main__":

    import os
    # To get the plot_date to write labels in Amarican English
    os.environ['LC_ALL'] = 'en_US'

    filenames = glob('./matchup_synop_all_npp.txt')
    this = MatchupAnalysis(mtype='synop')
    this.get_results(filenames)
    #print("Add sunz")
    # this.add_sunzenith()
    #print("Sun zenith angles derived...")

    res = this.data

    print("Matchup data retrieved")
    res = res[np.isfinite(res['sat'])]

    # Illumination filtering:
    # We should add the sunz angles to the data files
    # This will save processing time! FIXME!
    res = sunz_filter(res, [0, 80])  # Daytime
    print("Sun zenith angle filtering done!")

    cloudobs = np.array(res.loc[res.index, 'obs'].tolist())
    ppsclouds = np.array(res.loc[res.index, 'sat'].tolist())
    clear = cloudobs <= 0.25
    cloudy = cloudobs >= 0.75
    ppscloudy = ppsclouds >= 0.75
    ppsclear = ppsclouds <= 0.25

    #synop_validation(res, './ppsval_daytime.txt')

    r06 = np.array(res.loc[res.index, 'r06'].tolist())
    r13 = np.array(res.loc[res.index, 'r13'].tolist())
    r16 = np.array(res.loc[res.index, 'r16'].tolist())
    satz = np.array(res.loc[res.index, 'satz'].tolist())
    ciwv = np.array(res.loc[res.index, 'ciwv'].tolist())
    # nodata = np.logical_or(
    #    np.less_equal(r16, -999.), np.less_equal(r06, -999.))
    nodata = np.less_equal(r13, -999.)
    mask = np.logical_or(nodata, clear == False)
    r16 = np.ma.masked_array(r16, mask=mask)
    r06 = np.ma.masked_array(r06, mask=mask)
    r13 = np.ma.masked_array(r13, mask=mask)
    satz = np.ma.masked_array(satz, mask=mask)
    ciwv = np.ma.masked_array(ciwv, mask=mask)

    import matplotlib.pyplot as plt

    plotfile = "./pps_synop_r13_ciwv.png"
    cmap = plt.cm.OrRd
    # extract all colors from the standard map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # force the first color entry to be grey
    cmaplist[0] = (.5, .5, .5, 1.0)
    # create the new map
    mycmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    plt.figure(figsize=(14, 9))

    y = r13
    x = ciwv
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), 8.0
    plt.subplots_adjust(hspace=0.5)
    plt.subplot(131)
    plt.hexbin(x, y, cmap=mycmap)
    plt.axis([xmin, xmax, ymin, ymax])
    plt.title('Clear (clf < 0.25)')
    plt.xlabel('ciwv')
    plt.ylabel('r13')
    plt.ylim(ymax=ymax)

    plt.subplot(132)

    r13 = np.array(res.loc[res.index, 'r13'].tolist())
    ciwv = np.array(res.loc[res.index, 'ciwv'].tolist())
    nodata = np.logical_or(nodata, np.less(ppsclouds, 0))
    mask = np.logical_or(nodata, cloudy == False)
    r13 = np.ma.masked_array(r13, mask=mask)
    ciwv = np.ma.masked_array(ciwv, mask=mask)
    y = r13
    x = ciwv

    plt.hexbin(x, y, cmap=mycmap)
    plt.axis([xmin, xmax, ymin, ymax])
    plt.title('Cloudy (clf > 0.75)')
    plt.xlabel('ciwv')
    plt.ylabel('r13')
    plt.ylim(ymax=ymax)

    plt.subplot(133)
    r13 = np.array(res.loc[res.index, 'r13'].tolist())
    ciwv = np.array(res.loc[res.index, 'ciwv'].tolist())
    mask = np.logical_or(
        nodata, np.logical_or(cloudy == False, ppsclear == False))
    r13 = np.ma.masked_array(r13, mask=mask)
    ciwv = np.ma.masked_array(ciwv, mask=mask)
    y = r13
    x = ciwv

    plt.hexbin(x, y, cmap=mycmap)
    plt.axis([xmin, xmax, ymin, ymax])
    plt.title('PPS clear and Synop cloudy')
    plt.xlabel('ciwv')
    plt.ylabel('r13')
    plt.ylim(ymax=ymax)
    cb = plt.colorbar()
    cb.set_label('N')

    plt.savefig(plotfile)
    # plt.show()

    idx = res[res['r13'] > 0.5].index
    idx2 = res[res['ciwv'] > 4.0].index
    idx3 = res[res['sat'] >= 0].index
    index = [x for x in idx if x in idx2]
    index = [x for x in index if x in idx3]
    save_pps = res.loc[index, 'sat']

    res.loc[res[res['r13'] > 0.5].index, 'sat'] = 1
    synop_validation(res, './ppsval_daytime_13boost.txt')
