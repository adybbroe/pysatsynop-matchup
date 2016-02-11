#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2014, 2015, 2016 Adam.Dybbroe

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

from analyse_results import (MatchupAnalysis,
                             hist1d_plot,
                             synop_validation)

from datetime import datetime, timedelta
import numpy as np
from glob import glob

# --------------------------------------------------------
if __name__ == "__main__":

    import os
    # To get the plot_date to write labels in Amarican English
    os.environ['LC_ALL'] = 'en_US'

    #filenames = glob('./data/matchup_*txt')
    #filenames = glob('./radvaldata/matchup_*txt')
    #filenames = glob('./matchup_all_npp*txt')
    #filenames = glob('./matchup_synop_radval_npp.txt')
    filenames = glob('./matchup_synop_all_npp.txt')
    this = MatchupAnalysis(mtype='synop')
    #this = MatchupAnalysis(mtype='point')
    this.get_results(filenames)
    res = this.data[np.isfinite(this.data['sat'])]

    # Illumination filtering:
    # res = sunz_filter(res, [0, 80])  # Daytime
    # res = sunz_filter(res, [90, 180])  # Night
    # res = sunz_filter(res, [80, 90])  # Twilight

    hist1d_plot(res, 'sat', 'PPS cloud cover', color='blue')

    #synop_validation(res, './ppsval_nightime.txt')
    synop_validation(res, './ppsval.txt')

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
    # plt.savefig('./collocations_over_time_radval.png')
    # plt.savefig('./collocations_over_time_radval_synop.png')
    plt.savefig('./collocations_over_time_synop.png')

    # Make a map plot of the number of occurences and station locations
    names = res['station']
    land_stations = np.unique([s for s in names if s != 'SHIP'])
    land_stations = land_stations[land_stations != 'nan']
    numobs = np.array([len(res[res['station'] == s]) for s in land_stations])
    lons = np.array([res['lon'][res[res['station'] == s].index[0]]
                     for s in land_stations])
    lats = np.array([res['lat'][res[res['station'] == s].index[0]]
                     for s in land_stations])

    import pyresample as pr
    from matplotlib import cm
    area_id = 'europa'
    name = 'Europa'
    proj_id = 'tmp'
    proj4_args = 'proj=stere, ellps=bessel, lat_0=90, lon_0=14, lat_ts=60'
    x_size = 700
    y_size = 700
    area_extent = (-3700000, -7000000, 3300000, -500000)
    proj_dict = {'ellps': 'bessel', 'units': 'm', 'lon_0': '14',
                 'proj': 'stere', 'lat_0': '90'}
    area_def = pr.geometry.AreaDefinition(area_id, name, proj_id, proj_dict, x_size,
                                          y_size, area_extent)
    swath_def = pr.geometry.SwathDefinition(lons, lats)

    result = pr.kd_tree.resample_nearest(swath_def, numobs, area_def,
                                         radius_of_influence=50000,
                                         fill_value=None)

    bmap = pr.plot.area_def2basemap(area_def)
    bmng = bmap.drawlsmask(land_color=(0.2, 0.6, 0.2),
                           ocean_color=(0.4, 0.4, 1.0),
                           lakes=True)
    legend = cm.gist_heat
    legend = cm.OrRd
    col = bmap.imshow(result, origin='upper', cmap=legend)
    cbar = bmap.colorbar()
    cbar.set_label('Number of matchups')
    #plt.savefig('/tmp/bmap_radval.png', bbox_inches='tight')
    #plt.savefig('/tmp/bmap_radval_withsynop.png', bbox_inches='tight')
    plt.savefig('/tmp/bmap_withsynop.png', bbox_inches='tight')
