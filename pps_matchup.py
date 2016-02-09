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

"""Get the PPS cloud mask data for some scenes and do the satellite-synop
co-location. Write data to files. One file for each satellite swath.
"""

import os
from glob import glob
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from pypps_reader import NwcSafPpsData
from pyresample import spherical_geometry
from pykdtree.kdtree import KDTree
from synop_dwd import get_data

# The window size over which the satellite cloud cover is derived:
ROW_SZ, COL_SZ = (20, 20)
# ROW_SZ, COL_SZ = (6, 6)
# The time it takes for one sensor scan line:
# AVHRR, scans the earth at a rate of 6 scans per second
# GAC AVHRR, scan rate is 2 GAC scans per second
LINE_TIME = {}
LINE_TIME['avhrr_lac'] = timedelta(seconds=1. / 6.)
LINE_TIME['avhrr_gac'] = timedelta(seconds=1. / 2.)
LINE_TIME['viirs'] = timedelta(seconds=86. / (48. * 16))

time_thr = timedelta(seconds=1800)
# time_thr = timedelta(seconds=3600*6)
time_thr_swath = timedelta(seconds=(1800 + 100 * 60))

# ROOTDIR = "/media/Elements/data/pps_v2014_val"
# ROOTDIR = "/local_disk/data/pps_test"
# ROOTDIR = "/nobackup/smhid11/sm_ninha/pps/data_osisaf"
ROOTDIR = "/run/media/a000680/Elements/data/VIIRS_processed_with_ppsv2014patch_plus"
# ROOTDIR = "/nobackup/smhid11/sm_adam/pps/data_osisaf
# ROOTDIR = "/home/a000680/data/pps_val_v2014"

OVERWRITE = True
SYNOP_DATADIR = "./DataFromDwd"
# SYNOP_DATADIR =
# "/data/proj6/saf/adybbroe/satellite_synop_matchup/DataFromDwd"

# OUTPUT_DIR = "/local_disk/laptop/satellite_synop_matchup/data"
OUTPUT_DIR = "./data"

INSTRUMENT = {'npp': 'viirs', 'noaa18': 'avhrr', 'noaa19': 'avhrr'}
ANGLE_NAMES = ['SUNZ', 'SATZ', 'SSAZD']
SAT_FIELDS = {}
SAT_FIELDS['viirs'] = ['sunz', 'satz', 'ssazd', 'ciwv', 'tsur', 'm05',
                       'm07', 'm10', 'm12', 'm14', 'm15', 'm16']
SAT_FIELDS['avhrr'] = ['sunz', 'satz', 'ssazd', 'cwiv', 'tsur', '1',
                       '2', '3a', '3b', 'dummy', '4', '5']

# FILE_PATTERN =
FILE_PATTERN = "export/{start_month:%Y%m}/S_NWC_{product:s}_{satellite:s}_{orbit:5d}_{start_date:%Y%m%d}T{start_time:%H%M%S}{tenthsec_start:1d}Z_{end_time:%Y%m%dT%H%M%S}{tenthsec_end:1d}Z.h5"
# FILE_PATTERN = "export/S_NWC_{product:s}_{satellite:s}_{orbit:5d}_{start_date:%Y%m%d}T{start_time:%H%M%S}{tenthsec_start:1d}Z_{end_time:%Y%m%dT%H%M%S}{tenthsec_end:1d}Z.h5"

try:
    from trollsift import Parser
    pps_fparser = Parser(FILE_PATTERN)
except ImportError:
    print("No trollsift. Try cope without it...")
    pps_fparser = None


class SatellitePointData(object):

    def __init__(self):
        self.lon = 0
        self.lat = 0
        self.id = None
        self.time = None
        self.deltatime = 0
        self.cloudcover = 0
        self.cloudtype = 0
        self.channels = {}
        self.nwp = {}
        self.angles = {}

    def __str__(self):
        retv = ('Lon = ' + str(self.lon) + '\nLat = ' + str(self.lat) +
                '\nName/Id = ' + str(self.id) + '\n' +
                'Time = ' + str(self.time) + '\n' +
                'Time diff = ' + str(self.deltatime) + '\n' +
                'Cloud cover = ' + str(self.cloudcover) + '\n' +
                'Cloudtype = ' + str(self.cloudtype) + '\n')
        for item in [str(key) + ' = ' +
                     str(self.channels[key]) + ' '
                     for key in self.channels.keys()]:
            retv = retv + item
        retv = retv + '\n'
        for item in [str(key) + ' = ' +
                     str(self.angles[key]) + ' '
                     for key in self.angles.keys()]:
            retv = retv + item

        return retv


def get_satellite_data(point, imagerdata, sunsat_angles, nwp_ciwv, nwp_tsur):
    """Get the instrument data and viewing angles"""

    retv = {}
    for idx in range(1, 6):
        if hasattr(sunsat_angles, 'image%.1d' % idx):
            obj = getattr(sunsat_angles, 'image%.1d' % idx)
            info = getattr(obj, 'info')
            for angle_name in ANGLE_NAMES:
                if info['product'] == angle_name:
                    dname = angle_name.lower()
                    value = getattr(obj, 'data')[point[0], point[1]]
                    if value != info['nodata'] or value != info['missingdata']:
                        retv[dname] = value * info['gain'] + info['offset']
                    else:
                        retv[dname] = -999

    if hasattr(nwp_ciwv, 'ciwv'):
        obj = getattr(nwp_ciwv, 'ciwv')
        info = getattr(obj, 'info')
        value = getattr(obj, 'data')[point[0], point[1]]
        dname = 'ciwv'
        if value != info['nodata']:
            retv[dname] = value * info['gain'] + info['intercept']
        else:
            retv[dname] = -999

    if nwp_tsur and hasattr(nwp_tsur, 'tsur'):
        obj = getattr(nwp_tsur, 'tsur')
        info = getattr(obj, 'info')
        value = getattr(obj, 'data')[point[0], point[1]]
        dname = 'tsur'
        if value != info['nodata']:
            retv[dname] = value * info['gain'] + info['intercept']
        else:
            retv[dname] = -999

    for idx in range(1, 8):
        if hasattr(imagerdata, 'image%.1d' % idx):
            obj = getattr(imagerdata, 'image%.1d' % idx)
            info = getattr(obj, 'info')
            if info['product'] == 'SATCH':
                dname = info['channel'].lower()
                value = getattr(obj, 'data')[point[0], point[1]]
                if value != info['nodata'] or value != info['missingdata']:
                    retv[dname] = value * info['gain'] + info['offset']
                else:
                    retv[dname] = -999

    return retv


def get_cloud_cover(pps_ctype, point, box):
    """Get the cloud cover around the pixel *point* using a box with dimensions
    *box* """

    try:
        ctypedata = pps_ctype.cloudtype.data
    except AttributeError:
        ctypedata = pps_ctype.ct.data

    shape = ctypedata.shape

    center_val = ctypedata[point[0], point[1]]

    y_left, y_right = point[0] - box[0] / 2, point[0] + box[0] / 2
    x_left, x_right = point[1] - box[1] / 2, point[1] + box[1] / 2
    if (x_left < 0 or y_left < 0 or
            x_right >= shape[1] or y_right >= shape[0]):
        return None, center_val

    ctyarr = ctypedata[y_left: y_right,
                       x_left: x_right]
    ncl = np.sum(np.logical_and(np.greater(ctyarr, 4),
                                np.less(ctyarr, 20)))
    ntot = np.sum(np.logical_and(ctyarr != 0, ctyarr != 20))
    if ntot == 0:
        return None, center_val

    return float(ncl) / float(ntot), center_val


def mask_bounds(boundary, nodata=None):
    """Mask out no-data in the boundary lon,lat coordinates"""

    for idx in range(2):
        mask = np.logical_and(boundary[idx].side1 > -0.999,
                              boundary[idx].side1 < -0.998)
        if nodata:
            mask = np.logical_or(
                mask, np.abs(boundary[idx].side1 - nodata) < 0.01)
        boundary[idx].side1 = np.ma.masked_array(
            boundary[idx].side1, mask=mask)

        mask = np.logical_and(boundary[idx].side2 > -0.999,
                              boundary[idx].side2 < -0.998)
        if nodata:
            mask = np.logical_or(
                mask, np.abs(boundary[idx].side2 - nodata) < 0.01)
        boundary[idx].side2 = np.ma.masked_array(
            boundary[idx].side2, mask=mask)

        mask = np.logical_and(boundary[idx].side3 > -0.999,
                              boundary[idx].side3 < -0.998)
        if nodata:
            mask = np.logical_or(
                mask, np.abs(boundary[idx].side3 - nodata) < 0.01)
        boundary[idx].side3 = np.ma.masked_array(
            boundary[idx].side3, mask=mask)

        mask = np.logical_and(boundary[idx].side4 > -0.999,
                              boundary[idx].side4 < -0.998)
        if nodata:
            mask = np.logical_or(
                mask, np.abs(boundary[idx].side4 - nodata) < 0.01)
        boundary[idx].side4 = np.ma.masked_array(
            boundary[idx].side4, mask=mask)

    return boundary


class Matchup(object):

    """Satellite data and surface/point measurement collocation"""

    def __init__(self, ctype, avhrr, angles, ciwv, **kwargs):
        self.ctype = ctype
        self.avhrr = avhrr
        self.angles = angles
        self.ciwv = ciwv
        if 'tsur' in kwargs:
            self.tsur = kwargs['tsur']
        else:
            self.tsur = None

        self.platform = self.avhrr._how['platform']
        self.instrument = self.avhrr._how['instrument']

        if self.avhrr._how['instrument'] == 'viirs':
            self.data_type = 'viirs'
        elif self.avhrr._how['instrument'] == 'avhrr':
            if self.avhrr._how['orbit_number'] == 99999:
                self.data_type = 'avhrr_gac'
            else:
                self.data_type = 'avhrr_lac'
        else:
            raise IOError("Satellite instrument type not supported: " +
                          str(self.avhrr._how['instrument']))

        self.obstime = self.ctype.info['time']
        print("Observation time = " + str(self.obstime))
        self.resultfile = os.path.join(
            OUTPUT_DIR, './matchup_%s_%s.txt' % (self.platform,
                                                 self.obstime.strftime('%Y%m%d%H%M')))

        self.matchupdata = []

    def matchup(self, points, dtobj, **kwargs):
        """Do the avhrr/viirs - synop matchup and write data to file.
        """

        if 'names' in kwargs:
            names = kwargs['names']
        else:
            names = range(len(points))

        geodata = np.vstack((self.avhrr.area.lons.ravel(),
                             self.avhrr.area.lats.ravel())).T
        kd_tree = KDTree(geodata)
        satdata_shape = self.avhrr.area.lons.shape

        try:
            corners = self.avhrr.area.corners
            clons = np.array([s.lon for s in corners])
            clats = np.array([s.lat for s in corners])
        except ValueError:
            print "Failed getting the corners..."
            corners = None

        if not corners or (0. in np.diff(clons) and 0. in np.diff(clats)):
            print("Something suspecious with the geolocation corners!")
            # Lets set the bounding corners ignoring the first and last few lines
            # of swath!
            bounds = self.avhrr.area.get_boundary_lonlats()
            # Should mask out nodata values in lon,lat:
            nodata = self.avhrr.lon.info[
                'nodata'] * self.avhrr.lon.info['gain']
            # FIXME!

            bounds = mask_bounds(bounds, nodata)
            corners = [spherical_geometry.Coordinate(bounds[0].side4.compressed()[-1],
                                                     bounds[1].side4.compressed()[-1]),
                       spherical_geometry.Coordinate(bounds[0].side2.compressed()[0],
                                                     bounds[1].side2.compressed()[0]),
                       spherical_geometry.Coordinate(bounds[0].side2.compressed()[-1],
                                                     bounds[1].side2.compressed()[-1]),
                       spherical_geometry.Coordinate(bounds[0].side4.compressed()[0],
                                                     bounds[1].side4.compressed()[0])]

            clons = np.array([s.lon for s in corners])
            clats = np.array([s.lat for s in corners])
            if 0. in np.diff(clons) and 0. in np.diff(clats):
                raise ValueError(
                    "After adjusting corners, still something fishy...")

        for point, name in zip(points, names):
            lon, lat = point
            obs_loc = spherical_geometry.Coordinate(lon, lat)

            is_inside = True
            try:
                is_inside = spherical_geometry.point_inside(obs_loc, corners)
            except ZeroDivisionError:
                print("date, station, lon,lat: %r (%f,%f)" %
                      (dtobj.strftime('%Y-%m-%d %H:%M'), lon, lat))
                raise

            if not is_inside:
                print("Outside...")
                continue

            # Find the index of the closest pixel in the satellite data:
            req_point = np.array([[np.rad2deg(obs_loc.lon),
                                   np.rad2deg(obs_loc.lat)]])
            dist, kidx = kd_tree.query(req_point, k=1)
            if dist > 0.1:
                print("Point too far away from swath...")
                continue

            row, col = np.unravel_index(kidx[0], satdata_shape)
            # Now that we have the pixel position in swath, we can calculate the
            # actual observation time for that pixel, assuming the observation time
            # at hand applies to the first line:
            pixel_time = self.obstime + row * LINE_TIME[self.data_type]
            t1_ = pixel_time - time_thr
            t2_ = pixel_time + time_thr
            if dtobj < t1_ or dtobj > t2_:
                print("Pixel time outside window: " +
                      str(pixel_time) + " " + str(self.obstime))
                continue

            satdata = get_satellite_data((row, col),
                                         self.avhrr, self.angles, self.ciwv, self.tsur)

            cloudcover, ctype_center = get_cloud_cover(
                self.ctype, (row, col), (ROW_SZ, COL_SZ))
            if not cloudcover:
                print "Cloud cover not available from PPS..."
                cloudcover = -9

            tdelta = pixel_time - dtobj

            ppsdata = SatellitePointData()
            ppsdata.lon = lon
            ppsdata.lat = lat
            ppsdata.id = name
            ppsdata.time = dtobj
            ppsdata.deltatime = tdelta
            ppsdata.cloudcover = cloudcover
            ppsdata.cloudtype = ctype_center
            for key in satdata:
                if key in [s.lower() for s in ANGLE_NAMES]:
                    ppsdata.angles[key] = satdata[key]
                elif key == 'ciwv':
                    ppsdata.nwp[key] = satdata[key]
                elif key == 'tsur':
                    ppsdata.nwp[key] = satdata[key]
                else:
                    ppsdata.channels[key] = satdata[key]

            self.matchupdata.append(ppsdata)

    def writedata(self):
        """Write the matchup data to file"""

        fd_ = open(self.resultfile, 'w')
        fd_.write('# Platform: ' + str(self.platform) + '\n')
        fd_.write('# Instrument: ' + str(self.instrument) + '\n')
        headerline = ('# lon, lat, station-id, time, sat-synop time diff, ' +
                      'pps cloud cover, pps cloudtype')
        for field in SAT_FIELDS[self.instrument]:
            headerline = headerline + ', ' + field
        fd_.write(headerline + '\n')

        for item in self.matchupdata:
            line = (item.lon, item.lat,
                    str(item.id).rjust(7),
                    item.time.strftime('%Y%m%d%H%M'),
                    (item.deltatime.seconds +
                     item.deltatime.days * 24 * 3600) / 60,
                    item.cloudcover,
                    item.cloudtype)
            fd_.write("%6.2f %6.2f %s %s %5.1f %5.2f %3d " % line)

            for elem in SAT_FIELDS[self.instrument]:
                if item.channels.has_key(elem) and item.channels[elem] != -999:
                    fd_.write(' %6.2f' % item.channels[elem])
                elif item.angles.has_key(elem) and item.angles[elem] != -999:
                    fd_.write(' %6.2f' % item.angles[elem])
                elif item.nwp.has_key(elem) and item.nwp[elem] != -999:
                    fd_.write(' %6.2f' % item.nwp[elem])
                else:
                    fd_.write(' -999')

            fd_.write('\n')
        fd_.close()


def get_scenes(tstart, tend, satellite='npp'):
    """Get all scene files within specified time interval"""

    instr = INSTRUMENT.get(satellite, 'avhrr')
    scene_files = []
    oneday = timedelta(days=1)
    tslot = tstart
    while tslot < tend:
        # Find the cloudtype:
        if pps_fparser:
            matchstr = os.path.join(
                ROOTDIR, pps_fparser.globify({'product': 'CT',
                                              'satellite': satellite,
                                              'start_date': tslot,
                                              'start_month': tslot}))
        else:
            matchstr = os.path.join(ROOTDIR,
                                    "export/S_NWC_CT_%s_?????_%sT*_*Z.h5" % (satellite, tslot.strftime('%Y%m%d')))

        cty_files_aday = glob(matchstr)

        for ctyfile in cty_files_aday:
            fname = os.path.basename(ctyfile).replace('_CT_',
                                                      '_%s_' % instr)
            dirname = os.path.dirname(ctyfile).replace('export',
                                                       'import/PPS_data/remapped')
            viirsfile = os.path.join(dirname, fname)

            fname = os.path.basename(ctyfile).replace('_CT_',
                                                      '_sunsatangles_')
            dirname = os.path.dirname(ctyfile).replace('export',
                                                       'import/ANC_data')
            angles_file = os.path.join(dirname, fname)

            fname = os.path.basename(ctyfile).replace('_CT_',
                                                      '_nwp_ciwv_')
            dirname = os.path.dirname(ctyfile).replace('export',
                                                       'import/NWP_data/remapped')
            ciwv_file = os.path.join(dirname, fname)
            fname = os.path.basename(ctyfile).replace('_CT_',
                                                      '_nwp_tsur_')
            tsur_file = os.path.join(dirname, fname)

            if os.path.exists(viirsfile):
                scene_files.append(
                    (ctyfile, viirsfile, angles_file, ciwv_file, tsur_file))

        tslot = tslot + oneday

    return scene_files


def get_radvaldata(filename):
    """Read the lon,lat points from Øystein"""
    dtype = [('lon', 'f8'), ('lat', 'f8'), ('id', '|S5'), ]

    data = np.genfromtxt(filename,
                         skip_header=1,
                         dtype=dtype,
                         unpack=True)
    return data

# -------------------------------------------------------------------------
if __name__ == "__main__":

    starttime = datetime(2012, 10, 1, 0, 0)
    endtime = datetime(2014, 6, 1, 0, 0)
    scenes = []
    scenes = get_scenes(starttime, endtime, 'npp')
    scenes = scenes + get_scenes(starttime, endtime, 'noaa18')
    scenes = scenes + get_scenes(starttime, endtime, 'noaa19')

    datapoints = get_radvaldata("radval-stlist.txt")

    print scenes[0]
    import h5py

    for scene in scenes:
        print("Ctype file: %s" % os.path.basename(scene[0]))
        fileread_ok = True
        for filename in scene:
            try:
                h5f = h5py.File(filename, 'r')
                h5f.close()
            except IOError:
                print(
                    "Failed opening file! Skipping scene with file: %s" % filename)
                fileread_ok = False
                break

        if not fileread_ok:
            continue

        try:
            avhrr_obj = NwcSafPpsData(scene[1])
        except (ValueError, IOError):
            print("Skipping scene with file: %s" % scene[1])
            continue

        try:
            ctype_obj = NwcSafPpsData(scene[0])
        except IOError:
            print("Failed opening ctype file %s" % scene[0])
            continue
        try:
            angles_obj = NwcSafPpsData(scene[2])
        except IOError:
            print("Failed opening angles file %s" % scene[2])
            continue
        try:
            ciwv_obj = NwcSafPpsData(scene[3])
        except IOError:
            print("Failed opening nwp ciwv file %s" % scene[3])
            continue
        try:
            tsur_obj = NwcSafPpsData(scene[4])
        except IOError:
            print("Failed opening nwp ciwv file %s" % scene[4])
            continue

        this = Matchup(
            ctype_obj, avhrr_obj, angles_obj, ciwv_obj, tsur=tsur_obj)

        if OVERWRITE and os.path.exists(this.resultfile):
            print("File exists: %s ...continue" % this.resultfile)
            continue

        # Round the time to nearest hour:
        sattime = this.obstime
        dtobj = datetime(sattime.year,
                         sattime.month,
                         sattime.day,
                         sattime.hour)
        this.matchup(np.vstack((datapoints['lon'],
                                datapoints['lat'])).transpose(),
                     dtobj, names=datapoints['id'])
        dtobj = datetime(sattime.year,
                         sattime.month,
                         sattime.day,
                         sattime.hour) + timedelta(seconds=3600)
        this.matchup(np.vstack((datapoints['lon'],
                                datapoints['lat'])).transpose(),
                     dtobj, names=datapoints['id'])
        this.writedata()
