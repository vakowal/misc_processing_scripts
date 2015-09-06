# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 12:06:46 2015

@author: Ginger
"""

import os
import invest_natcap.sdr.sdr

raster_folder = r'C:\Users\Ginger\Documents\NatCap\GIS_local\Unilever\synthetic_landscapes\Spatial_data\single_slope'

outer_dir = r'C:\Users\Ginger\Documents\NatCap\GIS_local\Unilever\synthetic_landscapes\defaults_12.11.14'
lulc = 'grass_100.tif'
args = {
        u'workspace_dir': unicode(os.path.join(outer_dir, lulc[:-4])),
        u'results_suffix': u'',
        u'dem_uri': unicode(os.path.join(raster_folder, 'dem_15-.01.tif')),
        u'erosivity_uri': unicode(os.path.join(raster_folder, 'erosivity.tif')),
        u'erodibility_uri': unicode(os.path.join(raster_folder, 'erodibility.tif')),
        u'lulc_uri': unicode(os.path.join(raster_folder, lulc)),
        u'watersheds_uri': unicode(os.path.join(raster_folder, 'watershed.shp')),
        u'biophysical_table_uri': u'C:\\Users\\Ginger\\Dropbox\\NatCap_backup\\Unilever\\Synthetic_landscapes\\biophysical_coeffs_Brazil_Unilever_global.csv',
        u'threshold_flow_accumulation': u'21',
        u'k_param': u'2.0',
        u'sdr_max': u'0.8',
        u'ic_0_param': u'0.5',
        u'drainage_uri': u'',
}        
invest_natcap.sdr.sdr.execute(args)