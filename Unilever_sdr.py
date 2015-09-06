import sys
import os
import csv
import re
import invest_natcap.sdr.sdr
from arcpy import *

raster_folder = r'C:\Users\Ginger\Documents\NatCap\GIS_local\Unilever\synthetic_landscapes\Spatial_data'

data_files = [f for f in os.listdir(raster_folder) if os.path.isfile(os.path.join(raster_folder, f))]
lulc_files = [f for f in data_files if re.search('^forest', f)]
lulc_tif = [f for f in lulc_files if re.search('.tif$', f)]

#for ic0 in [0.25, 0.5, 0.75]:
outer_dir = r'C:\Users\Ginger\Desktop\test'
for lulc in lulc_tif:
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

out_name = os.path.join(outer_dir, ('synthetic_landscapes_summary_' + outer_dir[73:] + '.csv'))
with open(out_name, 'wb') as out:
    writer = csv.writer(out, delimiter = ',')
    header = ['lulc_filename', 'sed_export', 'usle', 'sed_retention']
    writer.writerow(header)
    for lulc in lulc_tif:
        name = lulc[:-4]
        row = [name]
        output_shp = os.path.join(outer_dir, name, 'output', 'watershed_outputs.shp')
        fields = ['sed_export', 'usle_tot', 'sed_retent']
        with da.SearchCursor(output_shp, fields) as cursor:
            cursor_row = cursor.next()
            row = row + list(cursor_row)
        writer.writerow(row)