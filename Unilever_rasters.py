from arcpy import *
import os
import re
import csv
import numpy as np

ascii_folder = r'C:\Users\Ginger\Dropbox\NatCap_backup\Unilever\Synthetic_landscapes\Ascii_text\double'
raster_folder = r'C:\Users\Ginger\Documents\NatCap\GIS_local\Unilever\synthetic_landscapes\Spatial_data\double'

# env.workspace = ascii_folder
# env.OverwriteOutput = True
# text_list = ListFiles()

# for ascii in text_list:
    # out_name = os.path.join(raster_folder, ascii[:-4] + '.tif')
    # ASCIIToRaster_conversion(ascii, out_name)
    
# env.workspace = raster_folder
# spatial_ref = Describe('watershed.shp').spatialReference

# for raster in ListRasters():
    # DefineProjection_management(raster, spatial_ref)

outer_dir = r'C:\Users\Ginger\Documents\NatCap\GIS_local\Unilever\synthetic_landscapes\broken'
data_files = [f for f in os.listdir(raster_folder) if os.path.isfile(os.path.join(raster_folder, f))]
lulc_files = [f for f in data_files if re.search('^forest', f)]
lulc_tif = [f for f in lulc_files if re.search('.tif$', f)]

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
