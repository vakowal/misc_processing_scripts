"""Collect landscape composition variables inside plot boundary buffers.

This script automates the calculation of the area of different habitat types in
buffers surrounding plot boundaries.

Ginger Kowal for IFPRI
gkowal@stanford.edu
December 2019
"""
import os
import shutil
import tempfile

import pandas
import re
import arcpy
from arcpy import sa

arcpy.CheckOutExtension("Spatial")

# GLOBAL VARIABLES: modify these to refer to local paths
PLOT_SHP_PATH = "<local_path>/SpatialData/SpatialData/FarmBoundaries.shp"

# this land cover raster includes a background value of 20 in areas not covered
# by the original land cover raster
LULC_TIF_PATH = "<local_path>/lclu_with_background.tif"

# output table where the results should be written
OUTPUT_TABLE_PATH = "<local_path>/landscape_variables_test.csv"

# set environment variables: extent and snap raster (so that internal
# resampling of plot boundaries is consistent with lulc grid structure)
arcpy.env.extent = LULC_TIF_PATH
arcpy.env.snapRaster = LULC_TIF_PATH

# temporary processing directory
processing_dir = tempfile.mkdtemp()

# fields to retrieve with search cursor
field_list = ['SHAPE@', 'HHID', 'PlotID', 'FID']

# list of buffer lengths to collect landscape composition, in meters
buffer_val_list = [100]  # 10, 1000]

# temporary shapefile to hold buffer
temp_buf_shp = os.path.join(processing_dir, 'temp_buf.shp')
buf_shp_zone_field = 'ORIG_FID'
temp_area_table = os.path.join(processing_dir, 'temp_table.dbf')

# for each buffer distance
for buf_val in buffer_val_list:
    # for each feature in PLOT_SHP_PATH
    with arcpy.da.SearchCursor(PLOT_SHP_PATH, field_list) as cursor:
        for row in cursor:
            selected_feature = row[0]
            # generate temporary buffer shapefile
            arcpy.Buffer_analysis(
                selected_feature, temp_buf_shp, buf_val)

            # calculate area of each lulc type inside the buffer
            arcpy.sa.TabulateArea(
                temp_buf_shp, buf_shp_zone_field, LULC_TIF_PATH, "VALUE",
                temp_area_table)

            # save table to temporary csv
            temp_csv_basename = 'temp_table_buf{}_HH{}_Plot{}_{}.csv'.format(
                buf_val, row[1], row[2], row[3])  # HHID, PlotID, FID
            arcpy.TableToTable_conversion(
                temp_area_table, processing_dir, temp_csv_basename)

            # delete temporary file targets so they can be reused
            arcpy.Delete_management(temp_buf_shp)
            arcpy.Delete_management(temp_area_table)

# combine csv tables
# unique values in the lclu raster (this is a hack, but expedient!)
raster_val_list = range(1, 13) + [16, 18, 19, 20]
table_col_list = ['VALUE_{}'.format(val) for val in raster_val_list]
df_list = []
csv_bn_list = [
    p for p in os.listdir(processing_dir) if p.startswith('temp_table_buf')
    and p.endswith('.csv')]
for csv_bn in csv_bn_list:
    buf_val = int(re.search('buf(.+?)_HH', csv_bn).group(1))
    HHID = int(re.search('HH(.+?)_Plot', csv_bn).group(1))
    PlotID = int(re.search('Plot(.+?)_', csv_bn).group(1))
    csv_path = os.path.join(processing_dir, csv_bn)
    df = pandas.read_csv(csv_path)
    # add missing columns, so that we can concatenate the data frames together
    cols_to_add = list(
        set(table_col_list).difference(set(df.columns.values)))
    for new_col in cols_to_add:
        df[new_col] = 0
    df = df[table_col_list]
    df['Buffer_distance_m'] = buf_val
    df['HHID'] = HHID
    df['PlotID'] = PlotID
    df_list.append(df)
concat_df = pandas.concat(df_list)
concat_df.to_csv(OUTPUT_TABLE_PATH)

# clean up
shutil.rmtree(processing_dir)
