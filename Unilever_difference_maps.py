#####
# Unilever rasters: difference maps and count changed cells
# 12.5.14
#####

from arcpy import *
import numpy as np

buf_dis = 1  # in units of raster cells

## create mask of buffer area of interest
downstream_distance = testRaster
cellsize = float((GetRasterProperties_management(downstream_distance,
            'CELLSIZEX')).getOutput(0))
            
dist_arr = RasterToNumPyArray(downstream_distance)
upper_limit = sqrt(
buf_arr = np.where(dist_arr >= cellsize and dist_arr <= , DEM_burn_arr, -9999)