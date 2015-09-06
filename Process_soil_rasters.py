## Process soil rasters from ISRIC (http://www.isric.org/data/soil-property-maps-africa-1-km)

# AfSIS soil layers:
# sd1: 0━5 cm
# sd2: 5━15 cm
# sd3: 15━30 cm
# sd4: 30━60 cm
# sd5: 60━100 cm
# sd6: 100━200 cm

from arcpy import *
import os
import numpy as np
import math

CheckOutExtension("Spatial")
env.overwriteOutput = 1

indir = r'C:\Users\Ginger\Documents\NatCap\GIS_local\Unilever\CENTURY_soil'
outdir = r'C:\Users\Ginger\Documents\NatCap\GIS_local\Unilever\CENTURY_soil\CENTURY_soil.gdb'
clip_feature = r'C:\Users\Ginger\Documents\NatCap\GIS_local\Laikipia_soil\Laikipia_soil.gdb\Laikipia_clip_area'
asciidir = r'C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Data\Kenya\Soil'
folder_list1 = ['SNDPPT_1km_glmrk.tif_', 'BLD_glmrk.tif_', 'CLYPPT_1km_glmrk.tif_']
folder_list2 = ['SLTPPT_1km_glmrk.tif_', 'PHIHO5_1km_glmrk.tif_']

def pH_to_h_conc(pH_arr):
    h_conc_arr = np.empty(pH_arr.shape)
    for row in xrange(len(pH_arr)):
        for col in xrange(len(pH_arr[0])):
            pH = pH_arr[row][col]
            if pH < 0:
                h_conc_arr[row][col] = np.nan
            else:
                h_conc_arr[row][col] = 10**(-pH) * 1000
    return h_conc_arr

def h_conc_to_pH(conc_arr):
    pH_arr = np.empty(conc_arr.shape)
    for row in xrange(len(conc_arr)):
        for col in xrange(len(conc_arr)):
            conc = float(conc_arr[row][col])
            if np.isnan(conc) or conc < 0:
                pH_arr[row][col] = -1
            elif conc == 0:
                pH_arr[row][col] = 0
            else:
                pH_arr[row][col] = -math.log10(conc / 1000)
    return pH_arr

def clip_rasters(folder_list, indir, clip_feature):
    for folder in folder_list:
        env.workspace = os.path.join(indir, folder)

        rasterList = ListRasters() #"*M.tif")
        for raster in rasterList:
            extracted = sa.ExtractByMask(raster, clip_feature)
            name = os.path.join(outdir, str(raster))[:-4]
            extracted.save(name)

#clip_rasters(folder_list2, indir, clip_feature)

def raster_to_array(prefix, suffixes, outdir):
    arcpy.env.workspace = outdir
    arr_list = []
    for suffix in suffixes:
        raster = prefix + suffix + '_m'
        arr = RasterToNumPyArray(raster)
        if 'phi' in prefix:
            arr = pH_to_h_conc(arr)
        arr_list.append(arr)
    return arr_list

def calc_weighted_list(arr_list, weights):
    weighted_list = []
    for i in xrange(len(weights)):
        arr = arr_list[i]
        weight = weights[i]
        mask = arr < 0
        weighted = arr * (weight / sum(weights))
        weighted[mask] = -1
        weighted[np.isnan(arr)] = -1
        weighted_list.append(weighted)
        # ave = np.average(weighted[weighted > 0])
        # print "weight: %f, arr average: %f" % (weight, ave)
    return weighted_list

def calc_total_comp(weighted_list):
    sum_arr = np.empty(weighted_list[0].shape)
    for arr in weighted_list:
        mask = arr < 0
        sum_arr = sum_arr + arr
        sum_arr[mask] = -1
    # ave = np.average(sum_arr[sum_arr > 0])
    # count = np.count_nonzero(sum_arr[sum_arr > 0])
    # print "value count: %d, average of summed arrays: %f" % (count, ave)
    return sum_arr

def calc_all_sums(prefixes, suffixes, weights, outdir):
    sum_arr_list = []
    if isinstance(prefixes, str):
        prefix = prefixes
        arr_list = raster_to_array(prefix, suffixes, outdir)
        weighted_list = calc_weighted_list(arr_list, weights)
        sum_arr = calc_total_comp(weighted_list)
        if 'phi' in prefix:
            sum_arr = h_conc_to_pH(sum_arr)
        sum_arr_list.append(sum_arr)
    else:
        for prefix in prefixes:
            arr_list = raster_to_array(prefix, suffixes, outdir)
            weighted_list = calc_weighted_list(arr_list, weights)
            sum_arr = calc_total_comp(weighted_list)
            if 'phi' in prefix:
                sum_arr = h_conc_to_pH(sum_arr)
            sum_arr_list.append(sum_arr)
    return sum_arr_list

def check_sums(comp_list):
    sum = np.empty(comp_list[0].shape)
    for arr in comp_list:
        sum = sum + arr
        sum[arr < 0] = 0
    print 'sum mean: %f' % np.mean(sum)
    print 'sum max: %f' % np.max(sum)

def array_to_raster(wkspace, raster, array, name):
    lower_left = Point((GetRasterProperties_management(raster,
        "LEFT")).getOutput(0), (GetRasterProperties_management(raster,
        "BOTTOM")).getOutput(0))
    spatial_ref = Describe(raster).spatialReference
    out_arr = np.where(array < 0, np.nan, array)
    out_raster = NumPyArrayToRaster(out_arr, lower_left, raster,
        raster, np.nan)
    out_raster.save(name)
    DefineProjection_management(name, spatial_ref)

    
prefixes = ['bld_', 'clyppt_', 'sndppt_', 'sltppt_', 'phihox_']
suffixes = ['sd1', 'sd2'] #, 'sd3', 'sd4', 'sd5', 'sd6']
weights = [5., 10.] # , 15., 30., 40., 100.]
# sum_arr_list = calc_all_sums(prefixes, suffixes, weights, outdir)

# comp_list1 = # check original compositions sum to 1
# check_sums(comp_list1)
# comp_list2 = sum_arr_list[1:3]  # clay, sand and silt
# check_sums(comp_list2)  

# if isinstance(prefixes, str):
    # raster = prefixes + suffixes[0] + '_m'
    # name = prefixes + '0_15_cm'
# else:
    # raster = prefixes[0] + suffixes[0] + '_m'  # example raster, for dimensions etc
# for i in xrange(len(sum_arr_list)):
    # arr = sum_arr_list[i]
    # name = prefixes[i] + '0_15_cm'
    # array_to_raster(outdir, raster, arr, name)

## extract values to points
env.workspace = outdir
raster_list = ListRasters("*0_15_cm")
points = 'weather_stations'
sa.ExtractMultiValuesToPoints(points, raster_list)