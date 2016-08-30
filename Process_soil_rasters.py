## Process soil rasters from ISRIC (http://www.isric.org/data/soil-property-maps-africa-1-km)

# AfSIS soil layers:
# sd1: 0 - 5 cm
# sd2: 5 - 15 cm
# sd3: 15 - 30 cm
# sd4: 30 - 60 cm
# sd5: 60 - 100 cm
# sd6: 100 - 200 cm

# from arcpy import *
import os
import numpy as np
import math
from ftplib import FTP
import glob
import fnmatch
import shutil
import gzip
from osgeo import gdal
import pygeoprocessing.geoprocessing

# CheckOutExtension("Spatial")
# env.overwriteOutput = 1

# indir = 'C:/Users/Ginger/Documents/NatCap/GIS_local/Kenya_forage/Laikipia_soil_250m'

# outdir = r'C:\Users\Ginger\Documents\NatCap\GIS_local\Unilever\CENTURY_soil\CENTURY_soil.gdb'
# clip_feature = r'C:/Users/Ginger/Dropbox/NatCap_backup/CGIAR/Peru/Spatial_data_study_area/Canete_basin_buffer_2km.shp'
# clip_feature =  r'C:\Users\Ginger\Documents\NatCap\GIS_local\Laikipia_soil_250m\Laikipia_soil_clip'
# asciidir = r'C:\Users\Ginger\Dropbox\NatCap_backup\Forage_model\Data\Kenya\Soil'
# folder_list1 = ['SNDPPT_1km_glmrk.tif_', 'BLD_glmrk.tif_', 'CLYPPT_1km_glmrk.tif_']
# folder_list2 = ['SLTPPT_1km_glmrk.tif_', 'PHIHO5_1km_glmrk.tif_']

# output_folder = "soil_grids"
# folder_list_3 = [output_folder]
# outdir = os.path.join(indir, "soil_grids_processed")
# if not os.path.exists(outdir):
    # os.makedirs(outdir)

def download_CENTURY_rasters(files_to_retrieve, output_folder):
    try:
        # Log onto ISRIC's ftp site via anonymous user
        ftp = FTP('ftp.soilgrids.org','soilgrids', 'soilgrids')
        parent_dir = ftp.pwd()

        # Access the sub-directory where the files are stored
        folder = '/data/recent/' 
    except:
        print "Error accessing the ftp site or connection error.  Please check the address to ensure the site is correct."
        raise
        
    try:
        # Open ftp site and download all files    
        ftp.cwd('{}/{}'.format(parent_dir, folder))
        filenames = ftp.nlst()
        for search in files_to_retrieve:
            for filename in fnmatch.filter(filenames, search):
                local_filename = os.path.join(output_folder, filename)
                raster_file = open(local_filename, 'wb')
                ftp.retrbinary('RETR ' + filename, raster_file.write)
            raster_file.close()
        ftp.quit() 
    except:
        print "Error downloading the ftp files to the output folder."
        raster_file.close()
        raise
        
    try:
        # Extract the data from the zip files but skip sub-directories if
        # present and extract the data
        gz_files = glob.glob(os.path.join(output_folder, '*.gz'))
        for gz_filename in gz_files:
            outF = open(gz_filename[0:-3], 'wb')
            with gzip.open(gz_filename, 'rb') as inF:
                shutil.copyfileobj(inF, outF)
            outF.close()
            inF.close()
            os.remove(gz_filename)
    except:
        print "Error extracting tar files."
        raise

def clip_rasters(folder_list, indir, clip_feature):
    for folder in folder_list:
        env.workspace = os.path.join(indir, folder)

        rasterList = ListRasters() #"*M.tif")
        for raster in rasterList:
            extracted = sa.ExtractByMask(raster, clip_feature)
            name = os.path.join(outdir, str(raster))
            extracted.save(name)

def depth_average(prefix_l, suffix_l, depth_l, indir, outdir):
    """Calculate the average of different rasters representing soil
    depth, according to the depths supplied in the depth_l list.  Prefixes
    should identify soil components that will be combined across horizons;
    suffixes should identify different depths.  The order of weights in the
    weight_l list should correspond to the order of horizons identified by the
    suffix_l list."""
    
    def w_avg_op(*rasters):
        """Calculate the average."""
        
        raster_list = list(rasters)
        sum_ras = np.sum(raster_list, axis=0)
        div_ras = sum_ras / float(len(raster_list))
        div_ras[raster_list[0] <= 0] = 255
        div_ras[raster_list[0] == 255] = 255
        return div_ras
    
    def div_by_10(raster):
        div_ras = np.divide(raster, 10.0)
        div_ras[raster <= 0] = 255
        div_ras[raster == 255] = 255
        return div_ras
        
    def div_by_1000(raster):
        div_ras = np.divide(raster, 1000.0)
        div_ras[raster <= 0] = 255
        div_ras[raster == 255] = 255
        return div_ras

    base_raster = os.path.join(indir, prefix_l[0] + suffix_l[0] + '.tif')
    out_pixel_size = pygeoprocessing.geoprocessing.\
                                    get_cell_size_from_uri(base_raster)    
    for prefix in prefix_l:
        rasters = [os.path.join(indir, prefix + suffix + '.tif') for suffix
                   in suffix_l]
        result_ras = os.path.join(outdir, '%s_%d-%d.tif' % (prefix,
                                                            min(depth_l),
                                                            max(depth_l)))
        pygeoprocessing.geoprocessing.vectorize_datasets(rasters,
                    w_avg_op, result_ras, gdal.GDT_Float32, 255,
                    out_pixel_size, "dataset", vectorize_op=False,
                    datasets_are_pre_aligned=True, dataset_to_bound_index=1)
        if 'phihox' in prefix:
            raster_l = [result_ras]
            result_ras = os.path.join(outdir, '%s_%d-%d_div10.tif' %
                                      (prefix, min(depth_l), max(depth_l)))
            pygeoprocessing.geoprocessing.vectorize_datasets(raster_l,
                    div_by_10, result_ras, gdal.GDT_Float32, 255,
                    out_pixel_size, "dataset", vectorize_op=False,
                    datasets_are_pre_aligned=True, dataset_to_bound_index=1)
        if 'bld' in prefix:
            raster_l = [result_ras]
            result_ras = os.path.join(outdir, '%s_%d-%d_div1000.tif' %
                                      (prefix, min(depth_l), max(depth_l)))
            pygeoprocessing.geoprocessing.vectorize_datasets(raster_l,
                    div_by_1000, result_ras, gdal.GDT_Float32, 255,
                    out_pixel_size, "dataset", vectorize_op=False,
                    datasets_are_pre_aligned=True, dataset_to_bound_index=1)
    
def calculate_weighted_sums(prefix_l, suffix_l, weight_l, indir, outdir):
    """Calculate the weighted sum of different rasters representing soil
    horizons, according to the weights supplied in the weight_l list.  Prefixes
    should identify soil components that will be combined across horizons;
    suffixes should identify horizons.  The order of weights in the weight_l
    list should correspond to the order of horizons identified by th
    suffix_l list."""
    
    def pH_to_h_conc(pH_arr):
        h_conc_arr = np.empty(pH_arr.shape)
        pH_arr = np.divide(pH_arr, 10.0)  # pH rasters supplied as pH * 10
        for row in xrange(pH_arr.shape[0]):
            for col in xrange(pH_arr.shape[1]):
                pH = pH_arr[row][col]
                if pH < 0:
                    h_conc_arr[row][col] = np.nan
                else:
                    h_conc_arr[row][col] = 10**(-pH) * 1000
        return h_conc_arr
    
    def h_conc_to_pH(conc_arr):
        pH_arr = np.empty(conc_arr.shape)
        
        for row in xrange(conc_arr.shape[0]):
            for col in xrange(conc_arr.shape[1]):
                conc = float(conc_arr[row][col])
                if np.isnan(conc) or conc < 0:
                    pH_arr[row][col] = -1
                elif conc == 0:
                    pH_arr[row][col] = 0
                else:
                    pH_arr[row][col] = -math.log10(conc / 1000)
        return pH_arr
    weight_sum = sum(weight_l)
    sum_of_weights_raster = os.path.join(outdir, "weight_sum.tif")
    base_raster = os.path.join(indir, prefix_l[0] + suffix_l[0] + '.tif')
    pygeoprocessing.geoprocessing.new_raster_from_base_uri(base_raster,
            sum_of_weights_raster, 'GTiff', -9999.0, gdal.GDT_Float32,
            fill_value=weight_sum)
                    
    for prefix in prefix_l:
        rasters = [os.path.join(indir, prefix + suffix + '.tif') for suffix
                   in suffix_l]
        rasters_to_sum = []
        if 'PHI' in prefix or 'phi' in prefix:
            h_conc_l = []
            for raster in rasters:
                out_name = os.path.join(outdir, os.path.basename(raster)[:-4] +
                                        "h_conc.tif")
                out_pixel_size = pygeoprocessing.geoprocessing.\
                                    get_cell_size_from_uri(raster)
                pygeoprocessing.geoprocessing.vectorize_datasets([raster],
                                pH_to_h_conc, out_name, gdal.GDT_Float32,
                                -9999.0, out_pixel_size, "dataset",
                                vectorize_op=False,
                                datasets_are_pre_aligned=True,
                                dataset_to_bound_index=1)
                h_conc_l.append(out_name)
            rasters = h_conc_l
        # calculate weighted raster, multiply each raster by its weight
        for i in xrange(len(rasters)):
            raster = rasters[i]
            weight = weight_l[i]
            weight_raster = os.path.join(outdir, 'weight_%f.tif' % weight)
            if not os.path.exists(weight_raster):
                pygeoprocessing.geoprocessing.new_raster_from_base_uri(raster,
                        weight_raster, 'GTiff', -9999.0, gdal.GDT_Float32,
                        fill_value=weight)
            def calc_weighted_raster(raster, weight_raster):
                mask = raster < 0
                weighted = raster * weight_raster
                weighted[mask] = -1
                weighted[np.isnan(raster)] = -1
                return weighted
            weighted_name = os.path.join(outdir, os.path.basename(raster)[:-4]
                                         + "_weighted_%f.tif" % weight)
            out_pixel_size = pygeoprocessing.geoprocessing.\
                                    get_cell_size_from_uri(raster)
            pygeoprocessing.geoprocessing.vectorize_datasets([raster,
                    weight_raster], calc_weighted_raster, weighted_name,
                    gdal.GDT_Float32, -9999.0, out_pixel_size, "dataset",
                    vectorize_op=False, datasets_are_pre_aligned=True,
                    dataset_to_bound_index=1)
            
            # divide each weighted raster by sum of weights
            def divided_by_sum(raster, weight_sum):
                mask = raster < 0
                divided = raster / weight_sum
                divided[mask] = -1
                divided[np.isnan(raster)] = -1
                return divided
            out_name = os.path.join(outdir, os.path.basename(raster)[:-4] +
                                    "weighted_%f_divided.tif" % weight)
            rasters_to_sum.append(out_name)
            pygeoprocessing.geoprocessing.vectorize_datasets([weighted_name,
                    sum_of_weights_raster], divided_by_sum, out_name,
                    gdal.GDT_Float32, -9999.0, out_pixel_size, "dataset",
                    vectorize_op=False, datasets_are_pre_aligned=True,
                    dataset_to_bound_index=1)
            
        # sum weighted rasters
        def sum_rasters(*rasters):
            result = np.add(*rasters)
            return result
        out_name = os.path.join(outdir, prefix + "weighted_sum.tif")
        pygeoprocessing.geoprocessing.vectorize_datasets(rasters_to_sum,
                    sum_rasters, out_name, gdal.GDT_Float32, -9999.0,
                    out_pixel_size, "dataset", vectorize_op=False,
                    datasets_are_pre_aligned=True, dataset_to_bound_index=1)
        if 'PHI' in prefix or 'phi' in prefix:
            interm = os.path.join(outdir, "pH_interm.tif")
            shutil.copyfile(out_name, interm)
            os.remove(out_name)
            pygeoprocessing.geoprocessing.vectorize_datasets([interm],
                    h_conc_to_pH, out_name, gdal.GDT_Float32, -9999.0,
                    out_pixel_size, "dataset", vectorize_op=False,
                    datasets_are_pre_aligned=True, dataset_to_bound_index=1)
        # extract by original raster
        example_raster = os.path.join(indir, prefix + suffix_l[0] + '.tif')
        def extract_by_mask(raster_to_extract, mask_raster):
            raster_to_extract[mask_raster == 0] = -9999.0
            return raster_to_extract
        interm = os.path.join(outdir, "intermediate.tif")
        shutil.copyfile(out_name, interm)
        os.remove(out_name)
        pygeoprocessing.geoprocessing.vectorize_datasets([interm,
                example_raster], extract_by_mask, out_name,
                gdal.GDT_Float32, -9999.0, out_pixel_size, "dataset",
                vectorize_op=False, datasets_are_pre_aligned=True,
                dataset_to_bound_index=1)
    
def raster_to_array(prefix, suffixes, outdir):
    arcpy.env.workspace = outdir
    arr_list = []
    for suffix in suffixes:
        raster = prefix + suffix + '.tif'
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


# prefix_l = ['bld_', 'clyppt_', 'sndppt_', 'sltppt_', 'phihox_']
# prefix_l = [str.upper(f) for f in prefix_l]
# suffix_l = ['sd1_M_1km_T263', 'sd2_M_1km_T263'] #, 'sd3', 'sd4', 'sd5', 'sd6']
# weight_l = [5., 10.] # , 15., 30., 40., 100.]

# clip_rasters(folder_list_3, indir, clip_feature)
# sum_arr_list = calc_all_sums(prefixes, suffixes, weights, outdir)

# comp_list1 = # check original compositions sum to 1
# check_sums(comp_list1)
# comp_list2 = sum_arr_list[1:3]  # clay, sand and silt
# check_sums(comp_list2)  

# if isinstance(prefixes, str):
    # raster = prefixes + suffixes[0] + '_m'
    # name = prefixes + '0_15_cm'
# else:
    # raster = prefixes[0] + suffixes[0] + '.tif'  # example raster, for dimensions etc
# for i in xrange(len(sum_arr_list)):
    # arr = sum_arr_list[i]
    # name = prefixes[i] + '0_15_cm.tif'
    # array_to_raster(outdir, raster, arr, name)

# calculate_weighted_sums(prefix_l, suffix_l, weight_l, indir, outdir)

## extract values to points
# env.workspace = outdir
# raster_list = ListRasters("*0_15_cm")
# points = 'weather_stations'
# sa.ExtractMultiValuesToPoints(points, raster_list)

if __name__ == "__main__":
    indir = r"C:\Users\Ginger\Documents\NatCap\GIS_local\Kenya_forage\Laikipia_soil_250m\raw_downloads"
    outdir = 'C:/Users/Ginger/Documents/NatCap/GIS_local/Kenya_forage/Laikipia_soil_250m/averaged'
    prefix_l = ['geonode-%s_m_sl' % n for n in ['bldfie', 'clyppt', 'phihox',
                                                'sltppt', 'sndppt']]
    suffix_l = ['%d_250m' % n for n in [1, 2, 3]]
    depth_l = [0, 5, 15]
    depth_average(prefix_l, suffix_l, depth_l, indir, outdir)