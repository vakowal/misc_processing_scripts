# calculate erosivity from worldclim

import arcpy
import os
import pygeoprocessing.geoprocessing
import shutil
from osgeo import gdal
import numpy as np

arcpy.CheckOutExtension("Spatial")

def mosaic_and_extract():
    aoi = r"C:\Users\Ginger\Documents\NatCap\GIS_local\Corinne\Volta\DEM_strm90m_subset_fill.tif"
    folder = r"C:\Users\Ginger\Documents\NatCap\GIS_local\Corinne\Volta\worldclim_volta"
    arcpy.env.workspace = folder
    
    example_raster = os.path.join(folder, 'prec1_26.tif')
    spatial_ref = arcpy.Describe(example_raster).spatialReference
    pixel_type = "32_BIT_FLOAT"
    cellsize = arcpy.GetRasterProperties_management(
                                          example_raster, "CELLSIZEX").getOutput(0)
    numberbands = arcpy.GetRasterProperties_management(
                                          example_raster, "BANDCOUNT").getOutput(0)

    mosaic_dir = os.path.join(folder, 'mosaic')
    if not os.path.exists(mosaic_dir):
        os.makedirs(mosaic_dir)

    ex_list = []
    for m in range(1, 13):
        rasters = [os.path.join(folder, 'prec{}_26.tif'.format(m)),
                   os.path.join(folder, 'prec{}_25.tif'.format(m))]
        result = 'prec{}_mosaic.tif'.format(m)
        ex_list.append(result)
        # arcpy.MosaicToNewRaster_management(
                                # rasters, mosaic_dir, result, spatial_ref,
                                # pixel_type, cellsize, numberbands, "LAST",
                                # "FIRST")

    ex_dir = os.path.join(folder, "extracted")
    if not os.path.exists(ex_dir):
        os.makedirs(ex_dir)

    aoi_in_mosaic_dir = os.path.join(mosaic_dir, os.path.basename(aoi))
    shutil.copyfile(aoi, aoi_in_mosaic_dir)
    
    arcpy.env.workspace = mosaic_dir
    for m_ras in ex_list:
        aoi_basename = os.path.basename(aoi_in_mosaic_dir)
        out = arcpy.sa.ExtractByMask(m_ras, aoi_basename)
        out.save(os.path.join(ex_dir, m_ras))

def project():
    folder = r"C:\Users\Ginger\Documents\NatCap\GIS_local\Corinne\Volta\worldclim_volta"
    source_dir = os.path.join(folder, "extracted")
    proj_dir = os.path.join(folder, 'projected')
    if not os.path.exists(proj_dir):
        os.makedirs(proj_dir)
        
    example_ras = r"C:\Users\Ginger\Documents\NatCap\GIS_local\Corinne\Volta\DEM_strm90m_subset_fill.tif"
    spatial_ref = arcpy.Describe(example_ras).spatialReference
    
    arcpy.env.workspace = source_dir
    rasters = arcpy.ListRasters()
    
    for ras in rasters:
        out_ras = "{}_proj.tif".format(ras)
        arcpy.ProjectRaster_management(ras, out_ras, spatial_ref,
                                       resampling_type="NEAREST")
        source_name = os.path.join(source_dir, out_ras)
        dest_name = os.path.join(proj_dir, out_ras)
        shutil.copyfile(source_name, dest_name)

def calc_erosivity():
    """Calculate erosivity from equation 5 in Vrieling et al 2010"""
    
    data_dir = r"C:\Users\Ginger\Documents\NatCap\GIS_local\Corinne\Volta\worldclim_volta\projected"
    raw_p_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if
                       f.endswith('.tif')]
    squared_dir = r"C:\Users\Ginger\Documents\NatCap\GIS_local\Corinne\Volta\worldclim_volta\squared"
    squared_list = [os.path.join(squared_dir, f) for f in
                       os.listdir(squared_dir) if f.endswith('.tif')]
                       
    def one_over_P_op(*rasters):
        mean_P = np.mean(np.array(rasters), axis=0)
        one_arr = np.full(rasters[0].shape, 1.0)
        one_over_P =  np.divide(one_arr, mean_P)
        return one_over_P
        
    def sum_squares(*rasters):
        return np.sum(np.array(rasters), axis=0)
        
    example_raster = raw_p_list[0]
    dataset = gdal.Open(example_raster)
    band = dataset.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    del dataset
    del band
    out_pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(
                                                                example_raster)
    
    one_over_P = r"C:\Users\Ginger\Documents\NatCap\GIS_local\Corinne\Volta\worldclim_volta\one_over_P.tif"
    pygeoprocessing.geoprocessing.vectorize_datasets(
                raw_p_list, one_over_P_op, one_over_P,
                gdal.GDT_Float32, nodata, out_pixel_size, "union",
                dataset_to_align_index=0, assert_datasets_projected=True,
                vectorize_op=False, datasets_are_pre_aligned=True)

    save_as = r"C:\Users\Ginger\Documents\NatCap\GIS_local\Corinne\Volta\worldclim_volta\sum_of_squares.tif"  
    pygeoprocessing.geoprocessing.vectorize_datasets(
                squared_list, sum_squares, save_as,
                gdal.GDT_Float32, nodata, out_pixel_size, "union",
                dataset_to_align_index=0, assert_datasets_projected=True,
                vectorize_op=False, datasets_are_pre_aligned=True)

if __name__ == "__main__":
    # mosaic_and_extract()
    # project()
    calc_erosivity()