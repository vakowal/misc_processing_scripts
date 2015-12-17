# miscellaneous snippets useful for CGIAR raster processing
# 12.15.15

import arcpy
import numpy as np
import pandas as pd
import os
import pygeoprocessing.geoprocessing
import re
from osgeo import gdal
import csv

arcpy.CheckOutExtension("Spatial")

canete = "C:/Users/Ginger/Dropbox/NatCap_backup/CGIAR/Peru/Spatial_data_study_area/Canete_basin_buffer_2km.shp"
highlands = "C:/Users/Ginger/Dropbox/NatCap_backup/CGIAR/Peru/Spatial_data_study_area/Estimated_highlands_districts.shp"
clim_folder = "C:/Users/Ginger/Documents/NatCap/GIS_local/CGIAR/Forage_model_data/climate_and_soil/Worldclim_current"
soil_folder = "C:/Users/Ginger/Documents/NatCap/GIS_local/CGIAR/Forage_model_data/climate_and_soil/soil_grids_processed"
clim_folder = "C:/Users/Ginger/Documents/NatCap/GIS_local/CGIAR/Forage_model_data/climate_and_soil/Worldclim_current/canete"
prj_folder = os.path.join(clim_folder, "projected")
if not os.path.exists(prj_folder):
    os.makedirs(prj_folder)

# extract by mask
# arcpy.env.workspace = clim_folder
# rasters = arcpy.ListRasters()
# outdir = os.path.join(clim_folder, "canete")
# for raster in rasters:
    # name = os.path.basename(raster)
    # out = arcpy.sa.ExtractByMask(raster, canete)
    # out.save(os.path.join(outdir, name))

def summarize_rasters():
    # calc average of climate rasters
    def return_average(*rasters):
        return np.mean(np.array(rasters), axis=0)

    arcpy.env.workspace = clim_folder    
    rasters = arcpy.ListRasters()
    out_coord = """PROJCS['WGS_1984_UTM_Zone_18S', GEOGCS['GCS_WGS_1984',
                   DATUM['D_WGS_1984',SPHEROID['WGS_1984',6378137.0,298.257223563]],
                   PRIMEM['Greenwich',0.0],UNIT['Degree',0.0174532925199433]],
                   PROJECTION['Transverse_Mercator'],PARAMETER['False_Easting',
                   500000.0],PARAMETER['False_Northing',10000000.0],
                   PARAMETER['Central_Meridian',-75.0],
                   PARAMETER['Scale_Factor',0.9996],
                   PARAMETER['Latitude_Of_Origin',0.0],UNIT['Meter',1.0]]"""
    in_coord = """GEOGCS['GCS_WGS_1984',DATUM['D_WGS_1984',SPHEROID['WGS_1984',
                  6378137.0,298.257223563]],PRIMEM['Greenwich',0.0],
                  UNIT['Degree',0.0174532925199433]]"""
               
    for raster in rasters:
        out_raster = os.path.join(prj_folder, raster)
        arcpy.ProjectRaster_management(in_raster=raster, out_raster=out_raster,
                                       out_coor_system=out_coord,
                                       resampling_type="BILINEAR",
                                       cell_size="916.940248645902 916.940248645901",
                                       geographic_transform="#",
                                       Registration_Point="#",
                                       in_coor_system=in_coord)

    tif_files = [f for f in os.listdir(prj_folder) if re.search(".tif$", f)]
    tmin_names = [f for f in tif_files if re.search("^tmin", f)]
    tmin_files = [os.path.join(clim_folder, f) for f in tmin_names]
    out_pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(
                        tmin_files[0])
    tmin_mean = os.path.join(clim_folder, "tmin_mean.tif")
    pygeoprocessing.geoprocessing.vectorize_datasets(
                tmin_files, return_average, tmin_mean,
                gdal.GDT_Float32, -9999, out_pixel_size, "union",
                dataset_to_align_index=0, assert_datasets_projected=False,
                vectorize_op=False, datasets_are_pre_aligned=True)
                
    tmax_names = [f for f in tif_files if re.search("^tmax", f)]
    tmax_files = [os.path.join(clim_folder, f) for f in tmax_names]
    out_pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(
                        tmax_files[0])
    tmax_mean = os.path.join(clim_folder, "tmax_mean.tif")
    pygeoprocessing.geoprocessing.vectorize_datasets(
                tmax_files, return_average, tmax_mean,
                gdal.GDT_Float32, -9999, out_pixel_size, "union",
                dataset_to_align_index=0, assert_datasets_projected=False,
                vectorize_op=False, datasets_are_pre_aligned=True)

    prec_names = [f for f in tif_files if re.search("^prec", f)]
    prec_files = [os.path.join(clim_folder, f) for f in prec_names]
    out_pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(
                        prec_files[0])
    prec_mean = os.path.join(clim_folder, "prec_mean.tif")
    pygeoprocessing.geoprocessing.vectorize_datasets(
                prec_files, return_average, prec_mean,
                gdal.GDT_Float32, -9999, out_pixel_size, "union",
                dataset_to_align_index=0, assert_datasets_projected=False,
                vectorize_op=False, datasets_are_pre_aligned=True)

    # calc standard deviation of climate rasters
    def return_stdev(*rasters):
        std_raster = np.std(np.array(rasters), axis=0)
        result_raster = np.full((rasters[0].shape), -9999)
        np.copyto(result_raster, std_raster, where=np.isfinite(rasters[0]))
        return result_raster

    tmin_std = os.path.join(clim_folder, "tmin_std.tif")
    tmax_std = os.path.join(clim_folder, "tmax_std.tif")
    prec_std = os.path.join(clim_folder, "prec_std.tif")    
    pygeoprocessing.geoprocessing.vectorize_datasets(
                tmin_files, return_stdev, tmin_std,
                gdal.GDT_Float32, -9999, out_pixel_size, "union",
                dataset_to_align_index=0, assert_datasets_projected=False,
                vectorize_op=False, datasets_are_pre_aligned=True)
    pygeoprocessing.geoprocessing.vectorize_datasets(
                tmax_files, return_stdev, tmax_std,
                gdal.GDT_Float32, -9999, out_pixel_size, "union",
                dataset_to_align_index=0, assert_datasets_projected=False,
                vectorize_op=False, datasets_are_pre_aligned=True)
    pygeoprocessing.geoprocessing.vectorize_datasets(
                prec_files, return_stdev, prec_std,
                gdal.GDT_Float32, -9999, out_pixel_size, "union",
                dataset_to_align_index=0, assert_datasets_projected=False,
                vectorize_op=False, datasets_are_pre_aligned=True)

def summarize_by_zone(raster_list, zonal_raster, outdir):
    """Summarize input rasters by zone."""
    
    arcpy.BuildRasterAttributeTable_management(zonal_raster)
    for raster in raster_list:
        intermediate_table = os.path.join(outdir, os.path.basename(raster)[:-4]
                                          + '.dbf')
        arcpy.sa.ZonalStatisticsAsTable(zonal_raster, 'VALUE', raster,
                                        intermediate_table,
                                        statistics_type="MEAN")

def combine_tables(table_dir, output_csv):
    sum_dict = {}
    arcpy.env.workspace = table_dir
    tables = arcpy.ListTables()
    for table in tables:
        sum_dict[table[:-4]] = []
        sum_dict['zone_' + table[:-4]] = []
        fields = arcpy.ListFields(table)
        field_names = ['Value', 'MEAN']
        with arcpy.da.SearchCursor(table, field_names) as cursor:
            for row in cursor:
                sum_dict['zone_' + table[:-4]].append(row[0])
                sum_dict[table[:-4]].append(row[1])
    sum_df = pd.DataFrame.from_dict(sum_dict)
    sum_df.to_csv(output_csv)
    
def process_climate_rasters(climate_dir):
    """The raw temp rasters from WorldClim are supplied as deg C * 10.  Divide
    them all by 10 to get deg C for CENTURY.  The precip rasters from WorldClim
    are supplied in mm.  Divide them by 10 to get cm for CENTURY."""
    
    outdir = os.path.join(climate_dir, 'converted')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    def div_by_ten(raster):
        return np.divide(raster, 10.0)  
    tif_names = [f for f in os.listdir(climate_dir) if re.search(".tif$", f)]
    example_raster = os.path.join(climate_dir, tif_names[0])
    out_pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(
                                                                example_raster)
    for name in tif_names:
        in_raster = os.path.join(climate_dir, name)
        out_name = os.path.join(outdir, name)
        pygeoprocessing.geoprocessing.vectorize_datasets(
                [in_raster], div_by_ten, out_name,
                gdal.GDT_Float32, -9999, out_pixel_size, "union",
                dataset_to_align_index=0, assert_datasets_projected=False,
                vectorize_op=False, datasets_are_pre_aligned=True)
    
    
if __name__ == "__main__":
    soil_raster_folder = "C:/Users/Ginger/Documents/NatCap/GIS_local/CGIAR/Forage_model_data/climate_and_soil/soil_grids_processed"
    soil_zones = "C:/Users/Ginger/Documents/NatCap/GIS_local/CGIAR/Forage_model_data/summarized_by_zone/bld_3_zones.tif"
    soil_outdir = "C:/Users/Ginger/Documents/NatCap/GIS_local/CGIAR/Forage_model_data/summarized_by_zone/soil_summary_tables"
    arcpy.env.workspace = soil_raster_folder
    soil_rasters = arcpy.ListRasters()
    # summarize_by_zone(soil_rasters, soil_zones, soil_outdir)
    summary_csv = os.path.join(soil_outdir, "summary.csv")
    # combine_tables(soil_outdir, summary_csv)
    
    climate_raster_folder = "C:/Users/Ginger/Documents/NatCap/GIS_local/CGIAR/Forage_model_data/climate_and_soil/Worldclim_current/canete"
    climate_zones = "C:/Users/Ginger/Documents/NatCap/GIS_local/CGIAR/Forage_model_data/summarized_by_zone/mean_prec_3_zones.tif"
    clim_outdir = "C:/Users/Ginger/Documents/NatCap/GIS_local/CGIAR/Forage_model_data/summarized_by_zone/clim_summary_tables"
    process_climate_rasters(climate_raster_folder)
    climate_dir = os.path.join(climate_raster_folder, 'converted')
    arcpy.env.workspace = climate_dir
    climate_rasters = arcpy.ListRasters()
    summarize_by_zone(climate_rasters, climate_zones, clim_outdir)
    summary_csv = os.path.join(clim_outdir, "summary.csv")
    combine_tables(clim_outdir, summary_csv)
    