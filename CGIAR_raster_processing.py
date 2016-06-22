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
graz_areas = "C:/Users/Ginger/Dropbox/NatCap_backup/CGIAR/Peru/Spatial_data_study_area/Estimated_grazing_areas.shp"
clim_folder = "C:/Users/Ginger/Documents/NatCap/GIS_local/CGIAR/Forage_model_data/climate_and_soil/Worldclim_current"
soil_folder = "C:/Users/Ginger/Documents/NatCap/GIS_local/CGIAR/Forage_model_data/climate_and_soil/soil_grids_processed"

def translate_solution(solution_table, HRU_raster, raster_out_uri):
    """"""
    
    out_datatype = 3
    source_dataset = gdal.Open(HRU_raster)
    band = source_dataset.GetRasterBand(1)
    out_nodata = band.GetNoDataValue()
    
    sol_df = pd.read_csv(solution_table)
    value_map = {row[3]: row[2] for row in sol_df.itertuples()}
    pygeoprocessing.geoprocessing.reclassify_dataset_uri(
               HRU_raster, value_map, raster_out_uri, out_datatype, out_nodata)
    
def create_mv_rasters(mv_table, subbasin_tif, outdir):
    """make marginal value rasters from livestock model that can be summarized
    by implementation unit polygons"""
    
    out_datatype = 6
    source_dataset = gdal.Open(subbasin_tif)
    band = source_dataset.GetRasterBand(1)
    out_nodata = band.GetNoDataValue()
    
    mv_df = pd.read_csv(mv_table)
    mv_grouped = mv_df.groupby(['animal', 'density'])
    for intervention in mv_grouped.groups.keys():
    # (each unique combination of animal and density)
        # make lookup dictionary from mv_table
        iv_df = mv_grouped.get_group(intervention)
        value_map = {row[1]: row[5] for row in iv_df.itertuples()}
        for subbasin in value_map.keys():
            if value_map[subbasin] == 'failed':
                value_map[subbasin] = out_nodata
            else:
                value_map[subbasin] = float(value_map[subbasin]) * 0.81
        raster_out_uri = os.path.join(outdir, 'mv_%s_%s.tif' % 
                                      (intervention[0], intervention[1]))
        print "generating raster %s_%s" % (intervention[0], intervention[1])
        pygeoprocessing.geoprocessing.reclassify_dataset_uri(
            subbasin_tif, value_map, raster_out_uri, out_datatype, out_nodata)

def summarize_mv_by_HRU():
    HRU_zones = r"C:\Users\Ginger\Documents\NatCap\GIS_local\CGIAR\Peru\Other_spatial_data\HRU_BLUG-FESC-RYEG.tif"
    zero_raster = r"C:\Users\Ginger\Documents\NatCap\GIS_local\CGIAR\Peru\Other_spatial_data\HRU_BLUG-FESC-RYEG-0.tif"
    objectives = ['livestock']  # ['sdr', 'swy', 'swyl']
    for obj in objectives:
        raster_folder = "C:/Users/Ginger/Documents/NatCap/GIS_local/CGIAR/Peru/summarized_by_zone/%s_mv_rasters_6.10.16" % obj
        arcpy.env.workspace = raster_folder
        mv_rasters = arcpy.ListRasters()
        save_dir = "C:/Users/Ginger/Documents/NatCap/GIS_local/CGIAR/Peru/summarized_by_zone/%s_mv_rasters_mosaic_6.10.16" % obj
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for raster in mv_rasters:
            inputs = [zero_raster, raster]
            save_name = os.path.basename(raster)
            arcpy.MosaicToNewRaster_management(inputs, save_dir, save_name, 
                                               "", "32_BIT_FLOAT", "", 1,
                                               "LAST", "")
        arcpy.env.workspace = save_dir
        mv_rasters = arcpy.ListRasters()
        HRU_outdir = "C:/Users/Ginger/Documents/NatCap/GIS_local/CGIAR/Peru/summarized_by_zone/%s_summary_tables_by_HRU_6.10.16" % obj
        if not os.path.exists(HRU_outdir):
            os.makedirs(HRU_outdir)
        summarize_by_zone(mv_rasters, HRU_zones, HRU_outdir)
        summary_csv = "C:/Users/Ginger/Documents/NatCap/GIS_local/CGIAR/Peru/summarized_by_zone/%s_mv_by_HRU_6.10.16.csv" % obj
        combine_tables(HRU_outdir, summary_csv)

def extract_by_mask(folder, mask, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir) 
    # extract by mask
    arcpy.env.workspace = folder
    rasters = arcpy.ListRasters()
    for raster in rasters:
        name = os.path.basename(raster)
        out = arcpy.sa.ExtractByMask(raster, mask)
        out.save(os.path.join(outdir, name))

def summarize_rasters(clim_folder):
    # calc average of climate rasters
    def return_average(*rasters):
        return np.mean(np.array(rasters), axis=0)
    tif_files = [f for f in os.listdir(clim_folder) if re.search(".tif$", f)]
    tmin_names = [f for f in tif_files if re.search("^tmin", f)]
    tmin_files = [os.path.join(clim_folder, f) for f in tmin_names]
    example_raster = tmin_files[0]
    dataset = gdal.Open(example_raster)
    band = dataset.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    del dataset
    del band
    out_pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(
                        example_raster)
    tmin_mean = os.path.join(clim_folder, "tmin_mean.tif")
    pygeoprocessing.geoprocessing.vectorize_datasets(
                tmin_files, return_average, tmin_mean,
                gdal.GDT_Float32, nodata, out_pixel_size, "union",
                dataset_to_align_index=0, assert_datasets_projected=False,
                vectorize_op=False, datasets_are_pre_aligned=True)
                
    tmax_names = [f for f in tif_files if re.search("^tmax", f)]
    tmax_files = [os.path.join(clim_folder, f) for f in tmax_names]
    out_pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(
                        tmax_files[0])
    tmax_mean = os.path.join(clim_folder, "tmax_mean.tif")
    pygeoprocessing.geoprocessing.vectorize_datasets(
                tmax_files, return_average, tmax_mean,
                gdal.GDT_Float32, nodata, out_pixel_size, "union",
                dataset_to_align_index=0, assert_datasets_projected=False,
                vectorize_op=False, datasets_are_pre_aligned=True)

    prec_names = [f for f in tif_files if re.search("^prec", f)]
    prec_files = [os.path.join(clim_folder, f) for f in prec_names]
    out_pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(
                        prec_files[0])
    prec_mean = os.path.join(clim_folder, "prec_mean.tif")
    pygeoprocessing.geoprocessing.vectorize_datasets(
                prec_files, return_average, prec_mean,
                gdal.GDT_Float32, nodata, out_pixel_size, "union",
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
                gdal.GDT_Float32, nodata, out_pixel_size, "union",
                dataset_to_align_index=0, assert_datasets_projected=False,
                vectorize_op=False, datasets_are_pre_aligned=True)
    pygeoprocessing.geoprocessing.vectorize_datasets(
                tmax_files, return_stdev, tmax_std,
                gdal.GDT_Float32, nodata, out_pixel_size, "union",
                dataset_to_align_index=0, assert_datasets_projected=False,
                vectorize_op=False, datasets_are_pre_aligned=True)
    pygeoprocessing.geoprocessing.vectorize_datasets(
                prec_files, return_stdev, prec_std,
                gdal.GDT_Float32, nodata, out_pixel_size, "union",
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
                                        statistics_type="SUM")

def combine_tables(table_dir, output_csv):
    sum_dict = {}
    arcpy.env.workspace = table_dir
    tables = arcpy.ListTables()
    for table in tables:
        sum_dict[table[:-4]] = []
        sum_dict['zone_' + table[:-4]] = []
        fields = arcpy.ListFields(table)
        field_names = ['Value', 'SUM']
        with arcpy.da.SearchCursor(table, field_names) as cursor:
            try:
                for row in cursor:
                    sum_dict['zone_' + table[:-4]].append(row[0])
                    sum_dict[table[:-4]].append(row[1])
            except:
                import pdb; pdb.set_trace()
                print table
    sum_df = pd.DataFrame.from_dict(sum_dict)
    sum_df.to_csv(output_csv)

def pH_to_h_conc(pH_raster):
    """Convert a raster of pH values to hydrogen ion concentration so that
    average can be calculated."""
    
    dataset = gdal.Open(pH_raster)
    band = dataset.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    del dataset
    del band
    out_pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(
                                                                pH_raster)                                                            
    out_name = os.path.join(os.path.dirname(pH_raster), "h_conc.tif")
    def convert_to_h_conc(pH_arr):
        h_conc_arr = np.empty(pH_arr.shape)
        for row in xrange(pH_arr.shape[0]):
            for col in xrange(pH_arr.shape[1]):
                pH = pH_arr[row][col]
                if pH < 0:
                    h_conc_arr[row][col] = np.nan
                else:
                    h_conc_arr[row][col] = 10**(-pH)
        return h_conc_arr
    
    pygeoprocessing.geoprocessing.vectorize_datasets(
                [pH_raster], convert_to_h_conc, out_name,
                gdal.GDT_Float32, nodata, out_pixel_size, "union",
                dataset_to_align_index=0, assert_datasets_projected=False,
                vectorize_op=False, datasets_are_pre_aligned=True)
                
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
    dataset = gdal.Open(example_raster)
    band = dataset.GetRasterBand(1)
    nodata = band.GetNoDataValue() / 10.0
    del dataset
    del band
    out_pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(
                                                                example_raster)
    for name in tif_names:
        in_raster = os.path.join(climate_dir, name)
        out_name = os.path.join(outdir, name)
        pygeoprocessing.geoprocessing.vectorize_datasets(
                [in_raster], div_by_ten, out_name,
                gdal.GDT_Float32, nodata, out_pixel_size, "union",
                dataset_to_align_index=0, assert_datasets_projected=False,
                vectorize_op=False, datasets_are_pre_aligned=True)
    
    
if __name__ == "__main__":
    # arcpy.env.overwriteOutput = 1
    
    # pH_raster = "C:/Users/Ginger/Documents/NatCap/GIS_local/CGIAR/Forage_model_data/climate_and_soil/soil_grids_processed/PHIHOX_weighted_sum_0-15.tif"
    # # pH_to_h_conc(pH_raster)
    # soil_raster_folder = "C:/Users/Ginger/Documents/NatCap/GIS_local/CGIAR/Forage_model_data/climate_and_soil/soil_grids_processed"
    # soil_zones = "C:/Users/Ginger/Documents/NatCap/GIS_local/CGIAR/Forage_model_data/summarized_by_zone/bld_3_zones_grz.tif"
    # soil_outdir = "C:/Users/Ginger/Documents/NatCap/GIS_local/CGIAR/Forage_model_data/summarized_by_zone/soil_summary_tables"
    # arcpy.env.workspace = soil_raster_folder
    # soil_rasters = arcpy.ListRasters()
    # # summarize_by_zone(soil_rasters, soil_zones, soil_outdir)
    # summary_csv = "C:/Users/Ginger/Documents/NatCap/GIS_local/CGIAR/Forage_model_data/summarized_by_zone/soil_summary.csv"
    # # combine_tables(soil_outdir, summary_csv)
    
    # climate_raster_folder = clim_folder
    # # summarize_rasters(climate_raster_folder)
    # # process_climate_rasters(climate_raster_folder)
    # climate_zones = "C:/Users/Ginger/Documents/NatCap/GIS_local/CGIAR/Forage_model_data/summarized_by_zone/mean_prec_3_zones_grz.tif"
    # clim_outdir = "C:/Users/Ginger/Documents/NatCap/GIS_local/CGIAR/Forage_model_data/summarized_by_zone/clim_summary_tables"
    # climate_dir = os.path.join(climate_raster_folder, 'converted')
    # arcpy.env.workspace = climate_dir
    # climate_rasters = arcpy.ListRasters()
    # # summarize_by_zone(climate_rasters, climate_zones, clim_outdir)
    # summary_csv = "C:/Users/Ginger/Documents/NatCap/GIS_local/CGIAR/Forage_model_data/summarized_by_zone/climate_summary.csv"
    # # combine_tables(clim_outdir, summary_csv)
    
    # elev_raster = [r"C:\Users\Ginger\Documents\NatCap\GIS_local\CGIAR\Forage_model_data\Elevation\strm_1k_canete.tif"]
    # clim_intermediate_dir = r"C:\Users\Ginger\Documents\NatCap\GIS_local\CGIAR\Forage_model_data\summarized_by_zone\elevation_by_clim"
    # soil_intermediate_dir = r"C:\Users\Ginger\Documents\NatCap\GIS_local\CGIAR\Forage_model_data\summarized_by_zone\elevation_by_soil"
    # # summarize_by_zone(elev_raster, climate_zones, clim_intermediate_dir)
    # # summarize_by_zone(elev_raster, soil_zones, soil_intermediate_dir)
    # summary_csv = "C:/Users/Ginger/Documents/NatCap/GIS_local/CGIAR/Forage_model_data/summarized_by_zone/elev_by_clim.csv"
    # # combine_tables(clim_intermediate_dir, summary_csv)
    # summary_csv = "C:/Users/Ginger/Documents/NatCap/GIS_local/CGIAR/Forage_model_data/summarized_by_zone/elev_by_soil.csv"
    # # combine_tables(soil_intermediate_dir, summary_csv)
    
    # soil_raster_folder = "C:/Users/Ginger/Documents/NatCap_4.15.16/GIS_local_4.26.16/CGIAR/Peru/climate_and_soil/soil_grids_processed"
    # soil_outdir = r"C:\Users\Ginger\Documents\NatCap_4.15.16\GIS_local_4.26.16\CGIAR\Peru\summarized_by_zone\soil_summary_tables_SWAT_subbasins"
    # arcpy.env.workspace = soil_raster_folder
    # soil_rasters = arcpy.ListRasters()
    # soil_zones = r"C:\Users\Ginger\Documents\NatCap_4.15.16\GIS_local_4.26.16\CGIAR\Peru\boundaries\SWAT_subbasins.tif"
    # # summarize_by_zone(soil_rasters, soil_zones, soil_outdir)
    # summary_csv = "C:/Users/Ginger/Documents/NatCap_4.15.16/GIS_local_4.26.16/CGIAR/Peru/summarized_by_zone/soil_summary_SWAT_subbasins.csv"
    # # combine_tables(soil_outdir, summary_csv)
    
    # ## create livestock marginal value tables and summarize them by SWAT HRU
    mv_table = r"C:\Users\Ginger\Dropbox\NatCap_backup\CGIAR\Peru\Forage_model_results\marginal_5.2.16_ed.csv"
    subbasin_tif = r"C:\Users\Ginger\Documents\NatCap\GIS_local\CGIAR\Peru\boundaries\SWAT_subbasins.tif"
    outdir = r"C:\Users\Ginger\Documents\NatCap\GIS_local\CGIAR\Peru\summarized_by_zone\livestock_mv_rasters_6.10.16"
    create_mv_rasters(mv_table, subbasin_tif, outdir)
    
    summarize_mv_by_HRU()
    
    