import os

import gdal
import numpy
import pygeoprocessing
import glob
import os
import re
import csv

from osgeo import ogr

def subtract_rasters(scenario_folder, baseline_raster):
    """Subtract each raster in scenario_folder from the baseline raster
    (baseline_raster), and store the output raster in a new folder inside the
    scenario folder called "subtracted"."""
    
    dataset = gdal.Open(baseline_raster)
    band = dataset.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    data_type = gdal.GetDataTypeName(band.DataType)
    del dataset
    del band
    out_pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(
                                                               baseline_raster)                                                      
    def subtract_op(scenario, baseline):
        return numpy.subtract(scenario, baseline)
    out_folder = os.path.join(scenario_folder, "subtracted")
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    tif_names = [f for f in os.listdir(scenario_folder) if
                                                         re.search(".tif$", f)]
    tif_files = [os.path.join(scenario_folder, name) for name in tif_names]
    for scenario in tif_files:
        out_name = os.path.join(out_folder, os.path.basename(scenario)[:-4] +
                                "_minus_baseline.tif")
        pygeoprocessing.geoprocessing.vectorize_datasets(
                [scenario, baseline_raster], subtract_op, out_name,
                gdal.GDT_Float32, nodata, out_pixel_size, "union",
                dataset_to_align_index=0, assert_datasets_projected=False,
                vectorize_op=False, datasets_are_pre_aligned=True)
                
def calc_baseline_MSA(scenario_folder, baseline_raster, shapefile_uri,
                      summary_table):
    """For each scenario in scenario_folder, calculate mean baseline MSA.
    Calculate this as the mean of baseline MSA values within pixels that differ
    in MSA between the baseline raster and the scenario raster."""
    
    dataset = gdal.Open(baseline_raster)
    band = dataset.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    data_type = gdal.GetDataTypeName(band.DataType)
    del dataset
    del band
    out_pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(
                                                               baseline_raster)                                                      
    def mean_baseline_msa(scenario, baseline):
        diff_raster = numpy.subtract(scenario, baseline)
        baseline[diff_raster == 0] = numpy.nan
        return baseline
    out_folder = os.path.join(scenario_folder, "mean_baseline_msa")
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    tif_names = [f for f in os.listdir(scenario_folder) if
                                                         re.search(".tif$", f)]
    tif_files = [os.path.join(scenario_folder, name) for name in tif_names]
    with open(summary_table, 'wb') as out:
        writer = csv.writer(out, delimiter=',')
        header = ['scenario', 'mean_baseline_msa']
        writer.writerow(header)
        for scenario in tif_files:
            scen_id = os.path.basename(scenario)[:-4]
            out_name = os.path.join(out_folder, scen_id + "_baseline_rest.tif")
            pygeoprocessing.geoprocessing.vectorize_datasets(
                    [scenario, baseline_raster], mean_baseline_msa, out_name,
                    gdal.GDT_Float32, nodata, out_pixel_size, "union",
                    dataset_to_align_index=0, assert_datasets_projected=False,
                    vectorize_op=False, datasets_are_pre_aligned=True)
            mean_val = pygeoprocessing.geoprocessing.aggregate_raster_values_uri(
                                      out_name, shapefile_uri).pixel_mean[9999]
            row = [scen_id, mean_val]
            writer.writerow(row)
        
        

def histogram(output_dir):
    for msa_path in glob.glob(r"C:\Users\Ginger\Documents\NatCap\GIS_local\Unilever\Uncertainty_scenario_results\GLOBIO\MT_MSA_results\subtracted\*.tif"):
        
            print "PROCESSING SCENARIO %s" % msa_path
            suffix = os.path.splitext(os.path.basename(msa_path))[0]
            print suffix
            raster = gdal.Open(msa_path)
            band = raster.GetRasterBand(1)
            array = band.ReadAsArray()
            unique_values = pygeoprocessing.unique_raster_values_uri(msa_path)
            table = open(os.path.join(output_dir, suffix + '.csv'), 'wb')
            table.write('MSAvalue,count\n')
            print array.shape
            for value in unique_values:
                table.write('%s,%s\n' % (str(value), str(numpy.count_nonzero(array==value))))

if __name__ == "__main__":
    ## BECKY set these values here
    
    scenario_folder = r"C:\Users\Ginger\Documents\NatCap\GIS_local\Unilever\Uncertainty_scenario_results\GLOBIO\MT_MSA_results"
    # scenario_folder = r"C:\Users\Ginger\Downloads\IA_scenarios"
    # baseline_raster = r"C:\Users\Ginger\Downloads\msa_IA_2007_baseline.tif"
    # shapefile_uri = r"C:\Users\Ginger\Documents\NatCap\GIS_local\Unilever\Becky_GLOBIO_reproduce_1.20.16\Iowa_globio_inputs\IA_aoi.shp"
    # baseline raster: the baseline raster you want to subtract from scenarios
    baseline_raster = r"C:\Users\Ginger\Documents\NatCap\GIS_local\Unilever\Uncertainty_scenario_results\GLOBIO\MT_MSA_results\msa_MT_scen_A_UTM.tif"
    # shapefile_uri = r"C:\Users\Ginger\Documents\NatCap\GIS_local\Unilever\Becky_GLOBIO_reproduce_1.20.16\MT_globio_inputs\MT_aoi.shp"
    # shapefile_uri = r"C:\Users\Ginger\Dropbox\NatCap_backup\Unilever\Unilever_model_inputs_10.6.15\Iowa_state_extent\Iowa_UTM.shp"
    # summary_table = r"C:\Users\Ginger\Desktop\baseline_mean_msa.csv"
    #Iowa
    # baseline_raster = r"C:\Users\Ginger\Downloads\msa_MT_2007_baseline.tif"
    
    subtract_rasters(scenario_folder, baseline_raster)
    output_dir = os.path.join(scenario_folder, 'histogram')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    histogram(output_dir)
    calc_baseline_MSA(scenario_folder, baseline_raster, shapefile_uri,
                      summary_table)