"""Point sampler for precipitation datasets."""
import collections
import subprocess
import os
import glob
import logging

from osgeo import ogr
from osgeo import gdal
import taskgraph
import pandas

GPM_DIR = "./GPM_IMERG"
CHIRPS_DIR = "C:/Users/Ginger/Documents/NatCap/GIS_local/Mongolia/CHIRPS/raw_data/CHIRPS"  # "./CHIRPS"
POINT_SHAPEFILE_PATH = "C:/Users/Ginger/Documents/NatCap/GIS_local/Mongolia/CHIRPS/CHIRPS_pixel_centroid_monitoring_soums.shp"  # r"./fetch_climate_data_for_points_in_this_shapefile/CBM_SCP_sites.shp"

logging.basicConfig(
    format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG)

def decompress_gzip(base_path):
    """Decompress gzip in base_path."""
    cmd = 'gzip -d "%s"' % base_path
    print cmd
    subprocess.call(cmd)


def main():
    """Entry point."""
    task_graph = taskgraph.TaskGraph('taskgraph_cache', 0)

    # decompress any .gz if necessary
    for chirps_path in glob.glob(os.path.join(CHIRPS_DIR, '*.gz')):
        gtiff_basename = os.path.splitext(chirps_path)[0]
        task_graph.add_task(
            func=decompress_gzip,
            args=(gtiff_basename,),
            target_path_list=[gtiff_basename])

    task_graph.close()
    task_graph.join()

    # index all raster paths by their basename
    raster_path_id_map = {}
    # for raster_path in [
            # path for dir_path in [GPM_DIR, CHIRPS_DIR]
            # for path in glob.glob(os.path.join(dir_path, '*.tif'))]:
    for raster_path in [
            path for dir_path in [CHIRPS_DIR]
            for path in glob.glob(os.path.join(dir_path, '*.tif'))]:
        basename = os.path.splitext(os.path.basename(raster_path))[0]
        raster_path_id_map[basename] = raster_path

    # pre-fetch point geometry and IDs
    point_vector = ogr.Open(POINT_SHAPEFILE_PATH)
    point_layer = point_vector.GetLayer()
    point_defn = point_layer.GetLayerDefn()

    # build up a list of the original field names so we can copy it to report
    point_field_name_list = []
    for field_index in xrange(point_defn.GetFieldCount()):
        point_field_name_list.append(
            point_defn.GetFieldDefn(field_index).GetName())
    # this will hold (x,y) coordinates for each point in its iterator order
    point_coord_list = []
    # this maps fieldnames to a list of the values associated with that
    # fieldname in the order that the points are read in and written to
    # `point_coord_list`.
    feature_attributes_fieldname_map = collections.defaultdict(list)
    for point_feature in point_layer:
        sample_point_geometry = point_feature.GetGeometryRef()
        for field_name in point_field_name_list:
            feature_attributes_fieldname_map[field_name].append(
                point_feature.GetField(field_name))
        point_coord_list.append(
            (sample_point_geometry.GetX(), sample_point_geometry.GetY()))
    del point_layer
    del point_vector

    # each element will hold the point samples for each raster in the order of
    # `point_coord_list`
    sampled_precip_data_list = []
    for field_name in point_field_name_list:
        sampled_precip_data_list.append(
            pandas.Series(
                data=feature_attributes_fieldname_map[field_name],
                name=field_name))
    for basename in sorted(raster_path_id_map.iterkeys()):
        path = raster_path_id_map[basename]
        raster = gdal.Open(path)
        band = raster.GetRasterBand(1)
        geotransform = raster.GetGeoTransform()
        sample_list = []
        for point_x, point_y in point_coord_list:
            raster_x = int((
                point_x - geotransform[0]) / geotransform[1])
            raster_y = int((
                point_y - geotransform[3]) / geotransform[5])
            sample_list.append(
                band.ReadAsArray(raster_x, raster_y, 1, 1)[0, 0])
        sampled_precip_data_list.append(
            pandas.Series(data=sample_list, name=basename))

    report_table = pandas.DataFrame(data=sampled_precip_data_list)
    report_table = report_table.transpose()
    save_as = r"C:\Users\Ginger\Dropbox\NatCap_backup\Mongolia\data\climate\CHIRPS\precip_data_CHIRPS_pixels.csv"
    report_table.to_csv(save_as)


if __name__ == '__main__':
    main()