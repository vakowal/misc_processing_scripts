"""Extract data from multi-band NDVI rasters on dates of biomass sampling."""
import os
import collections

import numpy
import pandas
from sklearn.linear_model import LinearRegression

from osgeo import gdal
from osgeo import ogr

import pygeoprocessing

# points at which biomass was sampled
POINT_SHP_PATH = "E:/GIS_local/Mongolia/From_Boogie/shapes/GK_reanalysis/CBM_SCP_sites.shp"

# table containing biomass and sampling date of biomass collection
BIOMASS_TABLE = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/data/CBM_SCP_sites_2016_2017_herb_biomass.csv"

# table containing match between sampling date and closest NDVI band
BAND_MATCH_TABLE = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/biomass_vs_NDVI/match_date_band.csv"


def raster_values_at_points(
        point_shp_path, raster_path, band, raster_field_name):
    """Collect values from a raster intersecting points in a shapefile.

    Parameters:
        point_shp_path (string): path to shapefile containing point features
            where raster values should be extracted. Must be in geographic
            coordinates and must have a site id field called 'site_id'
        raster_path (string): path to raster containing values that should be
            extracted at points
        band (int): band index of the raster to analyze
        raster_field_name (string): name to assign to the field in the data
            frame that contains values extracted from the raster

    Returns:
        a pandas data frame with one column 'site_id' containing site_id values
            of point features, and one column raster_field_name containing
            values from the raster at the point location

    """
    raster_nodata = pygeoprocessing.get_raster_info(raster_path)['nodata'][0]
    # TODO this is a hack
    raster_nodata = 32767
    point_vector = ogr.Open(point_shp_path)
    point_layer = point_vector.GetLayer()

    # build up a list of the original field names so we can copy it to report
    point_field_name_list = ['site_id']

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
    point_layer = None
    point_vector = None

    # each element will hold the point samples for each raster in the order of
    # `point_coord_list`
    sampled_precip_data_list = []
    for field_name in point_field_name_list:
        sampled_precip_data_list.append(
            pandas.Series(
                data=feature_attributes_fieldname_map[field_name],
                name=field_name))
    raster = gdal.Open(raster_path)
    band = raster.GetRasterBand(band)
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
        pandas.Series(data=sample_list, name=raster_field_name))

    raster = None
    band = None

    # set nodata values to NA
    report_table = pandas.DataFrame(data=sampled_precip_data_list)
    report_table = report_table.transpose()
    try:
        report_table.loc[
            numpy.isclose(report_table[raster_field_name], raster_nodata),
            raster_field_name] = None
    except TypeError:
        report_table[raster_field_name] = pandas.to_numeric(
            report_table[raster_field_name], errors='coerce')
        report_table.loc[
            numpy.isclose(report_table[raster_field_name], raster_nodata),
            raster_field_name] = None
    return report_table


def main():
    """Program entry point."""
    img_pattern = "E:/GIS_local/Mongolia/NDVI/fitted_NDVI_3day/H25V04_H26V04_fitted_NDVI_<year>.006_Mongolia.img"
    biomass_df = pandas.read_csv(BIOMASS_TABLE)

    # all sampling was done in August

    # band indices: the NDVI is in 3-daily bands, 122 bands for the year
    # August occupies days 213-243, hence bands 72-81
    # closest bands to sampling days:
    match_df = pandas.read_csv(BAND_MATCH_TABLE)

    # files and bands at which to extract NDVI
    year_list = [2016, 2017]
    band_list = match_df['band'].unique().tolist()

    # get NDVI at those dates
    df_list = []
    for year in year_list:
        ndvi_path = img_pattern.replace('<year>', str(year))
        for band in band_list:
            ndvi_df = raster_values_at_points(
                POINT_SHP_PATH, ndvi_path, band, 'NDVI')
            ndvi_df['band'] = band
            ndvi_df['Year'] = year
            df_list.append(ndvi_df)
    sum_df = pandas.concat(df_list)
    biomass_df = pandas.read_csv(BIOMASS_TABLE)
    merged_biomass_df = biomass_df.merge(
        match_df, on='Day', suffixes=(False, False))
    merged_ndvi_df = merged_biomass_df.merge(
        sum_df, on=['site_id', 'Year', 'band'])
    merged_ndvi_df.drop(columns='band', inplace=True)
    save_as = "C:/Users/ginge/Dropbox/NatCap_backup/Mongolia/biomass_vs_NDVI/biomass_NDVI_table.csv"
    merged_ndvi_df.to_csv(save_as, index=False)

    # covert biomass from g/m2 to kg/ha
    merged_ndvi_df['herb_biomass_kgha'] = merged_ndvi_df['Herb_biomass'] * 10.
    # fit linear regression
    ndvi_array = merged_ndvi_df['NDVI'].values.reshape(-1, 1)
    biomass_array = merged_ndvi_df['herb_biomass_kgha'].values.reshape(-1, 1)
    model = LinearRegression().fit(ndvi_array, biomass_array)
    Rsq = model.score(ndvi_array, biomass_array)
    print("R squared: {}".format(Rsq))
    print("intercept: {}".format(model.intercept_))
    print("coefficient: {}".format(model.coef_))


if __name__ == "__main__":
    main()
