"""Summarize watershed characteristics for Moore Amazon Hydro project."""
import os
import shutil
import tempfile
import collections
import time
import logging

import pandas
import numpy

from osgeo import gdal
from osgeo import ogr

import pygeoprocessing

LOGGER = logging.getLogger(__name__)
_LOGGING_PERIOD = 5.0  # min 5.0 seconds per update log message for the module

# watersheds for long-listed dams
_WATERSHEDS_PATH = "C:/Users/ginge/Desktop/watershed_subsets/watersheds_ESPG3857_fixgeom.gpkg"

# base data to summarize
_BASE_DATA_DICT = {
    'elevation': "G:/Shared drives/Moore Amazon Hydro/1_base_data/Raster_datasa_dem_30s_proj.tif",
    'slope': "G:/Shared drives/Moore Amazon Hydro/1_base_data/Raster_data/sa_30s_slope_proj.tif",
    'landcover': "G:/Shared drives/Moore Amazon Hydro/1_base_data/Raster_data/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2015-v2-0-7_clip_proj.tif",
}


def zonal_histogram(
        base_raster_path_band, aggregate_vector_path,
        aggregate_layer_name=None, ignore_nodata=True,
        polygons_might_overlap=True, working_dir=None):
    """Collect stats on pixel values which lie within polygons.

    This function counts the number of pixels of each unique value in a raster
    over the regions on the raster that are overlapped by the polygons in the
    vector layer. Overlapping polygons are correctly handled.

    Parameters:
        base_raster_path_band (tuple): a str/int tuple indicating the path to
            the base raster and the band index of that raster to analyze.
        aggregate_vector_path (string): a path to an ogr compatable polygon
            vector whose geometric features indicate the areas over
            ``base_raster_path_band`` to calculate statistics over.
        aggregate_layer_name (string): name of shapefile layer that will be
            used to aggregate results over.  If set to None, the first layer
            in the DataSource will be used as retrieved by ``.GetLayer()``.
            Note: it is normal and expected to set this field at None if the
            aggregating shapefile is a single layer as many shapefiles,
            including the common 'ESRI Shapefile', are.
        ignore_nodata: if true, then nodata pixels are not accounted for when
            calculating min, max, count, or mean.  However, the value of
            ``nodata_count`` will always be the number of nodata pixels
            aggregated under the polygon.
        polygons_might_overlap (boolean): if True the function calculates
            aggregation coverage close to optimally by rasterizing sets of
            polygons that don't overlap.  However, this step can be
            computationally expensive for cases where there are many polygons.
            Setting this flag to False directs the function rasterize in one
            step.
        working_dir (string): If not None, indicates where temporary files
            should be created during this run.

    Returns:
        nested dictionary indexed by aggregating feature id, and then by unique
        raster value.

    Raises:
        ValueError if ``base_raster_path_band`` is incorrectly formatted.
        RuntimeError(s) if the aggregate vector or layer cannot open.

    """
    if not pygeoprocessing._is_raster_path_band_formatted(
            base_raster_path_band):
        raise ValueError(
            "`base_raster_path_band` not formatted as expected.  Expects "
            "(path, band_index), received %s" % repr(base_raster_path_band))
    aggregate_vector = gdal.OpenEx(aggregate_vector_path, gdal.OF_VECTOR)
    if aggregate_vector is None:
        raise RuntimeError(
            "Could not open aggregate vector at %s" % aggregate_vector_path)
    LOGGER.debug(aggregate_vector)
    if aggregate_layer_name is not None:
        aggregate_layer = aggregate_vector.GetLayerByName(
            aggregate_layer_name)
    else:
        aggregate_layer = aggregate_vector.GetLayer()
    if aggregate_layer is None:
        raise RuntimeError(
            "Could not open layer %s on %s" % (
                aggregate_layer_name, aggregate_vector_path))

    # create a new aggregate ID field to map base vector aggregate fields to
    # local ones that are guaranteed to be integers.
    local_aggregate_field_name = 'original_fid'
    rasterize_layer_args = {
        'options': [
            'ALL_TOUCHED=FALSE',
            'ATTRIBUTE=%s' % local_aggregate_field_name]
        }

    # clip base raster to aggregating vector intersection
    raster_info = pygeoprocessing.get_raster_info(base_raster_path_band[0])
    # -1 here because bands are 1 indexed
    raster_nodata = raster_info['nodata'][base_raster_path_band[1]-1]
    with tempfile.NamedTemporaryFile(
            prefix='clipped_raster', suffix='.tif', delete=False,
            dir=working_dir) as clipped_raster_file:
        clipped_raster_path = clipped_raster_file.name
    try:
        pygeoprocessing.align_and_resize_raster_stack(
            [base_raster_path_band[0]], [clipped_raster_path], ['near'],
            raster_info['pixel_size'], 'intersection',
            base_vector_path_list=[aggregate_vector_path],
            raster_align_index=0)
        clipped_raster = gdal.OpenEx(clipped_raster_path, gdal.OF_RASTER)
        clipped_band = clipped_raster.GetRasterBand(base_raster_path_band[1])
    except ValueError as e:
        if 'Bounding boxes do not intersect' in repr(e):
            LOGGER.error(
                "aggregate vector %s does not intersect with the raster %s",
                aggregate_vector_path, base_raster_path_band)
            aggregate_stats = collections.defaultdict(
                lambda: {
                    'min': None, 'max': None, 'count': 0, 'nodata_count': 0,
                    'sum': 0.0})
            for feature in aggregate_layer:
                _ = aggregate_stats[feature.GetFID()]
            return dict(aggregate_stats)
        else:
            # this would be very unexpected to get here, but if it happened
            # and we didn't raise an exception, execution could get weird.
            raise

    # get unique values in raster
    raster_values_set = set()
    for offset_map, raster_block in pygeoprocessing.iterblocks(
            base_raster_path_band):
        raster_values_set.update(numpy.unique(raster_block))

    # make a shapefile that non-overlapping layers can be added to
    driver = ogr.GetDriverByName('MEMORY')
    disjoint_vector = driver.CreateDataSource('disjoint_vector')
    spat_ref = aggregate_layer.GetSpatialRef()

    # Initialize these dictionaries to have the shapefile fields in the
    # original datasource even if we don't pick up a value later
    LOGGER.info("build a lookup of aggregate field value to FID")

    aggregate_layer_fid_set = set(
        [agg_feat.GetFID() for agg_feat in aggregate_layer])

    # Loop over each polygon and aggregate
    if polygons_might_overlap:
        LOGGER.info("creating disjoint polygon set")
        disjoint_fid_sets = pygeoprocessing.calculate_disjoint_polygon_set(
            aggregate_vector_path, bounding_box=raster_info['bounding_box'])
    else:
        disjoint_fid_sets = [aggregate_layer_fid_set]

    with tempfile.NamedTemporaryFile(
            prefix='aggregate_fid_raster', suffix='.tif',
            delete=False, dir=working_dir) as agg_fid_raster_file:
        agg_fid_raster_path = agg_fid_raster_file.name

    agg_fid_nodata = -1
    pygeoprocessing.new_raster_from_base(
        clipped_raster_path, agg_fid_raster_path, gdal.GDT_Int32,
        [agg_fid_nodata])
    # fetch the block offsets before the raster is opened for writing
    agg_fid_offset_list = list(
        pygeoprocessing.iterblocks((agg_fid_raster_path, 1), offset_only=True))
    agg_fid_raster = gdal.OpenEx(
        agg_fid_raster_path, gdal.GA_Update | gdal.OF_RASTER)
    inner_dict = {val: 0 for val in raster_values_set}
    aggregate_stats = {
        fid: inner_dict.copy() for fid in aggregate_layer_fid_set}
    last_time = time.time()
    LOGGER.info("processing %d disjoint polygon sets", len(disjoint_fid_sets))
    for set_index, disjoint_fid_set in enumerate(disjoint_fid_sets):
        last_time = pygeoprocessing._invoke_timed_callback(
            last_time, lambda: LOGGER.info(
                "zonal stats approximately %.1f%% complete on %s",
                100.0 * float(set_index+1) / len(disjoint_fid_sets),
                os.path.basename(aggregate_vector_path)),
            _LOGGING_PERIOD)
        disjoint_layer = disjoint_vector.CreateLayer(
            'disjoint_vector', spat_ref, ogr.wkbPolygon)
        disjoint_layer.CreateField(
            ogr.FieldDefn(local_aggregate_field_name, ogr.OFTInteger))
        disjoint_layer_defn = disjoint_layer.GetLayerDefn()
        # add polygons to subset_layer
        disjoint_layer.StartTransaction()
        for index, feature_fid in enumerate(disjoint_fid_set):
            last_time = pygeoprocessing._invoke_timed_callback(
                last_time, lambda: LOGGER.info(
                    "polygon set %d of %d approximately %.1f%% processed "
                    "on %s", set_index+1, len(disjoint_fid_sets),
                    100.0 * float(index+1) / len(disjoint_fid_set),
                    os.path.basename(aggregate_vector_path)),
                _LOGGING_PERIOD)
            agg_feat = aggregate_layer.GetFeature(feature_fid)
            disjoint_feat = ogr.Feature(disjoint_layer_defn)
            disjoint_feat.SetGeometry(agg_feat.GetGeometryRef().Clone())
            disjoint_feat.SetField(
                local_aggregate_field_name, feature_fid)
            disjoint_layer.CreateFeature(disjoint_feat)
        disjoint_layer.CommitTransaction()

        LOGGER.info(
            "disjoint polygon set %d of %d 100.0%% processed on %s",
            set_index+1, len(disjoint_fid_sets), os.path.basename(
                aggregate_vector_path))

        # nodata out the mask
        agg_fid_band = agg_fid_raster.GetRasterBand(1)
        agg_fid_band.Fill(agg_fid_nodata)
        LOGGER.info(
            "rasterizing disjoint polygon set %d of %d %s", set_index+1,
            len(disjoint_fid_sets),
            os.path.basename(aggregate_vector_path))
        rasterize_callback = pygeoprocessing._make_logger_callback(
            "rasterizing polygon " + str(set_index+1) + " of " +
            str(len(disjoint_fid_set)) + " set %.1f%% complete")
        gdal.RasterizeLayer(
            agg_fid_raster, [1], disjoint_layer,
            callback=rasterize_callback, **rasterize_layer_args)
        agg_fid_raster.FlushCache()

        # Delete the features we just added to the subset_layer
        disjoint_layer = None
        disjoint_vector.DeleteLayer(0)

        # create a key array
        # and parallel min, max, count, and nodata count arrays
        LOGGER.info(
            "summarizing rasterized disjoint polygon set %d of %d %s",
            set_index+1, len(disjoint_fid_sets),
            os.path.basename(aggregate_vector_path))
        for agg_fid_offset in agg_fid_offset_list:
            agg_fid_block = agg_fid_band.ReadAsArray(**agg_fid_offset)
            clipped_block = clipped_band.ReadAsArray(**agg_fid_offset)
            valid_mask = (agg_fid_block != agg_fid_nodata)
            valid_agg_fids = agg_fid_block[valid_mask]
            valid_clipped = clipped_block[valid_mask]
            for agg_fid in numpy.unique(valid_agg_fids):
                masked_clipped_block = valid_clipped[
                    valid_agg_fids == agg_fid]
                if raster_nodata is not None:
                    clipped_nodata_mask = numpy.isclose(
                        masked_clipped_block, raster_nodata)
                else:
                    clipped_nodata_mask = numpy.zeros(
                        masked_clipped_block.shape, dtype=numpy.bool)
                if ignore_nodata:
                    masked_clipped_block = (
                        masked_clipped_block[~clipped_nodata_mask])
                if masked_clipped_block.size == 0:
                    continue

                block_elements, element_counts = numpy.unique(
                    masked_clipped_block, return_counts=True)
                for value_pair in zip(block_elements, element_counts):
                    if aggregate_stats[agg_fid][value_pair[0]] is None:
                        aggregate_stats[agg_fid][value_pair[0]] = value_pair[1]
                    else:
                        aggregate_stats[agg_fid][value_pair[0]] += (
                            value_pair[1])
    unset_fids = aggregate_layer_fid_set.difference(aggregate_stats)
    LOGGER.debug(
        "unset_fids: %s of %s ", len(unset_fids),
        len(aggregate_layer_fid_set))
    clipped_gt = numpy.array(
        clipped_raster.GetGeoTransform(), dtype=numpy.float32)
    LOGGER.debug("gt %s for %s", clipped_gt, base_raster_path_band)
    for unset_fid in unset_fids:
        unset_feat = aggregate_layer.GetFeature(unset_fid)
        unset_geom_envelope = list(unset_feat.GetGeometryRef().GetEnvelope())
        if clipped_gt[1] < 0:
            unset_geom_envelope[0], unset_geom_envelope[1] = (
                unset_geom_envelope[1], unset_geom_envelope[0])
        if clipped_gt[5] < 0:
            unset_geom_envelope[2], unset_geom_envelope[3] = (
                unset_geom_envelope[3], unset_geom_envelope[2])

        xoff = int((unset_geom_envelope[0] - clipped_gt[0]) / clipped_gt[1])
        yoff = int((unset_geom_envelope[2] - clipped_gt[3]) / clipped_gt[5])
        win_xsize = int(numpy.ceil(
            (unset_geom_envelope[1] - clipped_gt[0]) /
            clipped_gt[1])) - xoff
        win_ysize = int(numpy.ceil(
            (unset_geom_envelope[3] - clipped_gt[3]) /
            clipped_gt[5])) - yoff

        # clamp offset to the side of the raster if it's negative
        if xoff < 0:
            win_xsize += xoff
            xoff = 0
        if yoff < 0:
            win_ysize += yoff
            yoff = 0

        # clamp the window to the side of the raster if too big
        if xoff+win_xsize > clipped_band.XSize:
            win_xsize = clipped_band.XSize-xoff
        if yoff+win_ysize > clipped_band.YSize:
            win_ysize = clipped_band.YSize-yoff

        if win_xsize <= 0 or win_ysize <= 0:
            continue

        # here we consider the pixels that intersect with the geometry's
        # bounding box as being the proxy for the intersection with the
        # polygon itself. This is not a bad approximation since the case
        # that caused the polygon to be skipped in the first phase is that it
        # is as small as a pixel. There could be some degenerate cases that
        # make this estimation very wrong, but we do not know of any that
        # would come from natural data. If you do encounter such a dataset
        # please email the description and datset to richsharp@stanford.edu.
        # unset_fid_block = clipped_band.ReadAsArray(
        #     xoff=xoff, yoff=yoff, win_xsize=win_xsize, win_ysize=win_ysize)

        # if raster_nodata is not None:
        #     unset_fid_nodata_mask = numpy.isclose(
        #         unset_fid_block, raster_nodata)
        # else:
        #     unset_fid_nodata_mask = numpy.zeros(
        #         unset_fid_block.shape, dtype=numpy.bool)

        # valid_unset_fid_block = unset_fid_block[~unset_fid_nodata_mask]
        # if valid_unset_fid_block.size == 0:
        #     aggregate_stats[unset_fid]['min'] = 0.0
        #     aggregate_stats[unset_fid]['max'] = 0.0
        #     aggregate_stats[unset_fid]['sum'] = 0.0
        # else:
        #     aggregate_stats[unset_fid]['min'] = numpy.min(
        #         valid_unset_fid_block)
        #     aggregate_stats[unset_fid]['max'] = numpy.max(
        #         valid_unset_fid_block)
        #     aggregate_stats[unset_fid]['sum'] = numpy.sum(
        #         valid_unset_fid_block)
        # aggregate_stats[unset_fid]['count'] = valid_unset_fid_block.size
        # aggregate_stats[unset_fid]['nodata_count'] = numpy.count_nonzero(
        #     unset_fid_nodata_mask)

    unset_fids = aggregate_layer_fid_set.difference(aggregate_stats)
    LOGGER.debug(
        "remaining unset_fids: %s of %s ", len(unset_fids),
        len(aggregate_layer_fid_set))
    # fill in the missing polygon fids in the aggregate stats by invoking the
    # accessor in the defaultdict
    for fid in unset_fids:
        _ = aggregate_stats[fid]

    LOGGER.info(
        "all done processing polygon sets for %s", os.path.basename(
            aggregate_vector_path))

    # clean up temporary files
    gdal.Dataset.__swig_destroy__(agg_fid_raster)
    gdal.Dataset.__swig_destroy__(aggregate_vector)
    gdal.Dataset.__swig_destroy__(clipped_raster)
    clipped_band = None
    clipped_raster = None
    agg_fid_raster = None
    disjoint_layer = None
    disjoint_vector = None
    aggregate_layer = None
    aggregate_vector = None
    for filename in [agg_fid_raster_path, clipped_raster_path]:
        os.remove(filename)

    return dict(aggregate_stats)


def summarize_terrain(out_dir):
    elevation_dict = pygeoprocessing.zonal_statistics(
        (_BASE_DATA_DICT['elevation'], 1), _WATERSHEDS_PATH, 'watersheds')
    elevation_df_t = pandas.DataFrame(elevation_dict)
    elevation_df = elevation_df_t.transpose()
    elevation_df['fid'] = elevation_df.index
    elevation_df.rename(
        columns={'min': 'min_elevation', 'max': 'max_elevation'}, inplace=True)
    elevation_df.drop(
        ['sum', 'count', 'nodata_count'], axis='columns', inplace=True)

    slope_dict = pygeoprocessing.zonal_statistics(
        (_BASE_DATA_DICT['slope'], 1), _WATERSHEDS_PATH, 'watersheds')
    slope_df_t = pandas.DataFrame(slope_dict)
    slope_df = slope_df_t.transpose()
    slope_df['fid'] = slope_df.index
    slope_df['mean_slope'] = slope_df['sum'] / slope_df['count']
    slope_df.drop(
        ['min', 'max', 'count', 'nodata_count', 'sum'], axis='columns',
        inplace=True)

    terrain_df = elevation_df.merge(
        slope_df, on='fid', suffixes=(False, False))
    terrain_path = os.path.join(out_dir, 'terrain_characteristics.csv')
    terrain_df.to_csv(terrain_path, index=False)


def summarize_landcover(out_dir):
    """Get count and percent of specific landcover categories in watersheds."""
    urban = 190
    crop_rainfed = 10
    crop_irrigated = 20
    crop_mosaic = 30
    landcover_dict = zonal_histogram(
        (_BASE_DATA_DICT['landcover'], 1), _WATERSHEDS_PATH, 'watersheds')
    landcover_df_t = pandas.DataFrame(landcover_dict)
    landcover_df = landcover_df_t.transpose()
    landcover_df['fid'] = landcover_df.index
    count_df = landcover_df[
        ['fid', urban, crop_rainfed, crop_irrigated, crop_mosaic]]
    count_df.rename(
        columns={
            urban: 'urban_count',
            crop_rainfed: 'crop_rainfed_count',
            crop_irrigated: 'crop_irrigated_count',
            crop_mosaic: 'crop_mosaic_count',
        }, inplace=True)
    count_path = os.path.join(out_dir, 'landcover_pixel_count.csv')
    count_df.to_csv(count_path, index=False)

    landcover_df.set_index('fid', inplace=True)
    percent_df = landcover_df.div(landcover_df.sum(axis=1), axis=0)
    percent_path = os.path.join(out_dir, 'landcover_percent.csv')
    percent_df.to_csv(percent_path, index=True)


def summarize_ssp_landcover(out_dir):
    """Get count of all landcover categories for future SSPs in Chaglla."""
    # landcover_path_dict = {
    #     'current': "F:/Moore_Amazon_backups/Chaplin-Kramer_etal_future_land_use/Clipped_to_Chaglla/Globio4_landuse_10sec_2015.tif",
    #     'SSP1': "F:/Moore_Amazon_backups/Chaplin-Kramer_etal_future_land_use/Clipped_to_Chaglla/GLOBIO4_LU_10sec_2050_SSP1_RCP26.tif",
    #     'SSP3': "F:/Moore_Amazon_backups/Chaplin-Kramer_etal_future_land_use/Clipped_to_Chaglla/GLOBIO4_LU_10sec_2050_SSP3_RCP70.tif",
    #     'SSP5': "F:/Moore_Amazon_backups/Chaplin-Kramer_etal_future_land_use/Clipped_to_Chaglla/GLOBIO4_LU_10sec_2050_SSP5_RCP85.tif",
    # }
    # shp_path = "G:/Shared drives/Moore Amazon Hydro/1_base_data/Vector_data/Chaglla_dam_watershed.shp"
    landcover_path_dict = {
        'current': "F:/Moore_Amazon_backups/Johnson_SEALS_future_land_use/lulc_esa_2015_reclassified_to_seals_simplified.tif",
        'SSP3_2050': "F:/Moore_Amazon_backups/Johnson_SEALS_future_land_use/lulc_RCP7.0_SSP3_2050.tif",
        'SSP3_2070': "F:/Moore_Amazon_backups/Johnson_SEALS_future_land_use/lulc_RCP7.0_SSP3_2070.tif",
        'SSP5_2050': "F:/Moore_Amazon_backups/Johnson_SEALS_future_land_use/lulc_RCP8.5_SSP5_2050.tif",
        'SSP5_2070': "F:/Moore_Amazon_backups/Johnson_SEALS_future_land_use/lulc_RCP8.5_SSP5_2070.tif",
        'SSP1_2050': "F:/Moore_Amazon_backups/Johnson_SEALS_future_land_use/lulc_RCP2.6_SSP1_2050.tif",
        'SSP1_2070': "F:/Moore_Amazon_backups/Johnson_SEALS_future_land_use/lulc_RCP2.6_SSP1_2070.tif",
    }
    shp_path = "G:/Shared drives/Moore Amazon Hydro/1_base_data/Vector_data/Chaglla_dam_watershed_WGS84.shp"
    df_list = []
    for ssp in landcover_path_dict:
        landcover_dict = zonal_histogram(
            (landcover_path_dict[ssp], 1), shp_path)
        landcover_df = pandas.DataFrame(landcover_dict)
        landcover_df['lulc_code'] = landcover_df.index
        landcover_df.rename(columns={0: '{}_count'.format(ssp)}, inplace=True)
        df_list.append(landcover_df)
    combined_df = df_list[0]
    df_i = 1
    while df_i < len(df_list):
        combined_df = combined_df.merge(
            df_list[df_i], how='outer', on='lulc_code')
        df_i = df_i + 1
    save_as = os.path.join(out_dir, 'landcover_count_summary.csv')
    combined_df.to_csv(save_as, index=False)


def process_hansen_tiles(out_dir):
    # merge and project selected tiles in 4 subsets covering study watersheds
    # for subset in ['s2', 's3']:
    subset = 's4'
    warped_path = "C:/Users/ginge/Desktop/hansen_{}.tif".format(subset)
    frequency_dict = zonal_histogram(
        (warped_path, 1), _WATERSHEDS_PATH, 'watersheds')
    frequency_df_t = pandas.DataFrame(frequency_dict)
    frequency_df = frequency_df_t.transpose()
    frequency_df['fid'] = frequency_df.index
    frequency_path = os.path.join(
        out_dir, 'Hansen_lossyear_frequency_{}.csv'.format(subset))
    frequency_df.to_csv(frequency_path, index=False)
    # merge subsetted frequency tables


def calc_disjoint_set():
    disjoint_set = pygeoprocessing.calculate_disjoint_polygon_set(
        _WATERSHEDS_PATH, 'watersheds')
    print(disjoint_set)


def summarize_climate_change():
    """Calculate mean percent change in monthly precipitation in the future."""
    def id_change_from_zero(current, future):
        problem_mask = ((current == 0) & (future > 0))
        result = numpy.empty(current.shape, dtype=numpy.float32)
        result[:] = current_nodata
        result[problem_mask] = 100
        return result

    def calc_percent_change(current, future):
        """Calculate percent change in future relative to current."""
        valid_mask = (
            (~numpy.isclose(current, current_nodata)) &
            (~numpy.isclose(future, future_nodata)))
        zero_mask = ((current == 0) & (future == 0))
        divide_mask = ((current > 0) & valid_mask)
        result = numpy.empty(current.shape, dtype=numpy.float32)
        result[:] = current_nodata
        result[zero_mask] = 0
        result[divide_mask] = (
            (future[divide_mask] - current[divide_mask]) / current[divide_mask]
            * 100)
        return result

    def raster_mean_op(*raster_list):
        """Calculate the mean value pixel-wise from rasters in raster_list."""
        valid_mask = numpy.any(
            ~numpy.isclose(numpy.array(raster_list), current_nodata), axis=0)
        # get number of valid observations per pixel
        num_observations = numpy.count_nonzero(
            ~numpy.isclose(numpy.array(raster_list), current_nodata), axis=0)
        for r in raster_list:
            numpy.place(r, numpy.isclose(r, current_nodata), [0])
        sum_of_rasters = numpy.sum(raster_list, axis=0)

        divide_mask = (
            (num_observations > 0) &
            valid_mask)

        mean_of_rasters = numpy.empty(
            sum_of_rasters.shape, dtype=numpy.float32)
        mean_of_rasters[:] = current_nodata
        mean_of_rasters[valid_mask] = 0
        mean_of_rasters[divide_mask] = numpy.divide(
            sum_of_rasters[divide_mask], num_observations[divide_mask])
        return mean_of_rasters

    climate_dir = "F:/Moore_Amazon_backups/climate"
    year = 2050  # for year in [2050, 2070]:

    # pattern that can be used to identify current rasters, 1 for each month
    current_pattern = os.path.join(
        climate_dir, 'historical_precip/wc2.1_5m_prec_{:02d}.tif')
    # pattern that can be used to identify future rasters, 1 for each month
    if year == 2050:
        future_pattern = os.path.join(
            climate_dir, "future_precip/MIROC-ESM-CHEM/RCP4.5",
            '{}'.format(year), "mi45pr50{}.tif")
    else:
        future_pattern = os.path.join(
            climate_dir, "future_precip/MIROC-ESM-CHEM/RCP4.5",
            '{}'.format(year), "mi45pr70{}.tif")

    intermediate_dir = os.path.join(climate_dir, "intermediate")
    if not os.path.exists(intermediate_dir):
        os.makedirs(intermediate_dir)
    output_dir = os.path.join(climate_dir, "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # align current and future rasters
    base_raster_path_list = (
        [current_pattern.format(i) for i in range(1, 13)] +
        [future_pattern.format(i) for i in range(1, 13)])
    target_raster_path_list = (
        [os.path.join(
            intermediate_dir,
            'current{}.tif'.format(i)) for i in range(1, 13)] +
        [os.path.join(
            intermediate_dir,
            'future{}.tif'.format(i)) for i in range(1, 13)])
    raster_info = pygeoprocessing.get_raster_info(base_raster_path_list[0])
    pygeoprocessing.align_and_resize_raster_stack(
        base_raster_path_list, target_raster_path_list, ['near'] *
        len(base_raster_path_list), raster_info['pixel_size'], 'union')

    # calculate percent change per pixel for each month
    current_nodata = pygeoprocessing.get_raster_info(
        os.path.join(intermediate_dir, 'current1.tif'))['nodata'][0]
    future_nodata = pygeoprocessing.get_raster_info(
        os.path.join(intermediate_dir, 'future1.tif'))['nodata'][0]
    for mon in range(1, 13):
        target_path = os.path.join(
            intermediate_dir, 'perc_change_{}.tif'.format(mon))
        pygeoprocessing.raster_calculator([
            (os.path.join(intermediate_dir, 'current{}.tif'.format(mon)), 1),
            (os.path.join(intermediate_dir, 'future{}.tif'.format(mon)), 1)],
            calc_percent_change, target_path, gdal.GDT_Float32,
            current_nodata)

    # identify problematic areas
    # for mon in range(1, 13):
    #     target_path = os.path.join(
    #         intermediate_dir, 'problematic_{}.tif'.format(mon))
    #     pygeoprocessing.raster_calculator([
    #         (os.path.join(intermediate_dir, 'current{}.tif'.format(mon)), 1),
    #         (os.path.join(intermediate_dir, 'future{}.tif'.format(mon)), 1)],
    #         id_change_from_zero, target_path, gdal.GDT_Float32,
    #         current_nodata)
    # import pdb; pdb.set_trace()
    # print("Intermission: check out problematic areas")

    # calculate mean percent change across months
    monthly_path_list = [
        os.path.join(intermediate_dir, 'perc_change_{}.tif'.format(m)) for m
        in range(1, 13)]
    mean_ch_path = os.path.join(intermediate_dir, 'mean_perc_change.tif')
    pygeoprocessing.raster_calculator(
        [(path, 1) for path in monthly_path_list], raster_mean_op,
        mean_ch_path, gdal.GDT_Float32, current_nodata)

    # extract mean percent change to approximate aoi
    target_pixel_size = pygeoprocessing.get_raster_info(
        mean_ch_path)['pixel_size']
    target_bb = [-78, -29, -43, 4]  # I cheated
    mean_perc_masked = os.path.join(intermediate_dir, 'mean_perc_ch_mask.tif')
    pygeoprocessing.warp_raster(
        mean_ch_path, target_pixel_size, mean_perc_masked, 'near',
        target_bb=target_bb)

    # reproject mean change raster to conform to watersheds
    # this is not working
    # target_sr = pygeoprocessing.get_vector_info(
    #     _WATERSHEDS_PATH, 'watersheds')['projection']
    # proj_pixel_size = (9587.1095591824851, -9587.1095591824851)  # I cheated
    # pygeoprocessing.warp_raster(
    #     mean_perc_masked, proj_pixel_size, mean_ch_proj, 'near',
    #     target_sr_wkt=target_sr)
    mean_ch_proj = os.path.join(intermediate_dir, 'mean_perc_change_pr.tif')
    import pdb; pdb.set_trace()
    print("intermission; create projected raster in QGIS")

    # calculate zonal mean percent change within watersheds
    change_dict = pygeoprocessing.zonal_statistics(
        (mean_ch_proj, 1), _WATERSHEDS_PATH, 'watersheds')
    change_df_t = pandas.DataFrame(change_dict)
    change_df = change_df_t.transpose()
    change_df['fid'] = change_df.index
    change_df['mean_percent_change'] = change_df['sum'] / change_df['count']
    change_df.drop(
        ['min', 'max', 'count', 'nodata_count', 'sum'], axis='columns',
        inplace=True)
    table_path = os.path.join(
        output_dir, 'mean_percent_change_precip_{}.csv'.format(year))
    change_df.to_csv(table_path, index=False)

    # clean up
    shutil.rmtree(intermediate_dir)


def process_WDPA_intersect_tables():
    """Combine tables of protected area features intersecting watersheds."""
    # derive full table of WDPA features: rbind these two tables
    wdpa_point = pandas.read_csv("C:/Users/ginge/Desktop/wdpa_export_intermediate/WDPA_point.csv")
    wdpa_poly = pandas.read_csv("C:/Users/ginge/Desktop/wdpa_export_intermediate/WDPA_polygon.csv")
    wdpa_df = pandas.concat([wdpa_point, wdpa_poly])
    # a few protected areas are composed of sub-parcels.
    # here we keep info for just one parcel of the protected area
    wdpa_df.drop_duplicates(subset=['WDPAID'], inplace=True)

    df_list = []
    subs_dir = "C:/Users/ginge/Desktop/watershed_subsets/fixed_geom"
    for sub in range(9):
        poly_df = pandas.read_csv(
            os.path.join(subs_dir, 'polyjoin_s{}.csv'.format(sub)))
        poly_df = poly_df[['fid', 'WDPAID']]
        df_list.append(poly_df)
        point_df = pandas.read_csv(
            os.path.join(subs_dir, 'pointsjoin_s{}.csv'.format(sub)))
        point_df = point_df[['fid', 'WDPAID']]
        df_list.append(point_df)
    combined_df = pandas.concat(df_list)
    combined_df = combined_df.merge(
        wdpa_df, on='WDPAID', suffixes=(False, False), validate="many_to_one")
    save_as = "G:/Shared drives/Moore Amazon Hydro/1_base_data/Other/watershed_characteristics/WDPA_intersect_watersheds.csv"
    combined_df.to_csv(save_as, index=False)


def count_protected_areas():
    wdpa_df = pandas.read_csv("G:/Shared drives/Moore Amazon Hydro/1_base_data/Other/watershed_characteristics/WDPA_intersect_watersheds.csv")
    wdpa_count = wdpa_df[['fid', 'WDPAID']].groupby('fid').count()
    wdpa_count.rename(
        columns={'WDPAID': 'num_intersecting_PA'}, inplace=True)
    wdpa_count.reset_index(inplace=True)
    wdpa_count.to_csv(
        "C:/Users/ginge/Desktop/WDPA_intersect_count.csv", index=False)


def process_WDPA_area_tables():
    df_list = []
    subs_dir = "C:/Users/ginge/Desktop/watershed_subsets/fixed_geom/intersection"
    for sub in range(9):
        df_list.append(
            pandas.read_csv(os.path.join(subs_dir, 's{}.csv'.format(sub))))
    combined_df = pandas.concat(df_list)
    area_sum = combined_df[['fid', 'area_ha']].groupby('fid').sum()
    area_sum.rename(
        columns={'area_ha': 'area_ha_intersecting_PA'}, inplace=True)
    area_sum.reset_index(inplace=True)
    area_sum.to_csv(
        "C:/Users/ginge/Desktop/WDPA_intersect_area.csv", index=False)



if __name__ == "__main__":
    __spec__ = None  # for running with pdb
    out_dir = "F:/Moore_Amazon_backups/Johnson_SEALS_future_land_use"
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    # summarize_terrain(out_dir)
    # summarize_landcover(out_dir)
    # process_hansen_tiles(out_dir)
    # calc_disjoint_set()
    # summarize_climate_change()
    # process_WDPA_intersect_tables()
    # count_protected_areas()
    # process_WDPA_area_tables()
    summarize_ssp_landcover(out_dir)