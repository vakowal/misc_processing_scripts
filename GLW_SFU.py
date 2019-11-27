"""Calculate sheep forage units from animals in GLW3."""
import os
import tempfile
import shutil

import numpy
from osgeo import gdal

import pygeoprocessing

_DATA_DIR = "E:/GIS_local_archive/General_useful_data/GLW3_Livestock_of_the_World"
_AOI_SHP = "E:/GIS_local/Mongolia/boundaries_etc/soums_monitoring_area_dissolve.shp"
_TARGET_NODATA = -1


def get_animal_info_dict():
    """Retrieve ditionary of information about each livestock type.

    Returns:
        a dictionary whose keys indicate the livestock types for which I
        downloaded distribution data from Gridded Livestock of the World 3
        (GLW3), and values are dictionaries containing information about that
        animal type. Information includes sheep forage unit equivalents of each
        animal type, from Gao et al. 2015, "Is Overgrazing A Pervasive Problem
        Across Mongolia? An Examination of Livestock Forage Demand and Forage
        Availability from 2000 to 2014". Information also includes the
        abbreviation string identifying the animal type in the GLW3 datasets.

    """
    animal_info_dict = {
        'sheep': {
            'SFU_equivalent': 1,
            'abbreviation': 'Sh',
            },
        'goats': {
            'SFU_equivalent': 0.9,
            'abbreviation': 'Gt',
            },
        'cattle': {
            'SFU_equivalent': 6,
            'abbreviation': 'Ct',
            },
        'horses': {
            'SFU_equivalent': 7,
            'abbreviation': 'Ho',
        },
    }
    return animal_info_dict


def extract_by_mask(base_raster_path, target_raster_path):
    """Extract pixels from a raster falling within features in _AOI_SHP.

    Parameters:
        base_raster_path (string): path to raster whose pixels should be
            extracted
        target_raster_path (string): location of the extracted raster

    Returns:
        None

    """
    target_pixel_size = pygeoprocessing.get_raster_info(
        base_raster_path)['pixel_size']
    src_nodata = pygeoprocessing.get_raster_info(base_raster_path)['nodata'][0]
    pygeoprocessing.warp_raster(
        base_raster_path, target_pixel_size, target_raster_path, 'near',
        vector_mask_options={'mask_vector_path': _AOI_SHP},
        gdal_warp_options=['dstNodata=-1.0'])


def convert_to_SFU(raster_path, SFU_conversion_rate, target_raster_path):
    """Calculate sheep forage unit equivalents from a raster of animal numbers.

    Parameters:
        raster_path (string): path to raster containing the number of one type
            of livestock
        SFU_conversion_rate (float): number of sheep forage equivalents
            corresponding to one of the livestock
        target_raster_path (string): location of the SFU raster

    Returns:
        None

    """
    def multiply_op(livestock, conversion_rate):
        valid_mask = (~numpy.isclose(livestock, livestock_nodata))
        sfu = numpy.empty(livestock.shape, dtype=numpy.float64)
        sfu[:] = livestock_nodata
        sfu[valid_mask] = livestock[valid_mask] * conversion_rate
        return sfu
    livestock_nodata = pygeoprocessing.get_raster_info(
        raster_path)['nodata'][0]
    pygeoprocessing.raster_calculator(
        [(raster_path, 1), (SFU_conversion_rate, 'raw')],
        multiply_op, target_raster_path, gdal.GDT_Float64, livestock_nodata)


def raster_list_sum(raster_list, input_nodata, target_path, target_nodata):
    """Calculate the sum per pixel across rasters in a list.

    Sum the rasters in `raster_list` element-wise, treating nodata as zero.
    Areas where all inputs are nodata will be nodata in the output.

    Parameters:
        raster_list (list): list of paths to rasters to sum
        input_nodata (float or int): nodata value in the input rasters
        target_path (string): path to location to store the result
        target_nodata (float or int): nodata value for the result raster

    Side effects:
        modifies or creates the raster indicated by `target_path`

    Returns:
        None

    """
    def raster_sum_op_nodata_remove(*raster_list):
        """Add the rasters in raster_list, treating nodata as zero."""
        invalid_mask = numpy.all(
            numpy.isclose(numpy.array(raster_list), input_nodata), axis=0)
        for r in raster_list:
            numpy.place(r, numpy.isclose(r, input_nodata), [0])
        sum_of_rasters = numpy.sum(raster_list, axis=0)
        sum_of_rasters[invalid_mask] = target_nodata
        return sum_of_rasters

    pygeoprocessing.raster_calculator(
        [(path, 1) for path in raster_list], raster_sum_op_nodata_remove,
        target_path, gdal.GDT_Float64, target_nodata)


def main(result_dir):
    """Calculate sheep forage unit equivalents in the study area.

    For area-weighted and dasymetric data products, for each animal type of
    GLW3 that I downloaded: extract the GLW3 data by our study aoi, convert
    # animals to sheep forage unit equivalents according to values from Gao et
    al. 2015.  Sum sheep forage unit equivalents across animal types.

    Parameters:
        result_dir (string): path to directory where results should be stored.
            If it does not exist, it will be created.

    Side effects:
        Creates the following two rasters in the `result_dir` directory:
            sfu_aoi_Da.tif, sheep forage unit equivalents in the study area
                in animals per pixel according to the dasymetric weighting
                method (see GLW3 documentation)
            sfu_aoi_Aw.tif, sheep forage unit equivalents in the study area
                in animals per pixel according to the area-weighted method
                (see GLW3 documentation)

    Returns:
        temp_dir, path to temporary processing directory where intermediate
            results are stored

    """
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    animal_info_dict = get_animal_info_dict()
    temp_dir = tempfile.mkdtemp()
    temp_val_dict = {}
    for animal_type in animal_info_dict.keys():
        temp_val_dict['base_{}'.format(animal_type)] = ''
        temp_val_dict['extracted_{}'.format(animal_type)] = os.path.join(
            temp_dir, 'extracted_{}.tif'.format(animal_type))
        temp_val_dict['sfu_equivalent_{}'.format(animal_type)] = os.path.join(
            temp_dir, 'sfu_{}.tif'.format(animal_type))

    # for dasymetric and area-weighted methods (see GLW3 documentation)
    for resampling_method in ['Da', 'Aw']:
        if resampling_method == 'Da':
            file_index = 5
        else:
            file_index = 6
        raw_input_list = []
        for animal_type in animal_info_dict.keys():
            base_raster_path = os.path.join(
                _DATA_DIR, animal_type, '{}_{}_2010_{}.tif'.format(
                    file_index, animal_info_dict[animal_type]['abbreviation'],
                    resampling_method))
            temp_val_dict['base_{}'.format(animal_type)] = base_raster_path
            raw_input_list.append(base_raster_path)

        # make sure all rasters have same nodata value
        livestock_nodata_set = set([])
        for raster in raw_input_list:
            livestock_nodata_set.update(
                set([pygeoprocessing.get_raster_info(raster)['nodata'][0]]))
        if len(livestock_nodata_set) > 1:
            raise ValueError("Livestock rasters include >1 nodata value")
        livestock_nodata = list(livestock_nodata_set)[0]

        # extract by study area aoi and convert to SFU equivalents
        SFU_equivalent_path_list = []
        for animal_type in animal_info_dict.keys():
            extract_by_mask(
                temp_val_dict['base_{}'.format(animal_type)],
                temp_val_dict['extracted_{}'.format(animal_type)])
            convert_to_SFU(
                temp_val_dict['extracted_{}'.format(animal_type)],
                animal_info_dict[animal_type]['SFU_equivalent'],
                temp_val_dict['sfu_equivalent_{}'.format(animal_type)])
            SFU_equivalent_path_list.append(
                temp_val_dict['sfu_equivalent_{}'.format(animal_type)])

        # sum SFU equivalents across animal types
        target_path = os.path.join(
            result_dir, 'SFU_equivalent_sum_{}.tif'.format(resampling_method))
        raster_list_sum(
            SFU_equivalent_path_list, livestock_nodata,
            target_path, livestock_nodata)
    return temp_dir


def raster_list_mean(raster_list, input_nodata, target_path, target_nodata):
    """Calculate the mean value across rasters in a list.

    Sum the rasters in `raster_list` element-wise, treating nodata as zero, and
    divide the sum by the number of rasters in raster list.  Areas where all
    input rasters are nodata will be nodata in the output.

    Parameters:
        raster_list (list): list of paths to rasters to calculate mean from
        input_nodata (float or int): nodata value in the input rasters
        target_path (string): path to location to store the result
        target_nodata (float or int): nodata value for the result raster

    Side effects:
        modifies or creates the raster indicated by `target_path`

    Returns:
        None

    """
    def raster_mean_op_nodata_remove(*raster_list):
        """Calculate mean of values in list, treating nodata as zero."""
        invalid_mask = numpy.all(
            numpy.isclose(numpy.array(raster_list), input_nodata), axis=0)
        for r in raster_list:
            numpy.place(r, numpy.isclose(r, input_nodata), [0])
        sum_of_rasters = numpy.sum(raster_list, axis=0)
        mean = sum_of_rasters / float(len(raster_list))
        mean[invalid_mask] = target_nodata
        return mean

    pygeoprocessing.raster_calculator(
        [(path, 1) for path in raster_list], raster_mean_op_nodata_remove,
        target_path, gdal.GDT_Float64, target_nodata)


def calculate_mean_density_RPM():
    """Calculate mean animal density across months of RPM simulation."""
    rpm_outputs_dir = "E:/GIS_local/Mongolia/RPM_outputs_empirical_sd_7.30.19"
    density_dir = os.path.join(rpm_outputs_dir, 'animal_density')
    animal_density_path_list = [
        os.path.join(density_dir, bn) for bn in os.listdir(density_dir)
        if bn.endswith('.tif')]
    input_nodata = pygeoprocessing.get_raster_info(
        animal_density_path_list[0])['nodata'][0]
    target_path = os.path.join(rpm_outputs_dir, 'mean_animal_density.tif')
    raster_list_mean(
        animal_density_path_list, input_nodata, target_path, input_nodata)


if __name__ == '__main__':
    # result_dir = os.path.join(_DATA_DIR, 'SFU_equivalent')
    # temp_dir = main(result_dir)
    # import pdb; pdb.set_trace()

    # # clean up
    # shutil.rmtree(temp_dir)

    calculate_mean_density_RPM()
