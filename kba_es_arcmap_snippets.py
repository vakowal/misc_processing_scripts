"""Python snippets for cv habitat attribution to be run in ArcMap."""

import os
import arcpy

from arcpy import sa

arcpy.CheckOutExtension("Spatial")


def convert_m_to_deg_equator(distance_meters):
    """Convert meters to degrees at the equator."""
    distance_dict = {
        500: 0.004476516196036,
        1000: 0.008953032392071,
        2000: 0.017906064784142,
    }
    return distance_dict[distance_meters]


habitat_dict = {
        'mangrove': {
            'risk': 1,
            'effect_distance': 1000,
            'habitat_shp': "C:/Users/ginge/Documents/NatCap/GIS_local/KBA_ES/cv_workspace/mangrove_proj.shp",
        },
        'saltmarsh': {
            'risk': 2,
            'effect_distance': 1000,
            'habitat_shp': "C:/Users/ginge/Documents/NatCap/GIS_local/KBA_ES/cv_workspace/saltmarsh_proj.shp",
        },
        'coralreef': {
            'risk': 1,
            'effect_distance': 2000,
            'habitat_shp': "C:/Users/ginge/Documents/NatCap/GIS_local/KBA_ES/cv_workspace/coral_reef_proj.shp",
        },
        'seagrass': {
            'risk': 4,
            'effect_distance': 500,
            'habitat_shp': "C:/Users/ginge/Documents/NatCap/GIS_local/KBA_ES/cv_workspace/seagrass_proj.shp",
        },
    }


def point_statistics(cv_results_shp, scenario, hab_distance):
    service_field = 'Service_{}'.format(scenario)
    effect_distance = convert_m_to_deg_equator(hab_distance)
    target_pixel_size = effect_distance / 2
    point_stat_raster_path = os.path.join(
        workspace_dir,
        'pt_st_{}_{}.tif'.format(scenario, hab_distance))
    if not os.path.exists(point_stat_raster_path):
        print "generating point statistics:"
        print point_stat_raster_path
        neighborhood = arcpy.sa.NbrCircle(effect_distance, "MAP")
        point_statistics_raster = arcpy.sa.PointStatistics(
            cv_results_shp, field=service_field,
            cell_size=target_pixel_size, neighborhood=neighborhood,
            statistics_type="MEAN")
        point_statistics_raster.save(point_stat_raster_path)


workspace_dir = "F:/cv_workspace"
cv_pattern = "C:/Users/ginge/Documents/ArcGIS/Default.gdb/main_Project_pop_gt_0"
for scenario in ['cur', 'ssp1', 'ssp3', 'ssp5']:
    cv_results_shp = '{}_{}'.format(cv_pattern, scenario)
    for hab_distance in [2000, 1000, 500]:  # unique effect distance vals
        point_statistics(cv_results_shp, scenario, hab_distance)

