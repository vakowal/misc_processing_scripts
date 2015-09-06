import os
import csv
import re
import sys
import invest_natcap.routing.routedem

raster_folder = r'C:\Users\Ginger\Documents\NatCap\GIS_local\Unilever\DEM'

data_files = [f for f in os.listdir(raster_folder) if os.path.isfile(os.path.join(raster_folder, f))]
dem_files = [f for f in data_files if re.search('SRTM', f)]
dem_tif = [f for f in dem_files if re.search('.tif$', f)]

outer_dir = r'C:\Users\Ginger\Documents\NatCap\GIS_local\Unilever\RouteDEM'
for dem in dem_tif:
    args = {
            u'workspace_dir': unicode(os.path.join(outer_dir, dem[:-4])),
            u'dem_uri': unicode(dem),
            u'pit_filled_filename': unicode(dem + 'pit'),
            u'resolve_plateaus_filename': unicode(dem + 'plateau'),
            u'calculate_slope': 0,
            u'flow_direction_filename': unicode(dem + 'flowdir'),
            u'flow_accumulation_filename': unicode(dem + 'flowacc'),
            u'threshold_flow_accumulation': unicode(1000), ## ?????
    }        
    invest_natcap.routing.routedem.execute(args)
