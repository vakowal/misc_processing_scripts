# list all fields in shapefiles from Boogie

import os
from osgeo import ogr
import pandas as pd

if __name__ == '__main__':
    shapefile_dir = r"C:\Users\Ginger\Documents\NatCap\GIS_local\Mongolia\From_Boogie\shapes"
    shape_list = [s for s in os.listdir(shapefile_dir) if s.endswith('.shp')]
    result_dict = {'shapefile': [], 'field': []}
    for s in shape_list:
        shapefile = os.path.join(shapefile_dir, s)
        source = ogr.Open(shapefile)
        layer = source.GetLayer()
        fields = []
        ldefn = layer.GetLayerDefn()
        for n in range(ldefn.GetFieldCount()):
            fdefn = ldefn.GetFieldDefn(n)
            fields.append(fdefn.name)
        # fields = arcpy.ListFields(shapefile)
        result_dict['shapefile'].extend([s] * len(fields))
        result_dict['field'].extend(fields)
    save_as = r"C:/Users/Ginger/Desktop/shapefile_field_summary.csv"
    df = pd.DataFrame(result_dict)
    df.to_csv(save_as)