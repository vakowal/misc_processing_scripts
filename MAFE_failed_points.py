# This script is not meant to be run standalone, but instead as useful snippets
# to be run in the Python window of Arcmap

from arcpy import *

sshed = r'C:\Users\Ginger\Documents\NatCap\GIS_local\MAFE_Colombia\Failed_servicesheds_subs4\rogue_servicesheds_Colombia\sshed_t2.shp'
MakeTableView_management(sshed, 'sheds')
AddField_management('sheds', 'name_int', "SHORT")
with arcpy.da.SearchCursor(sshed, ["Name", "name_int"]) as cursor:
    for row in cursor:
        row
        

def unique_values(table, field):
    with arcpy.da.SearchCursor(table, [field]) as cursor:
        return sorted({row[0] for row in cursor})
        
def find_failed_points(points, servicesheds):
    """Find the points that did not create servicesheds, for one group of points
    and one group of resulting servicesheds."""
    
    Delete_management('orig_tab')
    MakeTableView_management(points, 'orig_tab')
    orig_names = unique_values('orig_tab', "CPOB_COD")
    MakeTableView_management(servicesheds, 'tab')
    succeeded_points = unique_values('tab', "Name")
    Delete_management('tab')
    failed_points = []
    for name in orig_names:
        if name not in succeeded_points:
            failed_points.append(name)
    return failed_points
        
orig_points = r'C:\Users\Ginger\Documents\NatCap\GIS_local\MAFE_Colombia\G_intermediate\Include.shp'
MakeTableView_management(orig_points, 'orig_tab')
orig_names = unique_values('orig_tab', "CPOB_COD")

subsets = [r'C:\Users\Ginger\Documents\NatCap\GIS_local\MAFE_Colombia\G_intermediate\servicesheds_subset1.shp',
    r'C:\Users\Ginger\Documents\NatCap\GIS_local\MAFE_Colombia\G_intermediate\servicesheds_subset2.shp',
    r'C:\Users\Ginger\Documents\NatCap\GIS_local\MAFE_Colombia\G_intermediate\servicesheds_subset3.shp']
subset4 = r'C:\Users\Ginger\Documents\NatCap\GIS_local\MAFE_Colombia\G_intermediate\Subset4.shp'

succeeded_points = []
for sh in subsets:
    MakeTableView_management(sh, 'tab')
    names = unique_values('tab', "Name")
    succeeded_points = succeeded_points + names
    Delete_management('tab')

MakeTableView_management(subset4, 'tab')
names = unique_values('tab', "CPOB_COD")
succeeded_points = succeeded_points + names
Delete_management('tab')

failed_points = []
for name in orig_names:
    if name not in succeeded_points:
        failed_points.append(name)

MakeFeatureLayer_management(orig_points, 'all_points')
SelectLayerByAttribute_management('all_points', "NEW_SELECTION", """ "CPOB_COD" IN ('SAN JOSE DE LA MONTA„A_54223', 'SAN JOSE DEL NUS (SAN JOSE DE NUESTRA SE„ORA)_05425', 'SAN JOSE DEL PE„ON (LAS PORQUERAS)_13657', 'SAN PEDRO DE MUCE„O_15425', 'SIERRA DE LA CA„ADA_41799', 'MONTA„A DEL TOTUMO_85250', 'LA MONTA„A DE ALONSO (MARTIN ALONSO)_13212', 'PUEBLO VIEJO SECTOR NI„O_25758', 'PUERTO NI„O (CHARANGA)_47161', 'SANTA ROSA LA CA„A_23419')""")
CopyFeatures_management('all_points', r'C:\Users\Ginger\Documents\NatCap\GIS_local\MAFE_Colombia\G_intermediate\redo.shp')
