import arcpy
import os

arcpy.CheckOutExtension("Spatial")

folder = r"C:\Users\Ginger\Downloads\connectivity\undefended_no_threshold\1in100"

arcpy.env.workspace = folder
rasters = arcpy.ListRasters()
example_raster = rasters[0]
spatial_ref = arcpy.Describe(example_raster).spatialReference
pixel_type = "32_BIT_FLOAT"
cellsize = arcpy.GetRasterProperties_management(
                                      example_raster, "CELLSIZEX").getOutput(0)
numberbands = arcpy.GetRasterProperties_management(
                                      example_raster, "BANDCOUNT").getOutput(0)

arcpy.MosaicToNewRaster_management(rasters, folder, "Mosaic.tif", spatial_ref,
                                   pixel_type, cellsize, numberbands, "LAST",
                                   "FIRST")