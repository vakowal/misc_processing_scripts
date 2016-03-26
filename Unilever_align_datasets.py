import os
import pygeoprocessing as pgp

def createTempFilePaths(workspace, inputRasterPaths):
    intermediateRasterPathsList = []

    # # Create required folders
    intermediate_folder_uri = os.path.join(workspace, 'intermediate')
    if not os.path.exists(intermediate_folder_uri):
        os.makedirs(intermediate_folder_uri)

    output_folder_uri = os.path.join(workspace, 'output')
    if not os.path.exists(output_folder_uri):
        os.makedirs(output_folder_uri)

    for inputRasterPath in inputRasterPaths:
        filepath, filename = os.path.split(inputRasterPath)
        inputRasterFile = os.path.join(workspace, 'intermediate', filename)
        intermediateRasterPathsList.append(inputRasterFile)

    return intermediateRasterPathsList

if __name__ == '__main__':
    workspace = r'C:\Users\Ginger\Dropbox\2015 Use Case Inputs\MT'
    
    # watershed extent 2012 dataset that we want to clip and resample
    orig_2012_ws = r"C:\Users\Ginger\Dropbox\2015 Use Case Inputs\MT\MT12_ws.tif"
    
    # state extent 2007 dataset that we want to align to
    state_2007 = r"C:\Users\Ginger\Dropbox\2015 Use Case Inputs\MT\MT_07_GKclip.tif"
    
    cell_size = 500
    #Create paths to store aligned rasters for all inputs, then call pygeoprocessing align
    rasterPathList = [orig_2012_ws, state_2007]
    rslist = ['nearest', 'nearest']

    alignedTempRasterPathsList = createTempFilePaths(workspace, rasterPathList)

    pgp.align_dataset_list(rasterPathList, alignedTempRasterPathsList, rslist,
                           cell_size, "dataset", 1, dataset_to_bound_index=1)
