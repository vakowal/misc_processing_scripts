# Author:  Brad Eichelberger, Natural Capital Project
# Last Updated: August 6, 2015
# Python Version: 2.7
# Libraries needed: gdal, numpy, pygeoprocessing, pygeometa
# Download zip files for 30 arcesecond erodibility from ISRIC ftp site and extract them to a working directory.
# The script will download the organic content and percent clay, silt, and sand content for the upper 10 cm.
# Soil texture for each soil unit was defined using the USDA NRCS soil texture classes (http://www.nrcs.usda.gov/wps/portal/nrcs/detail//?cid=nrcs142p2_054167) and erodibility values
# assigned based on the work of Wischmeier, Johnson and Cross (reported in Roose, 1996) using orgnic matter content and soil texture.  The OMAFRA fact sheet summarize these values in
# the following table (http://www.omafra.gov.on.ca/english/engineer/facts/12-051.pdf).
# This script requires two arguments.  The first argument is the location to store the output (i.e. "C:\erodibility_globe_ISRIC_30arcseconds\".  The second argument is the pathway to the
# mcf used to generate the metadata file and should be called "erodibility_globe_ISRICSoilGrids1km_30arcseconds.mcf".
#--------------------------------

try:

    try:
        from ftplib import FTP
        import os, sys, glob, fnmatch, shutil, gdal, pygeoprocessing, numpy, gzip
        from pygeometa import render_template

    except:
        print "Error in loading Python librries.  This script requires the gdal, numpy, pygeoprocessing, and pygeometa library."
        raise
    
    try:
        if len(sys.argv) == 3:
            # Location to store the data
            output_folder = sys.argv[1]
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)
            os.makedirs(output_folder)

            mcf_file = sys.argv[2]

        else:
            print """Error in reading arguments.  This script requires one argument - the location to store the output (i.e. "C:\\erodibility_globe_ISRIC_30arcseconds\".
            The second argument is the pathway to the mcf used to generate the metadata file and should be called "erodibility_globe_ISRICSoilGrids1km_30arcseconds.mcf"." """
            raise

    except:
        raise
        
    try:
        
        # Log onto ISRIC's ftp site via annoymous user
        ftp = FTP('ftp.soilgrids.org','soilgrids', 'soilgrids')
        parent_dir = ftp.pwd()

        # Access the sub-directory where the files are stored
        folder = '/data/recent/' 

    except:
        print "Error accessing the ftp site or connection error.  Please check the address to ensure the site is correct."
        raise
        
    try:
        # Open ftp site and download all files    
        ftp.cwd('{}/{}'.format(parent_dir, folder))
        filenames = ftp.nlst()
        for search in ['CLYPPT_sd2_M*.gz', 'ORCDRC_sd2_M*.gz', 'SLTPPT_sd2_M*.gz', 'SNDPPT_sd2_M*.gz']:  
            for filename in fnmatch.filter(filenames, search):
                local_filename = os.path.join(output_folder, filename)
                raster_file = open(local_filename, 'wb')
                ftp.retrbinary('RETR ' + filename, raster_file.write)
            raster_file.close()

        ftp.quit() 

    except:
        print "Error downloading the ftp files to the output folder."
        raster_file.close()
        raise
        
    try:
        # Extract the data from the zip files but skip sub-directories if present and extract the data
        gz_files = glob.glob(os.path.join(output_folder, '*.gz'))

        raster_uri = []
        for gz_filename in gz_files:
            outF = open(gz_filename[0:-3], 'wb')
            with gzip.open(gz_filename, 'rb') as inF:
                shutil.copyfileobj(inF, outF)
            outF.close()
            inF.close()
            os.remove(gz_filename)

    except:
        print "Error extracting tar files."
        raise


    try:
        for i in os.listdir(output_folder):
            if i.endswith(".tif"):
                raster_uri.append(os.path.join(output_folder, i))
        dataset_out_uri = os.path.join(output_folder, "erodibility_globe_ISRIC_30arcseconds.tif")
        nodata_out = -9999
        pixel_size_out = pygeoprocessing.get_cell_size_from_uri(raster_uri[0])
        unit_clay_nodata = pygeoprocessing.get_nodata_from_uri(raster_uri[0])
        unit_organic_nodata = pygeoprocessing.get_nodata_from_uri(raster_uri[1])
        unit_silt_nodata = pygeoprocessing.get_nodata_from_uri(raster_uri[2])
        unit_sand_nodata = pygeoprocessing.get_nodata_from_uri(raster_uri[3])

        
        def erod_func(unit_clay, unit_organic, unit_silt, unit_sand):
            nodata_mask = (unit_clay == unit_clay_nodata) | (unit_silt == unit_silt_nodata) | (unit_sand == unit_sand_nodata) | (unit_sand == unit_organic_nodata)
            unit_erod = numpy.empty(unit_clay.shape)
            unit_erod[:] = nodata_mask
            unit_organic2 = unit_organic / 10.0
                
            sand_mask_avg = (((unit_silt + 1.5*unit_clay) < 15) & (unit_organic2 == 2.0))
            unit_erod = numpy.where(sand_mask_avg, 0.02 * 0.1317, unit_erod)
            sand_mask_avg = None
            sand_mask_bel = (((unit_silt + 1.5*unit_clay) < 15) & (unit_organic2 < 2.0))
            unit_erod = numpy.where(sand_mask_bel, 0.01 * 0.1317, unit_erod)
            sand_mask_bel = None
            sand_mask_abo = (((unit_silt + 1.5*unit_clay) < 15) & (unit_organic2 > 2.0))
            unit_erod = numpy.where(sand_mask_abo, 0.03 * 0.1317, unit_erod)
            sand_mask_abo = None
            loamysand_mask = ((((unit_silt + 1.5*unit_clay) >= 15) & ((unit_silt + 2*unit_clay) < 30)) & (unit_organic2 <= 2.0))
            unit_erod = numpy.where(loamysand_mask, 0.04 * 0.1317, unit_erod)	
            loamysand_mask = None
            loamysand_mask_abo = ((((unit_silt + 1.5*unit_clay) >= 15) & ((unit_silt + 2*unit_clay) < 30)) & (unit_organic2 > 2.0))
            unit_erod = numpy.where(loamysand_mask_abo, 0.05 * 0.1317, unit_erod)	
            loamysand_mask_abo = None            
            sandyloam_mask = (((7 <= unit_clay) & (unit_clay < 20) & (unit_sand > 52) & ((unit_silt + 2*unit_clay) >= 30)) & (unit_organic2 == 2.0))
            unit_erod = numpy.where(sandyloam_mask, 0.13 * 0.1317, unit_erod)	
            sandyloam_mask = None
            sandyloam_mask_bel = (((7 <= unit_clay) & (unit_clay < 20) & (unit_sand > 52) & ((unit_silt + 2*unit_clay) >= 30)) & (unit_organic2 < 2.0))
            unit_erod = numpy.where(sandyloam_mask_bel, 0.12 * 0.1317, unit_erod)	
            sandyloam_mask_bel = None
            sandyloam_mask_abo = (((7 <= unit_clay) & (unit_clay < 20) & (unit_sand > 52) & ((unit_silt + 2*unit_clay) >= 30)) & (unit_organic2 > 2.0))
            unit_erod = numpy.where(sandyloam_mask_abo, 0.14 * 0.1317, unit_erod)	
            sandyloam_mask_abo = None
            sandyloam_mask2 = (((unit_clay < 7) & (unit_silt < 50) & ((unit_silt+2*unit_clay)>=30)) & (unit_organic2 == 2.0))
            unit_erod = numpy.where(sandyloam_mask2, 0.13 * 0.1317, unit_erod)	
            sandyloam_mask2 = None
            sandyloam_mask2_bel = (((unit_clay < 7) & (unit_silt < 50) & ((unit_silt+2*unit_clay)>=30)) & (unit_organic2 < 2.0))
            unit_erod = numpy.where(sandyloam_mask2_bel, 0.12 * 0.1317, unit_erod)	
            sandyloam_mask2_bel = None
            sandyloam_mask2_abo = (((unit_clay < 7) & (unit_silt < 50) & ((unit_silt+2*unit_clay)>=30)) & (unit_organic2 > 2.0))
            unit_erod = numpy.where(sandyloam_mask2_abo, 0.14 * 0.1317, unit_erod)	
            sandyloam_mask2_abo = None
            loam_mask = (((unit_clay >= 7) & (unit_clay < 27) & (unit_silt >= 28) & (unit_silt < 50) & (unit_sand <= 52)) & (unit_organic2 == 2.0))
            unit_erod = numpy.where(loam_mask,0.3 * 0.1317, unit_erod)
            loam_mask = None
            loam_mask_bel = (((unit_clay >= 7) & (unit_clay < 27) & (unit_silt >= 28) & (unit_silt < 50) & (unit_sand <= 52)) & (unit_organic2 < 2.0))
            unit_erod = numpy.where(loam_mask_bel,0.26 * 0.1317, unit_erod)
            loam_mask_bel = None
            loam_mask_abo = (((unit_clay >= 7) & (unit_clay < 27) & (unit_silt >= 28) & (unit_silt < 50) & (unit_sand <= 52)) & (unit_organic2 > 2.0))
            unit_erod = numpy.where(loam_mask_abo,0.34 * 0.1317, unit_erod)
            loam_mask_abo = None
            siltloam_mask =(((unit_silt >= 50) & (unit_clay >= 12) & (unit_clay < 27)) & (unit_organic2 == 2.0))
            unit_erod = numpy.where(siltloam_mask, 0.38 * 0.1317, unit_erod)
            siltloam_mask = None
            siltloam_mask_bel =(((unit_silt >= 50) & (unit_clay >= 12) & (unit_clay < 27)) & (unit_organic2 < 2.0))
            unit_erod = numpy.where(siltloam_mask_bel, 0.37 * 0.1317, unit_erod)
            siltloam_mask_bel = None
            siltloam_mask_abo =(((unit_silt >= 50) & (unit_clay >= 12) & (unit_clay < 27)) & (unit_organic2 > 2.0))
            unit_erod = numpy.where(siltloam_mask_abo, 0.41 * 0.1317, unit_erod)
            siltloam_mask_abo = None
            siltloam_mask2 = (((unit_silt >= 50) & (unit_silt < 80) & (unit_clay < 12)) & (unit_organic2 == 2.0))
            unit_erod = numpy.where(siltloam_mask2, 0.38 * 0.1317, unit_erod)
            siltloam_mask2 = None
            siltloam_mask2_bel = (((unit_silt >= 50) & (unit_silt < 80) & (unit_clay < 12)) & (unit_organic2 < 2.0))
            unit_erod = numpy.where(siltloam_mask2_bel, 0.37 * 0.1317, unit_erod)
            siltloam_mask2_bel = None
            siltloam_mask2_abo = (((unit_silt >= 50) & (unit_silt < 80) & (unit_clay < 12)) & (unit_organic2 > 2.0))
            unit_erod = numpy.where(siltloam_mask2_abo, 0.41 * 0.1317, unit_erod)
            siltloam_mask2_abo = None
            silt_mask = (((unit_silt >= 80) & (unit_clay < 12)) & (unit_organic2 == 2.0))
            unit_erod = numpy.where(silt_mask, 0.38 * 0.1317, unit_erod)
            silt_mask = None
            silt_mask_bel = (((unit_silt >= 80) & (unit_clay < 12)) & (unit_organic2 < 2.0))
            unit_erod = numpy.where(silt_mask_bel, 0.37 * 0.1317, unit_erod)
            silt_mask_bel = None
            silt_mask_abo = (((unit_silt >= 80) & (unit_clay < 12)) & (unit_organic2 > 2.0))
            unit_erod = numpy.where(silt_mask_abo, 0.41 * 0.1317, unit_erod)
            silt_mask_abo = None
            sandyclayloam_mask = ((unit_clay >= 20) & (unit_clay < 35) & (unit_silt < 28) & (unit_sand > 45))
            unit_erod = numpy.where(sandyclayloam_mask, 0.2 * 0.1317, unit_erod)
            sandyclayloam_mask = None
            clayloam_mask = ((((unit_clay >= 27) & (unit_clay < 40)) & ((unit_sand > 20) & (unit_sand <= 45))) & (unit_organic2 == 2.0))
            unit_erod = numpy.where(clayloam_mask, 0.3 * 0.1317, unit_erod)
            clayloam_mask = None
            clayloam_mask_bel = ((((unit_clay >= 27) & (unit_clay < 40)) & ((unit_sand > 20) & (unit_sand <= 45))) & (unit_organic2 < 2.0))
            unit_erod = numpy.where(clayloam_mask_bel, 0.28 * 0.1317, unit_erod)
            clayloam_mask_bel = None
            clayloam_mask_abo = ((((unit_clay >= 27) & (unit_clay < 40)) & ((unit_sand > 20) & (unit_sand <= 45))) & (unit_organic2 > 2.0))
            unit_erod = numpy.where(clayloam_mask_abo, 0.33 * 0.1317, unit_erod)
            clayloam_mask_abo = None
            siltyclayloam_mask = (((unit_clay >= 27) & (unit_clay < 40) & (unit_sand  <= 20)) & (unit_organic2 == 2.0))
            unit_erod = numpy.where(siltyclayloam_mask, 0.32 * 0.1317, unit_erod)
            siltyclayloam_mask = None
            siltyclayloam_mask_bel = (((unit_clay >= 27) & (unit_clay < 40) & (unit_sand  <= 20)) & (unit_organic2 < 2.0))
            unit_erod = numpy.where(siltyclayloam_mask_bel, 0.3 * 0.1317, unit_erod)
            siltyclayloam_mask_bel = None
            siltyclayloam_mask_abo = (((unit_clay >= 27) & (unit_clay < 40) & (unit_sand  <= 20)) & (unit_organic2 > 2.0))
            unit_erod = numpy.where(siltyclayloam_mask_abo, 0.35 * 0.1317, unit_erod)
            siltyclayloam_mask_abo = None           
            sandyclay_mask = ((unit_clay >= 35) & (unit_sand > 45))
            unit_erod = numpy.where(sandyclay_mask, 0.2 * 0.1317, unit_erod)
            sandyclay_mask = None
            siltyloam_mask = (((unit_clay >= 40) & (unit_silt >= 40)) & (unit_organic2 == 2.0))
            unit_erod = numpy.where(siltyloam_mask, 0.38 * 0.1317, unit_erod)
            siltyloam_mask = None
            siltyloam_mask_bel = (((unit_clay >= 40) & (unit_silt >= 40)) & (unit_organic2 < 2.0))
            unit_erod = numpy.where(siltyloam_mask_bel, 0.37 * 0.1317, unit_erod)
            siltyloam_mask_bel = None
            siltyloam_mask_abo = (((unit_clay >= 40) & (unit_silt >= 40)) & (unit_organic2 > 2.0))
            unit_erod = numpy.where(siltyloam_mask_abo, 0.41 * 0.1317, unit_erod)
            siltyloam_mask_abo = None
            clay_mask = (((unit_clay >= 40) & (unit_sand <= 45) & (unit_silt < 40)) & (unit_organic2 == 2.0))
            unit_erod = numpy.where(clay_mask, 0.22 * 0.1317, unit_erod)
            clay_mask = None
            clay_mask_bel = (((unit_clay >= 40) & (unit_sand <= 45) & (unit_silt < 40)) & (unit_organic2 < 2.0))
            unit_erod = numpy.where(clay_mask_bel, 0.22 * 0.1317, unit_erod)
            clay_mask_bel = None
            clay_mask_abo = (((unit_clay >= 40) & (unit_sand <= 45) & (unit_silt < 40)) & (unit_organic2 > 2.0))
            unit_erod = numpy.where(clay_mask_abo, 0.22 * 0.1317, unit_erod)
            clay_mask_abo = None

            return numpy.where(nodata_mask, nodata_out, unit_erod)


        pygeoprocessing.vectorize_datasets(
            raster_uri, erod_func, dataset_out_uri, gdal.GDT_Float32,
            nodata_out, pixel_size_out, 'intersection',
            vectorize_op=False, assert_datasets_projected=False)
        
    except:
        print "Error vectorizing datasets."
        raise

    try:
        # Delete the older tif files used in processing
        for tif_file in raster_uri:
            os.remove(tif_file)
            
    except:
        print "Error in removing temporary file."  
        raise

    try:
        # Read the mcf file designed for the file and render it as an .xml metadata file
        xml_string = render_template(mcf_file, schema='iso19139')
        with open(dataset_out_uri + '.xml', 'w') as ff:
            ff.write(xml_string)
    except:
        print "Error in processing the metadata file."  
        raise
except:
    print "Error in running the script."
