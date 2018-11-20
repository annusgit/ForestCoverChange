#!/usr/bin/env python
"""Some data rejiggling functions"""
import os
import matplotlib.pyplot as plt
import matplotlib.dates
import numpy as np
from osgeo import gdal

def interpolate_daily( ndvi ):
    """A function that interpolates monthly NDVI values to daily, assuming the
    data refers to the period 2001 to 2011."""
    ndvi_daily = []
    for year in xrange ( 2001, 2012):
        ndvi_i = ndvi [ (year-2001)*12:( year - 2001 + 1)*12 ]
        t = np.array( [ 16,  44,  75, 105, 136, 166, 197, 228,\
        258, 289, 319, 350 ] )
        # We will interpolate NDVI to be daily. For this we need the following array
        if year%4 == 0:
            ti = np.arange ( 1, 367 ) # Leap year
        else:
            ti = np.arange ( 1, 366 )
        # This is a simple linear interpolator. Strictly, *NOT* needed, but makes
        # everything else easier.
        ndvid = np.interp ( ti, t, ndvi_i )
        ndvi_daily = np.r_[ ndvi_daily, ndvid ]
    return ndvi_daily

def pixel_loc ( longitude, latitude ):
    """This function gets the pixel location from a given longitude and latitude.
    It is used by user-level functions, and assumes the global 1.5 degree 
    grid from ERA interim is being used."""
    
    igeoT = [ 120.0, 0.6666666666666666, 0.0, 60.0, 0.0, -0.6666666666666666]
    
    [ix, iy] = gdal.ApplyGeoTransform ( igeoT, longitude, latitude)
    return ( ix, iy )
    
def fit_ndvi ( ndvi, agdd, function="quadratic" ):
    """Fits the NDVI data to a given function. Two such functions are provided:
    
    * a simple quadratic
    * a double logistic function
    
    The second is clever, and flips itself in terms of hemisphere. Or at least
    I hope so."""
    # Import the leastsq solver
    from scipy.optimize import leastsq
    if function == "quadratic":
        fit_function = lambda p: p[0]*agdd*agdd + p[1]*agdd + p[2] - ndvi
        (xsol, cov_x, infodict, mesg, ier ) =leastsq( fit_function, \
                [0., 0., 0], full_output=True)
        rmse = infodict['fvec'].std()
        return ( rmse, xsol, cov_x, infodict, mesg, ier )
    elif function == "dbl_logistic":
        ndvi_w = ndvi.min()
        ndvi_m = ndvi.max()
        fit_function1 = lambda p: ndvi_w + (ndvi_m - ndvi_w)* ( \
            1./(1+np.exp(-p[0]*(agdd-p[1]))) + \
            1./(1+np.exp(p[2]*(agdd-p[3]))) - 1 )
        fit_function2 = lambda p: ndvi_m - (ndvi_m - ndvi_w)* ( \
            1./(1+np.exp(-p[0]*(agdd-p[1]))) + \
            1./(1+np.exp(p[2]*(agdd-p[3]))) - 1 )
        (xsol1, cov_x1, infodict1, mesg1, ier1 ) =leastsq( fit_function1, \
            [0., 0., 0], full_output=True)
        (xsol2, cov_x2, infodict2, mesg2, ier2 ) =leastsq( fit_function2, \
            [0., 0., 0], full_output=True)    
        rmse1 = infodict1['fvec'].std()
        rmse2 = infodict2['fvec'].std()
        if rmse1 < rmse2:
            return ( rmse1, xsol1, cov_x1, infodict1, mesg1, ier1 )
        else:
            return ( rmse2, xsol2, cov_x2, infodict2, mesg2, ier2 )
            
def resample_dataset ( fname, x_factor, y_factor, method="mean", \
            data_min=-1000, data_max=10000 ):
    """This function resamples a GDAL dataset (single band) by a factor of
    (``x_factor``, ``y_factor``) in x and y. By default, the only method used
    is to calculate the mean. The ``data_min`` and ``data_max`` parameters are
    used to mask out pixels in value"""
    QA_OK = np.array([0, 1, 4, 12, 8, 64, 512, 2048] )# VI OK
    # Table in http://gis.cri.fmach.it/modis-ndvi-evi/
    # First open the NDVI file
    fname = 'HDF4_EOS:EOS_GRID:"%s":' % fname + \
            'MOD_Grid_monthly_CMG_VI:CMG 0.05 Deg Monthly NDVI'
    gdal_data = gdal.Open ( fname )
    # Get raster sizes
    nx = gdal_data.RasterXSize
    ny = gdal_data.RasterYSize
    # Calculate output raster size
    nnx = nx/x_factor
    nny = ny/y_factor
    # Reshape the raster data...
    B = np.reshape ( gdal_data.ReadAsArray(), ( nny, y_factor, nnx, x_factor ) )
    # Now open QA file
    fname = fname.replace ("NDVI", "VI Quality" )
    gdal_data = gdal.Open ( fname )
    qa = gdal_data.ReadAsArray()
    # Check what goes through QA
    qa_pass = np.logical_or.reduce([qa==x for x in QA_OK ])
    
    B = np.ma.array ( B, mask=qa_pass )
    # Re-jiggle the dimensions so we can easily average over then
    C = np.transpose ( B, (0, 2, 1, 3 ) )
    if method == "mean":
        reduced_raster = np.mean ( np.mean ( C, axis=-1), axis=-1 )
    else:
        raise NotImplemented, "Only mean reduction supported by now"

    return reduced_raster


        
def save_raster ( fname_out, raster_in, cell_size, \
        driver="GTiff",dtype=gdal.GDT_Float32 ):
    """This function saves a raster to a filename. The raster must either be
    two-dimensional, or three dimensional, with the first dimension being the
    number of bands. By default, we use GeoTIFF output."""
    
    drv = gdal.GetDriverByName ( driver )
    # Get shapes
    try:
        ( n_bands, nx, ny ) = raster_in.shape
    except ValueError:
        ( nx, ny ) = raster_in.shape
        n_bands = 1
    # Create output file
    dst_ds = drv.Create ( fname_out, ny, nx, n_bands, dtype, \
            ["TFW=YES","TILED=YES","COMPRESS=LZW"] )
    dst_ds.SetGeoTransform( [-180, cell_size, 0.0, 90, 0.0, -cell_size])
    dst_ds.SetProjection ( 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84"' + \
    ',6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],' + \
    'PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",' + \
    '0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]' )

    for b in xrange ( n_bands ):
        try:
            dst_ds.GetRasterBand ( b+1 ).WriteArray ( \
                raster_in [ b, :, :].astype(np.float32) )
        except IndexError:
            dst_ds.GetRasterBand ( b+1 ).WriteArray ( \
                raster_in [ :, :].astype(np.float32) )
    dst_ds = None

def process_vi_files ( data_dir, fname_out, cell_size=1.5, vi="NDVI" ):
    """This function scans all the MODIS HDF files, and process them in annual
    chunks"""
    import glob
    # This globs all the files ending in HDF. I'm assuming that that's the ones
    # I want. Ok, will also do the M?D13C2 bit too...
    files = glob.glob ( "%s/M*D13C2.*.hdf" % data_dir )
    files.sort()
    files = np.array ( files )
    years = np.array( [int(s.split(".")[1][1:5]) for s in files] )
    nny = 180./cell_size
    nnx = 360./cell_size
    x_factor = cell_size/0.05 # CMG cell size is 0.05 degrees
    y_factor = cell_size/0.05 # CMG cell size is 0.05 degrees
    
    for y in np.unique ( years ):
        if y > 2000 and y < 2012:
            if not os.path.exists ( "%s_%04d.tif" % ( fname_out, y ) ):
                # 2000 only has 11 months' worth of data. Skip it
                # 2012 isn't yet finished...
                print "Doing year ", y
                
                year_sel = ( years == y )
                annual = np.zeros ( ( 12, nny, nnx ) )
                for ( i, f_in ) in enumerate ( files[ year_sel ] ):
                    annual [i, :, : ] = resample_dataset ( f_in, x_factor,\
                        y_factor )
                    print i, f_in, "Done..."
                save_raster ( "%s_%04d.tif" % ( fname_out, y ), annual, \
                        cell_size )
                print "Saved to %s_%04d.tif" % ( fname_out, y )
            
            
    print "Finished"
if __name__ == "__main__":
    
    process_vi_files ( "/data/geospatial_20/ucfajlg/MODIS/", \
        "/data/geospatial_20/ucfajlg/MODIS/output/NDVI" )