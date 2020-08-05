from userQueryhelpers import *
from sentinelsat.sentinel import SentinelAPI, read_geojson, geojson_to_wkt
from datetime import date
import geopandas
import shapely
import rasterio
from rasterio import plot
import matplotlib.pyplot as plt
import numpy as np
from rasterio.windows import Window
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.plot import show_hist, show
import rasterio.features
import rasterio.warp
from affine import Affine
import time
import geopandas as gpd
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
from rasterio.plot import plotting_extent
from numpy import save
from numpy import savetxt
from numpy import savez_compressed
from numpy import load
import geopandas as gpd
from datetime import date
from SentinelLoadstutzv2 import *


def main():
    userquery()

if __name__ == "__main__":
    pass

def userquery(input, ftype, hweather = True, whistory = 240, fweather = True, histndvi = True, cy = True, cs = True, userdir = 'rtest/'):
    "Takes a tif, tiff, jp2, or shp file or coordinates as input from user. ftype is the type of input. If hweather is true, historical weather data will be saved. whistory is the number of days of weather history gathered. histndvi determines if historical ndvi is given. userdir is the directory for files to be saved for the user.   "
    # instantiate sentinel download class
    bb, acres, poly = GetCoordinates(input, ftype)
    center = [(bb[0]+bb[2])/2, (bb[1]+bb[3]/2)]

    # current date, and date from whistory number of days in the past
    start, end = datehistory(whistory)

    username = 'croblitos' 
    pword = 'LucklessMonkey$30'
    userdate = start

    sl = Sentinel2Loaderv2('', 
                    username, pword,
                    apiUrl='https://scihub.copernicus.eu/apihub/', showProgressbars=True, loglevel=logging.DEBUG, cacheApiCalls=True, savepath=userdir)

    # creates save paths depending on if a user directory is specified
    bandpaths = makepathlist(userdir)
    if userdir == None:
        hndvipath = 'historicalndvi.png'
        hwpath = 'histweather.csv'
        fwpath = 'forecast.csv'
    else:
        hndvipath = os.path.join(userdir, 'historicalndvi.png')
        hwpath = os.path.join(userdir, 'histweather.csv')
        fwpath = os.path.join(userdir, 'forecast.csv')
    
    # saves historical weather including cold/heat stress events to hwpath
    if hweather == True:
        GetHistWeather(center, start, end, path = hwpath)
    # saves 7 day weather forecast to fwpath
    if fweather == True:
        GetWForecast(center[0], center[1], path=fwpath)

    if histndvi == True:
        hndvi = historicalndvi(sl, poly, count = 10) # need to add function to delete raw images once npz extracted
        matplotlib.image.imsave(hndvi, hndvipath)

       # get bands and save at imsavepath
        fpaths = getBands(bandpaths) 
        # saves bands at bandpaths
        getImages(bandpaths, bb)

    # create input for machine learning algorithms
    if cy == True:
        cyinput = cyformat(setofinputs)
        cropyield(cyinput)
    # collects soil, 
    if cs == True:
        csinput = csformat(csinputss)
        cropstage(csinput)
    
    
    


    

















# Weather forecast - COMPLETE

# # Historical Weather - COMPLETE

# Moisture map w/ NDWI image (8A and SWIR1) ---sampler - complete

# TC Image - Either TCI or if npz-> jpg for true color works, do that. ---sampler - eh

# Stress Indicators (SIPI+RECI+GCI) + Talk about drops in NDVI (12 images) ---sampler - cmplete

# Cold Stress/Heat Stress - Add to helpers, but is complete.


# Historical Productivity (overlaid NDVI)
# --- NDVI averaged over all historical images. Enforce ranges.
# ----- I can do this quickly if I have time w/ like 6 historical images. Acknowledge this is half (12 images)

# Vegetation Level Zoning
# --- NDVI * NDRE w/ 4 cutoffs. Cutoffs should not be based off actual values, 


# Predicted Water Demands

# Crop stage prediction




