import requests
import datetime
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
import matplotlib
import pickle
import csv
from datetime import date
from SentinelLoadstutzv2 import *

# just prints week weather forecast
def GetWForecast(lat, lon, path, daily = True, hourly=False, minutely = False, current=False, tempunit = 'metric'):
    Days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    key = '35e02b4dab539973840fc771425f3539'

    if daily == True:
        WeatherStats = WeekForecast(lat, lon, key, tempunit)
        
        dayofweek = datetime.datetime.now().weekday()
        #for i in range(7):
        #    print('The forecast for ', Days[(dayofweek+i)%7], 'is' )
        #    for k in WeatherStats:
        #        print(k, ': ', WeatherStats[k][i], WeatherStats[k][7]) 
    
        week = list()
        week.append('Weather Forecast')
        for i in range(7):
            dayofweek = Days[(datetime.datetime.now().weekday() + i)%7]
            week.append(dayofweek)
        week.append('Units')
        with open(path, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow([week])
            for k in WeatherStats:
                writer.writerow([k] + ["," + str(v) for v in WeatherStats[k]])
        


def WeekForecast(lat, lon, key, tempunit):
    exclude = 'minutely, current, hourly'
    WeatherData = requests.get('https://api.openweathermap.org/data/2.5/onecall?lat={}&lon={}&units={}&exclude={}&appid={}'.format(lat, lon, tempunit,exclude, key))
    Weather = WeatherData.json()
    DailyW = Weather['daily']
    if tempunit == 'metric':
        TempUnit = 'celsius'
    Rain = [0]*8
    Rain[7] = 'mm'
    MinTemp = [0]*8
    MinTemp[7] = TempUnit
    MaxTemp = [0]*8
    MaxTemp[7] = TempUnit
    Humid = [0]*8
    Humid[7] = '%'
    DewPoint = [0]*8
    DewPoint[7] = TempUnit
    Clouds = [0]*8
    Clouds[7] = '%'
    Descrip = [0]*8
    Descrip[7] = ''
    UVIs = [0]*8
    UVIs[7] = ''
    Morn = [0]*8
    Morn[7] = TempUnit
    Night = [0]*8
    Night[7] = TempUnit
    Day = [0]*8
    Day[7] = TempUnit
    WindSpeed = [0]*8
    WindSpeed[7] = 'meter/sec'
    WindDeg = [0]*8
    WindDeg[7] = 'degrees'
    Eve = [0]*8
    Eve[7] = TempUnit

    for i in range(7):
        Day[i] = str(DailyW[i]['temp']['day'])
        Morn[i] = DailyW[i]['temp']['morn']
        Eve[i] = DailyW[i]['temp']['eve']
        Night[i] = DailyW[i]['temp']['night']
        MaxTemp[i] = DailyW[i]['temp']['max']
        MinTemp[i] = DailyW[i]['temp']['min']
        Humid[i] = DailyW[i]['humidity']
        DewPoint[i] = DailyW[i]['dew_point']
        WindSpeed[i] = DailyW[i]['wind_speed']
        WindDeg[i] = DailyW[i]['wind_deg']
        Descrip[i] = DailyW[i]['weather'][0]['description']
        UVIs[i] = DailyW[i]['uvi']
        Clouds[i] = DailyW[i]['clouds']

        if 'rain' in DailyW[i]:
            print("Rain: ", DailyW[i]['rain'], "mm")
            Rain[i] = DailyW[i]['rain']
    WeatherStats = dict({'Morning': Morn, 'Day': Day, 'Evening': Eve, 'Night': Night, 'Minimum Temperature': MinTemp, 'Max Temperature':MaxTemp, 'Humidity':Humid, 'Dew Point': DewPoint,'UVI': UVIs,'Cloud Coverage(%)': Clouds, 'Rain': Rain, 'Wind Speed': WindSpeed, 'Wind Degrees': WindDeg, 'Description':Descrip}) #

   
    return WeatherStats

#FOR ANY POLYGON INPUT, JUST MAKE A LIST OF X, Y, AND THEN TAKE THE MIN, MAX OF EACH LIST GIVING YOU YOUR TOP, BOT, LEFT, RIGHT BOUNDS
def getImages(fpaths, bandpaths, bb):
    "Bandpaths: list of 9 bands from most recent satellite imagery. LatLon: tuple of EPSG:4326 coordinates. Returns: list of np arrays containing 11 spectral indices, NDVI, NDRE, RECI, SAVI, SIPI, ARVI, GVMI, NDMI, GCI, NDWI, MI. Saves directly to bandpaths"
    band2 = rasterio.open(fpaths[0], driver = 'JP2OpenJPEG') #blue - 10m
    band3 = rasterio.open(fpaths[1],  driver = 'JP2OpenJPEG') #green - 10m
    band4 = rasterio.open(fpaths[2],  driver = 'JP2OpenJPEG') #red - 10m
    band5 = rasterio.open(fpaths[3],  driver = 'JP2OpenJPEG') #red edge close to center - 20m, 5990
    band8 = rasterio.open(fpaths[4],  driver = 'JP2OpenJPEG') #nir - 10m, 10980
    band8A = rasterio.open(fpaths[5],  driver = 'JP2OpenJPEG') #narrownir - 20m, 5990
    band11 = rasterio.open(fpaths[6],  driver = 'JP2OpenJPEG') #swir1 - 20m
    band12 = rasterio.open(fpaths[7],  driver = 'JP2OpenJPEG') #swir2 - 20m
    bandTCI = rasterio.open(fpaths[8],  driver = 'JP2OpenJPEG') #true color image, 10m


    x, y = rasterio.warp.transform('EPSG:4326', band4.crs, [bb[0], bb[2]], [bb[1], bb[3]]) # check if lat, lon should be flipped

    row, col = band4.index(exx, exy)
    c = col[0]
    r = row[0]
    top = y[1]
    bot = y[0]
    left = x[0]
    right = x[1]
    

    ogRedBand = band4.read(1, window=Window(left, top, top-bot, right-left)).astype('float64')
    ogNIRBand = band8.read(1, window=Window(left, top, top-bot, right-left)).astype('float64')
    rededge = band5.read(1, window=Window(left/2, top/2, (top-bot)/2, (right-left)/2)).astype('float64')
    ogGreenBand = band3.read(1, window=Window(left, top, top-bot, right-left)).astype('float64')
    ogBlueBand = band2.read(1, window=Window(left, top, top-bot, right-left)).astype('float64')
    wnnir = band8A.read(1, window=Window(left/2, top/2, (top-bot)/2, (right-left)/2)).astype('float64')
    wswir1 = band11.read(1, window=Window(left/2, top/2, (top-bot)/2, (right-left)/2)).astype('float64')
    wswir2 = band12.read(1, window=Window(left/2, top/2, (top-bot)/2, (right-left)/2)).astype('float64')

    #Stretch all 20m bands to 10980x10980
    RedEdgeStretch = np.repeat(np.repeat(rededge,2, axis=0), 2, axis=1) 
    SWIR1Stretch = np.repeat(np.repeat(wswir1,2, axis=0), 2, axis=1) 
    SWIR2Stretch = np.repeat(np.repeat(wswir2,2, axis=0), 2, axis=1) 
    NNIRStretch = np.repeat(np.repeat(wnnir,2, axis=0), 2, axis=1) 

    ogRedEdgeStretch = RedEdgeStretch
    ogSWIR1Stretch = SWIR1Stretch
    ogSWIR2Stretch = SWIR2Stretch
    ogNNIRStretch= NNIRStretch

    # Calculate NDVI
    nNDVI = np.where(
        (ogNIRBand+ogRedBand)==0., 
        0,
        (ogNIRBand-ogRedBand)/(ogNIRBand+ogRedBand)
    )

    # NDRE (NIR-RE)/(NIR+RE)
    nNDRE = np.where(
        (ogNIRBand+ogRedEdgeStretch)==0.,
        0.,
        (ogNIRBand-ogRedEdgeStretch)/(ogNIRBand+ogRedEdgeStretch)
    ) 

    # RECI (NIR/Red Edge)
    nRECI = np.where(
        ogRedEdgeStretch==0.,
        0.,
        ogNIRBand/ogRedEdgeStretch
    )

    # SAVI ((NIR – Red) / (NIR + Red + L)) x (1 + L)
    L = 0 #L is variable from -1 to 1. For high green vegetation, L is set to 0, whereas for low green vegetation, it is set to 1.
    nSAVI = np.where(
        ((ogNIRBand+ogRedBand+L) * (1+L)) ==0.,
        0.,
        (ogNIRBand - ogRedBand) / ((ogNIRBand+ogRedBand+L) * (1+L))
    )

    # SIPI (NIR-Blue)/(NIR-Red)
    nSIPI = np.where(
        (ogNIRBand-ogRedBand)==0.,
        0.,
        (ogNIRBand-ogBlueBand)/(ogNIRBand-ogRedBand)
    )

    # ARVI (NIR-(2*Red)+Blue)/(NIR + (2*Red) + Blue))
    nARVI = np.where(
        (ogNIRBand+(2*ogRedBand)+ogBlueBand)==0.,
        0.,
        (ogNIRBand-(2*ogRedBand)+ogBlueBand)/(ogNIRBand+(2*ogRedBand)+ogBlueBand)
    )

    # GVMI (NIR+0.1) - (SWIR+0.2) // (NIR+0.1) + (SWIR+0.2) #currently attempt with SWIR1
    nGVMI = np.where(
        ((ogNIRBand+0.1)+(ogSWIR1Stretch+0.2))==0.,
        0.,
        (ogNIRBand+0.1)-(ogSWIR1Stretch+0.2)/(ogNIRBand+0.1)+(ogSWIR1Stretch+0.2)
    )

    # NDMI Narrow NIR - (SWIR1 - SWIR2) / NarrowNIR + (SWIR1-SWIR2)
    nNDMI = np.where(
        (ogNNIRStretch + (ogSWIR1Stretch-ogSWIR2Stretch))==0.,
        0.,
        (ogNNIRStretch - (ogSWIR1Stretch-ogSWIR2Stretch)) / (ogNNIRStretch + (ogSWIR1Stretch-ogSWIR2Stretch))
    )
    #NDMI limit to values between -1 and 1 for visibility
    nNDMI = np.where(
        nNDMI>1.,
        0.,
        nNDMI
    )
    nNDMI = np.where(
        nNDMI<-1.,
        0.,
        nNDMI
    )

    # GCI (NIR/Green)
    nGCI = np.where(
        ogGreenBand==0.,
        0.,
        ogNIRBand/ogGreenBand
    )
    # NDWI (B8A-B11)/(B11+B8A) THIS IS THE CORRECT ONE!!!!!! FOR MOISTURE MAPPING!
    nNDWI = np.where(
        (ogSWIR1Stretch+ogNNIRStretch)==0.,
        0.,
        (ogNNIRStretch- ogSWIR1Stretch)/(ogSWIR1Stretch+ogNNIRStretch)
    )

    # MOISTURE INDEX (B8A-B11)/(B8A+B11)
    nMI = np.where(
        (ogNNIRStretch+ogSWIR2Stretch)==0.,
        0.,
        (ogNNIRStretch-ogSWIR2Stretch)/(ogNNIRStretch+ogSWIR2Stretch)
    )
    indices = [nNDVI, nNDRE, nRECI, nSAVI, nSIPI, nARVI, nGVMI, nNDMI, nGCI, nNDWI, nMI]
    # save all indices
    [matplotlib.image.imsave(bandpaths[i], indices[i]) for i in range(len(indices))]

    # close all band files
    band2.close()
    band3.close()
    band4.close()
    band5.close()
    band8.close()
    band8A.close()
    band11.close()
    band12.close()

    

def getAll():
    LatLon = (-120.3, 37.81)
    lon= -120
    lat = 37
    WeatherCall(lat, lon)
    getImages(LatLon)
def GetStationString(LatLon, MyAlt):
    stationlist = list()
    with open("CSVFiles\stationsAlt.pkl", "rb") as stationdict:
        stationdicts = pickle.load(stationdict)
        for k in stationdicts:
            if CalcDist(LatLon, stationdicts[k])<0.20:
                if AltDif(MyAlt, stationdicts[k])<450:
                    stationlist.append(k)
                    if len(stationlist)>48: #CHANGE IT BACK TO 48!!!!
                        break
        if len(stationlist)<15:
            for k in stationdicts:
                if CalcDist(LatLon, stationdicts[k])<0.45:
                    if AltDif(MyAlt, stationdicts[k]) < 400:
                        stationlist.append(k)
                        if len(stationlist)>48: #CHANGE IT BACK TO 48!!!!!
                            break
    print(len(stationlist), "lenstationlist")
    stationstring = ",".join(stationlist)
    return stationstring

def GetAltitude(LatLon):
    LatLonString = str(LatLon[0]) + ','+ str(LatLon[1])
    # Access altitude from coordinates
    AltKey = 'P9JZGGsBywA7Sx2IceReALUSOjsGQ8XQ'
    AltURL = 'http://open.mapquestapi.com/elevation/v1/profile?key={AltKey}&shapeFormat=raw&latLngCollection={LatLonString}'.format(AltKey=AltKey, LatLonString=LatLonString)
    MyAlt = requests.get(AltURL)
    MyAlt = MyAlt.json()
    if (MyAlt['info']['statuscode']==0):
        Altitude = MyAlt['elevationProfile'][0]['height']
    else:
        Altitude = -10000
    return Altitude

# takes two float tuples, calculates euclidian distance
def CalcDist(LatLon, v):
    xb = float(v[1])
    yb = float(v[0])
    dist =  ((LatLon[0] - xb)**2 + (LatLon[1]-yb)**2)**(1/2)
    #print(dist)
    return dist
# calculates difference in altitude (they are both in meters)
def AltDif(MyAlt, v):
    stationAlt = float(v[2])
    #check if either is missing value
    if ((MyAlt == -10000) or (stationAlt==-999)):
        return 0
    AltDif = abs(stationAlt - MyAlt)
    return AltDif

def GetHW(WS):
    "Returns 5 "
    # Dicts to save dated values
    newWS = WS
    prcpDict = dict()
    tminDict = dict()
    snowDict = dict()
    tmaxDict = dict()
    snwdDict = dict()

    for i in range(len(newWS)):
        if 'PRCP' in newWS[i]:
            MMPrecip = float(newWS[i]['PRCP'])/10
            #if date already stored, average over values
            if newWS[i]['DATE'] in prcpDict:
                Collisions = prcpDict[newWS[i]['DATE']][1] + 1
                # date's value = new value* 1/collisions + old value* (1 - 1/collisions) 
                prcpDict[newWS[i]['DATE']] = ((MMPrecip*(1/Collisions)) + (prcpDict[newWS[i]['DATE']][0]*(1-(1/Collisions))), Collisions)
            else:    
                prcpDict[newWS[i]['DATE']] = (float(newWS[i]['PRCP']), 1)
    
        if 'TMIN' in newWS[i]:
            CelsiusTMIN = float(newWS[i]['TMIN'])/10
            if newWS[i]['DATE'] in tminDict:
                TMINCollisions = tminDict[newWS[i]['DATE']][1] + 1
                tminDict[newWS[i]['DATE']] = ((CelsiusTMIN*(1/TMINCollisions))+(tminDict[newWS[i]['DATE']][0]*(1-(1/TMINCollisions))), TMINCollisions)    
        
            else:
                tminDict[newWS[i]['DATE']] = (CelsiusTMIN, 1)
    
        if 'SNOW' in newWS[i]:
            if newWS[i]['DATE'] in snowDict:
                SNOWCollisions = snowDict[newWS[i]['DATE']][1] + 1
                snowDict[newWS[i]['DATE']] = ((float(newWS[i]['SNOW'])*(1/SNOWCollisions))+(snowDict[newWS[i]['DATE']][0]*(1-(1/SNOWCollisions))), SNOWCollisions)
            else:
                snowDict[newWS[i]['DATE']] = (float(newWS[i]['SNOW']), 1)
    
        if 'TMAX' in newWS[i]:
            CelsiusTMAX = float(newWS[i]['TMAX'])/10
            if newWS[i]['DATE'] in tmaxDict:
                TMAXCollisions = tmaxDict[newWS[i]['DATE']][1] + 1 
                tmaxDict[newWS[i]['DATE']] = ((CelsiusTMAX*(1/TMAXCollisions))+(tmaxDict[newWS[i]['DATE']][0]*(1-(1/TMAXCollisions))), TMAXCollisions)
                #print(CelsiusTMAX, ": ", newWS[i]['DATE'])
            else:
                tmaxDict[newWS[i]['DATE']] = (CelsiusTMAX, 1)
                #print(CelsiusTMAX, newWS[i]['DATE'])

        if 'SNWD' in newWS[i]:
            if newWS[i]['DATE'] in snwdDict:
                SNWDCollisions = snwdDict[newWS[i]['DATE']][1] + 1
                snwdDict[newWS[i]['DATE']] = ((float(newWS[i]['SNOW'])*(1/SNWDCollisions)) + (snwdDict[newWS[i]['DATE']][0]*(1-(1/SNWDCollisions))), SNWDCollisions)
    return prcpDict, tmaxDict, tminDict, snowDict, snwdDict, 


def HeatStressCheck(croptype, tmax):
    "Identifies heat stress events to be charted from historical temperature. "
    HeatStressDict = {'corn': 35, 'avocado': 35, 'wheat': 32, 'sugarcane': 40, 'jalapeno': 32
    }
    stress = HeatStressDict[croptype]
    events = dict()
    for item in tmax:
        if tmax[item][0]>=stress:
            events[item] = tmax[item][0]
    count = len(events)
    return events, count
    

def ColdStressCheck(croptype, tmin):
    "Identifies cold stress events to be charted from historical temperature. "
    ColdStressDict = {'corn': 8, 'avocado': 0, 'wheat': 9, 'sugarcane': 18, 'jalapeno': 12 
    }
    stress = ColdStressDict[croptype]
    events = dict()
    for item in tmin:
        if tmin[item][0]<=stress:
            events[item] = tmin[item][0]
    count = len(events)
    return events, count

def RemoveCollisions(stats):
    "Removes collision values (previously used to average weather values) from historical weather"
    # stats is list of dicts
    for d in stats:
        for v in d.keys():
            d[v] = d[v][0]
    return stats

def GetHistWeather(LatLon, start, end, path, crop = 'wheat'):
    "Saves historical weather including precipitation, max temp, min temp, snow, snow depth, cold and heat stress events at coordinate from START to END dates, at the path specified. Cold/Heat Stress available for 'wheat, avocado, corn, sugarcane, and jalapeno. Auto-setting gives stress events for wheat. Returns no values."
    LatLonString = str(LatLon[0]) + ','+ str(LatLon[1])
    MyAlt = GetAltitude(LatLon) #CHECK UNITS- meters for API
    
    stationstring = GetStationString(LatLon, MyAlt)
    dates = 'startDate={start}&endDate={end}'.format(start=start, end=end)
    WeatherStats = requests.get('https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries&stations={stationstring}8&dataTypes=SNOW,PRCP,TMIN,TMAX,SNWD&{dates}&includeAttributes=true&includeStationName:1&includeStationLocation:1&format=json'.format(stationstring=stationstring, dates = dates))
    newWS = WeatherStats.json()
    
    prcpDict, tmaxDict, tminDict, snowDict, snwdDict = GetHW(newWS)

    cs, cscount, = ColdStressCheck(crop, tminDict)
    hs, hscount = HeatStressCheck(crop, tmax=tmaxDict)

    stats = [prcpDict, tmaxDict, tminDict, snowDict, snwdDict]
    stats = RemoveCollisions(stats)
    stats.append(cs)
    stats.append(hs)

    #for stat in prcpDict:
        #this is me trying to write all stats together to the same CSV
    with open(path, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['Date', ',Precipitation (mm)', ',Max Temp (C)', ',Min Temp (C)', ',Snowfall (mm)', ',Snow Depth(mm)', ',Cold Stress Events(C)', ',Heat Stress Events (C)'])
        for key in sorted(prcpDict.keys(), key=lambda x: x):
            writer.writerow([key] + ["," + str(d.get(key, None)) for d in stats])


def Zoning(nparr, indextype):
    "nparr: An index to be zoned. indextype: The vegetation index. Returns: An np array of the same index but zoned by values that are similar, and the cutoff values for the given index."
    if indextype in ['ndvi', 'ndre', 'sipi', 'ndmi']: # list of indices going from -1 to 1
        mx = 1
        mn = -1
    if indextype in ['reci', 'gci', 'arvi', ]: # list of indices going really high
        mx = 100
        mn = -100
    # clip values
    nparr = np.where(
        nparr<mn,
        mn,
        nparr
    )
    nparr = np.where(
        nparr>mx,
        mx,
        nparr
    )
    zones = [np.min(nparr), np.quantile(nparr, .25), np.quantile(nparr, .5), np.quantile(nparr, 0.75), np.max(nparr)]
    npzone = np.copy(nparr)
    for i in range(len(zones)-1):
        npzone = np.where(
            zones[i] < npzone < zones[i+1],
            zones[i+1],
            npzone
        )
    return npzone, zones # returns zoned index, as well as cutoff values


def historicalndvi(sl, coordinate, count=10):
    "Gets ndvi images going back 180 days. sl = SentinelLoader object. coordinate = Polygon or point to be transformed to wkt for sentinel load query. Count = number of images to be used in historic info. Returns hndvi which is the averaged ndvi image over past count* ndvi images, and a list of np arrays with data from ndvi images"

    # get current date
    today = date.today()
    
    dlist = []
    b4plist = []
    b8plist = []
    nlist = []
    nirplist = []

    avgs = []
    for i in range(count):
        delta = i * 20
        day = append(today - datetime.timedelta(days = delta))
        dlist.append(day.strftime("%Y-%m-%d"))
    for date in dlist:
        b4plist.append(sl.getProductBandTiles(coordinate, 'b04', '10m', date))
        b8plist.append(sl.getProductBandTiles(coordinate, 'b08', '10m', date))
        
 #       nirplist.append(sdownload(date, 'b08', '10m')) #ognirbands
 #       redplist.append(sdownload(date, 'b04', '10m')) #ogredbands
    # save all historical ndvi for viewing
    for i in range(len(b4plist)):
        b4 = rasterio.open(b4plist[i], driver = 'JP2OpenJPEG')
        b8 = rasterio.open(b8plist[i], driver = 'JP2OpenJPEG')
        redplist.append(b4.read(1))
        nirplist.append(b8.read(1))
        b4.close()
        b8.close()

    for i in range(count):
        nlist[i] = np.where(
        (nirplist[i] + redplist[i])==0., 
        0,
        (nirplist[i] - redplist[i])/(nirplist[i] + redplist[i])
        )
    ndvis = np.array(nlist)
    ndvis.sum(axis=0)
    hndvi = ndvis/len(nlist)

    return hndvi, nlist
    # average by PIXEL all the 
    


def GetCoordinates(gfile, ftype, n = 1, acrlim = 10000000):
    "Gets coordinates from valid file. gfile is a tif, jp2, or shape file w/ coordinates available. ftype is the file type. n is an optional parameter, defining the number of shapes in shape file if there are more than one. Lastly, acrlim is the acreage limit on a request, ensuring the "
    
    if ftype == 'shp':
        f = gpd.read_file(gfile)
        f = f.to_crs(epsg=4326)
        shplist = list()
        bblist = list()
        for i in range(n):
            shplist.append(f['geometry'][i])
            bblist.append(f['geometry'][i].bounds)
        
        sizelist = list()  
        [sizelist.append(checkSize(bb, acrlim)[0]) for bb in bblist]
        if sum(sizelist)<acrlim:
            acr = sum(sizelist)
            return shplist, acr
        else:
            raise Exception('Total acreage exceeds acreage limit with ', acr, 'acres')
        
    if ftype == ('tif' or 'jp2' or 'tiff'):
        with rasterio.open(gfile) as f:
            bounds = f.bounds
            print(bounds)
            print(len(bounds))
            xs = [bounds[0], bounds[2]]
            ys = [bounds[1], bounds[3]]
            xy = rasterio.warp.transform(
                src_crs=f.crs, dst_crs='EPSG:4326', xs = xs, ys = ys)
            [print(c[0], c[1]) for c in xy]
            bb = [xy[0][0], xy[1][0], xy[0][1], xy[1][1]]
    
            
    if ftype == 'coordinates':
        bb = Polygon(gfile).bounds
        

    if ftype == 'wkt':
        bb = gfile.bounds
    
    acr, check = checkSize(bb, acrlim)
    if check == False:
        raise Exception('Total acreage exceeds acreage limit with ', acr, 'acres')
        return 0
    poly = Polygon((bb[0], bb[2]), (bb[0], bb[3]), (bb[1], bb[3]), (bb[1], bb[2]), (bb[0], bb[2])) 

    return bb, poly, acr

        

def checkSize(bb, acrlim):
    "takes bounding box and returns acreage, and true if less than acreage limit"
    #54.6 is 1 longitude
    #69 is 1 latitude

    acr = (((bb[2] - bb[0])*69.0000) * ((bb[3]-bb[1])*54.6000))/0.0015625
    if acr<acrlim:
        check = True
    else:
        check = False
    return acr, check

def datehistory(delta):
    today = date.today()
    past = today - datetime.timedelta(days = delta)
    today = today.strftime("%Y-%m-%d")
    past = past.strftime("%Y-%m-%d")
    return today, past

def makepathlist(dirpath = None):
    "Creates list of paths to place index bands at the selected directory path."
    today = date.today()
    today = today.strftime("%Y_%m_%d_")
    pathlist = list()
    indexlist = ['ndvi', 'ndre', 'reci', 'savi', 'sipi', 'arvi', 'gvmi', 'ndmi', 'gci', 'ndwi', 'mi']
    if dirpath==None:        
        [pathlist.append(today + v + '.png') for v in indexlist]
    else:
        [pathlist.append(os.path.join(dirpath, (today + v + '.png'))) for v in indexlist]
    return pathlist


def getbands(sl, bandpaths, poly, date):
    blist = [('B02', '10m'), ('B03', '10m'), ('B04', '10m'), ('B05', '20m'),('B08', '10m'),  ('B8A', '20m'), ('B11', '20m'), ('B12', '20m'), ('TCI', '10m') ]
    bdict = {'B02': '10m', 'B03': '10m', 'B04': '10m', 'B05': '20m', 'B08': '10m',  'B8A': '20m', 'B11': '20m', 'B12': '20m', 'TCI': '10m' }
    fpaths = list()
    for k in bdict.keys():
        fpaths.append(sl.getProductBandTiles(poly, k, bdict[k], date))
    return fpaths




def getcropstage():
    None


def getcropyield():
    None
    "applies trained ML to "
    "add in makebundle, "


LatLon = (-120.3, 37.81)
lon= -120
lat = 37
LL = (-120, 37)
