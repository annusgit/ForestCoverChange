

from __future__ import print_function
from __future__ import division
import ee
import ee.mapclient
import geetools


def reference_code():
    ## Initialize connection to server
    ee.Initialize()
    ## Define your image collection
    collection = ee.ImageCollection('COPERNICUS/S2') # LANDSAT/LC8_L1T_TOA
    ## Define time range
    collection_time = collection.filterDate('2017-04-11', '2018-01-01') #YYYY-MM-DD
    ## Select location based on location of tile
    path = collection_time.filter(ee.Filter.eq('WRS_PATH', 198))
    pathrow = path.filter(ee.Filter.eq('WRS_ROW', 24))
    # or via geographical location:
    #point_geom = ee.Geometry.Point(5, 52) #longitude, latitude
    #pathrow = collection_time.filterBounds(point_geom)
    ## Select imagery with less then 5% of image covered by clouds
    clouds = pathrow.filter(ee.Filter.lt('CLOUD_COVER', 5))
    ## Select bands
    bands = clouds.select(['B4', 'B3', 'B2'])
    ## Make 8 bit data
    def convertBit(image):
        return image.multiply(512).uint8()
    ## Convert bands to output video
    outputVideo = bands.map(convertBit)
    print("Starting to create a video")
    ## Export video to Google Drive
    out = ee.batch.Export.video.toDrive(outputVideo,
                                        description='Netherlands_video_region_L8_time',
                                        dimensions=720,
                                        framesPerSecond=2,
                                        region=([5.588144,51.993435],
                                                [5.727906, 51.993435],
                                                [5.727906, 51.944356],
                                                [5.588144, 51.944356]),
                                        maxFrames=10000)
    ## Process the image
    process = ee.batch.Task.start(out)
    print("Process sent to cloud")


def downloader():
    ee.Initialize() ################################ this must be done before anything
    ee.mapclient.centerMap(-110, 40, 5)
    germany = ee.Geometry.Polygon([[
        [-109.05, 37.0], [-102.05, 37.0], [-102.05, 41.0],  # colorado
        [-109.05, 41.0], [-111.05, 41.0], [-111.05, 42.0],  # utah
        [-114.05, 42.0], [-114.05, 37.0], [-109.05, 37.0]]])

    def cropout_aoi(image):
        return image.clip(germany)

    def scale(image):
        return image.divide(4096).multiply(255).uint8()

    count = 1
    month_interval = 5
    month_end = 11 # download one image for 6-month interval
    year_interval = 1
    year_end = 2019
    for year in range(2016, year_end, year_interval):
        for month in range(1, month_end, month_interval):
            next_month = month + month_interval
            print('currently on ({})/({}) -> ({})/({})'.format(month, year, next_month, year))
            # Now define the dates we want to get data on
            date_from = ee.Date(date='{}-{}-01'.format(year, month))
            date_to = ee.Date(date='{}-{}-01'.format(year, next_month))
            # import sentinel images, clip your area of interest, dates and cloud cover as needed
            image_collection = (ee.ImageCollection('COPERNICUS/S2') #('LANDSAT/LE07/C01/T1_RT')
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5))
                .filterDate(date_from, date_to)
                .filterBounds(germany) # locates ones close to this polygon
                .map(cropout_aoi)) # actually does the clipping part
                #.map(maskS2clouds))
            all_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12']
            total_count = image_collection.size()
            # print('log: total images in collection =', total_count)

            this_image = image_collection.median()
            # ee.mapclient.addtoMap(this_image.select('B4', 'B3', 'B2'), {'gain': [1.4, 1.4, 1.1]})

            image = this_image.select('B3', 'B2', 'B1')
            ee.mapclient.addToMap(image, {'gain': [1.4, 1.4, 1.1]})

            # ee.batch.Export.image.toDrive(image=this_image.select('B2', 'B3', 'B4', 'B5', 'B8'),
            #                               description='German_S2_median_{}'.format(count),
            #                               folder='germany_s2',
            #                               fileNamePrefix='g_median_{}'.format(count),
            #                               region= germany,
            #                               scale=10)
            # image1 = ee.Image('CGIAR/SRTM90_V4')
            path = image.getDownloadUrl({
                'scale': 10,
                # 'crs': 'EPSG:4326',
                'region': '[[-109.05, 40.0],'
                          '[-108.05, 40.0],'
                          '[-108.05, 41.0],'
                          '[-109.05, 41.0]]'
            })
            print(path)
            count += 1

    pass

downloader()







