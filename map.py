

from __future__ import print_function
from __future__ import division
import ee
import ee.mapclient

ee.Initialize()

# Get a download URL for an image.
image1 = ee.Image('CGIAR/SRTM90_V4')
path = image1.getDownloadUrl({
    'scale': 10,
    'crs': 'EPSG:4326',
    'region': '[[-120, 35], [-120.05, 35], [-119, 34.05], [-120, 34.05]]'
})
print(path)