

from geetools import tools
import ee
ee.Initialize()

col = ee.ImageCollection("MODIS/NTSG/MOD16A2/105")
region = ee.Geometry.Polygon([[[7.7324289943328495, 49.41413955834249],
          [7.813109841500818, 49.41413955834249],
          [7.813109841500818, 49.45745251709314],
          [7.7324289943328495, 49.45745251709314]]])

# help(tools.col2drive)  # See help for function col2drive

tasks = tools.col2drive(col, 'MyFolder', region=region)