

from __future__ import print_function
from __future__ import division
import matplotlib.pyplot as pl
import descarteslabs as dl
import numpy as np
import cv2


# Define a bounding box around Taos in a GeoJSON

taos = {
    "type": "Polygon",
    "coordinates": [
        [
            [7.9052313346861865, 49.128817914014034],
            [8.105420543670562, 49.12971381160097],
            [8.104047252654937, 49.26659938059896],
            [7.9034710807799365, 49.26570688270015]
        ]
    ]
}

# Create a SceneCollection
scenes, ctx = dl.scenes.search(taos,
                               products=["landsat:LE07:01:RT:TOAR"],
                               start_datetime="2010-01-01",
                               end_datetime="2010-12-01",
                               cloud_fraction=0.5,
                               limit=15)
print(scenes)
print(scenes.each.properties.id)
# Make a lower-resolution GeoContext
ctx_lowres = ctx.assign(resolution=10)

# Request a NumPy stack of all the scenes using the same GeoContext
arr_stack = scenes.stack("red green blue", ctx_lowres)

# Composite the scenes based on the median pixel value
composite = np.ma.median(arr_stack, axis=0)
print(type(composite), composite.shape)
# dl.scenes.display(composite, title="Taos Composite")
image = composite.data.transpose(1,2,0)
print(image.max())
new = (image.astype(np.float)*255/4096).astype(np.uint8)
new = new[:,:,[2,1,0]]
pl.imshow(new)
pl.show()
cv2.imwrite('test.png', new)






