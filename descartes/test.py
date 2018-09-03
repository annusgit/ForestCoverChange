

from __future__ import print_function
from __future__ import division
import matplotlib.pyplot as pl
import descarteslabs as dl
import numpy as np
import cv2
from libtiff import TIFF

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
                               products=["landsat:LC08:01:RT:TOAR", "sentinel-2:L1C"],
                               start_datetime="2018-01-01",
                               end_datetime="2018-12-01",
                               cloud_fraction=0.1,
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
image = composite.data.transpose(1,2,0).astype(np.int16)
# print(image.max())
# new = (image.astype(np.float)*255/4096).astype(np.uint8)
# pl.imshow(new)
# pl.show()
# print(new.shape)
# new = new[:,:,:]
# cv2.imwrite('test.tif', image)
# tiff = TIFF.open('test.tiff', mode='w')
# tiff.write_image(new)
# tiff.close()

save_image = {
        'pixels': image,
        'size': image.size,
        'mode': None,
    }

import png
# from scipy.misc import imsave
# imsave('test.png', image)
with open('test.png', 'wb') as f:
    writer = png.Writer(width=image.shape[1], height=image.shape[0], bitdepth=16)
    # Convert z to the Python list of lists expected by
    # the png writer.
    z2list = image.reshape(-1, image.shape[1]*image.shape[2]).tolist()
    writer.write(f, z2list)

# import pickle as p
# with open('test.pkl', 'wb') as this_file:
#     p.dump(save_image, this_file, protocol=p.HIGHEST_PROTOCOL)




