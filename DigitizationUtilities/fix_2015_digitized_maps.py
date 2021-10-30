from skimage.transform import resize
import matplotlib.image as matimg
import matplotlib.pyplot as plt
import numpy as np
import gdal
import os

all_districts = ["abbottabad", "battagram", "buner", "chitral", "hangu", "haripur", "karak", "kohat", "kohistan", "lower_dir", "malakand", "mansehra",
                 "nowshehra", "shangla", "swat", "tor_ghar", "upper_dir"]
rasterized_shapefiles_path = "E:\\Forest Cover - Redo 2020\\Google Cloud - Training\\Training Data\\District_Shapefiles_as_Clipping_bands\\"
unclipped_maps_path = "E:\\Forest Cover - Redo 2020\\Georeferenced_maps_clipped_using_shapefile_rasters\\"


def fix_2015_image_sizes():
    global all_districts, rasterized_shapefiles_path
    for district in all_districts:
        this_shapefile_path = os.path.join(rasterized_shapefiles_path, f"{district}_shapefile.tif")
        ds = gdal.Open(this_shapefile_path)
        assert ds.RasterCount == 1
        shapefile_mask = np.array(ds.GetRasterBand(1).ReadAsArray(), dtype=np.uint8)
        map_ds = gdal.Open(os.path.join(unclipped_maps_path, f"{district}.tif"))
        unclipped_map_rgb = list()
        print("{}: Shapefile Size: {}".format(district, shapefile_mask.shape))
        for x in range(1, map_ds.RasterCount + 1):
            this_band = map_ds.GetRasterBand(x).ReadAsArray()
            print("{}: Band-{} Size: {}".format(district, x, this_band.shape))
            unclipped_map_rgb.append(np.multiply(resize(this_band, shapefile_mask.shape), shapefile_mask))
        x_prev, y_prev = unclipped_map_rgb[0].shape
        x_fixed, y_fixed = int(128 * np.ceil(x_prev / 128)), int(128 * np.ceil(y_prev / 128))
        diff_x, diff_y = x_fixed - x_prev, y_fixed - y_prev
        diff_x_before, diff_y_before = diff_x // 2, diff_y // 2
        clipped_map_rgb = [np.pad(x, [(diff_x_before, diff_x - diff_x_before), (diff_y_before, diff_y - diff_y_before)], mode='constant')
                           for x in unclipped_map_rgb]
        clipped_map_rgb_stacked_image = np.dstack(clipped_map_rgb)
        matimg.imsave(f'Resized_Clipped_Georeferenced_Maps/{district}_2015.png', clipped_map_rgb_stacked_image)
        # plt.title(f"District: {district}; Shape: {clipped_map_rgb_stacked_image.shape}")
        # plt.imshow(clipped_map_rgb_stacked_image[:, :, :])
        # plt.show()
    pass


if __name__ == "__main__":
    fix_2015_image_sizes()