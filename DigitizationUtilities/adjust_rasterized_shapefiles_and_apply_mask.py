from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
import gdal
import os

all_districts = ["abbottabad", "battagram", "buner", "chitral", "hangu", "haripur", "karak", "kohat", "kohistan", "lower_dir", "malakand", "mansehra",
                 "nowshehra", "shangla", "swat", "tor_ghar", "upper_dir"]


def visualize_rasterized_shapefiles():
    global all_districts, rasterized_shapefiles_path
    for district in all_districts:
        this_shapefile_path = os.path.join(rasterized_shapefiles_path, f"{district}_shapefile.tif")
        ds = gdal.Open(this_shapefile_path)
        assert ds.RasterCount == 1
        shapefile_mask = ds.GetRasterBand(1).ReadAsArray()
        plt.title(f"District: {district}; Shape: {shapefile_mask.shape}")
        plt.imshow(shapefile_mask)
        plt.show()
    pass


def mask_raster_using_shapefiles(data_path, shapefile_path, do_resize):
    ds = gdal.Open(shapefile_path)
    assert ds.RasterCount == 1
    shapefile_mask = np.array(ds.GetRasterBand(1).ReadAsArray(), dtype=np.float)  # we convert to float so that small float values may survive
    image_ds = gdal.Open(data_path)
    clipped_full_spectrum = list()
    # print("{}: Input Shapefile Size: {}".format(district, shapefile_mask.shape))
    for x in range(1, image_ds.RasterCount + 1):
        this_band = np.nan_to_num(image_ds.GetRasterBand(x).ReadAsArray())
        # print("{}: Input Band-{} Size: {}".format(district, x, this_band.shape))
        if do_resize:
            clipped_full_spectrum.append(np.multiply(resize(this_band, shapefile_mask.shape, order=0), shapefile_mask))
        else:
            clipped_full_spectrum.append(np.multiply(this_band, shapefile_mask))
    x_prev, y_prev = clipped_full_spectrum[0].shape
    x_fixed, y_fixed = int(128 * np.ceil(x_prev / 128)), int(128 * np.ceil(y_prev / 128))
    diff_x, diff_y = x_fixed - x_prev, y_fixed - y_prev
    diff_x_before, diff_y_before = diff_x//2, diff_y//2
    clipped_full_spectrum_resized = [np.pad(x, [(diff_x_before, diff_x-diff_x_before), (diff_y_before, diff_y-diff_y_before)], mode='constant')
                                     for x in clipped_full_spectrum]
    return clipped_full_spectrum_resized


if __name__ == "__main__":
    # visualize_rasterized_shapefiles()
    # mask the images first, then mask the label, then visualize and verify
    image_data_path = "E:\\Forest Cover - Redo 2020\\Google Cloud - Training\\Training Data\\30m_4326_btt_2020_unclipped_images\\"
    label_data_path = "E:\\Forest Cover - Redo 2020\\Google Cloud - Training\\Training Data\\Clipped dataset\\GroundTruth\\"
    shapefile_path = "E:\\Forest Cover - Redo 2020\\Google Cloud - Training\\Training Data\\District_Shapefiles_as_Clipping_bands\\"
    destination_path = "E:\\Forest Cover - Redo 2020\\Google Cloud - Training\\Training Data\\Big Test\\"
    for district in all_districts:
        # 1. do the image
        image_clipped_resized = mask_raster_using_shapefiles(data_path=os.path.join(image_data_path, f"landsat8_4326_30_2015_region_{district}.tif"),
                                                             shapefile_path=os.path.join(shapefile_path, f"{district}_shapefile.tif"), do_resize=False)
        rows, cols = image_clipped_resized[0].shape
        # write the new image
        driver = gdal.GetDriverByName("GTiff")
        modified_image_save_path = os.path.join(destination_path, f"clipped_landsat8_4326_30_2015_region_{district}.tif")
        driver_man = driver.Create(modified_image_save_path, rows, cols, len(image_clipped_resized), gdal.GDT_Float32)
        try:
            for idx, band in enumerate(image_clipped_resized):
                driver_man.GetRasterBand(idx+1).WriteArray(image_clipped_resized[idx])
        except ValueError:
            print("(ERROR): Array Size Error {} vs ({}, {}, {})".format(image_clipped_resized[0].shape, rows, cols, len(image_clipped_resized)))
        driver_man.FlushCache()  # saves to disk!!
        driver, driver_man = None, None
        print("(LOG): Saved {} to disk".format(modified_image_save_path))
        # visualize
        f, (ax1, ax2) = plt.subplots(1, 2)
        ds = gdal.Open(modified_image_save_path)
        assert ds.RasterCount == 11
        rgb_image = np.dstack([255*ds.GetRasterBand(x).ReadAsArray() for x in [4,3,2]]).astype(dtype=np.uint8)
        print("(LOG): {} Modified Image Size: {}".format(district, rgb_image.shape))
        ax1.imshow(rgb_image)
        ax1.set_title('{}-{}'.format(district, rgb_image.shape))
        # 2. do the label
        label_clipped_resized = mask_raster_using_shapefiles(data_path=os.path.join(label_data_path, f"{district}_2015.tif"),
                                                             shapefile_path=os.path.join(shapefile_path, f"{district}_shapefile.tif"), do_resize=True)
        rows, cols = label_clipped_resized[0].shape
        # write the new label
        driver = gdal.GetDriverByName("GTiff")
        modified_label_save_path = os.path.join(destination_path, f"clipped_label_{district}.tif")
        driver_man = driver.Create(modified_label_save_path, rows, cols, len(label_clipped_resized), gdal.GDT_Byte)
        try:
            for idx, band in enumerate(label_clipped_resized):
                driver_man.GetRasterBand(idx + 1).WriteArray(label_clipped_resized[idx])
        except ValueError:
            print("(ERROR): Array Size Error {} vs ({}, {}, {})".format(label_clipped_resized[0].shape, rows, cols, len(label_clipped_resized)))
        driver_man.FlushCache()  # saves to disk!!
        driver, driver_man = None, None
        print("(LOG): Saved {} to disk".format(modified_label_save_path))
        # visualize
        ds = gdal.Open(modified_label_save_path)
        assert ds.RasterCount == 1
        label_image = np.array(ds.GetRasterBand(1).ReadAsArray()).astype(dtype=np.uint8)
        print("(LOG): {} Modified Label Size: {}".format(district, label_image.shape))
        ax2.imshow(label_image)
        ax2.set_title('{}-{}'.format(district, label_image.shape))
        # plt.title(f"District: {district}; Shape: {clipped_full_spectrum_stacked_image.shape}")
        # plt.imshow(clipped_full_spectrum_stacked_image[:,:,[3,2,1]])
        plt.show()
