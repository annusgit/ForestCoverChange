# from main import get_clipped_image
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import gdal
import os

all_districts = ["abbottabad", "battagram", "buner", "chitral", "hangu", "haripur", "karak", "kohat", "kohistan", "lower_dir", "malakand", "mansehra",
                 "nowshehra", "shangla", "swat", "tor_ghar", "upper_dir"]

# local PC
# shapefile_masks_path = "E:\\Forest Cover - Redo 2020\\Shapefiles_clipped_to_district_vectors\\"
# raster_masks_path = "E:\\Forest Cover - Redo 2020\\Google Cloud - Training\\Training Data\\District_Shapefiles_as_Clipping_bands\\"
# images_path = "E:\\Forest Cover - Redo 2020\\Google Cloud - Training\\Training Data\\30m_4326_btt_2020_unclipped_images\\"

# remote GCP
shapefile_masks_path = "/home/azulfiqar_bee15seecs/btt_shapefiles/"
images_path = "/home/azulfiqar_bee15seecs/BTT_2014_2020_unclipped_images/"
destination_path = "/home/azulfiqar_bee15seecs/BTT_2014_2020_clipped_images/"
"""
    gdalwarp -of GTiff -tr 0.00026949458523585804 -0.00026949458523585566 -tap -cutline "E:\\Forest Cover - Redo 2020\\Shapefiles_clipped_to_district_vectors\\shangla.shp" -cl shangla -crop_to_cutline "E:\\Forest Cover - Redo 2020\\Google Cloud - Training\\Training Data\\30m_4326_btt_2020_unclipped_images\\landsat8_4326_30_2015_region_shangla.tif" C:/Users/annus/AppData/Local/Temp/processing_DHAjdp/93bb4beb509b4ad99595d252f8886cc2/OUTPUT.tif
"""


# def mask_this_district(district):
#     global all_districts, shapefile_masks_path, raster_masks_path, images_path
#     # read and prepare mask
#     this_shapefile_path = os.path.join(raster_masks_path, f"{district}_shapefile.tif")
#     ds = gdal.Open(this_shapefile_path)
#     assert ds.RasterCount == 1
#     shapefile_mask = ds.GetRasterBand(1).ReadAsArray()
#     # print(f"Shape: {shapefile_mask.shape}")
#     # read and prepare image
#     this_image_path = os.path.join(images_path, f"landsat8_4326_30_2015_region_{district}.tif")
#     ds = gdal.Open(this_image_path)
#     # mask the actual image
#     masked_bands = list()
#     for x in range(1, ds.RasterCount+1):
#         this_band = ds.GetRasterBand(x).ReadAsArray()
#         try:
#             assert this_band.shape == shapefile_mask.shape
#             masked_bands.append(np.multiply(this_band, shapefile_mask))
#         except AssertionError:
#             print(f"(ERROR): Fault in {district}. {this_band.shape} vs. {shapefile_mask.shape}")
#             return None
#     ds = None
#     return np.dstack(masked_bands)


def gdal_warp_masking(district, shapefile_path, image_path, destination_path):
    subprocess.call('sudo gdalwarp -of GTiff -tr 0.00026949458523585804 -0.00026949458523585566 -tap -cutline "{}" -cl "{}" -crop_to_cutline "{}" "{}"'
                    .format(shapefile_path, district, image_path, destination_path), shell=True)
    pass


if __name__ == "__main__":
    for year in [2014, 2016, 2017, 2018, 2019, 2020]:
        for district in all_districts:
            shapefile_path = os.path.join(shapefile_masks_path, '{}.shp'.format(district))
            image_path = os.path.join(images_path, 'landsat8_4326_30_{}_region_{}.tif'.format(year, district))
            dest_path = os.path.join(destination_path, 'clipped_{}_{}.tif'.format(district, year))
            print("Working with: ")
            print("\t {}\n\t {}\n\t {}".format(shapefile_path, image_path, dest_path))
            gdal_warp_masking(district=district, shapefile_path=shapefile_path, image_path=image_path, destination_path=dest_path)
            # streched_image = get_clipped_image(this_path=f"clipped_images\\clipped_{district}.tif")
            # # show the masked image
            # if streched_image is not None:
            #     plt.imshow(streched_image)
            #     plt.show()
            pass
        pass
    pass
