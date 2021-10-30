import matplotlib.image as matimg
import matplotlib.pyplot as plt
import numpy as np
import imageio
import gdal
import cv2
import os

forest_label, non_forest_label, null_pixel_label = 2, 1, 0


def min_max_stretch(this_image, null_mask):
    # https://theailearner.com/tag/min-max-stretching/
    min_max_stretched_bands = []
    for i in range(3):
        this_band = this_image[:,:,i]
        sorted_pixel_values = np.unique(np.sort(this_band.flatten()))
        min_pixel, max_pixel = sorted_pixel_values[1], sorted_pixel_values[-1]
        this_band_stretched = 255*(this_band-min_pixel)/(max_pixel-min_pixel)
        this_band_stretched[null_mask] = 0
        min_max_stretched_bands.append(this_band_stretched)
    return np.dstack(min_max_stretched_bands).astype(dtype=np.uint8)


def get_clipped_image(this_path):
    ds = gdal.Open(this_path)
    rgb = [ds.GetRasterBand(x).ReadAsArray() for x in [4, 3, 2]]  # 4, 3, 2 are the red, green, blue bands
    null_pixel_mask = (rgb[0] == 0)
    stacked_image = np.dstack(rgb)
    stretched_image = min_max_stretch(this_image=stacked_image, null_mask=null_pixel_mask)
    return stretched_image


def get_digitized_clipped_map(this_path, this_shape):
    # 51 is the mean value of green forest pixels
    mean_forest_value, tolerance = 51, 10
    global forest_label, non_forest_label, null_pixel_label
    clipped_map = imageio.imread(this_path)[:,:,:3]
    mean_map = np.mean(clipped_map, axis=-1).astype(np.int)
    forest_map = np.ones_like(mean_map)
    null_pixels_mask = (mean_map == 0)
    forest_pixels_mask = (mean_map > (mean_forest_value-tolerance))  # 51 is the mean value of green forest pixels
    forest_pixels_mask &= (mean_map < (mean_forest_value+tolerance))  # 51 is the mean value of green forest pixels
    forest_map[null_pixels_mask] = null_pixel_label
    forest_map[forest_pixels_mask] = forest_label
    # forest map generated at this point, resize it according to requirement
    # forest_map = cv2.resize(forest_map, dsize=this_shape, interpolation=cv2.INTER_NEAREST)
    forest_pixel_count = np.count_nonzero(forest_map == forest_label)
    non_forest_pixel_count = np.count_nonzero(forest_map == non_forest_label)
    forest_percent = forest_pixel_count*100/(forest_pixel_count+non_forest_pixel_count)
    return [clipped_map, forest_map, forest_percent]


if __name__ == "__main__":
    forest_percentages_by_districts = {
        "abbottabad":   [39.25],
        "battagram":    [33.28],
        "buner":        [32.84],
        "chitral":      [9.71],
        "hangu":        [6.56],
        "haripur":      [30.64],
        "karak":        [14.85],
        "kohat":        [20.03],
        "kohistan":     [37.74],
        "lower_dir":    [20.09],
        "malakand":     [15.90],
        "mansehra":     [31.21],
        "nowshehra":    [12.20],
        "shangla":      [39.74],
        "swat":         [24.64],
        "tor_ghar":     [29.93],
        "upper_dir":    [21.08]
    }
    forest_area_by_districts = {
        "abbottabad":   [70553],
        "battagram":    [48957],
        "buner":        [55887],
        "chitral":      [144416],
        "hangu":        [8985],
        "haripur":      [55043],
        "karak":        [39263],
        "kohat":        [59670],
        "kohistan":     [286421],
        "lower_dir":    [33371],
        "malakand":     [14925],
        "mansehra":     [129647],
        "nowshehra":    [22207],
        "shangla":      [56345],
        "swat":         [132538],
        "tor_ghar":     [13590],
        "upper_dir":    [186935]
    }
    path_1, path_2 = "E:\\Forest Cover - Redo 2020\\", "Resized_Clipped_Adjusted_Maps"
    destination_path = "E:\\Forest Cover - Redo 2020\\Resized_Clipped_Adjusted_Colored_Labelled_2015_Maps"
    all_districts = ["abbottabad", "battagram", "buner", "chitral", "hangu", "haripur", "karak", "kohat", "kohistan", "lower_dir", "malakand", "mansehra",
                     "nowshehra", "shangla", "swat", "tor_ghar", "upper_dir"]
    for district in all_districts:
        # stretched_image = get_clipped_image(this_path=os.path.join(path_1, path_2, f"{district}_image.tif"))
        clipped_map, forest_map, forest_percentage = get_digitized_clipped_map(this_path=os.path.join(path_1, path_2, f"{district}_2015.png"),
                                                                               this_shape=None)
        # output_npy_file = f"{district}_2015_npy"
        # np.save(output_npy_file, forest_map)
        # print(f"(LOG): Written {output_npy_file} to disk")
        # output_file = f"{district}_2015.tif"
        # driver = gdal.GetDriverByName("GTiff")
        # cols, rows = forest_map.shape
        # outdata = driver.Create(output_file, rows, cols, 1, gdal.GDT_UInt16)
        # outdata.SetGeoTransform(original_ds.GetGeoTransform())  # sets same geotransform as input
        # outdata.SetProjection(original_ds.GetProjection())  # sets same projection as input
        # outdata.GetRasterBand(1).WriteArray(forest_map)
        # outdata.FlushCache()  # saves to disk!!
        # print(f"(LOG): Written {output_file} to disk")
        # print(f"(LOG): Testing {output_file} from disk")
        # ds = gdal.Open(output_file)
        # result = ds.GetRasterBand(1).ReadAsArray()
        # plt.imshow(result)
        # plt.show()
        sq_meter_to_hectare = 0.0001
        land_area_of_district = int(900*np.count_nonzero(forest_map)*sq_meter_to_hectare)
        forest_area_of_district = int(0.01*forest_percentage*land_area_of_district)
        print("{}: Forest Percentage: {:.2f}% (vs {:.2f}%)".format(district, forest_percentage, forest_percentages_by_districts[district][0]))
        print("{}: Land Area: {} ha".format(district, land_area_of_district))
        print("{}: Forest Area: {} ha (vs {} ha)".format(district, forest_area_of_district, forest_area_by_districts[district][0]))
        print(f"{district} -> Shapes: ({forest_map.shape[0]}, {forest_map.shape[1]})")
        print("----------------------------------------------------------------------------------------------")
        forest_percentages_by_districts[district].append(forest_percentage)
        forest_area_by_districts[district].append(forest_area_of_district)
        f, axarr = plt.subplots(1, 2)
        # axarr[0].imshow(stretched_image)
        axarr[0].imshow(clipped_map)
        axarr[1].imshow(forest_map)
        # plt.title("{} -> Forest Percentage: {:.2f}%".format(forest_map.shape, forest_percentage))
        plt.title("{}-{}: Forest Percentage: {:.2f}% (vs {:.2f}%)".format(district, forest_map.shape, forest_percentage,
                                                                          forest_percentages_by_districts[district][0]))
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')  # works fine on Windows!
        plt.show()
        # forest_map_rband = np.zeros_like(forest_map)
        # forest_map_gband = np.zeros_like(forest_map)
        # forest_map_bband = np.zeros_like(forest_map)
        # forest_map_rband[forest_map == non_forest_label] = 255
        # forest_map_gband[forest_map == forest_label] = 255
        # forest_map_for_visualization = np.dstack([forest_map_rband, forest_map_gband, forest_map_bband]).astype(np.uint8)
        # matimg.imsave(os.path.join(destination_path, f'{district}_2015.png'), forest_map_for_visualization)
        # print('Saved: {}'.format(os.path.join(destination_path, f'{district}_2015.png')))
        pass
    # exit()
    # data to plot
    # forest_percentages_by_districts = {'hangu': [6.56, 4.385070243239075], 'chitral': [9.71, 33.42828609053335], 'nowshehra': [12.2, 12.78935115378584],
    #                                    'karak': [14.85, 14.59616342204001], 'malakand': [15.9, 15.196281390945275], 'kohat': [20.03, 15.31498685770034],
    #                                    'lower_dir': [20.09, 16.494481326224104], 'upper_dir': [21.08, 45.861086495509824], 'swat': [24.64, 22.268590285695545],
    #                                    'tor_ghar': [29.93, 26.8571455925553], 'haripur': [30.64, 26.055457059339695], 'mansehra': [31.21, 25.775518226295976],
    #                                    'buner': [32.84, 30.330167623826235], 'battagram': [33.28, 29.764684818054878], 'kohistan': [37.74, 32.25041713996332],
    #                                    'abbottabad': [39.25, 35.52242931070733], 'shangla': [39.74, 37.53508904135487]}
    # forest_percentages_by_districts = {k: v for k, v in sorted(forest_percentages_by_districts.items(), key=lambda item: item[1])}
    # print(forest_percentages_by_districts)
    n_groups = len(all_districts)
    original_percentages, digitized_percentages, district_names = list(), list(), list()
    original_areas_in_ha, digitized_areas_in_ha = list(), list()
    for x, y in forest_percentages_by_districts.items():
        original_percentages.append(y[0])
        digitized_percentages.append(y[1])
        district_names.append(x)
    forest_area_by_districts = {k: v for k, v in sorted(forest_area_by_districts.items(), key=lambda item: item[1])}
    print(forest_area_by_districts)
    for x, y in forest_area_by_districts.items():
        original_areas_in_ha.append(y[0])
        digitized_areas_in_ha.append(y[1])
    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8
    rects1 = plt.bar(index, original_percentages, bar_width, alpha=opacity, color='b', label='Original')
    rects2 = plt.bar(index + bar_width, digitized_percentages, bar_width, alpha=opacity, color='r', label='Digitized')
    plt.xlabel('Districts')
    plt.xticks(rotation=45)
    plt.ylabel('Forest Percentage %')
    plt.title('Forest Percentages (Original Vs. Digitized)')
    plt.xticks(index + bar_width, district_names)
    plt.legend()
    plt.show()
    # create plot
    fig_1, ax_1 = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8
    rects_1 = plt.bar(index, original_areas_in_ha, bar_width, alpha=opacity, color='b', label='Original')
    rects_2 = plt.bar(index + bar_width, digitized_areas_in_ha, bar_width, alpha=opacity, color='r', label='Digitized')
    plt.xlabel('Districts')
    plt.xticks(rotation=45)
    plt.ylabel('Forest Area (ha)')
    plt.title('Forest Areas (ha - Original Vs. Digitized)')
    plt.xticks(index + bar_width, district_names)
    plt.legend()
    plt.show()
    pass
