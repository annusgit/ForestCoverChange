from plotly.subplots import make_subplots
from skimage.transform import resize
import matplotlib.image as matimg
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from skimage import io
import numpy as np
import statistics
import os

NULL_PIXEL = 0
NON_FOREST_PIXEL = 1
FOREST_PIXEL = 2
BTT_Forest_Percentages = {
    "abbottabad":   {2015: 39.25},
    "battagram":    {2015: 33.28},
    "buner":        {2015: 32.84},
    "chitral":      {2015: 9.71},
    "hangu":        {2015: 6.56},
    "haripur":      {2015: 30.64},
    "karak":        {2015: 14.85},
    "kohat":        {2015: 20.03},
    "kohistan":     {2015: 37.74},
    "lower_dir":    {2015: 20.09},
    "malakand":     {2015: 15.90},
    "mansehra":     {2015: 31.21},
    "nowshehra":    {2015: 12.20},
    "shangla":      {2015: 39.74},
    "swat":         {2015: 24.64},
    "tor_ghar":     {2015: 29.93},
    "upper_dir":    {2015: 21.08}
}


def decipher_this_array(this_path):
    global NULL_PIXEL, NON_FOREST_PIXEL, FOREST_PIXEL
    this_map = np.load(this_path)
    forest_pixel_count = (this_map == FOREST_PIXEL).sum()
    non_forest_pixel_count = (this_map == NON_FOREST_PIXEL).sum()
    return forest_pixel_count*100/(forest_pixel_count+non_forest_pixel_count), this_map


def decipher_this_png_map(this_image_path):
    map_png = io.imread(this_image_path)
    # get the red and green band for visualization of forest cover
    red_band = map_png[:,:,0]
    green_band = map_png[:,:,1]
    # get forest percentage
    forest_pixel_count = (green_band != 0).sum()
    non_forest_pixel_count = (red_band != 0).sum()
    this_map = np.zeros_like(red_band)
    this_map[green_band != 0] = FOREST_PIXEL
    this_map[red_band != 0] = NON_FOREST_PIXEL
    return forest_pixel_count * 100 / (forest_pixel_count + non_forest_pixel_count), this_map, map_png


if __name__ == "__main__":
    do_work, save_png_image, show_image, show_forest_change_trend = True, False, False, True
    all_districts = ["abbottabad", "battagram", "buner", "chitral", "hangu", "haripur", "karak", "kohat", "kohistan", "lower_dir", "malakand", "mansehra",
                     "nowshehra", "shangla", "swat", "tor_ghar", "upper_dir"]
    all_years = [2014, 2015, 2016, 2017, 2018, 2019, 2020]
    # infered_png_maps_path = "E:\\Forest Cover - Redo 2020\\Digitized_Forest_Maps_2014_2020_png\\rgb"
    # infered_png_maps_path = "E:\\Forest Cover - Redo 2020\\Digitized_Forest_Maps_2014_2020_png\\full-spectrum"
    # infered_png_maps_path = "E:\\Forest Cover - Redo 2020\\Digitized_Forest_Maps_2014_2020_png\\statistical_models\\logistic regression"
    infered_png_maps_path = "E:\\Forest Cover - Redo 2020\\Digitized_Forest_Maps_2014_2020_png\\statistical_models\\Random Forests"
    # fig = make_subplots(rows=17, cols=7)
    if do_work:
        for k, district in enumerate(all_districts):
            # fig, axs = plt.subplots(1, len(all_years))
            # fig.suptitle(f'{district}', fontsize=16)
            # col_count = 0
            for i, year in enumerate(all_years):
                forest_percentage, forest_map, map_png = decipher_this_png_map(this_image_path=os.path.join(infered_png_maps_path, "{}_{}.png".format(district,
                                                                                                                                                      year)))
                print("District: {}; Year: {}; Size: {}; Forest Percentage: {:.2f}%".format(district, year, forest_map.shape, forest_percentage))
                # fig.add_trace(go.Image(z=map_png), k + 1, i + 1)
                if year == 2015:
                    continue
                BTT_Forest_Percentages[district][year] = forest_percentage
                # visualize the maps
                # axs[col_count].imshow(forest_map)
                # axs[col_count].set_title("{} @ {:.2f}%".format(year, BTT_Forest_Percentages[district][year]))
                # axs[col_count].axis('off')
                # col_count += 1
                pass
            # mng = plt.get_current_fig_manager()
            # mng.window.state('zoomed')
            # plt.show()
            pass
    # fig.update_layout(height=17 * 200, width=1100, title_text="Logistic Regression Full-Spectrum Model - Forest Cover Change Trends")
    # fig.show()
    if show_forest_change_trend:
        fig = go.Figure()
        for district in all_districts:
            y = np.array([BTT_Forest_Percentages[district][year] for year in all_years])
            fig.add_trace(go.Scatter(x=all_years, y=y, mode='lines+markers', name=f'{district}'))
        fig.add_trace(go.Scatter(x=all_years, y=[statistics.mean([BTT_Forest_Percentages[district][year] for district in all_districts]) for year in all_years],
                                 name='Average Trend', line=dict(color='royalblue', width=4, dash='dash')))
        fig.update_layout(hovermode="x", title_text="Random Forests Full-Spectrum Model - Forest Cover Change Trend")
        fig.show()
    # images_path = 'E:\\Forest Cover - Redo 2020\\Digitized_Forest_Maps_2014_2020_png\\'
    # row_count, col_count = 0, 0
    # for district in all_districts:
    #     print(f"Adding District: {district}")
    #     fig, axs = plt.subplots(1, len(all_years))
    #     fig.suptitle(f'{district}', fontsize=16)
    #     col_count = 0
    #     for year in all_years:
    #         image = io.imread(os.path.join(images_path, f'{district}_{year}.png'))
    #         axs[col_count].imshow(image)
    #         axs[col_count].set_title("{} @ {:.2f}%".format(year, BTT_Forest_Percentages[district][year]))
    #         axs[col_count].axis('off')
    #         col_count += 1
    #     mng = plt.get_current_fig_manager()
    #     mng.window.state('zoomed')
    #     plt.show()
    pass
