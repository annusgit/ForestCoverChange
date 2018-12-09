

% this file is meant for reading and converting only full images downloaded
% from sentinel, for full tile images, use the other file (google_earth_image_convert_to_png.m)

clc, clear all
%%
% this is for the german image
r = '/home/annus/Desktop/muzaffarabad_senitnel/GRANULE/L1C_T43SCU_A011771_20170923T055111/IMG_DATA/T43SCU_20170923T054631_B04.jp2';             
g = '/home/annus/Desktop/muzaffarabad_senitnel/GRANULE/L1C_T43SCU_A011771_20170923T055111/IMG_DATA/T43SCU_20170923T054631_B03.jp2';
b = '/home/annus/Desktop/muzaffarabad_senitnel/GRANULE/L1C_T43SCU_A011771_20170923T055111/IMG_DATA/T43SCU_20170923T054631_B02.jp2';

% ni = 'LC08_L1TP_151036_20160609_20170324_01_T1_sr_band5.tif';
% sh_ir2 = 'LC08_L1TP_151036_20160609_20170324_01_T1_sr_band7.tif';

red = imread(r);
green = imread(g);
blue = imread(b);

% red = histeq(imread(r));
% green = histeq(imread(g));
% blue = histeq(imread(b));
% near_ir = histeq(imread(ni));
% short_wave_ir2 = histeq(imread(sh_ir2));

%%

rgb = cat(3, red, green, blue);
imwrite(rgb, 'muzaffarabad_sentinel.png', 'BitDepth', 16);

% enhanced_veg = im2uint8(cat(3, short_wave_ir2, near_ir, green));
% imwrite(enhanced_veg, 'enhanc_veg.png');
% 
% false_color_veg = im2uint8(cat(3, near_ir, red, green));
% imwrite(false_color_veg, 'false_color_forvegetation.png');

%%
% ndvi = 'LC08_L1TP_151036_20160609_20170324_01_T1_sr_ndvi.tif';
% ndvi_img = imread(ndvi);



