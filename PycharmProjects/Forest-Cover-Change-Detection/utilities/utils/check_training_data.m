

clc, clear all
%%
base = '/home/annus/Desktop/forest_cover_change/eurosat/images/tif/Highway/';
full_image = strcat(base, 'Highway_1.tif');
image = imread(full_image);
red = image(:,:,4);
green = image(:,:,3);
blue = image(:,:,2);

rgb = cat(3, red, green, blue);
imshow(rgb)
% imwrite(rgb, 'rgb.png');




















