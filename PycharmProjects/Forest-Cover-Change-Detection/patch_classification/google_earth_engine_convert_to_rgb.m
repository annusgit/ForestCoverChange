

% for converting images downloaded via google-earth engine

clc, clear all
%%
% this is for the german image
file_path = '/home/annus/Desktop/german_sentinel_series/g_6.tif';
full_image = imread(file_path);
bgr = full_image(:,:,2:4);
bgr = im2uint8(bgr*255/4096); 
rgb = bgr; cat(3, bgr(:,:,3),bgr(:,:,2),bgr(:,:,1));
disp(max(max(rgb)));
% disp(rgb);

%%

% imshow(rgb);
imwrite(rgb, 'g6.png', 'BitDepth', 16);






