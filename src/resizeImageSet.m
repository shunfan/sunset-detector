% Reference:
% https://www.mathworks.com/matlabcentral/answers/
% 314902-how-to-store-resize-images-into-new-directory#answer_297261
srcFiles = dir('../images/*/*/*.jpg');
for i = 1 : length(srcFiles)
    filename = strcat(srcFiles(i).folder, '/', srcFiles(i).name);
    image = imread(filename);
    k = imresize(image, [224, 224]);
    newFilename = strcat( ...
        strrep(srcFiles(i).folder, 'images', 'images_resized2'), ...
            '/', ...
            srcFiles(i).name);
    imwrite(k, newFilename, 'jpg');
end
