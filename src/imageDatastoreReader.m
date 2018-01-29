function features = imageDatastoreReader(datastore)
% Example of using an image datastore.

nBlocks = 7; % 
nImages = numel(datastore.Files);

features = zeros(nImages, nBlocks * nBlocks * 6); 
row = 1;
for i = 1:nImages
    [img, fileinfo] = readimage(datastore, i);
    % fileinfo struct with filename and another field.
    fprintf('Processing %s\n', fileinfo.Filename);
    % TODO: Write and call a feature extraction here to operate on image.
    % Hint: debug this code ELSEWHERE on 1-2 images BEFORE looping over lots of them...
    featureVector = featureExtract(img, nBlocks);
    features(row,:) = featureVector;
    row = row + 1;
end
