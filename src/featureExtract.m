function [featureVector] = featureExtract(img, nBlocks)
    img = double(img);
    R = img(:, :, 1);
    G = img(:, :, 2);
    B = img(:, :, 3);
    L = R + G + B;
    S = R - B;
    T = R - 2 * G + B;

    [row, col, ~] = size(img);

    rowChunk = floor(row / nBlocks);
    rowSplitSizes = zeros(1, nBlocks);
    rowSplitSizes(:) = rowChunk;

    colChunk = floor(col / nBlocks);
    colSplitSizes = zeros(1, nBlocks);
    colSplitSizes(:) = colChunk;

    featureVector = zeros(1, 294);
    currentFeature = 1;
    currentRow = 1;
    currentCol = 1;
    for i = 1:nBlocks
        for j = 1:nBlocks
            LChunk = L(currentRow:currentRow + rowSplitSizes(i) - 1, ...
                currentCol:currentCol + colSplitSizes(j) - 1);
            LChunkMean = mean2(LChunk);
            LChunkSTD = std2(LChunk);

            SChunk = S(currentRow:currentRow + rowSplitSizes(i) - 1, ...
                currentCol:currentCol + colSplitSizes(j) - 1);
            SChunkMean = mean2(SChunk);
            SChunkSTD = std2(SChunk);

            TChunk = T(currentRow:currentRow + rowSplitSizes(i) - 1, ...
                currentCol:currentCol + colSplitSizes(j) - 1);
            TChunkMean = mean2(TChunk);
            TChunkSTD = std2(TChunk);

            featureVector(currentFeature * 6 - 5) = LChunkMean;
            featureVector(currentFeature * 6 - 4) = LChunkSTD;
            featureVector(currentFeature * 6 - 3) = SChunkMean;
            featureVector(currentFeature * 6 - 2) = SChunkSTD;
            featureVector(currentFeature * 6 - 1) = TChunkMean;
            featureVector(currentFeature * 6) = TChunkSTD;
            
            currentFeature = currentFeature + 1;
            currentCol = currentCol + colSplitSizes(j);
        end

        currentCol = 1;
        currentRow = currentRow + rowSplitSizes(i);
    end
end
