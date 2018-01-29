function [stdR] = std2(vector)
    % Source: https://stackoverflow.com/a/26236431
    % Input: single or double vector
    stdR = std(reshape(vector, 1, [])); 
end
