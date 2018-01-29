%% Reference

% openExample('nnet/TransferLearningUsingAlexNetExample');

%% Set up

clc;

if exist('images224.mat', 'file') == 0
    rootDir = '../images224/';
    trainDir = [rootDir 'train'];
    validateDir = [rootDir 'validate'];
    testDir = [rootDir 'test'];

    trainImages = imageDatastore(...
        trainDir, ...
        'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames');
    validateImages = imageDatastore(...
        validateDir, ...
        'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames');
    testImages = imageDatastore(...
        testDir, ...
        'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames');

    fprintf('Read images into datastores\n');

    save('images224.mat', 'trainImages', 'validateImages', 'testImages');
else
    load('images224.mat');
end

%% Configure the Network

net = googlenet;
lgraph = layerGraph(net);
lgraph = removeLayers(lgraph, {'loss3-classifier','prob','output'});
numClasses = numel(categories(trainImages.Labels));
newLayers = [
    fullyConnectedLayer(numClasses, ...
        'Name', 'fc', ...
        'WeightLearnRateFactor', 20, ...
        'BiasLearnRateFactor', 20)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
lgraph = addLayers(lgraph,newLayers);
lgraph = connectLayers(lgraph, 'pool5-drop_7x7_s1', 'fc');

%% Train Network

miniBatchSize = 10;
numIterationsPerEpoch = floor(numel(trainImages.Labels) / miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize', miniBatchSize, ...
    'MaxEpochs', 1, ...
    'InitialLearnRate', 1e-4, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'ValidationData', validateImages, ...
    'ValidationFrequency', numIterationsPerEpoch);

if exist('netTransferGoogLeNet.mat', 'file') == 0
    netTransfer = trainNetwork(trainImages, lgraph, options);
    save('netTransferGoogLeNet.mat', 'netTransfer');
else
    load('netTransferGoogLeNet.mat');
end

%% Classify Validation Images

if exist('classifyLabelsGoogLeNet.mat', 'file') == 0
    tic;
    predictedLabels = classify(netTransfer, validateImages);
    toc;
    save('classifyLabelsGoogLeNet.mat', 'predictedLabels');
else
    load('classifyLabelsGoogLeNet.mat');
end
valLabels = validateImages.Labels;
accuracy = mean(predictedLabels == valLabels);
display(['The accuracy of our CNN on validate set is ', ...
    num2str(accuracy)]);

%% Create an ROC curve and Choose the threshold

tprList = zeros(1, 101);
fprList = zeros(1, 101);
thresholdIndex = 0;
disp('Start to classify...');
if exist('classifyScoresGoogLeNet.mat', 'file') == 0
    tic;
    [~, scores] = classify(netTransfer, testImages);
    toc;
    save('classifyScoresGoogLeNet.mat', 'scores');
else
    load('classifyScoresGoogLeNet.mat');
end
for threshold = -5:0.1:5
    display(['Threshold: ', num2str(threshold)]);
    thresholdIndex = thresholdIndex + 1;
    labels = scores(:, 2) > threshold;
    labels = categorical(labels, [0 1], {'nonsunset', 'sunset'});
    tprCounter = 0;
    fprCounter = 0;
    for i = 1:length(testImages.Labels)
        if labels(i) == 'sunset'
            if labels(i) == testImages.Labels(i)
                tprCounter = tprCounter + 1;
            else
                fprCounter = fprCounter + 1;
            end
        end
    end
    tprList(thresholdIndex) = tprCounter / ...
                              size(find(testImages.Labels=='sunset'), 1);
    fprList(thresholdIndex) = fprCounter / ...
                              size(find(testImages.Labels=='nonsunset'), 1);
end

dMin = 999;
dMinIndex = -1;
for i = 1:101
    x = fprList(i);
    y = tprList(i);
    points = [0, 1; x, y];
    d = pdist(points, 'euclidean');
    if d < dMin
        dMin = d;
        dMinIndex = i;
    end
end

% Threshold Selected
thresholdSelected = (dMinIndex - 1) * 0.1 - 5;
display(['ThresholdSelected: ', ...
    num2str(thresholdSelected)]);
roc(tprList, fprList);

%% Test our SVM

% [~, scores] = classify(netTransfer, testImages);
realScores = scores(:, 2);
labels = scores(:, 2) > thresholdSelected;
labels = categorical(labels, [0 1], {'nonsunset', 'sunset'});
trueMatches = 0;
for i = 1:length(testImages.Labels)
    if labels(i) == testImages.Labels(i)
        trueMatches = trueMatches + 1;
    end
end
accuracyFinal = trueMatches / size(testImages.Labels, 1);
display(['The accuracy of our CNN on test set is ', ...
    num2str(accuracyFinal)]);
