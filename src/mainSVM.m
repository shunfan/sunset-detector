%% Set up

clc;

if exist('features.mat', 'file') == 0
    rootDir = '../images/';
    trainDir = [rootDir 'train'];
    validateDir = [rootDir 'validate'];
    testDir = [rootDir 'test'];

    trainImages = imageDatastore(...
        trainDir, ...
        'IncludeSubfolders',true, ...
        'LabelSource', 'foldernames');
    validateImages = imageDatastore(...
        validateDir, ...
        'IncludeSubfolders',true, ...
        'LabelSource', 'foldernames');
    testImages = imageDatastore(...
        testDir, ...
        'IncludeSubfolders',true, ...
        'LabelSource', 'foldernames');

    fprintf('Read images into datastores\n');

    xTrain = imageDatastoreReader(trainImages);
    yTrain = trainImages.Labels;
    xValidate = imageDatastoreReader(validateImages);
    yValidate = validateImages.Labels;
    xTest = imageDatastoreReader(testImages);
    yTest = testImages.Labels;

    save('features.mat', ...
         'xTrain', 'yTrain', ...
         'xValidate', 'yValidate', ...
         'xTest', 'yTest');
else
    load('features.mat');
end

%% Train and evaluate an SVM

if exist('tuning.mat', 'file') == 0
    maxKernalScale = 16;
    maxBoxConstraint = 50;
    accuracy = zeros(1, maxKernalScale * maxBoxConstraint);
    numOfSupportVectors = zeros(1, maxKernalScale * maxBoxConstraint);

    for kernalScale = 1:maxKernalScale
        display(['Training using kernal scale = ', num2str(kernalScale)]);
        for boxConstraint = 1:maxBoxConstraint
            net = fitcsvm(xTrain, yTrain, ...
                          'Standardize',true, ...
                          'KernelFunction', 'rbf', ...
                          'KernelScale', kernalScale, ...
                          'BoxConstraint', boxConstraint);
            [detectedClasses, distances] = predict(net, xValidate);

            trueMatches = 0;
            for i = 1:length(yValidate)
                if detectedClasses(i) == yValidate(i)
                    trueMatches = trueMatches + 1;
                end
            end
            index = (kernalScale - 1) * maxBoxConstraint + boxConstraint;
            accuracy(index) = trueMatches / size(yValidate, 1);
            numOfSupportVectors(index) = size(net.SupportVectorLabels, 1);
        end
    end
    
    save('tuning.mat', ...
     'maxKernalScale', 'maxBoxConstraint', ...
     'accuracy', 'numOfSupportVectors');
else
    load('tuning.mat');
end

maxAccuracyIndices = find(accuracy == max(accuracy));
candidates = numOfSupportVectors(maxAccuracyIndices);
minCandidatesIndices = find(candidates == ...
                         min(candidates));
maxIndexSelected = maxAccuracyIndices(minCandidatesIndices(1));
% Box Constraint Selected
boxConstraintSelected = mod(maxIndexSelected, maxBoxConstraint);
% Kernal Scale Selected
kernalScaleSelected = ceil(maxIndexSelected/maxBoxConstraint);

figure
hold on;
title('Attempt Index vs. Accuracy & Number of Support Vectors', 'fontSize', 18);
xlabel('Attempt Index', 'fontWeight', 'bold');
ylabel('Accuracy (Red) & Number of Support Vectors (Green)', 'fontWeight', 'bold');
scatter(1:size(accuracy, 2), accuracy, 'red');
scatter(1:size(numOfSupportVectors, 2), ...
        numOfSupportVectors ./ size(yTrain, 1), 'green');

%% Create an ROC curve

tprList = zeros(1, 101);
fprList = zeros(1, 101);
thresholdIndex = 0;
for threshold = -5:0.1:5
    thresholdIndex = thresholdIndex + 1;
    net = fitcsvm(xTrain, yTrain, ...
                  'Standardize',true, ...
                  'KernelFunction', 'rbf', ...
                  'KernelScale', kernalScaleSelected, ...
                  'BoxConstraint', boxConstraintSelected);
    [~, scores] = predict(net, xTest);
    labels = scores(:, 2) > threshold;
    labels = categorical(labels, [0 1], {'nonsunset', 'sunset'});
    tprCounter = 0;
    fprCounter = 0;
    for i = 1:length(yTest)
        if labels(i) == 'sunset'
            if labels(i) == yTest(i)
                tprCounter = tprCounter + 1;
            else
                fprCounter = fprCounter + 1;
            end
        end
    end
    tprList(thresholdIndex) = tprCounter / ...
                              size(find(yTest=='sunset'), 1);
    fprList(thresholdIndex) = fprCounter / ...
                              size(find(yTest=='nonsunset'), 1);
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
roc(tprList, fprList);

%% Test our SVM

net = fitcsvm(xTrain, yTrain, ...
              'Standardize',true, ...
              'KernelFunction', 'rbf', ...
              'KernelScale', kernalScaleSelected, ...
              'BoxConstraint', boxConstraintSelected);
[~, scores] = predict(net, xTest);
realScores = scores(:, 2);
labels = scores(:, 2) > thresholdSelected;
labels = categorical(labels, [0 1], {'nonsunset', 'sunset'});
trueMatches = 0;

fpMax = -999;
fpMaxIndex = -1;
fpMin = 999;
fpMinIndex = -1;
tpMax = -999;
tpMaxIndex = -1;
tpMin = 999;
tpMinIndex = -1;
fnMax = -999;
fnMaxIndex = -1;
fnMin = 999;
fnMinIndex = -1;
tnMax = -999;
tnMaxIndex = -1;
tnMin = 999;
tnMinIndex = -1;
for i = 1:length(yTest)
    if labels(i) == yTest(i)
        trueMatches = trueMatches + 1;
        
        if labels(i) == 'sunset'
            if (realScores(i) > tpMax)
                tpMax = realScores(i);
                tpMaxIndex = i;
            elseif (realScores(i) < tpMin)
                tpMin = realScores(i);
                tpMinIndex = i;
            end
        else
            if (realScores(i) > tnMax)
                tnMax = realScores(i);
                tnMaxIndex = i;
            elseif (realScores(i) < tnMin)
                tnMin = realScores(i);
                tnMinIndex = i;
            end
        end
    else
        if labels(i) == 'sunset'
            if (realScores(i) > fpMax)
                fpMax = realScores(i);
                fpMaxIndex = i;
            elseif (realScores(i) < fpMin)
                fpMin = realScores(i);
                fpMinIndex = i;
            end
        else
            if (realScores(i) > fnMax)
                fnMax = realScores(i);
                fnMaxIndex = i;
            elseif (realScores(i) < fnMin)
                fnMin = realScores(i);
                fnMinIndex = i;
            end
        end
    end
end

display(['FP Max Score: ', num2str(fpMax), ...
         ', Index: ', num2str(fpMaxIndex)]);
display(['FP Min Score: ', num2str(fpMin), ...
         ', Index: ', num2str(fpMinIndex)]);
display(['TP Max Score: ', num2str(tpMax), ...
         ', Index: ', num2str(tpMaxIndex)]);
display(['TP Min Score: ', num2str(tpMin), ...
         ', Index: ', num2str(tpMinIndex)]);
display(['FN Max Score: ', num2str(fnMax), ...
         ', Index: ', num2str(fnMaxIndex)]);
display(['FN Min Score: ', num2str(fnMin), ...
         ', Index: ', num2str(fnMinIndex)]);
display(['TN Max Score: ', num2str(tnMax), ...
         ', Index: ', num2str(tnMaxIndex)]);
display(['TN Min Score: ', num2str(tnMin), ...
         ', Index: ', num2str(tnMinIndex)]);

accuracyFinal = trueMatches / size(yTest, 1);
display(['The accuracy of our SVM is ', num2str(accuracyFinal)]);
