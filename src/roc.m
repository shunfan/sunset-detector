function roc(truePosRate, falsePosRate)
% Plot an ROC curve for an image. 
% This is basically just an example of the plot function.
% The help documents provide nice ways of changing these parameters 
% to suit your tastes.
% I make things bold because they show up better in print.

% The next points are just shown as an example.
% You will need to calculate your own TPR and FPR, of course.
% Using more points than 10 will give you a smoother graph, too.
% truePosRate = [.5 .6 .7 .8 .85 .9 .92 .96 .98 .99];
% falsePosRate = [0.01 0.03 0.05 0.07 0.1 0.16 0.23 0.30 0.45 0.6];

% Create a new figure. You can also number it: figure(1)
figure;
% Hold on means all subsequent plot data will be overlaid on a single plot
hold on;
% Plots using a blue line (see 'help plot' for shape and color codes 
plot(falsePosRate, truePosRate, 'b-', 'LineWidth', 2);
% Overlaid with circles at the data points
plot(falsePosRate, truePosRate, 'bo', 'MarkerSize', 6, 'LineWidth', 2);

% You could repeat here with a different color/style if you made 
% an enhancement and wanted to show that it outperformed the baseline.

% Title, labels, range for axes
title('TPR vs. FPR', 'fontSize', 18); % Really. Change this title.
xlabel('False Positive Rate', 'fontWeight', 'bold');
ylabel('True Positive Rate', 'fontWeight', 'bold');
% TPR and FPR range from 0 to 1. You can change these if you want to zoom in on part of the graph.
axis([0 1 0 1]);
end
