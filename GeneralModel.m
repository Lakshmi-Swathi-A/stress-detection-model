%% Subject Independent multiclass classification model using GSR dataset
clear all;
clc;

%% Loading Galvanic Skin Response (GSR)/ Electrodermal Activity (EDA) data and labels from WESAD dataset

WESADdata_mat = dir(fullfile('/Users/swathiakkiraju/Desktop/Capstone Project/MatlabCode/data_matfiles', '*.mat'));

labels = cell(1, numel(WESADdata_mat))';
Wrist_EDA_data = cell(1, numel(WESADdata_mat))';

for i = 1:numel(WESADdata_mat)
    data = load(fullfile('/Users/swathiakkiraju/Desktop/Capstone Project/MatlabCode/data_matfiles', WESADdata_mat(i).name));
    
    labels{i} = data.pickle_data.label';
    Wrist_EDA_data{i} = data.pickle_data.signal.wrist.EDA;
end

%% Downsample the label data from 700 Hz to 4 Hz

downsampled_labels = cell(1, numel(WESADdata_mat))';

for i = 1:numel(WESADdata_mat)
    downsampled_labels{i} = downsample(labels{i}, 175);
end

%% Combining the wrist EDA data and the labels 

Combine_GSR = cell(1, numel(WESADdata_mat))';

for i = 1:numel(WESADdata_mat)
    data_combine = Wrist_EDA_data{i};
    downsampled_label_combine = downsampled_labels{i};

    Combine_GSR{i} = table(data_combine, downsampled_label_combine);
end

%% Applying Moving Median Filter and Moving Mean Filter 

for i = 1:length(Combine_GSR)

    data = Combine_GSR{i};
    gsr_values = data.data_combine;

    data.median_filter = medfilt1(gsr_values, 50); % moving median filter with a window size = 50 
    data.mean_filter = movmean(data.median_filter, 100); % moving mean filter with a window size = 100 samples to the filtered data
    Combine_GSR{i} = data;
end

%% Plot of raw, filtered and smoothed GSR data 

data = Combine_GSR{1};
time = (0:length(data.data_combine)-1) / 4;  % Sampling frequency = 4Hz

figure; % plot of raw gsr signal from 1st subject data
subplot(3, 1, 1);
plot(time, data.data_combine, 'k','LineWidth',1); 
xlabel('Time (seconds)','FontSize',12);
ylabel('GSR (microsiemens)','FontSize',12);
xlim([1000 5000]);
ylim([0 2]);

subplot(3, 1, 2);  % plot of GSR signal post median filtering 
plot(time, data.median_filter, 'k','LineWidth',1);
xlabel('Time (seconds)','FontSize',12);
ylabel('GSR (microsiemens)','FontSize',12);
xlim([1000 5000]);
ylim([0 2]);

subplot(3, 1, 3); % plot of GSR signal post mean filtering implementation
plot(time, data.mean_filter, 'k','LineWidth',1);
xlabel('Time (seconds)','FontSize',12);
ylabel('GSR (microsiemens)','FontSize',12);
xlim([1000 5000]);
ylim([0 2]);

%% Plot of GSR components

Filtered_gsr = Combine_GSR{1}.median_filter;
time = (0:length(Filtered_gsr)-1) / 4;

[r, ~, t, ~, ~, ~, ~] = cvxEDA(Filtered_gsr, 1/4); %decomposition of GSR into phasic and tonic components

figure; %original filtered GSR from 1st subject data
subplot(3, 1, 1);
plot(time, Filtered_gsr, 'k', 'LineWidth', 1);
xlabel('Time (seconds)', 'FontSize', 12);
ylabel('y (microsiemens)', 'FontSize', 12);
xlim([1600 3000]);

subplot(3, 1, 2);% plot phasic component of GSR segment
plot(time, r, 'k', 'LineWidth', 1);
xlabel('Time (seconds)', 'FontSize', 12);
ylabel('p (microsiemens)', 'FontSize', 12);
xlim([1600 3000]);

subplot(3, 1, 3);% plot of tonic component of GSR segment
plot(time, t, 'k', 'LineWidth', 1);
xlabel('Time (seconds)', 'FontSize', 12);
ylabel('t (microsiemens)', 'FontSize', 12);
xlim([1600 3000]);

%% Segmenting the filtered GSR data using a fixed window size of 60 seconds with 50% overlap under each label.  

L = [1, 2, 3, 4]; % labels

GSR_segment_length = 60 * 4; % window size x sampling frequency
step = GSR_segment_length * (1 - 0.5); % 50% overlap

Segmented_data = cell(size(Combine_GSR));
Segmented_labels = cell(size(Combine_GSR));


for i = 1:numel(Combine_GSR)
    data = Combine_GSR{i};
    Segmented_data{i} = cell(numel(L), 1);
    Segmented_labels{i} = cell(numel(L), 1);
    
    for j = 1:numel(L)
        label = L(j);
        loc = data.downsampled_label_combine == label; % Find the location of the data for each label
        wrist_EDA = data.mean_filter(loc);
        
        numofsegments = floor((length(wrist_EDA) - GSR_segment_length) / step) + 1; %calculating total segments under each label according to length and overlap 
        Segmented_data{i}{j} = cell(numofsegments, 1);
        Segmented_labels{i}{j} = cell(numofsegments, 1);
        
        for neig = 1:numofsegments
            % start and end points of the segment
            start_point = (neig - 1) * step + 1;
            end_point = start_point + GSR_segment_length - 1;
            
            Segmented_data{i}{j}{neig} = wrist_EDA(start_point:end_point);
            Segmented_labels{i}{j}{neig} = label;
        end
    end
end

%% Removing any GSR segments with same values and its label before classification

for i = 1:numel(Segmented_data)
    for j = 1:numel(Segmented_data{i})
        x = {};
        y = {};
        segments = Segmented_data{i}{j};
        labels = Segmented_labels{i}{j};
        for neig = 1:numel(segments)
            segment = segments{neig};
            if length(unique(segment)) ~= 1 % keeping only the segments whose values are not same
                x{end+1} = segment;
                y{end+1} = labels{neig};
            end
        end
        Segmented_data{i}{j} = x;
        Segmented_labels{i}{j} = y;
    end
end

%% Plot of Peak Analysis

segment_data = Segmented_data{2,1}{2,1}{1,9}; % Second subject's ninth stress segment
time_segment = (0:length(segment_data)-1) / 4;

[r, ~, ~, ~, ~, ~, ~] = cvxEDA(segment_data, 1/4); % extracting phasic component of GSR data segment

minimum_height = 0.01; % Minimum height to consider a  peak in microsiemens
minimum_distance = 10; % Minimum number of samples between peaks
[peaks, locs] = findpeaks(r, 'MinPeakHeight', minimum_height, 'MinPeakDistance', minimum_distance);%local maxima

figure; 
subplot(1, 1, 1); 
plot(time_segment, r, 'k', 'LineWidth', 1);
hold on;
scatter(time_segment(locs), peaks, 'ro','filled');
hold off;
xlabel('Time (seconds)', 'FontSize', 12);
ylabel('Phasic Component (microsiemens)', 'FontSize', 12);

%% Applying Feature Extraction

samplingrate=4;

for i = 1:numel(Segmented_data)
    for j = 1:numel(Segmented_data{i})
        for neig = 1:numel(Segmented_data{i}{j})
            dataSegment = Segmented_data{i}{j}{neig};
            extractedfeatures = GSR_FE(dataSegment, 4);
            Segmented_data{i}{j}{neig} = extractedfeatures;
        end
    end
end

%% Removing feature segments with nan values

for i = 1:numel(Segmented_data)
    for j = 1:numel(Segmented_data{i})
        without_nan_seg = {};
        without_nan_lab = {};
        for neig = 1:numel(Segmented_data{i}{j})
            extractedfeatures = Segmented_data{i}{j}{neig};
            if ~any(isnan(extractedfeatures))
                without_nan_seg{end+1} = extractedfeatures;
                without_nan_lab{end+1} = Segmented_labels{i}{j}{neig};
            end
        end
        Segmented_data{i}{j} = without_nan_seg;
        Segmented_labels{i}{j} = without_nan_lab;
    end
end

%% Converting labels from numerical to string.

str_lab = containers.Map([1, 2, 3, 4], {'baseline', 'stress', 'amusement', 'meditation'});

for i = 1:numel(Segmented_labels)
    for j = 1:numel(Segmented_labels{i})
        for neig = 1:numel(Segmented_labels{i}{j})
            num = Segmented_labels{i}{j}{neig};
            str = str_lab(num);
            Segmented_labels{i}{j}{neig} = str;
        end
    end
end

%% Combining the feature segments and their corresponding labels

combinedfeatures = {};
combinedlabels = {};

for i = 1:numel(Segmented_data)
    for j = 1:numel(Segmented_data{i})
        for neig = 1:numel(Segmented_data{i}{j})
            dataSegment = Segmented_data{i}{j}{neig};
            labelSegment = Segmented_labels{i}{j}{neig};
            combinedfeatures = [combinedfeatures; dataSegment];
            combinedlabels = categorical([combinedlabels; labelSegment]);
        end
    end
end

%% Dataset Splitting

setPartitions = cvpartition(combinedlabels, 'HoldOut', 0.3, 'Stratify', true); 
% Training set 70%
Train_fe = combinedfeatures(setPartitions.training, :);
Train_lab = combinedlabels(setPartitions.training);

testPartitions = cvpartition(setPartitions.test, 'HoldOut', 0.5, 'Stratify', true); 
% Validation set 15%
Validation_fe = combinedfeatures(testPartitions.training, :);
Validation_lab = combinedlabels(testPartitions.training);

% Test set 15%
Test_fe = combinedfeatures(testPartitions.test, :);
Test_lab = combinedlabels(testPartitions.test);


%% Padding the feature vectors and converting them from cell array to matrix

max_l = max(cellfun(@(c) numel(c), Train_fe));
max_lv = max(cellfun(@(c) numel(c), Validation_fe));
max_lt = max(cellfun(@(c) numel(c), Test_fe));
% Pad each feature vector with zeros to ensure they are all the same length
Train_fe_p = cellfun(@(c) padarray(c, [0, max_l - numel(c)], 'post'), Train_fe, 'UniformOutput', false);
Validation_fe_p = cellfun(@(c) padarray(c, [0, max_lv - numel(c)], 'post'), Validation_fe, 'UniformOutput', false);
Test_fe_p = cellfun(@(c) padarray(c, [0, max_lt - numel(c)], 'post'), Test_fe, 'UniformOutput', false);

% Converting the padded features cell array to a matrix
Train_mtx = cell2mat(Train_fe_p);
Validation_mtx = cell2mat(Validation_fe_p);
Test_mtx = cell2mat(Test_fe_p);


%% SVM Model

KT= {'linear', 'rbf', 'polynomial'}; %kernel type
BC = [0.1, 1, 10]; %box connstraint 
KS = [1, 5, 10]; % kernel scale

svm_validation_accuracy = 0;
svm_model = [];
best_kernel = ''; 
best_bc = 0;   
best_ks = 0;     

for kernel_type = 1:length(KT)
    kt = KT{kernel_type};
    for bc = BC
        for ks = KS

            t = templateSVM('KernelFunction', kt, 'BoxConstraint', bc, 'KernelScale', ks);
            svm_mod = fitcecoc(Train_mtx, Train_lab, 'Learners', t); 

            svm_val = predict(svm_mod, Validation_mtx); 
            svm_val_accuracy = sum(svm_val == Validation_lab) / numel(Validation_lab) * 100;

            if svm_val_accuracy > svm_validation_accuracy %selecting best model config based on validation accuracy
                svm_validation_accuracy = svm_val_accuracy; %validation accuracy
                svm_model = svm_mod;
                best_kernel = kt;
                best_bc = bc;
                best_ks = ks;
            end
        end
    end
end

svm_test = predict(svm_model, Test_mtx); % testing the predictive performance of selected SVM model config on test set
svm_test_accuracy = sum(svm_test == Test_lab) / numel(Test_lab) * 100; %testing accuracy

svm_confusion_mtx = confusionmat(Test_lab, svm_test); %confusion matrix
svm_p = diag(svm_confusion_mtx) ./ sum(svm_confusion_mtx, 2); %precision
svm_r = diag(svm_confusion_mtx) ./ sum(svm_confusion_mtx, 1)'; %recall
svm_f1 = 2 * (svm_p .* svm_r) ./ (svm_p + svm_r); %F1 score
svm_f1_all = sum(svm_f1 .* sum(svm_confusion_mtx, 2)) / sum(svm_confusion_mtx(:)); %weighted f1

fprintf('SVM validation accuracy: %.2f%%\n', svm_validation_accuracy);
fprintf('Best kernel: %s\n', best_kernel);
fprintf('Best boxconstraint: %.2f\n', best_bc);
fprintf('Best kernel scale: %.2f\n', best_ks);
fprintf('SVM Test Accuracy: %.2f%%\n', svm_test_accuracy);
fprintf('\nConfusion matrix:\n');
disp(svm_confusion_mtx);
fprintf('\nPrecision per label:\n');
disp(svm_p);
fprintf('\nRecall per label:\n');
disp(svm_r);
fprintf('\nF1 Score per label:\n');
disp(svm_f1);
fprintf('\nWeighted F1 Score: %.2f\n', svm_f1_all);

%% k-NN model

NN = [2,3,4, 5,6, 7]; %no of neighbors
DM = {'euclidean', 'cityblock', 'minkowski', 'cosine'}; %distance metrics

Knn_Val_accuracy = 0;
Knn_model = [];

for neig = NN
    for dist = 1:length(DM)
        dm = DM{dist};

        knn_mod = fitcknn(Train_mtx, Train_lab, 'NumNeighbors', neig, 'Distance', dm);

        knn_val = predict(knn_mod, Validation_mtx);
        knn_val_accuracy = sum(knn_val == Validation_lab) / numel(Validation_lab) * 100; % validation accuracy

        if knn_val_accuracy > Knn_Val_accuracy
            Knn_Val_accuracy = knn_val_accuracy;
            Knn_model = knn_mod;
        end
    end
end

knn_test = predict(Knn_model, Test_mtx);
knn_test_accuracy = sum(knn_test == Test_lab) / numel(Test_lab) * 100;%test accuracy

knn_confusion_mtx = confusionmat(Test_lab, knn_test);%confusion matrix
knn_p = diag(knn_confusion_mtx) ./ sum(knn_confusion_mtx, 2);%precision
knn_r = diag(knn_confusion_mtx) ./ sum(knn_confusion_mtx, 1)';%recall
knn_f1 = 2 * (knn_p .* knn_r) ./ (knn_p + knn_r);%F1 score
knn_f1_all = sum(knn_f1 .* sum(knn_confusion_mtx, 2)) / sum(knn_confusion_mtx(:));%Weighted f1 score

fprintf('k-NN validation accuracy: %.2f%%\n', Knn_Val_accuracy);
fprintf('No. of neighbors: %d\n', Knn_model.NumNeighbors);
fprintf('Distance metric: %s\n', Knn_model.Distance);
fprintf('k-NN test accuracy: %.2f%%\n', knn_test_accuracy);
fprintf('\nConfusion matrix:\n');
disp(knn_confusion_mtx);
fprintf('\nPrecision per label:\n');
disp(knn_p);
fprintf('\nRecall per label:\n');
disp(knn_r);
fprintf('\nF1 Score per label:\n');
disp(knn_f1);
fprintf('\nWeighted F1 Score: %.2f\n', knn_f1_all);

%% Decision Trees

MD = [100,115,120,125,130];  % Maximum depths
MLS = [1,2,3,4 5, 10];  % Minimum leaf sizes

dt_val_accuracy = 0;
dt_model = [];
best_md = 0;
best_mls = 0;
for md = MD
    for mls = MLS
        dt_mod = fitctree(Train_mtx, Train_lab, 'MaxNumSplits', md, 'MinLeafSize', mls);

        dt_val = predict(dt_mod, Validation_mtx);
        Dt_val_accuracy = sum(dt_val == Validation_lab) / numel(Validation_lab) * 100; %validation accuracy

        if Dt_val_accuracy > dt_val_accuracy
            dt_val_accuracy = Dt_val_accuracy;
            dt_model = dt_mod;
            best_md = md;
            best_mls = mls;
        end
    end
end

dt_test = predict(dt_model, Test_mtx);
dt_test_accuracy = sum(dt_test == Test_lab) / numel(Test_lab) * 100;%test accuracy

dt_conf_mtx = confusionmat(Test_lab, dt_test); %confusion matrix
dt_p = diag(dt_conf_mtx) ./ sum(dt_conf_mtx, 2);%precision  
dt_r = diag(dt_conf_mtx) ./ sum(dt_conf_mtx, 1)';%recall    
dt_f1 = 2 * (dt_p .* dt_r) ./ (dt_p + dt_r);%F1 score   
dt_f1_all = sum(dt_f1 .* sum(dt_conf_mtx, 2)) / sum(dt_conf_mtx(:));%Weighted f1 score

fprintf('Decision Trees validation accuracy: %.2f%%\n', dt_val_accuracy);
fprintf('Best maximum depth: %d\n', best_md);
fprintf('Best minimum leaf size: %d\n', best_mls);
fprintf('Decision Tree test accuracy: %.2f%%\n', dt_test_accuracy);
fprintf('\nConfusion Matrix:\n');
disp(dt_conf_mtx);
fprintf('\nPrecision per label:\n');
disp(dt_p);
fprintf('\nRecall per label:\n');
disp(dt_r);
fprintf('\nF1 Score per label:\n');
disp(dt_f1);
fprintf('\nWeighted F1 Score: %.2f\n', dt_f1_all);

%% Random Forest

LC = [10,20,30,40,50, 75, 100];%No. of learning cycles
MSR = [50, 100, 200,225, 250,275, 300];% maximum no. of splits  
EM = {'Bag', 'AdaBoostM2', 'TotalBoost', };%ensemble method

rf_val_accuracy = 0;
rf_model = [];
best_LC = 0;
best_MSR = 0;
best_EM = '';

for em = EM
    for lc = LC
        for msr = MSR
            z = char(em);
            template = templateTree('MaxNumSplits', msr);
            RF_mod = fitcensemble(Train_mtx, Train_lab, 'Method', z, ...
                                      'NumLearningCycles', lc,'Learners', template);

            RF_val = predict(RF_mod, Validation_mtx);
            RF_val_accuracy = sum(RF_val == Validation_lab) / numel(Validation_lab) * 100;% validation accuracy

            if RF_val_accuracy > rf_val_accuracy
                rf_val_accuracy = RF_val_accuracy;
                rf_model = RF_mod;
                best_LC = lc;
                best_MSR = msr;
                best_EM = z;
            end
        end
    end
end

rf_test = predict(rf_model, Test_mtx);
rf_test_accuracy = sum(rf_test == Test_lab) / numel(Test_lab) * 100;% testing accuracy

rf_conf_mtx= confusionmat(Test_lab, rf_test);%confusion matrix
rf_p = diag(rf_conf_mtx) ./ sum(rf_conf_mtx, 2);%precision  
rf_r = diag(rf_conf_mtx) ./ sum(rf_conf_mtx, 1)';%recall
rf_f1 = 2 * (rf_p .* rf_r) ./ (rf_p + rf_r);%f1 score   
rf_f1_all = sum(rf_f1 .* sum(rf_conf_mtx, 2)) / sum(rf_conf_mtx(:));%Weighted f1 score

fprintf('Random Forest validation accuracy: %.2f%%\n', rf_val_accuracy);
fprintf('Best ensemble method: %s\n', best_EM);
fprintf('Best number of learning cycles: %d\n', best_LC);
fprintf('Best maximum number of splits: %d\n', best_MSR);
fprintf('Random Forest test accuracy: %.2f%%\n', rf_test_accuracy);
fprintf('\nConfusion Matrix:\n');
disp(rf_conf_mtx);
fprintf('\nPrecision per label:\n');
disp(rf_p);
fprintf('\nRecall per label:\n');
disp(rf_r);
fprintf('\nF1 Score per label:\n');
disp(rf_f1);
fprintf('\nWeighted F1 Score: %.2f\n', rf_f1_all);

% 
%% Naive Bayes

D = {'kernel'}; %Kernel Naive Bayes
W = [0.05,0.1,0.15,0.2, 0.25, 0.3, 0.5, 1.0];%Kernel width

nb_val_accuray = 0;
nb_model = [];
best_W = 0;

for d = D
    for w = W
        nb_c = char(d); 
        nb_mod = fitcnb(Train_mtx, Train_lab, 'DistributionNames', nb_c, 'Width', w);

        NB_val = predict(nb_mod, Validation_mtx);
        NB_val_acuracy = sum(NB_val == Validation_lab) / numel(Validation_lab) * 100;%validation accuract

        if NB_val_acuracy > nb_val_accuray
            nb_val_accuray = NB_val_acuracy;
            nb_model = nb_mod;
            best_W = w;
        end
    end
end

NB_test = predict(nb_model, Test_mtx);
NB_test_accuracy = sum(NB_test == Test_lab) / numel(Test_lab) * 100;%test accuracy
nb_conf_matx = confusionmat(Test_lab, NB_test);%confusion matrix
nb_p = diag(nb_conf_matx) ./ sum(nb_conf_matx, 2);%precision
nb_r = diag(nb_conf_matx) ./ sum(nb_conf_matx, 1)';%recall  
nb_f1 = 2 * (nb_p .* nb_r) ./ (nb_p + nb_r);%f1 score                   
nb_f1_all = sum(nb_f1 .* sum(nb_conf_matx, 2)) / sum(nb_conf_matx(:));%Weigted f1score

fprintf('Naive Bayes validation accuracy: %.2f%%\n', nb_val_accuray);
fprintf('Best kernel width: %.2f\n', best_W);
fprintf('Naive Bayes test accuracy: %.2f%%\n', NB_test_accuracy);
fprintf('\nConfusion Matrix:\n');
disp(nb_conf_matx);
fprintf('\nPrecision per label:\n');
disp(nb_p);
fprintf('\nRecall per label:\n');
disp(nb_r);
fprintf('\nF1 Score per label:\n');
disp(nb_f1);
fprintf('\nWeighted F1 Score: %.2f\n', nb_f1_all);

%% 1D Convolutional Neural Network (1D CNN)

FS = [5,10, 15,20, 25,30 ]; %filter sizes
NF = [16,32,64,128]; %numbers of filters
LR = [0.001,0.005,0.01,0.05, 0.1]; %learning rates

features = 1;
classes = 4;
best_FS = [];
best_NF = [];
best_LR = [];
cnn_validation_accuracy = 0;
cnn_model = [];

for fs = FS
    for nf = NF
        for lr = LR
            layers = [
                sequenceInputLayer(features)
                convolution1dLayer(fs, nf, 'Padding', 'causal')
                leakyReluLayer
                layerNormalizationLayer
                globalAveragePooling1dLayer
                fullyConnectedLayer(classes)
                softmaxLayer
                classificationLayer];
            options = trainingOptions("adam", ...
                "MiniBatchSize",128,...
                'MaxEpochs', 120, ...
                'InitialLearnRate', lr, .....
                'ValidationData', {Validation_fe, Validation_lab}, ...
                'Verbose', true);

            cnn_mod = trainNetwork(Train_fe, Train_lab, layers, options);
            cnn_val = classify(cnn_mod, Validation_fe);
            cnn_val_accuracy = sum(cnn_val == Validation_lab)/numel(Validation_lab)*100;% validation accuracy

            if cnn_val_accuracy > cnn_validation_accuracy
                cnn_validation_accuracy = cnn_val_accuracy;
                best_FS = fs;
                best_NF = nf;
                best_LR = lr;
                cnn_model = cnn_mod;
            end
        end
    end
end

cnn_test = classify(cnn_model, Test_fe);
cnn_test_accuracy = sum(cnn_test == Test_lab)/numel(Test_lab)*100; %test accuracy

cnn_confusion_mtx = confusionmat(Test_lab, cnn_test);%confusion matrix
cnn_p = diag(cnn_confusion_mtx) ./ sum(cnn_confusion_mtx, 2);%precision
cnn_r = diag(cnn_confusion_mtx) ./ sum(cnn_confusion_mtx, 1)';%recall
cnn_f1 = 2 * (cnn_p .* cnn_r) ./ (cnn_p + cnn_r);%f1 score
cnn_f1_all = sum(cnn_f1_all .* sum(cnn_confusion_mtx, 2)) / sum(cnn_confusion_mtx(:));%weighted fe score

fprintf('CNN validation accuracy: %.2f%%\n', cnn_validation_accuracy);
fprintf('Best filter size: %s\n', best_FS);
fprintf('Best no. of filters: %.2f\n', best_NF);
fprintf('Best learning rate: %.2f\n', best_LR);
fprintf('CNN test accuracy: %.2f%%\n', cnn_test_accuracy);
fprintf('\nConfusion matrix:\n');
disp(cnn_confusion_mtx);
fprintf('\nPrecision per label:\n');
disp(cnn_p);
fprintf('\nRecall per label:\n');
disp(cnn_r);
fprintf('\nF1 Score per label:\n');
disp(cnn_f1);
fprintf('\nWeighted F1 Score: %.2f\n', cnn_f1_all);

%% Long Short-Term Memory(LSTM)

HU = [25,50,75,100,125]; %numbers of hidden units
LR_l = [0.001,0.005,0.01,0.1]; %learning rates

features =1;
classes =4;
best_HU = [];
best_LR_l = [];
lstm_validation_accuracy = 0;
lstm_model = [];

for hu = HU
    for lr = LR_l
        layers = [
            sequenceInputLayer(features)
            lstmLayer(hu,'OutputMode','last')
            fullyConnectedLayer(classes)
            softmaxLayer
            classificationLayer];

        options = trainingOptions("adam", ...
            "MiniBatchSize", 128, ...
            'MaxEpochs', 120, ...
            'InitialLearnRate', lr, ...
            'ValidationData', {Validation_fe, Validation_lab}, ...
            'Verbose', true);

        lstm_mod = trainNetwork(Train_fe, Train_lab, layers, options);
        lstm_val = classify(lstm_mod, Validation_fe);
        lstm_val_accuracy = sum(lstm_val == Validation_lab)/numel(Validation_lab)*100;%validation accuracy

        if lstm_val_accuracy > lstm_validation_accuracy
            lstm_validation_accuracy = lstm_val_accuracy;
            best_HU = hu;
            best_LR_l = lr;
            lstm_model = lstm_mod;
        end
    end
end

lstm_test = classify(lstm_model, Test_fe);
lstm_test_accuracy = sum(lstm_test == Test_lab)/numel(Test_lab)*100;%test accuracy

lstm_confusion_mtx = confusionmat(Test_lab, lstm_test);%confusion matrix
lstm_p = diag(lstm_confusion_mtx) ./ sum(lstm_confusion_mtx, 2);%precision
lstm_r = diag(lstm_confusion_mtx) ./ sum(lstm_confusion_mtx, 1)';%recall
lstm_f1 = 2 * (lstm_p .* lstm_r) ./ (lstm_p + lstm_r);%f1 score
lstm_f1_all = sum(lstm_f1 .* sum(lstm_confusion_mtx, 2)) / sum(lstm_confusion_mtx(:));%weighted f1 score

fprintf('LSTM validation accuracy: %.2f%%\n', lstm_validation_accuracy);
fprintf('Best number of hidden units: %d\n', best_HU);
fprintf('Best learning rate: %.2f\n', best_LR_l);
fprintf('LSTM test accuracy: %.2f%%\n', lstm_test_accuracy);
fprintf('\nConfusion Matrix:\n');
disp(lstm_confusion_mtx);
fprintf('\nPrecision per class:\n');
disp(lstm_p);
fprintf('\nRecall per class:\n');
disp(lstm_r);
fprintf('\nF1 Score per class:\n');
disp(lstm_f1);
fprintf('\nWeighted F1 Score: %.2f\n', lstm_f1_all);
