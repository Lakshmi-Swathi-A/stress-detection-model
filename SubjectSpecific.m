% Subject specific multiclass classification using Random Forest Classifier

clear all;
clc;

% Loading dataset
WESADdata_mat = dir(fullfile('/Users/swathiakkiraju/Desktop/Capstone Project/MatlabCode/data_matfiles', '*.mat'));

LC = [10,20,30,40,50, 75, 100];%No. of learning cycles
MSR = [50, 100, 200,225, 250,275, 300];% maximum no. of splits
EM =  {'Bag', 'AdaBoostM2', 'TotalBoost', };%ensemble method
rf_val_accuracy = 0;
rf_model = [];
best_LC = 0;
best_MSR = 0;
best_EM = '';

for subj = 1 % change subject index
    % Load data for the current subject
    data = load(fullfile('/Users/swathiakkiraju/Desktop/Capstone Project/MatlabCode/data_matfiles', WESADdata_mat(subj).name));
    labels = data.pickle_data.label';
    Wrist_EDA_data = data.pickle_data.signal.wrist.EDA;

    % Downsampling the label data from 700 Hz to 4 Hz
    downsampled_labels = downsample(labels, 175);

    %Applying Moving Median Filter and Moving Mean Filter 
    median_filter = medfilt1(Wrist_EDA_data, 50); % moving median filter with a window size = 50 
    mean_filter = movmean(median_filter, 100); % moving mean filter with window size = 100 samples

    % Keeping labels 1: baseline, 2: stress, 3: amusement, 4: meditation
    L = [1, 2, 3, 4];
    valid_labels = ismember(downsampled_labels, L);
    wrist_EDA = mean_filter(valid_labels);
    downsampled_label_up = downsampled_labels(valid_labels);

    % Segmenting the GSR data using window size 60 seconds with 50% overlap under each label
    GSR_segment_length = 60 * 4; % window size x sampling frequency
    step = GSR_segment_length / 2; % 50% overlap
    numofsegments = floor((length(wrist_EDA) - GSR_segment_length) / step) + 1;

    Features = [];
    Labels = [];
    for i = 1:numofsegments
        start_point = (i - 1) * step + 1;
        end_point = start_point + GSR_segment_length - 1;
        Segmented_data = wrist_EDA(start_point:end_point);
        Segmented_labels = mode(downsampled_label_up(start_point:end_point)); % Most frequent label in segment
        extractedfeatures = GSR_FE(Segmented_data, 4); % feature extraction

        featureLength = numel(extractedfeatures);
        maxLength = 53;
        if featureLength < maxLength
            extractedfeatures = [extractedfeatures, zeros(1, maxLength - featureLength)]; % Pad with zeros
        elseif featureLength > maxLength
            extractedfeatures = extractedfeatures(1:maxLength); % Truncate
        end

        Features = [Features; extractedfeatures];
        Labels = [Labels; Segmented_labels];
    end

    % data splitting
    train_ratio = 0.7;% Training set 70%
    validation_ratio = 0.15;% Validation set 15%
    test_ratio = 0.15;% Test set 15%% Test set 15%
    
    num_samples = length(Labels);
    indices = 1:num_samples;
    % Randomly shuffle the indices to ensure randomness in splitting
    shuffled_indices = randperm(num_samples);
    shuffled_labels = Labels(shuffled_indices);
    shuffled_features = Features(shuffled_indices, :);

    % number of samples for each set
    num_train_samples = round(train_ratio * num_samples);
    num_validation_samples = round(validation_ratio * num_samples);
    num_test_samples = num_samples - num_train_samples - num_validation_samples;
    % Splitind the shuffled data into training, validation, and test sets
    XTrain = shuffled_features(1:num_train_samples, :);
    YTrain = shuffled_labels(1:num_train_samples);
    XValidation = shuffled_features(num_train_samples+1:num_train_samples+num_validation_samples, :);
    YValidation = shuffled_labels(num_train_samples+1:num_train_samples+num_validation_samples);
    XTest = shuffled_features(num_train_samples+num_validation_samples+1:end, :);
    YTest = shuffled_labels(num_train_samples+num_validation_samples+1:end);
    
    % Model generaion
    for em = EM
        for lc = LC
            for msr = MSR
                em = char(em); % Ensure it's a character vector
                template = templateTree('MaxNumSplits', msr);
                Rf_mod = fitcensemble(XTrain, YTrain, 'Method', em, ...
                                      'NumLearningCycles', lc, 'Learners', template);

                RF_val = predict(Rf_mod, XValidation);
                RF_val_accuracy = sum(RF_val == YValidation) / numel(YValidation) * 100; %validation accuracy

                if RF_val_accuracy > rf_val_accuracy
                    rf_val_accuracy = RF_val_accuracy;
                    rf_model = Rf_mod;
                    best_LC = lc;
                    best_MSR = msr;
                    best_EM = em;
                end
            end
        end
    end
    
    rf_test = predict(rf_model, XTest);
    rf_test_accuracy = sum(rf_test == YTest) / numel(YTest) * 100; %testing accuracy

    rf_conf_mtx = confusionmat(YTest, rf_test);%confusion matrix
    rf_p = diag(rf_conf_mtx) ./ sum(rf_conf_mtx, 2);%precision
    rf_r = diag(rf_conf_mtx) ./ sum(rf_conf_mtx, 1)';%recall
    rf_f1 = 2 * (rf_p .* rf_r) ./ (rf_p + rf_r);%F1 score
    rf_f1_all = sum(rf_f1 .* sum(rf_conf_mtx, 2)) / sum(rf_conf_mtx(:));%Weighted f1 score

    fprintf('Random Forest validation accuracy: %.2f%%\n', rf_val_accuracy);
    fprintf('Best ensemble method: %s\n', best_EM);
    fprintf('Best number of learning cycles: %d\n', best_LC);
    fprintf('Best maximum number of splits: %d\n', best_MSR);
    fprintf('Random Forest testing accuracy: %.2f%%\n', rf_test_accuracy);
    fprintf('\nConfusion Matrix:\n');
    disp(rf_conf_mtx);
    fprintf('\nPrecision per class:\n');
    disp(rf_p);
    fprintf('\nRecall per class:\n');
    disp(rf_r);
    fprintf('\nF1 Score per class:\n');
    disp(rf_f1);
    fprintf('\nWeighted F1 Score: %.2f\n', rf_f1_all);

end


