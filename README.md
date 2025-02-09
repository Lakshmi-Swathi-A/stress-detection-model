# stress-detection-model
Chronic stress affects health, making accurate detection vital. Using WESAD data and GSR signals, this project applies machine learning. Random Forest achieved 89.5% accuracy and a 0.91 F1 score, excelling in stress detection. The study highlights personalised AI's role in better stress management and mental health support.

![image](https://github.com/user-attachments/assets/26bf1565-89df-4c8b-a446-ac795e4fd234)

1. Dataset: 
Used the WESAD dataset (Wristband-based Affect and Stress Detection).
Collected physiological signals using the Empatica E4 wristband from 15 participants.
Sampling rate: 4 Hz (data collected every 0.25 seconds).

3. Data Preprocessing:
Synchronized labels to match physiological signal timestamps.
Applied moving median filter (50 samples) to remove outliers.
Used a moving mean filter (100 samples) for smoothing.
Segmented data into 60-second windows with 50% overlap.
Removed uniform segments to eliminate irrelevant data.

5. Feature Engineering:
Extracted phasic (skin conductance level) & tonic (skin conductance level) components from EDA signals using cvxEDA algorithm.
Identified peak features (count, amplitude, local minima using min distance = 10 samples).
Computed 53 statistical features, including: Mean, RMS, Variance, Skewness, Kurtosis.
Frequency domain features using Welch’s method (Power Spectral Density - PSD).

7. Machine Learning Models:
Data Split: Using stratified partitioning approach, 70% Train, 15% Validation, 15% Test.
Used multiple classifiers: Support Vector Machine (SVM), k-Nearest Neighbors (k-NN), Decision Tree (DT), Random Forest (RF), Naïve Bayes (NB), Convolutional Neural Network (CNN),& Long Short-Term Memory (LSTM)
Performed grid search for hyperparameter tuning.
Evaluated models using Accuracy, Precision, Recall, and F1-score.
Tested subject-specific classification performance.

Reference:
1. Schmidt, P. R. (2018). Introducing wesad, a multimodal dataset for wearable stress and affect detection. 20th ACM international conference on multimodal interaction, (pp. 400-408).
2. Greco, A. V. (2015). cvxEDA: A convex optimization approach to electrodermal activity processing. IEEE transactions on biomedical engineering( 63(4)), 797-804.
