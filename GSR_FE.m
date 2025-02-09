% Time and frequency domain features extraction from galvanic skin response (GSR) data and its componenets.

function features = GSR_FE(segment, samplingrate)

    %Decomposition of GSR segmens using 'cvxEDA' algorithm
    [r, ~, t, ~, ~, ~, ~] = cvxEDA(segment, 1/samplingrate); % r: phasic component, t: tonic component

    %Peak analysis using phasic component
    minimum_height = 0.01;  % Minimum height to consider for a peak
    minimum_distance = 10;  % Minimum number of samples between two peaks
    [pk, loc] = findpeaks(r, 'MinPeakHeight', minimum_height, 'MinPeakDistance', minimum_distance);
    peak_count = length(pk); % no. of peaks in the phasic segment
    maximum_amplitude = max(r);% maximum peak amplitude i the segment 
    
    % Time domain analysis for original GSR segment

    mean_gsr = mean(segment);%mean
    median_gsr = median(segment);%median
    standard_deviation_gsr = std(segment);%standard deviation
    rms_gsr = rms(segment);%root men square
    skewness_gsr = skewness(segment);%skewness
    kurtosis_gsr = kurtosis(segment);%kurtosis
    maximum_gsr = max(segment);%maximum value
    minimum_gsr = min(segment);%minimum value
    iqr_gsr = iqr(segment);%inter quartile region
    sum_gsr = sum(segment);%sum
    meandiff_gsr = mean(abs(diff(segment)));% mean of absolute differences 

    % Frequency domain analysis for original GSR segment
    [psd, freq] = pwelch(segment, [], [], [], samplingrate);% power spectral density
    psd_sum_gsr = sum(psd); %sum of power spectral density
    psd_mean_gsr = mean(psd);% mean of power spectral density

    % Time domain analysis of phasic and tonic components
    mean_r = mean(r);
    standard_deviation_r = std(r); %standard deviation of phasic segment
    skewness_r = skewness(r);% skewness of phasic segment
    kurtosis_r = kurtosis(r);% kurtosis of phasic segment

    mean_t = mean(t);% mean of tonic segment
    standard_deviation_t = std(t);% standard deviation of tonic segment
    skewness_t = skewness(t);% skewness of tonic segmentt
    kurtosis_t = kurtosis(t);% kurtosis of tonic segment

    diff_r = diff(r);
    mean_diff_r = mean(abs(diff_r)); %mean of absolute first difference of phasic component
    secdiff_r = diff(diff_r);
    mean_secdiff_r = mean(abs(secdiff_r)); %mean of absolute second difference of phasic component
    
    diff_t = diff(t);
    mean_diff_t = mean(abs(diff_t));%mean of absolute first difference of tonic component
    secdiff_t = diff(diff_t);
    mean_secdiff_t = mean(abs(secdiff_t));%mean of absolute first difference of tonic component
    
    mean_neg_diff_r = mean(abs(diff_r(diff_r < 0))); %mean of absolute of negative difference ofphasic component
    mean_neg_diff_t = mean(abs(diff_t(diff_t < 0)));%mean of absolute of negative difference of tonic component
    
    ratio_neg_diff_r = sum(diff_r < 0) / length(diff_r); %ration of negative differences for phasic component
    ratio_neg_diff_t = sum(diff_t < 0) / length(diff_t);%ration of negative differences for tonic component
    
    local_minima_r = sum(islocalmin(r)); %local maxima for phasic component
    local_maxima_r = sum(islocalmax(r));%local minima for phasic component
    
    local_minima_t = sum(islocalmin(t));%local maxima for tonic component
    local_maxima_t = sum(islocalmax(t));%local minima for tonic component

    % Frequency domain analysis of phasic and tonic components
    [psd_t, freq_t] = pwelch(t, [], [], [], samplingrate); % power spectral density for tonic component 
    [psd_r, freq_r] = pwelch(r, [], [], [], samplingrate); %power spectral density for phasic component
    
    tonic_LF_band = [0, 0.1];%low frequency band for tonic component
    tonic_HF_band = [0.1, 0.2];%high frequency band for tonic component
    phasic_LF_band = [0.2, 1];%low frequency band for phasic component
    phasic_HF_band = [1, 2];%high frequency band for phasic component
    %power within the low and high frequency bands 
    tonic_LF_band_power = bandpower(t, samplingrate, tonic_LF_band); 
    tonic_HF_band_power = bandpower(t, samplingrate, tonic_HF_band);
    phasic_LF_band_power = bandpower(r, samplingrate, phasic_LF_band);
    phasic_HF_band_power = bandpower(r, samplingrate, phasic_HF_band);
    % finding indices of the frequency bands in the PSD
    tonicLF_indices = (freq_t >= tonic_LF_band(1) & freq_t <= tonic_LF_band(2));
    tonicHF_indices = (freq_t >= tonic_HF_band(1) & freq_t <= tonic_HF_band(2));
    phasicLF_indices = (freq_r >= phasic_LF_band(1) & freq_r <= phasic_LF_band(2));
    phasicHF_indices = (freq_r >= phasic_HF_band(1) & freq_r <= phasic_HF_band(2));

    % Mean, maximum, minimum, and sum of the PSD values within each frequency band
    tonic_LF_psd = [mean(psd_t(tonicLF_indices)), max(psd_t(tonicLF_indices)), min(psd_t(tonicLF_indices)),...
        sum(psd_t(tonicLF_indices))];% using tonic low freq bad indices
    tonic_HF_psd = [mean(psd_t(tonicHF_indices)), max(psd_t(tonicHF_indices)), min(psd_t(tonicHF_indices)),...
        sum(psd_t(tonicHF_indices))];% using tonic high freq bad indices
    phasic_LF_psd = [mean(psd_r(phasicLF_indices)), max(psd_r(phasicLF_indices)), min(psd_r(phasicLF_indices)),...
        sum(psd_r(phasicLF_indices))];% using phasic low freq bad indices
    phasic_HF_psd = [mean(psd_r(phasicHF_indices)), max(psd_r(phasicHF_indices)), min(psd_r(phasicHF_indices)),...
        sum(psd_r(phasicHF_indices))];% using phasic high freq bad indices

    % Calculate the ratio of LF power to HF power
    tonic_ratio_LFbyHF = tonic_LF_band_power / tonic_HF_band_power;%ratio of low frequency power to high frequency power for tonic segment
    phasic_ratio_LFbyHF = phasic_LF_band_power / phasic_HF_band_power;%ratio of low frequency power to high frequency power for phasic segment
    
    %Creating a feature array containing 53 features
    features = [mean_gsr, peak_count, maximum_amplitude, median_gsr, standard_deviation_gsr,rms_gsr,skewness_gsr,kurtosis_gsr, maximum_gsr, ...
        minimum_gsr ,iqr_gsr, sum_gsr,meandiff_gsr, psd_sum_gsr,  psd_mean_gsr, mean_r, standard_deviation_r, skewness_r, kurtosis_r, mean_diff_r, mean_secdiff_r,...
        mean_neg_diff_r, ratio_neg_diff_r, local_minima_r, local_maxima_r, mean_t, standard_deviation_t, skewness_t, kurtosis_t, mean_diff_t, mean_secdiff_t,...
        mean_neg_diff_t, ratio_neg_diff_t, local_minima_t, local_maxima_t, tonic_LF_psd, tonic_HF_psd, tonic_ratio_LFbyHF,...
        phasic_LF_psd, phasic_HF_psd, phasic_ratio_LFbyHF];
end
