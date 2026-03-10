# camera-ready?
import os
import numpy as np
import pickle
import datetime
import wfdb

from utils import preprocess_ecg_no_truncate, preprocess_ppg_no_truncate
from utils.extract_ppg_feature import extract_ppg_features
# from utils.extract_ppg_feature_stdout_suppression import extract_ppg_features # (to suppress all the "no more peaks" prints)

import torch


mcmed_dir_path = "INSERT-PATH-HERE" #################################################################
waveforms_dir_path = mcmed_dir_path + "/waveforms"

signalmcmed_dir_path = "INSERT-PATH-HERE" ###########################################################

sec_to_extract = 600
batch_size = 1 ##########################
expected_ecg_fs = 500
expected_ppg_fs = 125


with open(signalmcmed_dir_path + "/signalmc-med_csns.pkl", "rb") as f:
    csns = pickle.load(f)
num_csns = len(csns)
print("number of CSNs: %d" % num_csns)


class SignalDataset(torch.utils.data.Dataset):
    def __init__(self, signals):
        self.signals = signals

        self.num_examples = self.signals.shape[0]

    def __getitem__(self, index):
        signal = self.signals[index]

        return signal

    def __len__(self):
        return self.num_examples


preprocessed_csns = []
features_10min_ppg_list = []
features_5min_ppg_list = []
features_2min_ppg_list = []
features_1min_ppg_list = []
for i, csn in enumerate(csns):
    if i % 10 == 0:
        print("%d/%d" % (i+1, num_csns))
        print("- - - - - - %d" % len(preprocessed_csns))
        print("- - - - - - %d" % len(features_10min_ppg_list))

    csn_string = str(csn)

    try:
        ecg_signal, ecg_fields = wfdb.rdsamp(waveforms_dir_path + "/%s/%s/II/%s_1" % (csn_string[-3:], csn_string, csn_string))
    except:
        print("Error when trying to read the ECG segment for %s" % csn_string)
        continue
    ecg_signal = ecg_signal[:, 0] # (shape: [num_ecg_samples,])

    ecg_start_time = datetime.datetime.combine(ecg_fields["base_date"], ecg_fields["base_time"])

    ecg_fs = ecg_fields["fs"]
    assert ecg_fs == expected_ecg_fs

    try:
        ppg_signal, ppg_fields = wfdb.rdsamp(waveforms_dir_path + "/%s/%s/Pleth/%s_1" % (csn_string[-3:], csn_string, csn_string))
    except:
        print("Error when trying to read the PPG segment for %s" % csn_string)
        continue
    ppg_signal = ppg_signal[:, 0] # (shape: [num_ppg_samples,])

    ppg_start_time = datetime.datetime.combine(ppg_fields["base_date"], ppg_fields["base_time"])

    ppg_fs = ppg_fields["fs"]
    assert ppg_fs == expected_ppg_fs

    # print(ecg_signal.shape)
    # print(ppg_signal.shape)

    if ecg_start_time > ppg_start_time:
        ecg_start_index = 0
        target_time = ecg_start_time
        delta_seconds = (target_time - ppg_start_time).total_seconds()
        ppg_start_index = round(delta_seconds*ppg_fs)
    else:
        ppg_start_index = 0
        target_time = ppg_start_time
        delta_seconds = (target_time - ecg_start_time).total_seconds()
        ecg_start_index = round(delta_seconds*ecg_fs)

    ecg_synced_signal = ecg_signal[ecg_start_index:(ecg_start_index + sec_to_extract*ecg_fs)]
    ppg_synced_signal = ppg_signal[ppg_start_index:(ppg_start_index + sec_to_extract*ppg_fs)]

    # print(ecg_synced_signal.shape)
    # print(ppg_synced_signal.shape)

    try:
        ppg_signal_preprocessed = preprocess_ppg_no_truncate(ppg_synced_signal, expected_ppg_fs) # (shape: [1, sec_to_extract*250])
        # print(ppg_signal_preprocessed.shape)
    except:
        print("Error when trying to preprocess the PPG signal for %s" % csn_string)
        continue

    # print(ppg_signal_preprocessed.shape)

    try:
        ppg_signal_preprocessed = torch.from_numpy(ppg_signal_preprocessed.astype(np.float32)) # (shape: [1, sec_to_extract*250])
        # print(ppg_signal_preprocessed.shape)
        ppg_signal_preprocessed = ppg_signal_preprocessed.squeeze(0)  # (shape: [sec_to_extract*250])
        # print(ppg_signal_preprocessed.shape)
        ppg_60sec_segments = ppg_signal_preprocessed.view(10, 15000) # (shape: [10, 15000])
        # print(ppg_60sec_segments.shape)
        ppg_60sec_segments = ppg_60sec_segments.unsqueeze(1) # (shape: [10, 1, 15000])
        # print(ppg_60sec_segments.shape)
        #
        dataset = SignalDataset(ppg_60sec_segments)
        loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
        #
        features_list_ppg = []
        for batch_i, ppgs in enumerate(loader):
            ppgs = ppgs.squeeze(0) # (shape: (1, 15000))
            ppgs = ppgs.squeeze(0) # (shape: (15000))
            # print(ppgs.shape)
            features_ppg = torch.from_numpy(extract_ppg_features(ppgs, fs=250)).unsqueeze(0) # (shape: [1, 306])
            # print(features_ppg.shape)
            features_list_ppg.append(features_ppg)
        features_ppg = torch.cat(features_list_ppg, dim=0)  # (shape: [10, 306])
        # print(features_ppg.shape)
    except:
        print("Error when trying to extract PPG features for %s" % csn_string)
        continue

    mean_feature_10min_ppg = torch.mean(features_ppg, dim=0).unsqueeze(0) # (shape: [1, 306])
    mean_feature_5min_ppg = torch.mean(features_ppg[0:5], dim=0).unsqueeze(0) # (shape: [1, 306])
    mean_feature_2min_ppg = torch.mean(features_ppg[0:2], dim=0).unsqueeze(0) # (shape: [1, 306])
    mean_feature_1min_ppg = torch.mean(features_ppg[0:1], dim=0).unsqueeze(0) # (shape: [1, 306])

    features_10min_ppg_list.append(mean_feature_10min_ppg)
    features_5min_ppg_list.append(mean_feature_5min_ppg)
    features_2min_ppg_list.append(mean_feature_2min_ppg)
    features_1min_ppg_list.append(mean_feature_1min_ppg)
    #
    preprocessed_csns.append(csn)

    # # (debug:)
    # if i == 5:
    #     break


print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
print(len(preprocessed_csns))
print(len(features_10min_ppg_list))
print(len(features_5min_ppg_list))
print(len(features_2min_ppg_list))
print(len(features_1min_ppg_list))

features_10min_ppg = torch.cat(features_10min_ppg_list, dim=0).numpy()  # (shape: [num_preprocessdd_csns, 306])
features_5min_ppg = torch.cat(features_5min_ppg_list, dim=0).numpy()  # (shape: [num_preprocessdd_csns, 306])
features_2min_ppg = torch.cat(features_2min_ppg_list, dim=0).numpy()  # (shape: [num_preprocessdd_csns, 306])
features_1min_ppg = torch.cat(features_1min_ppg_list, dim=0).numpy()  # (shape: [num_preprocessdd_csns, 306])
print(features_10min_ppg.shape)
print(features_5min_ppg.shape)
print(features_2min_ppg.shape)
print(features_1min_ppg.shape)


with open(signalmcmed_dir_path + "/extract-features_ppg-domain-features-60sec_csns.pkl", "wb") as f:
    pickle.dump(preprocessed_csns, f)

with open(signalmcmed_dir_path + "/extract-features_ppg-domain-features-60sec_10min_ppg.pkl", "wb") as f:
    pickle.dump(features_10min_ppg, f)
with open(signalmcmed_dir_path + "/extract-features_ppg-domain-features-60sec_5min_ppg.pkl", "wb") as f:
    pickle.dump(features_5min_ppg, f)
with open(signalmcmed_dir_path + "/extract-features_ppg-domain-features-60sec_2min_ppg.pkl", "wb") as f:
    pickle.dump(features_2min_ppg, f)
with open(signalmcmed_dir_path + "/extract-features_ppg-domain-features-60sec_1min_ppg.pkl", "wb") as f:
    pickle.dump(features_1min_ppg, f)
