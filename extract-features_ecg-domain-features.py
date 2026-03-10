import os
import numpy as np
import pickle
import datetime
import wfdb

from utils import preprocess_ecg_no_truncate, preprocess_ppg_no_truncate
from utils import extract_ecg_feature

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
features_10min_ecg_list = []
features_5min_ecg_list = []
features_2min_ecg_list = []
features_1min_ecg_list = []
features_30sec_ecg_list = []
features_10sec_ecg_list = []
for i, csn in enumerate(csns):
    if i % 10 == 0:
        print("%d/%d" % (i+1, num_csns))
        print("- - - - - - %d" % len(preprocessed_csns))
        print("- - - - - - %d" % len(features_10min_ecg_list))

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
        ecg_signal_preprocessed = preprocess_ecg_no_truncate(ecg_synced_signal, expected_ecg_fs) # (shape: [1, sec_to_extract*250])
        # print(ecg_signal_preprocessed.shape)
    except:
        print("Error when trying to preprocess the ECG signal for %s" % csn_string)
        continue

    # print(ecg_signal_preprocessed.shape)

    try:
        ecg_signal_preprocessed = torch.from_numpy(ecg_signal_preprocessed.astype(np.float32)) # (shape: [1, sec_to_extract*250])
        # print(ecg_signal_preprocessed.shape)
        ecg_signal_preprocessed = ecg_signal_preprocessed.squeeze(0)  # (shape: [sec_to_extract*250])
        # print(ecg_signal_preprocessed.shape)
        ecg_10sec_segments = ecg_signal_preprocessed.view(60, 2500) # (shape: [60, 2500])
        # print(ecg_10sec_segments.shape)
        ecg_10sec_segments = ecg_10sec_segments.unsqueeze(1) # (shape: [60, 1, 2500])
        # print(ecg_10sec_segments.shape)
        #
        dataset = SignalDataset(ecg_10sec_segments)
        loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
        #
        features_list_ecg = []
        for batch_i, ecgs in enumerate(loader):
            ecgs = ecgs.squeeze(0) # (shape: (1, 2500))
            ecgs = ecgs.squeeze(0) # (shape: (2500))
            # print(ecgs.shape)
            features_ecg = torch.from_numpy(extract_ecg_feature(ecgs, fs=250)).unsqueeze(0) # (shape: [1, 54])
            # print(features_ecg.shape)
            features_list_ecg.append(features_ecg)
        features_ecg = torch.cat(features_list_ecg, dim=0)  # (shape: [60, 54])
        # print(features_ecg.shape)
    except:
        print("Error when trying to extract ECG features for %s" % csn_string)
        continue

    mean_feature_10min_ecg = torch.mean(features_ecg, dim=0).unsqueeze(0) # (shape: [1, 54])
    mean_feature_5min_ecg = torch.mean(features_ecg[0:30], dim=0).unsqueeze(0) # (shape: [1, 54])
    mean_feature_2min_ecg = torch.mean(features_ecg[0:12], dim=0).unsqueeze(0) # (shape: [1, 54])
    mean_feature_1min_ecg = torch.mean(features_ecg[0:6], dim=0).unsqueeze(0) # (shape: [1, 54])
    mean_feature_30sec_ecg = torch.mean(features_ecg[0:3], dim=0).unsqueeze(0) # (shape: [1, 54])
    mean_feature_10sec_ecg = features_ecg[0].unsqueeze(0) # (shape: [1, 54])

    features_10min_ecg_list.append(mean_feature_10min_ecg)
    features_5min_ecg_list.append(mean_feature_5min_ecg)
    features_2min_ecg_list.append(mean_feature_2min_ecg)
    features_1min_ecg_list.append(mean_feature_1min_ecg)
    features_30sec_ecg_list.append(mean_feature_30sec_ecg)
    features_10sec_ecg_list.append(mean_feature_10sec_ecg)
    #
    preprocessed_csns.append(csn)

    # # (debug:)
    # if i == 15:
    #     break


print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
print(len(preprocessed_csns))
print(len(features_10min_ecg_list))
print(len(features_5min_ecg_list))
print(len(features_2min_ecg_list))
print(len(features_1min_ecg_list))
print(len(features_30sec_ecg_list))
print(len(features_10sec_ecg_list))

features_10min_ecg = torch.cat(features_10min_ecg_list, dim=0).numpy()  # (shape: [num_preprocessdd_csns, 54])
features_5min_ecg = torch.cat(features_5min_ecg_list, dim=0).numpy()  # (shape: [num_preprocessdd_csns, 54])
features_2min_ecg = torch.cat(features_2min_ecg_list, dim=0).numpy()  # (shape: [num_preprocessdd_csns, 54])
features_1min_ecg = torch.cat(features_1min_ecg_list, dim=0).numpy()  # (shape: [num_preprocessdd_csns, 54])
features_30sec_ecg = torch.cat(features_30sec_ecg_list, dim=0).numpy()  # (shape: [num_preprocessdd_csns, 54])
features_10sec_ecg = torch.cat(features_10sec_ecg_list, dim=0).numpy()  # (shape: [num_preprocessdd_csns, 54])
print(features_10min_ecg.shape)
print(features_5min_ecg.shape)
print(features_2min_ecg.shape)
print(features_1min_ecg.shape)
print(features_30sec_ecg.shape)
print(features_10sec_ecg.shape)

with open(signalmcmed_dir_path + "/extract-features_ecg-domain-features_csns.pkl", "wb") as f:
    pickle.dump(preprocessed_csns, f)

with open(signalmcmed_dir_path + "/extract-features_ecg-domain-features_10min_ecg.pkl", "wb") as f:
    pickle.dump(features_10min_ecg, f)
with open(signalmcmed_dir_path + "/extract-features_ecg-domain-features_5min_ecg.pkl", "wb") as f:
    pickle.dump(features_5min_ecg, f)
with open(signalmcmed_dir_path + "/extract-features_ecg-domain-features_2min_ecg.pkl", "wb") as f:
    pickle.dump(features_2min_ecg, f)
with open(signalmcmed_dir_path + "/extract-features_ecg-domain-features_1min_ecg.pkl", "wb") as f:
    pickle.dump(features_1min_ecg, f)
with open(signalmcmed_dir_path + "/extract-features_ecg-domain-features_30sec_ecg.pkl", "wb") as f:
    pickle.dump(features_30sec_ecg, f)
with open(signalmcmed_dir_path + "/extract-features_ecg-domain-features_10sec_ecg.pkl", "wb") as f:
    pickle.dump(features_10sec_ecg, f)
