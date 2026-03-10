# camera-ready?
import os
import numpy as np
import pickle
import datetime
import wfdb

from utils import preprocess_ecg_no_truncate_500hz, preprocess_ppg_no_truncate_500hz

import torch


mcmed_dir_path = "INSERT-PATH-HERE" #################################################################
waveforms_dir_path = mcmed_dir_path + "/waveforms"

signalmcmed_dir_path = "INSERT-PATH-HERE" ###########################################################

model_path = "INSERT-PATH-HERE/1_lead_ECGFounder.pth" ########################################################

sec_to_extract = 600
batch_size = 128 # (128 should be fine on 40 GB GPU, for both single-lead and 12-lead ECGs)
expected_ecg_fs = 500
expected_ppg_fs = 125


################################################################################
# load the model:
################################################################################
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "ECGFounder"))
from ECGFounder.finetune_model import ft_1lead_ECGFounder

model = ft_1lead_ECGFounder("cuda", model_path, 1, linear_prob=False)
model.return_features = True
model.eval()
model.cuda()
print(model)

print("model is loaded!")
print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
################################################################################


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
#
features_10min_ppg_list = []
features_5min_ppg_list = []
features_2min_ppg_list = []
features_1min_ppg_list = []
features_30sec_ppg_list = []
features_10sec_ppg_list = []
for i, csn in enumerate(csns):
    if i % 10 == 0:
        print("%d/%d" % (i+1, num_csns))
        print("- - - - - - %d" % len(preprocessed_csns))
        print("- - - - - - %d" % len(features_10min_ecg_list))
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
        ecg_signal_preprocessed = preprocess_ecg_no_truncate_500hz(ecg_synced_signal, expected_ecg_fs) # (shape: [1, sec_to_extract*500])
        # print(ecg_signal_preprocessed.shape)
    except:
        print("Error when trying to preprocess the ECG signal for %s" % csn_string)
        continue

    try:
        ppg_signal_preprocessed = preprocess_ppg_no_truncate_500hz(ppg_synced_signal, expected_ppg_fs) # (shape: [1, sec_to_extract*500])
        # print(ppg_signal_preprocessed.shape)
    except:
        print("Error when trying to preprocess the PPG signal for %s" % csn_string)
        continue

    # print(ecg_signal_preprocessed.shape)
    # print(ppg_signal_preprocessed.shape)

    ecg_signal_preprocessed = torch.from_numpy(ecg_signal_preprocessed.astype(np.float32)).cuda() # (shape: [1, sec_to_extract*500])
    # print(ecg_signal_preprocessed.shape)
    ecg_signal_preprocessed = ecg_signal_preprocessed.squeeze(0)  # (shape: [sec_to_extract*500])
    # print(ecg_signal_preprocessed.shape)
    ecg_10sec_segments = ecg_signal_preprocessed.view(60, 5000) # (shape: [60, 5000])
    # print(ecg_10sec_segments.shape)
    ecg_10sec_segments = ecg_10sec_segments.unsqueeze(1) # (shape: [60, 1, 5000])
    # print(ecg_10sec_segments.shape)
    #
    dataset = SignalDataset(ecg_10sec_segments)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    #
    features_list_ecg = []
    for batch_i, ecgs in enumerate(loader):
        ecgs = ecgs.cuda() # (shape: (batch_size, 1, 5000))
        # print(ecgs.shape)
        with torch.no_grad():
            _, features_ecg = model(ecgs) # (shape: [batch_size, 1024])
            features_ecg = features_ecg.cpu()
            # print(features_ecg.shape)
            features_list_ecg.append(features_ecg)
    features_ecg = torch.cat(features_list_ecg, dim=0)  # (shape: [60, 1024])
    # print(features_ecg.shape)

    ppg_signal_preprocessed = torch.from_numpy(ppg_signal_preprocessed.astype(np.float32)).cuda() # (shape: [1, sec_to_extract*500])
    # print(ppg_signal_preprocessed.shape)
    ppg_signal_preprocessed = ppg_signal_preprocessed.squeeze(0)  # (shape: [sec_to_extract*500])
    # print(ppg_signal_preprocessed.shape)
    ppg_10sec_segments = ppg_signal_preprocessed.view(60, 5000) # (shape: [60, 5000])
    # print(ppg_10sec_segments.shape)
    ppg_10sec_segments = ppg_10sec_segments.unsqueeze(1) # (shape: [60, 1, 5000])
    # print(ppg_10sec_segments.shape)
    #
    dataset = SignalDataset(ppg_10sec_segments)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    #
    features_list_ppg = []
    for batch_i, ppgs in enumerate(loader):
        ppgs = ppgs.cuda() # (shape: (batch_size, 1, 5000))
        # print(ppgs.shape)
        with torch.no_grad():
            _, features_ppg = model(ppgs) # (shape: [batch_size, 1024])
            features_ppg = features_ppg.cpu()
            # print(features_ppg.shape)
            features_list_ppg.append(features_ppg)
    features_ppg = torch.cat(features_list_ppg, dim=0)  # (shape: [60, 1024])
    # print(features_ppg.shape)

    mean_feature_10min_ecg = torch.mean(features_ecg, dim=0).unsqueeze(0) # (shape: [1, 1024])
    mean_feature_5min_ecg = torch.mean(features_ecg[0:30], dim=0).unsqueeze(0) # (shape: [1, 1024])
    mean_feature_2min_ecg = torch.mean(features_ecg[0:12], dim=0).unsqueeze(0) # (shape: [1, 1024])
    mean_feature_1min_ecg = torch.mean(features_ecg[0:6], dim=0).unsqueeze(0) # (shape: [1, 1024])
    mean_feature_30sec_ecg = torch.mean(features_ecg[0:3], dim=0).unsqueeze(0) # (shape: [1, 1024])
    mean_feature_10sec_ecg = features_ecg[0].unsqueeze(0) # (shape: [1, 1024])
    #
    mean_feature_10min_ppg = torch.mean(features_ppg, dim=0).unsqueeze(0) # (shape: [1, 1024])
    mean_feature_5min_ppg = torch.mean(features_ppg[0:30], dim=0).unsqueeze(0) # (shape: [1, 1024])
    mean_feature_2min_ppg = torch.mean(features_ppg[0:12], dim=0).unsqueeze(0) # (shape: [1, 1024])
    mean_feature_1min_ppg = torch.mean(features_ppg[0:6], dim=0).unsqueeze(0) # (shape: [1, 1024])
    mean_feature_30sec_ppg = torch.mean(features_ppg[0:3], dim=0).unsqueeze(0) # (shape: [1, 1024])
    mean_feature_10sec_ppg = features_ppg[0].unsqueeze(0) # (shape: [1, 1024])

    features_10min_ecg_list.append(mean_feature_10min_ecg)
    features_5min_ecg_list.append(mean_feature_5min_ecg)
    features_2min_ecg_list.append(mean_feature_2min_ecg)
    features_1min_ecg_list.append(mean_feature_1min_ecg)
    features_30sec_ecg_list.append(mean_feature_30sec_ecg)
    features_10sec_ecg_list.append(mean_feature_10sec_ecg)
    #
    features_10min_ppg_list.append(mean_feature_10min_ppg)
    features_5min_ppg_list.append(mean_feature_5min_ppg)
    features_2min_ppg_list.append(mean_feature_2min_ppg)
    features_1min_ppg_list.append(mean_feature_1min_ppg)
    features_30sec_ppg_list.append(mean_feature_30sec_ppg)
    features_10sec_ppg_list.append(mean_feature_10sec_ppg)
    #
    preprocessed_csns.append(csn)

    # # (debug:)
    # if i == 25:
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
print(len(features_10min_ppg_list))
print(len(features_5min_ppg_list))
print(len(features_2min_ppg_list))
print(len(features_1min_ppg_list))
print(len(features_30sec_ppg_list))
print(len(features_10sec_ppg_list))

features_10min_ecg = torch.cat(features_10min_ecg_list, dim=0).numpy()  # (shape: [num_preprocessdd_csns, 1024])
features_5min_ecg = torch.cat(features_5min_ecg_list, dim=0).numpy()  # (shape: [num_preprocessdd_csns, 1024])
features_2min_ecg = torch.cat(features_2min_ecg_list, dim=0).numpy()  # (shape: [num_preprocessdd_csns, 1024])
features_1min_ecg = torch.cat(features_1min_ecg_list, dim=0).numpy()  # (shape: [num_preprocessdd_csns, 1024])
features_30sec_ecg = torch.cat(features_30sec_ecg_list, dim=0).numpy()  # (shape: [num_preprocessdd_csns, 1024])
features_10sec_ecg = torch.cat(features_10sec_ecg_list, dim=0).numpy()  # (shape: [num_preprocessdd_csns, 1024])
print(features_10min_ecg.shape)
print(features_5min_ecg.shape)
print(features_2min_ecg.shape)
print(features_1min_ecg.shape)
print(features_30sec_ecg.shape)
print(features_10sec_ecg.shape)

features_10min_ppg = torch.cat(features_10min_ppg_list, dim=0).numpy()  # (shape: [num_preprocessdd_csns, 1024])
features_5min_ppg = torch.cat(features_5min_ppg_list, dim=0).numpy()  # (shape: [num_preprocessdd_csns, 1024])
features_2min_ppg = torch.cat(features_2min_ppg_list, dim=0).numpy()  # (shape: [num_preprocessdd_csns, 1024])
features_1min_ppg = torch.cat(features_1min_ppg_list, dim=0).numpy()  # (shape: [num_preprocessdd_csns, 1024])
features_30sec_ppg = torch.cat(features_30sec_ppg_list, dim=0).numpy()  # (shape: [num_preprocessdd_csns, 1024])
features_10sec_ppg = torch.cat(features_10sec_ppg_list, dim=0).numpy()  # (shape: [num_preprocessdd_csns, 1024])
print(features_10min_ppg.shape)
print(features_5min_ppg.shape)
print(features_2min_ppg.shape)
print(features_1min_ppg.shape)
print(features_30sec_ppg.shape)
print(features_10sec_ppg.shape)

with open(signalmcmed_dir_path + "/extract-features_ecgfounder_csns.pkl", "wb") as f:
    pickle.dump(preprocessed_csns, f)

with open(signalmcmed_dir_path + "/extract-features_ecgfounder_10min_ecg.pkl", "wb") as f:
    pickle.dump(features_10min_ecg, f)
with open(signalmcmed_dir_path + "/extract-features_ecgfounder_5min_ecg.pkl", "wb") as f:
    pickle.dump(features_5min_ecg, f)
with open(signalmcmed_dir_path + "/extract-features_ecgfounder_2min_ecg.pkl", "wb") as f:
    pickle.dump(features_2min_ecg, f)
with open(signalmcmed_dir_path + "/extract-features_ecgfounder_1min_ecg.pkl", "wb") as f:
    pickle.dump(features_1min_ecg, f)
with open(signalmcmed_dir_path + "/extract-features_ecgfounder_30sec_ecg.pkl", "wb") as f:
    pickle.dump(features_30sec_ecg, f)
with open(signalmcmed_dir_path + "/extract-features_ecgfounder_10sec_ecg.pkl", "wb") as f:
    pickle.dump(features_10sec_ecg, f)

with open(signalmcmed_dir_path + "/extract-features_ecgfounder_10min_ppg.pkl", "wb") as f:
    pickle.dump(features_10min_ppg, f)
with open(signalmcmed_dir_path + "/extract-features_ecgfounder_5min_ppg.pkl", "wb") as f:
    pickle.dump(features_5min_ppg, f)
with open(signalmcmed_dir_path + "/extract-features_ecgfounder_2min_ppg.pkl", "wb") as f:
    pickle.dump(features_2min_ppg, f)
with open(signalmcmed_dir_path + "/extract-features_ecgfounder_1min_ppg.pkl", "wb") as f:
    pickle.dump(features_1min_ppg, f)
with open(signalmcmed_dir_path + "/extract-features_ecgfounder_30sec_ppg.pkl", "wb") as f:
    pickle.dump(features_30sec_ppg, f)
with open(signalmcmed_dir_path + "/extract-features_ecgfounder_10sec_ppg.pkl", "wb") as f:
    pickle.dump(features_10sec_ppg, f)
