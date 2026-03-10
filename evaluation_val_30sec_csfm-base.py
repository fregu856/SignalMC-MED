import os
import numpy as np
import pickle
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression


debug = False


mcmed_dir_path = "INSERT-PATH-HERE" #################################################################
signalmcmed_dir_path = "INSERT-PATH-HERE" ###########################################################
save_dir_path = signalmcmed_dir_path + "/evaluation_val_30sec_csfm-base"
os.makedirs(save_dir_path, exist_ok=True)


if not debug:
    train_props = [1.0, 0.75, 0.50, 0.25, 0.10, 0.05]
else:
    train_props = [0.10, 0.05]

if not debug:
    number_of_runs = 5
else:
    number_of_runs = 2


seed = 42
rng = np.random.RandomState(seed=seed)

if not debug:
    alpha_values = np.logspace(-6, 6, 31)
    c_values = np.logspace(-4, 4, 25)
else:
    alpha_values = [1.0]
    c_values = [1.0]


with open(signalmcmed_dir_path + "/extract-features_csfm-base_30sec_ecg.pkl", "rb") as f:
    features_30sec_ecg = pickle.load(f)
print(features_30sec_ecg.shape)
with open(signalmcmed_dir_path + "/extract-features_csfm-base_30sec_ppg.pkl", "rb") as f:
    features_30sec_ppg = pickle.load(f)
print(features_30sec_ppg.shape)
with open(signalmcmed_dir_path + "/extract-features_csfm-base_30sec_ecg_ppg.pkl", "rb") as f:
    features_30sec_ecg_ppg = pickle.load(f)
print(features_30sec_ecg_ppg.shape)
# (all have shape: [num_csns, feature_dim])

with open(signalmcmed_dir_path + "/extract-features_csfm-base_csns.pkl", "rb") as f:
    preprocessed_csns = pickle.load(f)
print(len(preprocessed_csns))


with open(signalmcmed_dir_path + "/signalmc-med_csns.pkl", "rb") as f:
    signalmcmed_csns = pickle.load(f)
print(len(signalmcmed_csns))
preprocessed_csns_filter = []
features_30sec_ecg_list = []
features_30sec_ppg_list = []
features_30sec_ecg_ppg_list =[]
for i in range(len(preprocessed_csns)):
    if preprocessed_csns[i] in signalmcmed_csns:
        features_30sec_ecg_list.append(features_30sec_ecg[i])
        features_30sec_ppg_list.append(features_30sec_ppg[i])
        features_30sec_ecg_ppg_list.append(features_30sec_ecg_ppg[i])
        preprocessed_csns_filter.append(preprocessed_csns[i])
features_30sec_ecg = np.stack(features_30sec_ecg_list, axis=0)
features_30sec_ppg = np.stack(features_30sec_ppg_list, axis=0)
features_30sec_ecg_ppg = np.stack(features_30sec_ecg_ppg_list, axis=0)
preprocessed_csns = preprocessed_csns_filter
print(features_30sec_ecg.shape)
print(features_30sec_ppg.shape)
print(features_30sec_ecg_ppg.shape)
print(len(preprocessed_csns))
assert len(preprocessed_csns) == len(signalmcmed_csns)


all_train_csns = pd.read_csv(mcmed_dir_path + "/split_chrono_train.csv", header=None)[0].tolist()
print(len(all_train_csns))
all_val_csns = pd.read_csv(mcmed_dir_path + "/split_chrono_val.csv", header=None)[0].tolist()
print(len(all_val_csns))
all_test_csns = pd.read_csv(mcmed_dir_path + "/split_chrono_test.csv", header=None)[0].tolist()
print(len(all_test_csns))

train_csns = [csn for csn in preprocessed_csns if csn in all_train_csns]
val_csns = [csn for csn in preprocessed_csns if csn in all_val_csns]
test_csns = [csn for csn in preprocessed_csns if csn in all_test_csns]
print(len(train_csns))
print(len(val_csns))
print(len(test_csns))

print("total number of CSNs/visits: %d" % len(preprocessed_csns))
print("number of train CSNs/visits: %d" % len(train_csns))
print("number of val   CSNs/visits: %d" % len(val_csns))
print("number of test  CSNs/visits: %d" % len(test_csns))

visits_df = pd.read_csv(mcmed_dir_path + "/visits.csv")
# print(visits_df)

labs_df = pd.read_csv(mcmed_dir_path + "/labs.csv")
# print(labs_df)

pmh_df = pd.read_csv(mcmed_dir_path + "/pmh.csv")
# print(pmh_df)




all_results_data = {}
for train_prop_i, train_prop in enumerate(train_props):
    results_data = {}
    for run_i in range(number_of_runs):
        print("train_prop: %.2f (%d/%d) -- run %d/%d" % (train_prop, train_prop_i, len(train_props)-1, run_i, number_of_runs-1))

        train_ages = []
        train_features_list_30sec_ecg = []
        train_features_list_30sec_ppg = []
        train_features_list_30sec_ecg_ppg = []
        for i, csn in enumerate(preprocessed_csns):
            if csn in train_csns:
                age_values = visits_df[visits_df["CSN"] == csn]["Age"].values
                assert len(age_values) == 1
                train_ages.append(age_values[0])
                train_features_list_30sec_ecg.append(features_30sec_ecg[i])
                train_features_list_30sec_ppg.append(features_30sec_ppg[i])
                train_features_list_30sec_ecg_ppg.append(features_30sec_ecg_ppg[i])
        train_features_30sec_ecg = np.stack(train_features_list_30sec_ecg, axis=0)
        train_features_30sec_ppg = np.stack(train_features_list_30sec_ppg, axis=0)
        train_features_30sec_ecg_ppg = np.stack(train_features_list_30sec_ecg_ppg, axis=0)
        print(len(train_ages))
        print(train_features_30sec_ecg.shape)
        print(train_features_30sec_ppg.shape)
        print(train_features_30sec_ecg_ppg.shape)

        val_ages = []
        val_features_list_30sec_ecg = []
        val_features_list_30sec_ppg = []
        val_features_list_30sec_ecg_ppg = []
        for i, csn in enumerate(preprocessed_csns):
            if csn in val_csns:
                age_values = visits_df[visits_df["CSN"] == csn]["Age"].values
                assert len(age_values) == 1
                val_ages.append(age_values[0])
                val_features_list_30sec_ecg.append(features_30sec_ecg[i])
                val_features_list_30sec_ppg.append(features_30sec_ppg[i])
                val_features_list_30sec_ecg_ppg.append(features_30sec_ecg_ppg[i])
        val_features_30sec_ecg = np.stack(val_features_list_30sec_ecg, axis=0)
        val_features_30sec_ppg = np.stack(val_features_list_30sec_ppg, axis=0)
        val_features_30sec_ecg_ppg = np.stack(val_features_list_30sec_ecg_ppg, axis=0)
        print(len(val_ages))
        print(val_features_30sec_ecg.shape)
        print(val_features_30sec_ppg.shape)
        print(val_features_30sec_ecg_ppg.shape)

        X_train_30sec_ecg = train_features_30sec_ecg
        X_train_30sec_ppg = train_features_30sec_ppg
        X_train_30sec_ecg_ppg = train_features_30sec_ecg_ppg
        X_train_30sec_ecg_ppg_mean = (X_train_30sec_ecg + X_train_30sec_ppg)/2.0
        y_train = np.array(train_ages)
        #######
        indices = np.arange(X_train_30sec_ecg.shape[0])
        train_inds = rng.choice(indices, size=int(train_prop*len(indices)), replace=True)
        X_train_30sec_ecg = X_train_30sec_ecg[train_inds]
        X_train_30sec_ppg = X_train_30sec_ppg[train_inds]
        X_train_30sec_ecg_ppg = X_train_30sec_ecg_ppg[train_inds]
        X_train_30sec_ecg_ppg_mean = X_train_30sec_ecg_ppg_mean[train_inds]
        y_train = y_train[train_inds]
        print(len(y_train))
        print(X_train_30sec_ecg.shape)
        print(X_train_30sec_ppg.shape)
        print(X_train_30sec_ecg_ppg.shape)
        print(X_train_30sec_ecg_ppg_mean.shape)

        X_val_30sec_ecg = val_features_30sec_ecg
        X_val_30sec_ppg = val_features_30sec_ppg
        X_val_30sec_ecg_ppg = val_features_30sec_ecg_ppg
        X_val_30sec_ecg_ppg_mean = (X_val_30sec_ecg + X_val_30sec_ppg)/2.0
        y_val = np.array(val_ages)

        maes_30sec_ecg = []
        pearsons_30sec_ecg = []
        maes_30sec_ppg = []
        pearsons_30sec_ppg = []
        maes_30sec_ecg_ppg = []
        pearsons_30sec_ecg_ppg = []
        maes_30sec_ecg_ppg_mean = []
        pearsons_30sec_ecg_ppg_mean = []
        for alpha in alpha_values:
            print(alpha)

            pipe = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            pipe.fit(X_train_30sec_ecg, y_train)
            y_val_pred = pipe.predict(X_val_30sec_ecg)
            mae = np.mean(np.abs(y_val - y_val_pred))
            maes_30sec_ecg.append(mae)
            pearson_corr, _ = pearsonr(y_val_pred, y_val)
            pearsons_30sec_ecg.append(pearson_corr)

            pipe = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            pipe.fit(X_train_30sec_ppg, y_train)
            y_val_pred = pipe.predict(X_val_30sec_ppg)
            mae = np.mean(np.abs(y_val - y_val_pred))
            maes_30sec_ppg.append(mae)
            pearson_corr, _ = pearsonr(y_val_pred, y_val)
            pearsons_30sec_ppg.append(pearson_corr)

            pipe = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            pipe.fit(X_train_30sec_ecg_ppg, y_train)
            y_val_pred = pipe.predict(X_val_30sec_ecg_ppg)
            mae = np.mean(np.abs(y_val - y_val_pred))
            maes_30sec_ecg_ppg.append(mae)
            pearson_corr, _ = pearsonr(y_val_pred, y_val)
            pearsons_30sec_ecg_ppg.append(pearson_corr)

            pipe = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            pipe.fit(X_train_30sec_ecg_ppg_mean, y_train)
            y_val_pred = pipe.predict(X_val_30sec_ecg_ppg_mean)
            mae = np.mean(np.abs(y_val - y_val_pred))
            maes_30sec_ecg_ppg_mean.append(mae)
            pearson_corr, _ = pearsonr(y_val_pred, y_val)
            pearsons_30sec_ecg_ppg_mean.append(pearson_corr)

        best_idx_ecg = np.argmax(pearsons_30sec_ecg)
        best_alpha_ecg = alpha_values[best_idx_ecg]
        #
        best_idx_ppg = np.argmax(pearsons_30sec_ppg)
        best_alpha_ppg = alpha_values[best_idx_ppg]
        #
        best_idx_ecg_ppg = np.argmax(pearsons_30sec_ecg_ppg)
        best_alpha_ecg_ppg = alpha_values[best_idx_ecg_ppg]
        #
        best_idx_ecg_ppg_mean = np.argmax(pearsons_30sec_ecg_ppg_mean)
        best_alpha_ecg_ppg_mean = alpha_values[best_idx_ecg_ppg_mean]

        print("###############")
        print("###############")
        print("evaluated alpha range: %f -- %f" % (alpha_values[0], alpha_values[-1]))
        print(alpha_values)
        print("optimal alpha according to ECG-only Pearson corr: %f" % best_alpha_ecg)
        print("optimal alpha according to PPG-only Pearson corr: %f" % best_alpha_ppg)
        print("optimal alpha according to ECG + PPG Pearson corr: %f" % best_alpha_ecg_ppg)
        print("optimal alpha according to ECG-only + PPG-only (mean of features) Pearson corr: %f" % best_alpha_ecg_ppg_mean)

        plt.figure(1)
        plt.plot(alpha_values, pearsons_30sec_ecg, "^-", color="C0", label="ECG-only -- 30sec")
        plt.plot(alpha_values, pearsons_30sec_ppg, "^-", color="C1", label="PPG-only -- 30sec")
        plt.plot(alpha_values, pearsons_30sec_ecg_ppg, "^-", color="C2", label="ECG + PPG -- 30sec")
        plt.plot(alpha_values, pearsons_30sec_ecg_ppg_mean, "^-", color="C3", label="ECG-only + PPG-only (mean of features) -- 30sec")
        plt.ylabel("Pearson corr")
        plt.xlabel("alpha")
        plt.xscale("log")
        plt.legend()
        plt.savefig("%s/Ridge_val_pearsons_combined_30sec_age_alpha_values_train_prop%.2f_run_i%d.png" % (save_dir_path, train_prop, run_i), dpi=300)
        plt.close(1)




        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        train_sexs = []
        train_features_list_30sec_ecg = []
        train_features_list_30sec_ppg = []
        train_features_list_30sec_ecg_ppg = []
        for i, csn in enumerate(preprocessed_csns):
            if csn in train_csns:
                sex_values = visits_df[visits_df["CSN"] == csn]["Gender"].values
                assert len(sex_values) == 1
                if sex_values[0] in ["M", "F"]:
                    train_sexs.append(sex_values[0])
                    train_features_list_30sec_ecg.append(features_30sec_ecg[i])
                    train_features_list_30sec_ppg.append(features_30sec_ppg[i])
                    train_features_list_30sec_ecg_ppg.append(features_30sec_ecg_ppg[i])
        train_features_30sec_ecg = np.stack(train_features_list_30sec_ecg, axis=0)
        train_features_30sec_ppg = np.stack(train_features_list_30sec_ppg, axis=0)
        train_features_30sec_ecg_ppg = np.stack(train_features_list_30sec_ecg_ppg, axis=0)
        train_sexs = list(map({"M": 0, "F": 1}.get, train_sexs))
        print(len(train_sexs))
        print(train_features_30sec_ecg.shape)
        print(train_features_30sec_ppg.shape)
        print(train_features_30sec_ecg_ppg.shape)

        val_sexs = []
        val_features_list_30sec_ecg = []
        val_features_list_30sec_ppg = []
        val_features_list_30sec_ecg_ppg = []
        for i, csn in enumerate(preprocessed_csns):
            if csn in val_csns:
                sex_values = visits_df[visits_df["CSN"] == csn]["Gender"].values
                assert len(sex_values) == 1
                if sex_values[0] in ["M", "F"]:
                    val_sexs.append(sex_values[0])
                    val_features_list_30sec_ecg.append(features_30sec_ecg[i])
                    val_features_list_30sec_ppg.append(features_30sec_ppg[i])
                    val_features_list_30sec_ecg_ppg.append(features_30sec_ecg_ppg[i])
        val_features_30sec_ecg = np.stack(val_features_list_30sec_ecg, axis=0)
        val_features_30sec_ppg = np.stack(val_features_list_30sec_ppg, axis=0)
        val_features_30sec_ecg_ppg = np.stack(val_features_list_30sec_ecg_ppg, axis=0)
        val_sexs = list(map({"M": 0, "F": 1}.get, val_sexs))
        print(len(val_sexs))
        print(val_features_30sec_ecg.shape)
        print(val_features_30sec_ppg.shape)
        print(val_features_30sec_ecg_ppg.shape)

        X_train_30sec_ecg = train_features_30sec_ecg
        X_train_30sec_ppg = train_features_30sec_ppg
        X_train_30sec_ecg_ppg = train_features_30sec_ecg_ppg
        X_train_30sec_ecg_ppg_mean = (X_train_30sec_ecg + X_train_30sec_ppg)/2.0
        y_train = np.array(train_sexs)
        #######
        indices = np.arange(X_train_30sec_ecg.shape[0])
        train_inds = rng.choice(indices, size=int(train_prop*len(indices)), replace=True)
        X_train_30sec_ecg = X_train_30sec_ecg[train_inds]
        X_train_30sec_ppg = X_train_30sec_ppg[train_inds]
        X_train_30sec_ecg_ppg = X_train_30sec_ecg_ppg[train_inds]
        X_train_30sec_ecg_ppg_mean = X_train_30sec_ecg_ppg_mean[train_inds]
        y_train = y_train[train_inds]
        print(len(y_train))
        print(X_train_30sec_ecg.shape)
        print(X_train_30sec_ppg.shape)
        print(X_train_30sec_ecg_ppg.shape)
        print(X_train_30sec_ecg_ppg_mean.shape)

        X_val_30sec_ecg = val_features_30sec_ecg
        X_val_30sec_ppg = val_features_30sec_ppg
        X_val_30sec_ecg_ppg = val_features_30sec_ecg_ppg
        X_val_30sec_ecg_ppg_mean = (X_val_30sec_ecg + X_val_30sec_ppg)/2.0
        y_val = np.array(val_sexs)

        auprcs_30sec_ecg = []
        rocaucs_30sec_ecg = []
        auprcs_30sec_ppg = []
        rocaucs_30sec_ppg = []
        auprcs_30sec_ecg_ppg = []
        rocaucs_30sec_ecg_ppg = []
        auprcs_30sec_ecg_ppg_mean = []
        rocaucs_30sec_ecg_ppg_mean = []
        for c_value in c_values:
            print(c_value)

            pipe = make_pipeline(StandardScaler(), LogisticRegression(C=c_value, penalty="l2", solver="lbfgs", max_iter=2000))
            pipe.fit(X_train_30sec_ecg, y_train)
            y_val_pred = pipe.predict(X_val_30sec_ecg)
            y_val_proba = pipe.predict_proba(X_val_30sec_ecg)[:, 1]
            rocauc = roc_auc_score(y_val, y_val_proba)
            rocaucs_30sec_ecg.append(rocauc)
            auprc = average_precision_score(y_val, y_val_proba)
            auprcs_30sec_ecg.append(auprc)

            pipe = make_pipeline(StandardScaler(), LogisticRegression(C=c_value, penalty="l2", solver="lbfgs", max_iter=2000))
            pipe.fit(X_train_30sec_ppg, y_train)
            y_val_pred = pipe.predict(X_val_30sec_ppg)
            y_val_proba = pipe.predict_proba(X_val_30sec_ppg)[:, 1]
            rocauc = roc_auc_score(y_val, y_val_proba)
            rocaucs_30sec_ppg.append(rocauc)
            auprc = average_precision_score(y_val, y_val_proba)
            auprcs_30sec_ppg.append(auprc)

            pipe = make_pipeline(StandardScaler(), LogisticRegression(C=c_value, penalty="l2", solver="lbfgs", max_iter=2000))
            pipe.fit(X_train_30sec_ecg_ppg, y_train)
            y_val_pred = pipe.predict(X_val_30sec_ecg_ppg)
            y_val_proba = pipe.predict_proba(X_val_30sec_ecg_ppg)[:, 1]
            rocauc = roc_auc_score(y_val, y_val_proba)
            rocaucs_30sec_ecg_ppg.append(rocauc)
            auprc = average_precision_score(y_val, y_val_proba)
            auprcs_30sec_ecg_ppg.append(auprc)

            pipe = make_pipeline(StandardScaler(), LogisticRegression(C=c_value, penalty="l2", solver="lbfgs", max_iter=2000))
            pipe.fit(X_train_30sec_ecg_ppg_mean, y_train)
            y_val_pred = pipe.predict(X_val_30sec_ecg_ppg_mean)
            y_val_proba = pipe.predict_proba(X_val_30sec_ecg_ppg_mean)[:, 1]
            rocauc = roc_auc_score(y_val, y_val_proba)
            rocaucs_30sec_ecg_ppg_mean.append(rocauc)
            auprc = average_precision_score(y_val, y_val_proba)
            auprcs_30sec_ecg_ppg_mean.append(auprc)

        best_idx_ecg = np.argmax(rocaucs_30sec_ecg)
        best_c_ecg = c_values[best_idx_ecg]
        #
        best_idx_ppg = np.argmax(rocaucs_30sec_ppg)
        best_c_ppg = c_values[best_idx_ppg]
        #
        best_idx_ecg_ppg = np.argmax(rocaucs_30sec_ecg_ppg)
        best_c_ecg_ppg = c_values[best_idx_ecg_ppg]
        #
        best_idx_ecg_ppg_mean = np.argmax(rocaucs_30sec_ecg_ppg_mean)
        best_c_ecg_ppg_mean = c_values[best_idx_ecg_ppg_mean]

        print("###############")
        print("###############")
        print("evaluated C range: %f -- %f" % (c_values[0], c_values[-1]))
        print(c_values)
        print("optimal C according to ECG-only AUROC: %f" % best_c_ecg)
        print("optimal C according to PPG-only AUROC: %f" % best_c_ppg)
        print("optimal C according to ECG + PPG AUROC: %f" % best_c_ecg_ppg)
        print("optimal C according to ECG-only + PPG-only (mean of features) AUROC: %f" % best_c_ecg_ppg_mean)

        plt.figure(1)
        plt.plot(c_values, rocaucs_30sec_ecg, "^-", color="C0", label="ECG-only -- 30sec")
        plt.plot(c_values, rocaucs_30sec_ppg, "^-", color="C1", label="PPG-only -- 30sec")
        plt.plot(c_values, rocaucs_30sec_ecg_ppg, "^-", color="C2", label="ECG + PPG -- 30sec")
        plt.plot(c_values, rocaucs_30sec_ecg_ppg_mean, "^-", color="C3", label="ECG-only + PPG-only (mean of features) -- 30sec")
        plt.ylabel("AUROC")
        plt.xlabel("C")
        plt.xscale("log")
        plt.legend()
        plt.savefig("%s/LogisticRegression_val_rocaucs_combined_30sec_sex_c_values_train_prop%.2f_run_i%d.png" % (save_dir_path, train_prop, run_i), dpi=300)
        plt.close(1)








        all_tasks_pearson_values_ecg = []
        all_tasks_mae_values_ecg = []
        all_tasks_auroc_values_ecg = []
        all_tasks_auprc_values_ecg = []
        #
        all_tasks_pearson_values_ppg = []
        all_tasks_mae_values_ppg = []
        all_tasks_auroc_values_ppg = []
        all_tasks_auprc_values_ppg = []
        #
        all_tasks_pearson_values_ecg_ppg = []
        all_tasks_mae_values_ecg_ppg = []
        all_tasks_auroc_values_ecg_ppg = []
        all_tasks_auprc_values_ecg_ppg = []
        #
        all_tasks_pearson_values_ecg_ppg_mean = []
        all_tasks_mae_values_ecg_ppg_mean = []
        all_tasks_auroc_values_ecg_ppg_mean = []
        all_tasks_auprc_values_ecg_ppg_mean = []

        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("1. Age regression:")
        train_ages = []
        train_features_list_30sec_ecg = []
        train_features_list_30sec_ppg = []
        train_features_list_30sec_ecg_ppg = []
        for i, csn in enumerate(preprocessed_csns):
            if csn in train_csns:
                age_values = visits_df[visits_df["CSN"] == csn]["Age"].values
                assert len(age_values) == 1
                train_ages.append(age_values[0])
                train_features_list_30sec_ecg.append(features_30sec_ecg[i])
                train_features_list_30sec_ppg.append(features_30sec_ppg[i])
                train_features_list_30sec_ecg_ppg.append(features_30sec_ecg_ppg[i])
        train_features_30sec_ecg = np.stack(train_features_list_30sec_ecg, axis=0)
        train_features_30sec_ppg = np.stack(train_features_list_30sec_ppg, axis=0)
        train_features_30sec_ecg_ppg = np.stack(train_features_list_30sec_ecg_ppg, axis=0)
        print(len(train_ages))
        print(train_features_30sec_ecg.shape)
        print(train_features_30sec_ppg.shape)
        print(train_features_30sec_ecg_ppg.shape)

        val_ages = []
        val_features_list_30sec_ecg = []
        val_features_list_30sec_ppg = []
        val_features_list_30sec_ecg_ppg = []
        for i, csn in enumerate(preprocessed_csns):
            if csn in val_csns:
                age_values = visits_df[visits_df["CSN"] == csn]["Age"].values
                assert len(age_values) == 1
                val_ages.append(age_values[0])
                val_features_list_30sec_ecg.append(features_30sec_ecg[i])
                val_features_list_30sec_ppg.append(features_30sec_ppg[i])
                val_features_list_30sec_ecg_ppg.append(features_30sec_ecg_ppg[i])
        val_features_30sec_ecg = np.stack(val_features_list_30sec_ecg, axis=0)
        val_features_30sec_ppg = np.stack(val_features_list_30sec_ppg, axis=0)
        val_features_30sec_ecg_ppg = np.stack(val_features_list_30sec_ecg_ppg, axis=0)
        print(len(val_ages))
        print(val_features_30sec_ecg.shape)
        print(val_features_30sec_ppg.shape)
        print(val_features_30sec_ecg_ppg.shape)

        X_train_30sec_ecg = train_features_30sec_ecg
        X_train_30sec_ppg = train_features_30sec_ppg
        X_train_30sec_ecg_ppg = train_features_30sec_ecg_ppg
        X_train_30sec_ecg_ppg_mean = (X_train_30sec_ecg + X_train_30sec_ppg)/2.0
        y_train = np.array(train_ages)
        #######
        indices = np.arange(X_train_30sec_ecg.shape[0])
        train_inds = rng.choice(indices, size=int(train_prop*len(indices)), replace=True)
        X_train_30sec_ecg = X_train_30sec_ecg[train_inds]
        X_train_30sec_ppg = X_train_30sec_ppg[train_inds]
        X_train_30sec_ecg_ppg = X_train_30sec_ecg_ppg[train_inds]
        X_train_30sec_ecg_ppg_mean = X_train_30sec_ecg_ppg_mean[train_inds]
        y_train = y_train[train_inds]
        print(len(y_train))
        print(X_train_30sec_ecg.shape)
        print(X_train_30sec_ppg.shape)
        print(X_train_30sec_ecg_ppg.shape)
        print(X_train_30sec_ecg_ppg_mean.shape)

        X_val_30sec_ecg = val_features_30sec_ecg
        X_val_30sec_ppg = val_features_30sec_ppg
        X_val_30sec_ecg_ppg = val_features_30sec_ecg_ppg
        X_val_30sec_ecg_ppg_mean = (X_val_30sec_ecg + X_val_30sec_ppg)/2.0
        y_val = np.array(val_ages)

        pipe = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha_ecg))
        pipe.fit(X_train_30sec_ecg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg)
        mae = np.mean(np.abs(y_val - y_val_pred))
        pearson_corr, _ = pearsonr(y_val_pred, y_val)
        all_tasks_pearson_values_ecg.append(pearson_corr)
        all_tasks_mae_values_ecg.append(mae)
        print("30sec -- alpha: %f -- ECG-only -- Pearson: %.3f, MAE: %.3f" % (best_alpha_ecg, pearson_corr, mae))

        pipe = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha_ppg))
        pipe.fit(X_train_30sec_ppg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ppg)
        mae = np.mean(np.abs(y_val - y_val_pred))
        pearson_corr, _ = pearsonr(y_val_pred, y_val)
        all_tasks_pearson_values_ppg.append(pearson_corr)
        all_tasks_mae_values_ppg.append(mae)
        print("30sec -- alpha: %f -- PPG-only -- Pearson: %.3f, MAE: %.3f" % (best_alpha_ppg, pearson_corr, mae))

        pipe = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha_ecg_ppg))
        pipe.fit(X_train_30sec_ecg_ppg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg_ppg)
        mae = np.mean(np.abs(y_val - y_val_pred))
        pearson_corr, _ = pearsonr(y_val_pred, y_val)
        all_tasks_pearson_values_ecg_ppg.append(pearson_corr)
        all_tasks_mae_values_ecg_ppg.append(mae)
        print("30sec -- alpha: %f -- ECG + PPG -- Pearson: %.3f, MAE: %.3f" % (best_alpha_ecg_ppg, pearson_corr, mae))

        pipe = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha_ecg_ppg_mean))
        pipe.fit(X_train_30sec_ecg_ppg_mean, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg_ppg_mean)
        mae = np.mean(np.abs(y_val - y_val_pred))
        pearson_corr, _ = pearsonr(y_val_pred, y_val)
        all_tasks_pearson_values_ecg_ppg_mean.append(pearson_corr)
        all_tasks_mae_values_ecg_ppg_mean.append(mae)
        print("30sec -- alpha: %f -- ECG-only + PPG-only (mean of features) -- Pearson: %.3f, MAE: %.3f" % (best_alpha_ecg_ppg_mean, pearson_corr, mae))




        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("2. Sex classification ('M' (negative class) vs 'F'):")
        train_sexs = []
        train_features_list_30sec_ecg = []
        train_features_list_30sec_ppg = []
        train_features_list_30sec_ecg_ppg = []
        for i, csn in enumerate(preprocessed_csns):
            if csn in train_csns:
                sex_values = visits_df[visits_df["CSN"] == csn]["Gender"].values
                assert len(sex_values) == 1
                if sex_values[0] in ["M", "F"]:
                    train_sexs.append(sex_values[0])
                    train_features_list_30sec_ecg.append(features_30sec_ecg[i])
                    train_features_list_30sec_ppg.append(features_30sec_ppg[i])
                    train_features_list_30sec_ecg_ppg.append(features_30sec_ecg_ppg[i])
        train_features_30sec_ecg = np.stack(train_features_list_30sec_ecg, axis=0)
        train_features_30sec_ppg = np.stack(train_features_list_30sec_ppg, axis=0)
        train_features_30sec_ecg_ppg = np.stack(train_features_list_30sec_ecg_ppg, axis=0)
        train_sexs = list(map({"M": 0, "F": 1}.get, train_sexs))
        print(len(train_sexs))
        print(train_features_30sec_ecg.shape)
        print(train_features_30sec_ppg.shape)
        print(train_features_30sec_ecg_ppg.shape)
        print("proportion of positive class in train: %.4f" % (train_sexs.count(1)/float(len(train_sexs))))

        val_sexs = []
        val_features_list_30sec_ecg = []
        val_features_list_30sec_ppg = []
        val_features_list_30sec_ecg_ppg = []
        for i, csn in enumerate(preprocessed_csns):
            if csn in val_csns:
                sex_values = visits_df[visits_df["CSN"] == csn]["Gender"].values
                assert len(sex_values) == 1
                if sex_values[0] in ["M", "F"]:
                    val_sexs.append(sex_values[0])
                    val_features_list_30sec_ecg.append(features_30sec_ecg[i])
                    val_features_list_30sec_ppg.append(features_30sec_ppg[i])
                    val_features_list_30sec_ecg_ppg.append(features_30sec_ecg_ppg[i])
        val_features_30sec_ecg = np.stack(val_features_list_30sec_ecg, axis=0)
        val_features_30sec_ppg = np.stack(val_features_list_30sec_ppg, axis=0)
        val_features_30sec_ecg_ppg = np.stack(val_features_list_30sec_ecg_ppg, axis=0)
        val_sexs = list(map({"M": 0, "F": 1}.get, val_sexs))
        print(len(val_sexs))
        print(val_features_30sec_ecg.shape)
        print(val_features_30sec_ppg.shape)
        print(val_features_30sec_ecg_ppg.shape)
        print("proportion of positive class in train: %.4f" % (val_sexs.count(1)/float(len(val_sexs))))

        X_train_30sec_ecg = train_features_30sec_ecg
        X_train_30sec_ppg = train_features_30sec_ppg
        X_train_30sec_ecg_ppg = train_features_30sec_ecg_ppg
        X_train_30sec_ecg_ppg_mean = (X_train_30sec_ecg + X_train_30sec_ppg)/2.0
        y_train = np.array(train_sexs)
        #######
        indices = np.arange(X_train_30sec_ecg.shape[0])
        train_inds = rng.choice(indices, size=int(train_prop*len(indices)), replace=True)
        X_train_30sec_ecg = X_train_30sec_ecg[train_inds]
        X_train_30sec_ppg = X_train_30sec_ppg[train_inds]
        X_train_30sec_ecg_ppg = X_train_30sec_ecg_ppg[train_inds]
        X_train_30sec_ecg_ppg_mean = X_train_30sec_ecg_ppg_mean[train_inds]
        y_train = y_train[train_inds]
        print(len(y_train))
        print(X_train_30sec_ecg.shape)
        print(X_train_30sec_ppg.shape)
        print(X_train_30sec_ecg_ppg.shape)
        print(X_train_30sec_ecg_ppg_mean.shape)

        X_val_30sec_ecg = val_features_30sec_ecg
        X_val_30sec_ppg = val_features_30sec_ppg
        X_val_30sec_ecg_ppg = val_features_30sec_ecg_ppg
        X_val_30sec_ecg_ppg_mean = (X_val_30sec_ecg + X_val_30sec_ppg)/2.0
        y_val = np.array(val_sexs)

        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best_c_ecg, penalty="l2", solver="lbfgs", max_iter=2000))
        pipe.fit(X_train_30sec_ecg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg)
        y_val_proba = pipe.predict_proba(X_val_30sec_ecg)[:, 1]
        rocauc = roc_auc_score(y_val, y_val_proba)
        auprc = average_precision_score(y_val, y_val_proba)
        all_tasks_auroc_values_ecg.append(rocauc)
        all_tasks_auprc_values_ecg.append(auprc)
        print("30sec -- C: %f -- ECG-only -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ecg, rocauc, auprc))

        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best_c_ppg, penalty="l2", solver="lbfgs", max_iter=2000))
        pipe.fit(X_train_30sec_ppg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ppg)
        y_val_proba = pipe.predict_proba(X_val_30sec_ppg)[:, 1]
        rocauc = roc_auc_score(y_val, y_val_proba)
        auprc = average_precision_score(y_val, y_val_proba)
        all_tasks_auroc_values_ppg.append(rocauc)
        all_tasks_auprc_values_ppg.append(auprc)
        print("30sec -- C: %f -- PPG-only -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ppg, rocauc, auprc))

        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best_c_ecg_ppg, penalty="l2", solver="lbfgs", max_iter=2000))
        pipe.fit(X_train_30sec_ecg_ppg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg_ppg)
        y_val_proba = pipe.predict_proba(X_val_30sec_ecg_ppg)[:, 1]
        rocauc = roc_auc_score(y_val, y_val_proba)
        auprc = average_precision_score(y_val, y_val_proba)
        all_tasks_auroc_values_ecg_ppg.append(rocauc)
        all_tasks_auprc_values_ecg_ppg.append(auprc)
        print("30sec -- C: %f -- ECG + PPG -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ecg_ppg, rocauc, auprc))

        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best_c_ecg_ppg_mean, penalty="l2", solver="lbfgs", max_iter=2000))
        pipe.fit(X_train_30sec_ecg_ppg_mean, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg_ppg_mean)
        y_val_proba = pipe.predict_proba(X_val_30sec_ecg_ppg_mean)[:, 1]
        rocauc = roc_auc_score(y_val, y_val_proba)
        auprc = average_precision_score(y_val, y_val_proba)
        all_tasks_auroc_values_ecg_ppg_mean.append(rocauc)
        all_tasks_auprc_values_ecg_ppg_mean.append(auprc)
        print("30sec -- C: %f -- ECG-only + PPG-only (mean of features) -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ecg_ppg_mean, rocauc, auprc))




        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("3. ED_dispo classification ('Discharge' (negative class) vs 'Inpatient'+'Observation'+'ICU' (positive class)):")
        train_dispos = []
        train_features_list_30sec_ecg = []
        train_features_list_30sec_ppg = []
        train_features_list_30sec_ecg_ppg = []
        for i, csn in enumerate(preprocessed_csns):
            if csn in train_csns:
                dispo_values = visits_df[visits_df["CSN"] == csn]["ED_dispo"].values
                assert len(dispo_values) == 1
                train_dispos.append(dispo_values[0])
                train_features_list_30sec_ecg.append(features_30sec_ecg[i])
                train_features_list_30sec_ppg.append(features_30sec_ppg[i])
                train_features_list_30sec_ecg_ppg.append(features_30sec_ecg_ppg[i])
        train_features_30sec_ecg = np.stack(train_features_list_30sec_ecg, axis=0)
        train_features_30sec_ppg = np.stack(train_features_list_30sec_ppg, axis=0)
        train_features_30sec_ecg_ppg = np.stack(train_features_list_30sec_ecg_ppg, axis=0)
        train_dispos = list(map({"Discharge": 0, "Inpatient": 1, "ICU": 1, "Observation": 1}.get, train_dispos))
        print(len(train_dispos))
        print(train_features_30sec_ecg.shape)
        print(train_features_30sec_ppg.shape)
        print(train_features_30sec_ecg_ppg.shape)
        print("proportion of positive class in train: %.4f" % (train_dispos.count(1)/float(len(train_dispos))))

        val_dispos = []
        val_features_list_30sec_ecg = []
        val_features_list_30sec_ppg = []
        val_features_list_30sec_ecg_ppg = []
        for i, csn in enumerate(preprocessed_csns):
            if csn in val_csns:
                dispo_values = visits_df[visits_df["CSN"] == csn]["ED_dispo"].values
                assert len(dispo_values) == 1
                val_dispos.append(dispo_values[0])
                val_features_list_30sec_ecg.append(features_30sec_ecg[i])
                val_features_list_30sec_ppg.append(features_30sec_ppg[i])
                val_features_list_30sec_ecg_ppg.append(features_30sec_ecg_ppg[i])
        val_features_30sec_ecg = np.stack(val_features_list_30sec_ecg, axis=0)
        val_features_30sec_ppg = np.stack(val_features_list_30sec_ppg, axis=0)
        val_features_30sec_ecg_ppg = np.stack(val_features_list_30sec_ecg_ppg, axis=0)
        val_dispos = list(map({"Discharge": 0, "Inpatient": 1, "ICU": 1, "Observation": 1}.get, val_dispos))
        print(len(val_dispos))
        print(val_features_30sec_ecg.shape)
        print(val_features_30sec_ppg.shape)
        print(val_features_30sec_ecg_ppg.shape)
        print("proportion of positive class in val: %.4f" % (val_dispos.count(1)/float(len(val_dispos))))

        X_train_30sec_ecg = train_features_30sec_ecg
        X_train_30sec_ppg = train_features_30sec_ppg
        X_train_30sec_ecg_ppg = train_features_30sec_ecg_ppg
        X_train_30sec_ecg_ppg_mean = (X_train_30sec_ecg + X_train_30sec_ppg)/2.0
        y_train = np.array(train_dispos)
        #######
        indices = np.arange(X_train_30sec_ecg.shape[0])
        train_inds = rng.choice(indices, size=int(train_prop*len(indices)), replace=True)
        X_train_30sec_ecg = X_train_30sec_ecg[train_inds]
        X_train_30sec_ppg = X_train_30sec_ppg[train_inds]
        X_train_30sec_ecg_ppg = X_train_30sec_ecg_ppg[train_inds]
        X_train_30sec_ecg_ppg_mean = X_train_30sec_ecg_ppg_mean[train_inds]
        y_train = y_train[train_inds]
        print(len(y_train))
        print(X_train_30sec_ecg.shape)
        print(X_train_30sec_ppg.shape)
        print(X_train_30sec_ecg_ppg.shape)
        print(X_train_30sec_ecg_ppg_mean.shape)

        X_val_30sec_ecg = val_features_30sec_ecg
        X_val_30sec_ppg = val_features_30sec_ppg
        X_val_30sec_ecg_ppg = val_features_30sec_ecg_ppg
        X_val_30sec_ecg_ppg_mean = (X_val_30sec_ecg + X_val_30sec_ppg)/2.0
        y_val = np.array(val_dispos)

        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best_c_ecg, penalty="l2", solver="lbfgs", max_iter=2000))
        pipe.fit(X_train_30sec_ecg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg)
        y_val_proba = pipe.predict_proba(X_val_30sec_ecg)[:, 1]
        rocauc = roc_auc_score(y_val, y_val_proba)
        auprc = average_precision_score(y_val, y_val_proba)
        all_tasks_auroc_values_ecg.append(rocauc)
        all_tasks_auprc_values_ecg.append(auprc)
        print("30sec -- C: %f -- ECG-only -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ecg, rocauc, auprc))

        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best_c_ppg, penalty="l2", solver="lbfgs", max_iter=2000))
        pipe.fit(X_train_30sec_ppg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ppg)
        y_val_proba = pipe.predict_proba(X_val_30sec_ppg)[:, 1]
        rocauc = roc_auc_score(y_val, y_val_proba)
        auprc = average_precision_score(y_val, y_val_proba)
        all_tasks_auroc_values_ppg.append(rocauc)
        all_tasks_auprc_values_ppg.append(auprc)
        print("30sec -- C: %f -- PPG-only -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ppg, rocauc, auprc))

        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best_c_ecg_ppg, penalty="l2", solver="lbfgs", max_iter=2000))
        pipe.fit(X_train_30sec_ecg_ppg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg_ppg)
        y_val_proba = pipe.predict_proba(X_val_30sec_ecg_ppg)[:, 1]
        rocauc = roc_auc_score(y_val, y_val_proba)
        auprc = average_precision_score(y_val, y_val_proba)
        all_tasks_auroc_values_ecg_ppg.append(rocauc)
        all_tasks_auprc_values_ecg_ppg.append(auprc)
        print("30sec -- C: %f -- ECG + PPG -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ecg_ppg, rocauc, auprc))

        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best_c_ecg_ppg_mean, penalty="l2", solver="lbfgs", max_iter=2000))
        pipe.fit(X_train_30sec_ecg_ppg_mean, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg_ppg_mean)
        y_val_proba = pipe.predict_proba(X_val_30sec_ecg_ppg_mean)[:, 1]
        rocauc = roc_auc_score(y_val, y_val_proba)
        auprc = average_precision_score(y_val, y_val_proba)
        all_tasks_auroc_values_ecg_ppg_mean.append(rocauc)
        all_tasks_auprc_values_ecg_ppg_mean.append(auprc)
        print("30sec -- C: %f -- ECG-only + PPG-only (mean of features) -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ecg_ppg_mean, rocauc, auprc))








        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("4. Potassium regression ('POTASSIUM'):")
        potassium_df = labs_df[labs_df["Component_name"] == "POTASSIUM"]
        # print(potassium_df)
        filtered = potassium_df.groupby("CSN").filter(lambda x: len(x) == 1) # (only keep visits with a single POTASSIUM value, NOTE! NOTE! NOTE!)
        # print(filtered)
        csn_to_potassium = dict(zip(filtered["CSN"], filtered["Component_value"]))
        potassium_values = [float(csn_to_potassium[csn]) for csn in preprocessed_csns if ((csn in csn_to_potassium) and (csn_to_potassium[csn] != '>10.0'))] ################
        print(len(potassium_values))
        print(np.min(potassium_values))
        print(np.mean(potassium_values))
        print(np.max(potassium_values))

        train_potassiums = []
        train_features_list_30sec_ecg = []
        train_features_list_30sec_ppg = []
        train_features_list_30sec_ecg_ppg = []
        for i, csn in enumerate(preprocessed_csns):
            if csn in train_csns:
                if (csn in csn_to_potassium) and (csn_to_potassium[csn] != '>10.0'): ####################
                    train_potassiums.append(float(csn_to_potassium[csn]))
                    train_features_list_30sec_ecg.append(features_30sec_ecg[i])
                    train_features_list_30sec_ppg.append(features_30sec_ppg[i])
                    train_features_list_30sec_ecg_ppg.append(features_30sec_ecg_ppg[i])
        train_features_30sec_ecg = np.stack(train_features_list_30sec_ecg, axis=0)
        train_features_30sec_ppg = np.stack(train_features_list_30sec_ppg, axis=0)
        train_features_30sec_ecg_ppg = np.stack(train_features_list_30sec_ecg_ppg, axis=0)
        print(len(train_potassiums))
        print(train_features_30sec_ecg.shape)
        print(train_features_30sec_ppg.shape)
        print(train_features_30sec_ecg_ppg.shape)
        print("train label stats:")
        print("min: %.2f" % np.min(train_potassiums))
        print("mean: %.2f" % np.mean(train_potassiums))
        print("max: %.2f" % np.max(train_potassiums))

        val_potassiums = []
        val_features_list_30sec_ecg = []
        val_features_list_30sec_ppg = []
        val_features_list_30sec_ecg_ppg = []
        for i, csn in enumerate(preprocessed_csns):
            if csn in val_csns:
                if (csn in csn_to_potassium) and (csn_to_potassium[csn] != '>10.0'): ####################
                    val_potassiums.append(float(csn_to_potassium[csn]))
                    val_features_list_30sec_ecg.append(features_30sec_ecg[i])
                    val_features_list_30sec_ppg.append(features_30sec_ppg[i])
                    val_features_list_30sec_ecg_ppg.append(features_30sec_ecg_ppg[i])
        val_features_30sec_ecg = np.stack(val_features_list_30sec_ecg, axis=0)
        val_features_30sec_ppg = np.stack(val_features_list_30sec_ppg, axis=0)
        val_features_30sec_ecg_ppg = np.stack(val_features_list_30sec_ecg_ppg, axis=0)
        print(len(val_potassiums))
        print(val_features_30sec_ecg.shape)
        print(val_features_30sec_ppg.shape)
        print(val_features_30sec_ecg_ppg.shape)
        print("val label stats:")
        print("min: %.2f" % np.min(val_potassiums))
        print("mean: %.2f" % np.mean(val_potassiums))
        print("max: %.2f" % np.max(val_potassiums))

        X_train_30sec_ecg = train_features_30sec_ecg
        X_train_30sec_ppg = train_features_30sec_ppg
        X_train_30sec_ecg_ppg = train_features_30sec_ecg_ppg
        X_train_30sec_ecg_ppg_mean = (X_train_30sec_ecg + X_train_30sec_ppg)/2.0
        y_train = np.array(train_potassiums)
        #######
        indices = np.arange(X_train_30sec_ecg.shape[0])
        train_inds = rng.choice(indices, size=int(train_prop*len(indices)), replace=True)
        X_train_30sec_ecg = X_train_30sec_ecg[train_inds]
        X_train_30sec_ppg = X_train_30sec_ppg[train_inds]
        X_train_30sec_ecg_ppg = X_train_30sec_ecg_ppg[train_inds]
        X_train_30sec_ecg_ppg_mean = X_train_30sec_ecg_ppg_mean[train_inds]
        y_train = y_train[train_inds]
        print(len(y_train))
        print(X_train_30sec_ecg.shape)
        print(X_train_30sec_ppg.shape)
        print(X_train_30sec_ecg_ppg.shape)
        print(X_train_30sec_ecg_ppg_mean.shape)

        X_val_30sec_ecg = val_features_30sec_ecg
        X_val_30sec_ppg = val_features_30sec_ppg
        X_val_30sec_ecg_ppg = val_features_30sec_ecg_ppg
        X_val_30sec_ecg_ppg_mean = (X_val_30sec_ecg + X_val_30sec_ppg)/2.0
        y_val = np.array(val_potassiums)

        pipe = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha_ecg))
        pipe.fit(X_train_30sec_ecg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg)
        mae = np.mean(np.abs(y_val - y_val_pred))
        pearson_corr, _ = pearsonr(y_val_pred, y_val)
        all_tasks_pearson_values_ecg.append(pearson_corr)
        all_tasks_mae_values_ecg.append(mae)
        print("30sec -- alpha: %f -- ECG-only -- Pearson: %.3f, MAE: %.3f" % (best_alpha_ecg, pearson_corr, mae))

        pipe = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha_ppg))
        pipe.fit(X_train_30sec_ppg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ppg)
        mae = np.mean(np.abs(y_val - y_val_pred))
        pearson_corr, _ = pearsonr(y_val_pred, y_val)
        all_tasks_pearson_values_ppg.append(pearson_corr)
        all_tasks_mae_values_ppg.append(mae)
        print("30sec -- alpha: %f -- PPG-only -- Pearson: %.3f, MAE: %.3f" % (best_alpha_ppg, pearson_corr, mae))

        pipe = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha_ecg_ppg))
        pipe.fit(X_train_30sec_ecg_ppg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg_ppg)
        mae = np.mean(np.abs(y_val - y_val_pred))
        pearson_corr, _ = pearsonr(y_val_pred, y_val)
        all_tasks_pearson_values_ecg_ppg.append(pearson_corr)
        all_tasks_mae_values_ecg_ppg.append(mae)
        print("30sec -- alpha: %f -- ECG + PPG -- Pearson: %.3f, MAE: %.3f" % (best_alpha_ecg_ppg, pearson_corr, mae))

        pipe = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha_ecg_ppg_mean))
        pipe.fit(X_train_30sec_ecg_ppg_mean, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg_ppg_mean)
        mae = np.mean(np.abs(y_val - y_val_pred))
        pearson_corr, _ = pearsonr(y_val_pred, y_val)
        all_tasks_pearson_values_ecg_ppg_mean.append(pearson_corr)
        all_tasks_mae_values_ecg_ppg_mean.append(mae)
        print("30sec -- alpha: %f -- ECG-only + PPG-only (mean of features) -- Pearson: %.3f, MAE: %.3f" % (best_alpha_ecg_ppg_mean, pearson_corr, mae))




        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("5. Calcium regression ('CALCIUM'):")
        calcium_df = labs_df[labs_df["Component_name"] == "CALCIUM"]
        # print(calcium_df)
        filtered = calcium_df.groupby("CSN").filter(lambda x: len(x) == 1) # (only keep visits with a single calcium value, NOTE! NOTE! NOTE!)
        # print(filtered)
        csn_to_calcium = dict(zip(filtered["CSN"], filtered["Component_value"]))
        calcium_values = [float(csn_to_calcium[csn]) for csn in preprocessed_csns if csn in csn_to_calcium]
        print(len(calcium_values))
        print(np.min(calcium_values))
        print(np.mean(calcium_values))
        print(np.max(calcium_values))

        train_calciums = []
        train_features_list_30sec_ecg = []
        train_features_list_30sec_ppg = []
        train_features_list_30sec_ecg_ppg = []
        for i, csn in enumerate(preprocessed_csns):
            if csn in train_csns:
                if csn in csn_to_calcium:
                    train_calciums.append(float(csn_to_calcium[csn]))
                    train_features_list_30sec_ecg.append(features_30sec_ecg[i])
                    train_features_list_30sec_ppg.append(features_30sec_ppg[i])
                    train_features_list_30sec_ecg_ppg.append(features_30sec_ecg_ppg[i])
        train_features_30sec_ecg = np.stack(train_features_list_30sec_ecg, axis=0)
        train_features_30sec_ppg = np.stack(train_features_list_30sec_ppg, axis=0)
        train_features_30sec_ecg_ppg = np.stack(train_features_list_30sec_ecg_ppg, axis=0)
        print(len(train_calciums))
        print(train_features_30sec_ecg.shape)
        print(train_features_30sec_ppg.shape)
        print(train_features_30sec_ecg_ppg.shape)
        print("train label stats:")
        print("min: %.2f" % np.min(train_calciums))
        print("mean: %.2f" % np.mean(train_calciums))
        print("max: %.2f" % np.max(train_calciums))

        val_calciums = []
        val_features_list_30sec_ecg = []
        val_features_list_30sec_ppg = []
        val_features_list_30sec_ecg_ppg = []
        for i, csn in enumerate(preprocessed_csns):
            if csn in val_csns:
                if csn in csn_to_calcium:
                    val_calciums.append(float(csn_to_calcium[csn]))
                    val_features_list_30sec_ecg.append(features_30sec_ecg[i])
                    val_features_list_30sec_ppg.append(features_30sec_ppg[i])
                    val_features_list_30sec_ecg_ppg.append(features_30sec_ecg_ppg[i])
        val_features_30sec_ecg = np.stack(val_features_list_30sec_ecg, axis=0)
        val_features_30sec_ppg = np.stack(val_features_list_30sec_ppg, axis=0)
        val_features_30sec_ecg_ppg = np.stack(val_features_list_30sec_ecg_ppg, axis=0)
        print(len(val_calciums))
        print(val_features_30sec_ecg.shape)
        print(val_features_30sec_ppg.shape)
        print(val_features_30sec_ecg_ppg.shape)
        print("val label stats:")
        print("min: %.2f" % np.min(val_calciums))
        print("mean: %.2f" % np.mean(val_calciums))
        print("max: %.2f" % np.max(val_calciums))

        X_train_30sec_ecg = train_features_30sec_ecg
        X_train_30sec_ppg = train_features_30sec_ppg
        X_train_30sec_ecg_ppg = train_features_30sec_ecg_ppg
        X_train_30sec_ecg_ppg_mean = (X_train_30sec_ecg + X_train_30sec_ppg)/2.0
        y_train = np.array(train_calciums)
        #######
        indices = np.arange(X_train_30sec_ecg.shape[0])
        train_inds = rng.choice(indices, size=int(train_prop*len(indices)), replace=True)
        X_train_30sec_ecg = X_train_30sec_ecg[train_inds]
        X_train_30sec_ppg = X_train_30sec_ppg[train_inds]
        X_train_30sec_ecg_ppg = X_train_30sec_ecg_ppg[train_inds]
        X_train_30sec_ecg_ppg_mean = X_train_30sec_ecg_ppg_mean[train_inds]
        y_train = y_train[train_inds]
        print(len(y_train))
        print(X_train_30sec_ecg.shape)
        print(X_train_30sec_ppg.shape)
        print(X_train_30sec_ecg_ppg.shape)
        print(X_train_30sec_ecg_ppg_mean.shape)

        X_val_30sec_ecg = val_features_30sec_ecg
        X_val_30sec_ppg = val_features_30sec_ppg
        X_val_30sec_ecg_ppg = val_features_30sec_ecg_ppg
        X_val_30sec_ecg_ppg_mean = (X_val_30sec_ecg + X_val_30sec_ppg)/2.0
        y_val = np.array(val_calciums)

        pipe = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha_ecg))
        pipe.fit(X_train_30sec_ecg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg)
        mae = np.mean(np.abs(y_val - y_val_pred))
        pearson_corr, _ = pearsonr(y_val_pred, y_val)
        all_tasks_pearson_values_ecg.append(pearson_corr)
        all_tasks_mae_values_ecg.append(mae)
        print("30sec -- alpha: %f -- ECG-only -- Pearson: %.3f, MAE: %.3f" % (best_alpha_ecg, pearson_corr, mae))

        pipe = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha_ppg))
        pipe.fit(X_train_30sec_ppg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ppg)
        mae = np.mean(np.abs(y_val - y_val_pred))
        pearson_corr, _ = pearsonr(y_val_pred, y_val)
        all_tasks_pearson_values_ppg.append(pearson_corr)
        all_tasks_mae_values_ppg.append(mae)
        print("30sec -- alpha: %f -- PPG-only -- Pearson: %.3f, MAE: %.3f" % (best_alpha_ppg, pearson_corr, mae))

        pipe = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha_ecg_ppg))
        pipe.fit(X_train_30sec_ecg_ppg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg_ppg)
        mae = np.mean(np.abs(y_val - y_val_pred))
        pearson_corr, _ = pearsonr(y_val_pred, y_val)
        all_tasks_pearson_values_ecg_ppg.append(pearson_corr)
        all_tasks_mae_values_ecg_ppg.append(mae)
        print("30sec -- alpha: %f -- ECG + PPG -- Pearson: %.3f, MAE: %.3f" % (best_alpha_ecg_ppg, pearson_corr, mae))

        pipe = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha_ecg_ppg_mean))
        pipe.fit(X_train_30sec_ecg_ppg_mean, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg_ppg_mean)
        mae = np.mean(np.abs(y_val - y_val_pred))
        pearson_corr, _ = pearsonr(y_val_pred, y_val)
        all_tasks_pearson_values_ecg_ppg_mean.append(pearson_corr)
        all_tasks_mae_values_ecg_ppg_mean.append(mae)
        print("30sec -- alpha: %f -- ECG-only + PPG-only (mean of features) -- Pearson: %.3f, MAE: %.3f" % (best_alpha_ecg_ppg_mean, pearson_corr, mae))




        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("6. eGFR regression ('EGFR REFIT WITHOUT RACE (2021)'):")
        egfr_df = labs_df[labs_df["Component_name"] == "EGFR REFIT WITHOUT RACE (2021)"]
        # print(egfr_df)
        filtered = egfr_df.groupby("CSN").filter(lambda x: len(x) == 1) # (only keep visits with a single egfr value, NOTE! NOTE! NOTE!)
        # print(filtered)
        csn_to_egfr = dict(zip(filtered["CSN"], filtered["Component_value"]))
        egfr_values = [float(csn_to_egfr[csn]) for csn in preprocessed_csns if csn in csn_to_egfr]
        print(len(egfr_values))
        print(np.min(egfr_values))
        print(np.mean(egfr_values))
        print(np.max(egfr_values))

        train_egfrs = []
        train_features_list_30sec_ecg = []
        train_features_list_30sec_ppg = []
        train_features_list_30sec_ecg_ppg = []
        for i, csn in enumerate(preprocessed_csns):
            if csn in train_csns:
                if csn in csn_to_egfr:
                    train_egfrs.append(float(csn_to_egfr[csn]))
                    train_features_list_30sec_ecg.append(features_30sec_ecg[i])
                    train_features_list_30sec_ppg.append(features_30sec_ppg[i])
                    train_features_list_30sec_ecg_ppg.append(features_30sec_ecg_ppg[i])
        train_features_30sec_ecg = np.stack(train_features_list_30sec_ecg, axis=0)
        train_features_30sec_ppg = np.stack(train_features_list_30sec_ppg, axis=0)
        train_features_30sec_ecg_ppg = np.stack(train_features_list_30sec_ecg_ppg, axis=0)
        print(len(train_egfrs))
        print(train_features_30sec_ecg.shape)
        print(train_features_30sec_ppg.shape)
        print(train_features_30sec_ecg_ppg.shape)
        print("train label stats:")
        print("min: %.2f" % np.min(train_egfrs))
        print("mean: %.2f" % np.mean(train_egfrs))
        print("max: %.2f" % np.max(train_egfrs))

        val_egfrs = []
        val_features_list_30sec_ecg = []
        val_features_list_30sec_ppg = []
        val_features_list_30sec_ecg_ppg = []
        for i, csn in enumerate(preprocessed_csns):
            if csn in val_csns:
                if csn in csn_to_egfr:
                    val_egfrs.append(float(csn_to_egfr[csn]))
                    val_features_list_30sec_ecg.append(features_30sec_ecg[i])
                    val_features_list_30sec_ppg.append(features_30sec_ppg[i])
                    val_features_list_30sec_ecg_ppg.append(features_30sec_ecg_ppg[i])
        val_features_30sec_ecg = np.stack(val_features_list_30sec_ecg, axis=0)
        val_features_30sec_ppg = np.stack(val_features_list_30sec_ppg, axis=0)
        val_features_30sec_ecg_ppg = np.stack(val_features_list_30sec_ecg_ppg, axis=0)
        print(len(val_egfrs))
        print(val_features_30sec_ecg.shape)
        print(val_features_30sec_ppg.shape)
        print(val_features_30sec_ecg_ppg.shape)
        print("val label stats:")
        print("min: %.2f" % np.min(val_egfrs))
        print("mean: %.2f" % np.mean(val_egfrs))
        print("max: %.2f" % np.max(val_egfrs))

        X_train_30sec_ecg = train_features_30sec_ecg
        X_train_30sec_ppg = train_features_30sec_ppg
        X_train_30sec_ecg_ppg = train_features_30sec_ecg_ppg
        X_train_30sec_ecg_ppg_mean = (X_train_30sec_ecg + X_train_30sec_ppg)/2.0
        y_train = np.array(train_egfrs)
        #######
        indices = np.arange(X_train_30sec_ecg.shape[0])
        train_inds = rng.choice(indices, size=int(train_prop*len(indices)), replace=True)
        X_train_30sec_ecg = X_train_30sec_ecg[train_inds]
        X_train_30sec_ppg = X_train_30sec_ppg[train_inds]
        X_train_30sec_ecg_ppg = X_train_30sec_ecg_ppg[train_inds]
        X_train_30sec_ecg_ppg_mean = X_train_30sec_ecg_ppg_mean[train_inds]
        y_train = y_train[train_inds]
        print(len(y_train))
        print(X_train_30sec_ecg.shape)
        print(X_train_30sec_ppg.shape)
        print(X_train_30sec_ecg_ppg.shape)
        print(X_train_30sec_ecg_ppg_mean.shape)

        X_val_30sec_ecg = val_features_30sec_ecg
        X_val_30sec_ppg = val_features_30sec_ppg
        X_val_30sec_ecg_ppg = val_features_30sec_ecg_ppg
        X_val_30sec_ecg_ppg_mean = (X_val_30sec_ecg + X_val_30sec_ppg)/2.0
        y_val = np.array(val_egfrs)

        pipe = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha_ecg))
        pipe.fit(X_train_30sec_ecg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg)
        mae = np.mean(np.abs(y_val - y_val_pred))
        pearson_corr, _ = pearsonr(y_val_pred, y_val)
        all_tasks_pearson_values_ecg.append(pearson_corr)
        all_tasks_mae_values_ecg.append(mae)
        print("30sec -- alpha: %f -- ECG-only -- Pearson: %.3f, MAE: %.3f" % (best_alpha_ecg, pearson_corr, mae))

        pipe = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha_ppg))
        pipe.fit(X_train_30sec_ppg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ppg)
        mae = np.mean(np.abs(y_val - y_val_pred))
        pearson_corr, _ = pearsonr(y_val_pred, y_val)
        all_tasks_pearson_values_ppg.append(pearson_corr)
        all_tasks_mae_values_ppg.append(mae)
        print("30sec -- alpha: %f -- PPG-only -- Pearson: %.3f, MAE: %.3f" % (best_alpha_ppg, pearson_corr, mae))

        pipe = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha_ecg_ppg))
        pipe.fit(X_train_30sec_ecg_ppg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg_ppg)
        mae = np.mean(np.abs(y_val - y_val_pred))
        pearson_corr, _ = pearsonr(y_val_pred, y_val)
        all_tasks_pearson_values_ecg_ppg.append(pearson_corr)
        all_tasks_mae_values_ecg_ppg.append(mae)
        print("30sec -- alpha: %f -- ECG + PPG -- Pearson: %.3f, MAE: %.3f" % (best_alpha_ecg_ppg, pearson_corr, mae))

        pipe = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha_ecg_ppg_mean))
        pipe.fit(X_train_30sec_ecg_ppg_mean, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg_ppg_mean)
        mae = np.mean(np.abs(y_val - y_val_pred))
        pearson_corr, _ = pearsonr(y_val_pred, y_val)
        all_tasks_pearson_values_ecg_ppg_mean.append(pearson_corr)
        all_tasks_mae_values_ecg_ppg_mean.append(mae)
        print("30sec -- alpha: %f -- ECG-only + PPG-only (mean of features) -- Pearson: %.3f, MAE: %.3f" % (best_alpha_ecg_ppg_mean, pearson_corr, mae))




        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("7. Glucose regression ('GLUCOSE'):")
        glucose_df = labs_df[labs_df["Component_name"] == "GLUCOSE"]
        # print(glucose_df)
        filtered = glucose_df.groupby("CSN").filter(lambda x: len(x) == 1) # (only keep visits with a single glucose value, NOTE! NOTE! NOTE!)
        # print(filtered)
        csn_to_glucose = dict(zip(filtered["CSN"], filtered["Component_value"]))
        glucose_values = [float(csn_to_glucose[csn]) for csn in preprocessed_csns if csn in csn_to_glucose]
        print(len(glucose_values))
        print(np.min(glucose_values))
        print(np.mean(glucose_values))
        print(np.max(glucose_values))

        train_glucoses = []
        train_features_list_30sec_ecg = []
        train_features_list_30sec_ppg = []
        train_features_list_30sec_ecg_ppg = []
        for i, csn in enumerate(preprocessed_csns):
            if csn in train_csns:
                if csn in csn_to_glucose:
                    train_glucoses.append(float(csn_to_glucose[csn]))
                    train_features_list_30sec_ecg.append(features_30sec_ecg[i])
                    train_features_list_30sec_ppg.append(features_30sec_ppg[i])
                    train_features_list_30sec_ecg_ppg.append(features_30sec_ecg_ppg[i])
        train_features_30sec_ecg = np.stack(train_features_list_30sec_ecg, axis=0)
        train_features_30sec_ppg = np.stack(train_features_list_30sec_ppg, axis=0)
        train_features_30sec_ecg_ppg = np.stack(train_features_list_30sec_ecg_ppg, axis=0)
        print(len(train_glucoses))
        print(train_features_30sec_ecg.shape)
        print(train_features_30sec_ppg.shape)
        print(train_features_30sec_ecg_ppg.shape)
        print("train label stats:")
        print("min: %.2f" % np.min(train_glucoses))
        print("mean: %.2f" % np.mean(train_glucoses))
        print("max: %.2f" % np.max(train_glucoses))

        val_glucoses = []
        val_features_list_30sec_ecg = []
        val_features_list_30sec_ppg = []
        val_features_list_30sec_ecg_ppg = []
        for i, csn in enumerate(preprocessed_csns):
            if csn in val_csns:
                if csn in csn_to_glucose:
                    val_glucoses.append(float(csn_to_glucose[csn]))
                    val_features_list_30sec_ecg.append(features_30sec_ecg[i])
                    val_features_list_30sec_ppg.append(features_30sec_ppg[i])
                    val_features_list_30sec_ecg_ppg.append(features_30sec_ecg_ppg[i])
        val_features_30sec_ecg = np.stack(val_features_list_30sec_ecg, axis=0)
        val_features_30sec_ppg = np.stack(val_features_list_30sec_ppg, axis=0)
        val_features_30sec_ecg_ppg = np.stack(val_features_list_30sec_ecg_ppg, axis=0)
        print(len(val_glucoses))
        print(val_features_30sec_ecg.shape)
        print(val_features_30sec_ppg.shape)
        print(val_features_30sec_ecg_ppg.shape)
        print("val label stats:")
        print("min: %.2f" % np.min(val_glucoses))
        print("mean: %.2f" % np.mean(val_glucoses))
        print("max: %.2f" % np.max(val_glucoses))

        X_train_30sec_ecg = train_features_30sec_ecg
        X_train_30sec_ppg = train_features_30sec_ppg
        X_train_30sec_ecg_ppg = train_features_30sec_ecg_ppg
        X_train_30sec_ecg_ppg_mean = (X_train_30sec_ecg + X_train_30sec_ppg)/2.0
        y_train = np.array(train_glucoses)
        #######
        indices = np.arange(X_train_30sec_ecg.shape[0])
        train_inds = rng.choice(indices, size=int(train_prop*len(indices)), replace=True)
        X_train_30sec_ecg = X_train_30sec_ecg[train_inds]
        X_train_30sec_ppg = X_train_30sec_ppg[train_inds]
        X_train_30sec_ecg_ppg = X_train_30sec_ecg_ppg[train_inds]
        X_train_30sec_ecg_ppg_mean = X_train_30sec_ecg_ppg_mean[train_inds]
        y_train = y_train[train_inds]
        print(len(y_train))
        print(X_train_30sec_ecg.shape)
        print(X_train_30sec_ppg.shape)
        print(X_train_30sec_ecg_ppg.shape)
        print(X_train_30sec_ecg_ppg_mean.shape)

        X_val_30sec_ecg = val_features_30sec_ecg
        X_val_30sec_ppg = val_features_30sec_ppg
        X_val_30sec_ecg_ppg = val_features_30sec_ecg_ppg
        X_val_30sec_ecg_ppg_mean = (X_val_30sec_ecg + X_val_30sec_ppg)/2.0
        y_val = np.array(val_glucoses)

        pipe = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha_ecg))
        pipe.fit(X_train_30sec_ecg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg)
        mae = np.mean(np.abs(y_val - y_val_pred))
        pearson_corr, _ = pearsonr(y_val_pred, y_val)
        all_tasks_pearson_values_ecg.append(pearson_corr)
        all_tasks_mae_values_ecg.append(mae)
        print("30sec -- alpha: %f -- ECG-only -- Pearson: %.3f, MAE: %.3f" % (best_alpha_ecg, pearson_corr, mae))

        pipe = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha_ppg))
        pipe.fit(X_train_30sec_ppg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ppg)
        mae = np.mean(np.abs(y_val - y_val_pred))
        pearson_corr, _ = pearsonr(y_val_pred, y_val)
        all_tasks_pearson_values_ppg.append(pearson_corr)
        all_tasks_mae_values_ppg.append(mae)
        print("30sec -- alpha: %f -- PPG-only -- Pearson: %.3f, MAE: %.3f" % (best_alpha_ppg, pearson_corr, mae))

        pipe = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha_ecg_ppg))
        pipe.fit(X_train_30sec_ecg_ppg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg_ppg)
        mae = np.mean(np.abs(y_val - y_val_pred))
        pearson_corr, _ = pearsonr(y_val_pred, y_val)
        all_tasks_pearson_values_ecg_ppg.append(pearson_corr)
        all_tasks_mae_values_ecg_ppg.append(mae)
        print("30sec -- alpha: %f -- ECG + PPG -- Pearson: %.3f, MAE: %.3f" % (best_alpha_ecg_ppg, pearson_corr, mae))

        pipe = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha_ecg_ppg_mean))
        pipe.fit(X_train_30sec_ecg_ppg_mean, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg_ppg_mean)
        mae = np.mean(np.abs(y_val - y_val_pred))
        pearson_corr, _ = pearsonr(y_val_pred, y_val)
        all_tasks_pearson_values_ecg_ppg_mean.append(pearson_corr)
        all_tasks_mae_values_ecg_ppg_mean.append(mae)
        print("30sec -- alpha: %f -- ECG-only + PPG-only (mean of features) -- Pearson: %.3f, MAE: %.3f" % (best_alpha_ecg_ppg_mean, pearson_corr, mae))




        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("8. Hemoglobin regression ('HEMOGLOBIN (HGB)'):")
        hemoglobin_df = labs_df[labs_df["Component_name"] == "HEMOGLOBIN (HGB)"]
        # print(hemoglobin_df)
        filtered = hemoglobin_df.groupby("CSN").filter(lambda x: len(x) == 1) # (only keep visits with a single hemoglobin value, NOTE! NOTE! NOTE!)
        # print(filtered)
        csn_to_hemoglobin = dict(zip(filtered["CSN"], filtered["Component_value"]))
        hemoglobin_values = [float(csn_to_hemoglobin[csn]) for csn in preprocessed_csns if csn in csn_to_hemoglobin]
        print(len(hemoglobin_values))
        print(np.min(hemoglobin_values))
        print(np.mean(hemoglobin_values))
        print(np.max(hemoglobin_values))

        train_hemoglobins = []
        train_features_list_30sec_ecg = []
        train_features_list_30sec_ppg = []
        train_features_list_30sec_ecg_ppg = []
        for i, csn in enumerate(preprocessed_csns):
            if csn in train_csns:
                if csn in csn_to_hemoglobin:
                    train_hemoglobins.append(float(csn_to_hemoglobin[csn]))
                    train_features_list_30sec_ecg.append(features_30sec_ecg[i])
                    train_features_list_30sec_ppg.append(features_30sec_ppg[i])
                    train_features_list_30sec_ecg_ppg.append(features_30sec_ecg_ppg[i])
        train_features_30sec_ecg = np.stack(train_features_list_30sec_ecg, axis=0)
        train_features_30sec_ppg = np.stack(train_features_list_30sec_ppg, axis=0)
        train_features_30sec_ecg_ppg = np.stack(train_features_list_30sec_ecg_ppg, axis=0)
        print(len(train_hemoglobins))
        print(train_features_30sec_ecg.shape)
        print(train_features_30sec_ppg.shape)
        print(train_features_30sec_ecg_ppg.shape)
        print("train label stats:")
        print("min: %.2f" % np.min(train_hemoglobins))
        print("mean: %.2f" % np.mean(train_hemoglobins))
        print("max: %.2f" % np.max(train_hemoglobins))

        val_hemoglobins = []
        val_features_list_30sec_ecg = []
        val_features_list_30sec_ppg = []
        val_features_list_30sec_ecg_ppg = []
        for i, csn in enumerate(preprocessed_csns):
            if csn in val_csns:
                if csn in csn_to_hemoglobin:
                    val_hemoglobins.append(float(csn_to_hemoglobin[csn]))
                    val_features_list_30sec_ecg.append(features_30sec_ecg[i])
                    val_features_list_30sec_ppg.append(features_30sec_ppg[i])
                    val_features_list_30sec_ecg_ppg.append(features_30sec_ecg_ppg[i])
        val_features_30sec_ecg = np.stack(val_features_list_30sec_ecg, axis=0)
        val_features_30sec_ppg = np.stack(val_features_list_30sec_ppg, axis=0)
        val_features_30sec_ecg_ppg = np.stack(val_features_list_30sec_ecg_ppg, axis=0)
        print(len(val_hemoglobins))
        print(val_features_30sec_ecg.shape)
        print(val_features_30sec_ppg.shape)
        print(val_features_30sec_ecg_ppg.shape)
        print("val label stats:")
        print("min: %.2f" % np.min(val_hemoglobins))
        print("mean: %.2f" % np.mean(val_hemoglobins))
        print("max: %.2f" % np.max(val_hemoglobins))

        X_train_30sec_ecg = train_features_30sec_ecg
        X_train_30sec_ppg = train_features_30sec_ppg
        X_train_30sec_ecg_ppg = train_features_30sec_ecg_ppg
        X_train_30sec_ecg_ppg_mean = (X_train_30sec_ecg + X_train_30sec_ppg)/2.0
        y_train = np.array(train_hemoglobins)
        #######
        indices = np.arange(X_train_30sec_ecg.shape[0])
        train_inds = rng.choice(indices, size=int(train_prop*len(indices)), replace=True)
        X_train_30sec_ecg = X_train_30sec_ecg[train_inds]
        X_train_30sec_ppg = X_train_30sec_ppg[train_inds]
        X_train_30sec_ecg_ppg = X_train_30sec_ecg_ppg[train_inds]
        X_train_30sec_ecg_ppg_mean = X_train_30sec_ecg_ppg_mean[train_inds]
        y_train = y_train[train_inds]
        print(len(y_train))
        print(X_train_30sec_ecg.shape)
        print(X_train_30sec_ppg.shape)
        print(X_train_30sec_ecg_ppg.shape)
        print(X_train_30sec_ecg_ppg_mean.shape)

        X_val_30sec_ecg = val_features_30sec_ecg
        X_val_30sec_ppg = val_features_30sec_ppg
        X_val_30sec_ecg_ppg = val_features_30sec_ecg_ppg
        X_val_30sec_ecg_ppg_mean = (X_val_30sec_ecg + X_val_30sec_ppg)/2.0
        y_val = np.array(val_hemoglobins)

        pipe = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha_ecg))
        pipe.fit(X_train_30sec_ecg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg)
        mae = np.mean(np.abs(y_val - y_val_pred))
        pearson_corr, _ = pearsonr(y_val_pred, y_val)
        all_tasks_pearson_values_ecg.append(pearson_corr)
        all_tasks_mae_values_ecg.append(mae)
        print("30sec -- alpha: %f -- ECG-only -- Pearson: %.3f, MAE: %.3f" % (best_alpha_ecg, pearson_corr, mae))

        pipe = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha_ppg))
        pipe.fit(X_train_30sec_ppg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ppg)
        mae = np.mean(np.abs(y_val - y_val_pred))
        pearson_corr, _ = pearsonr(y_val_pred, y_val)
        all_tasks_pearson_values_ppg.append(pearson_corr)
        all_tasks_mae_values_ppg.append(mae)
        print("30sec -- alpha: %f -- PPG-only -- Pearson: %.3f, MAE: %.3f" % (best_alpha_ppg, pearson_corr, mae))

        pipe = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha_ecg_ppg))
        pipe.fit(X_train_30sec_ecg_ppg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg_ppg)
        mae = np.mean(np.abs(y_val - y_val_pred))
        pearson_corr, _ = pearsonr(y_val_pred, y_val)
        all_tasks_pearson_values_ecg_ppg.append(pearson_corr)
        all_tasks_mae_values_ecg_ppg.append(mae)
        print("30sec -- alpha: %f -- ECG + PPG -- Pearson: %.3f, MAE: %.3f" % (best_alpha_ecg_ppg, pearson_corr, mae))

        pipe = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha_ecg_ppg_mean))
        pipe.fit(X_train_30sec_ecg_ppg_mean, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg_ppg_mean)
        mae = np.mean(np.abs(y_val - y_val_pred))
        pearson_corr, _ = pearsonr(y_val_pred, y_val)
        all_tasks_pearson_values_ecg_ppg_mean.append(pearson_corr)
        all_tasks_mae_values_ecg_ppg_mean.append(mae)
        print("30sec -- alpha: %f -- ECG-only + PPG-only (mean of features) -- Pearson: %.3f, MAE: %.3f" % (best_alpha_ecg_ppg_mean, pearson_corr, mae))




        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("9. Albumin regression ('ALBUMIN'):")
        albumin_df = labs_df[labs_df["Component_name"] == "ALBUMIN"]
        # print(albumin_df)
        filtered = albumin_df.groupby("CSN").filter(lambda x: len(x) == 1) # (only keep visits with a single albumin value, NOTE! NOTE! NOTE!)
        # print(filtered)
        csn_to_albumin = dict(zip(filtered["CSN"], filtered["Component_value"]))
        albumin_values = [float(csn_to_albumin[csn]) for csn in preprocessed_csns if csn in csn_to_albumin]
        print(len(albumin_values))
        print(np.min(albumin_values))
        print(np.mean(albumin_values))
        print(np.max(albumin_values))

        train_albumins = []
        train_features_list_30sec_ecg = []
        train_features_list_30sec_ppg = []
        train_features_list_30sec_ecg_ppg = []
        for i, csn in enumerate(preprocessed_csns):
            if csn in train_csns:
                if csn in csn_to_albumin:
                    train_albumins.append(float(csn_to_albumin[csn]))
                    train_features_list_30sec_ecg.append(features_30sec_ecg[i])
                    train_features_list_30sec_ppg.append(features_30sec_ppg[i])
                    train_features_list_30sec_ecg_ppg.append(features_30sec_ecg_ppg[i])
        train_features_30sec_ecg = np.stack(train_features_list_30sec_ecg, axis=0)
        train_features_30sec_ppg = np.stack(train_features_list_30sec_ppg, axis=0)
        train_features_30sec_ecg_ppg = np.stack(train_features_list_30sec_ecg_ppg, axis=0)
        print(len(train_albumins))
        print(train_features_30sec_ecg.shape)
        print(train_features_30sec_ppg.shape)
        print(train_features_30sec_ecg_ppg.shape)
        print("train label stats:")
        print("min: %.2f" % np.min(train_albumins))
        print("mean: %.2f" % np.mean(train_albumins))
        print("max: %.2f" % np.max(train_albumins))

        val_albumins = []
        val_features_list_30sec_ecg = []
        val_features_list_30sec_ppg = []
        val_features_list_30sec_ecg_ppg = []
        for i, csn in enumerate(preprocessed_csns):
            if csn in val_csns:
                if csn in csn_to_albumin:
                    val_albumins.append(float(csn_to_albumin[csn]))
                    val_features_list_30sec_ecg.append(features_30sec_ecg[i])
                    val_features_list_30sec_ppg.append(features_30sec_ppg[i])
                    val_features_list_30sec_ecg_ppg.append(features_30sec_ecg_ppg[i])
        val_features_30sec_ecg = np.stack(val_features_list_30sec_ecg, axis=0)
        val_features_30sec_ppg = np.stack(val_features_list_30sec_ppg, axis=0)
        val_features_30sec_ecg_ppg = np.stack(val_features_list_30sec_ecg_ppg, axis=0)
        print(len(val_albumins))
        print(val_features_30sec_ecg.shape)
        print(val_features_30sec_ppg.shape)
        print(val_features_30sec_ecg_ppg.shape)
        print("val label stats:")
        print("min: %.2f" % np.min(val_albumins))
        print("mean: %.2f" % np.mean(val_albumins))
        print("max: %.2f" % np.max(val_albumins))

        X_train_30sec_ecg = train_features_30sec_ecg
        X_train_30sec_ppg = train_features_30sec_ppg
        X_train_30sec_ecg_ppg = train_features_30sec_ecg_ppg
        X_train_30sec_ecg_ppg_mean = (X_train_30sec_ecg + X_train_30sec_ppg)/2.0
        y_train = np.array(train_albumins)
        #######
        indices = np.arange(X_train_30sec_ecg.shape[0])
        train_inds = rng.choice(indices, size=int(train_prop*len(indices)), replace=True)
        X_train_30sec_ecg = X_train_30sec_ecg[train_inds]
        X_train_30sec_ppg = X_train_30sec_ppg[train_inds]
        X_train_30sec_ecg_ppg = X_train_30sec_ecg_ppg[train_inds]
        X_train_30sec_ecg_ppg_mean = X_train_30sec_ecg_ppg_mean[train_inds]
        y_train = y_train[train_inds]
        print(len(y_train))
        print(X_train_30sec_ecg.shape)
        print(X_train_30sec_ppg.shape)
        print(X_train_30sec_ecg_ppg.shape)
        print(X_train_30sec_ecg_ppg_mean.shape)

        X_val_30sec_ecg = val_features_30sec_ecg
        X_val_30sec_ppg = val_features_30sec_ppg
        X_val_30sec_ecg_ppg = val_features_30sec_ecg_ppg
        X_val_30sec_ecg_ppg_mean = (X_val_30sec_ecg + X_val_30sec_ppg)/2.0
        y_val = np.array(val_albumins)

        pipe = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha_ecg))
        pipe.fit(X_train_30sec_ecg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg)
        mae = np.mean(np.abs(y_val - y_val_pred))
        pearson_corr, _ = pearsonr(y_val_pred, y_val)
        all_tasks_pearson_values_ecg.append(pearson_corr)
        all_tasks_mae_values_ecg.append(mae)
        print("30sec -- alpha: %f -- ECG-only -- Pearson: %.3f, MAE: %.3f" % (best_alpha_ecg, pearson_corr, mae))

        pipe = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha_ppg))
        pipe.fit(X_train_30sec_ppg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ppg)
        mae = np.mean(np.abs(y_val - y_val_pred))
        pearson_corr, _ = pearsonr(y_val_pred, y_val)
        all_tasks_pearson_values_ppg.append(pearson_corr)
        all_tasks_mae_values_ppg.append(mae)
        print("30sec -- alpha: %f -- PPG-only -- Pearson: %.3f, MAE: %.3f" % (best_alpha_ppg, pearson_corr, mae))

        pipe = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha_ecg_ppg))
        pipe.fit(X_train_30sec_ecg_ppg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg_ppg)
        mae = np.mean(np.abs(y_val - y_val_pred))
        pearson_corr, _ = pearsonr(y_val_pred, y_val)
        all_tasks_pearson_values_ecg_ppg.append(pearson_corr)
        all_tasks_mae_values_ecg_ppg.append(mae)
        print("30sec -- alpha: %f -- ECG + PPG -- Pearson: %.3f, MAE: %.3f" % (best_alpha_ecg_ppg, pearson_corr, mae))

        pipe = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha_ecg_ppg_mean))
        pipe.fit(X_train_30sec_ecg_ppg_mean, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg_ppg_mean)
        mae = np.mean(np.abs(y_val - y_val_pred))
        pearson_corr, _ = pearsonr(y_val_pred, y_val)
        all_tasks_pearson_values_ecg_ppg_mean.append(pearson_corr)
        all_tasks_mae_values_ecg_ppg_mean.append(mae)
        print("30sec -- alpha: %f -- ECG-only + PPG-only (mean of features) -- Pearson: %.3f, MAE: %.3f" % (best_alpha_ecg_ppg_mean, pearson_corr, mae))




        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("10. BUN regression ('BLOOD UREA NITROGEN (BUN)'):")
        bun_df = labs_df[labs_df["Component_name"] == "BLOOD UREA NITROGEN (BUN)"]
        # print(bun_df)
        filtered = bun_df.groupby("CSN").filter(lambda x: len(x) == 1) # (only keep visits with a single bun value, NOTE! NOTE! NOTE!)
        # print(filtered)
        csn_to_bun = dict(zip(filtered["CSN"], filtered["Component_value"]))
        bun_values = [float(csn_to_bun[csn]) for csn in preprocessed_csns if ((csn in csn_to_bun) and (csn_to_bun[csn] != '<2'))]
        print(len(bun_values))
        print(np.min(bun_values))
        print(np.mean(bun_values))
        print(np.max(bun_values))

        train_buns = []
        train_features_list_30sec_ecg = []
        train_features_list_30sec_ppg = []
        train_features_list_30sec_ecg_ppg = []
        for i, csn in enumerate(preprocessed_csns):
            if csn in train_csns:
                if (csn in csn_to_bun) and (csn_to_bun[csn] != '<2'):
                    train_buns.append(float(csn_to_bun[csn]))
                    train_features_list_30sec_ecg.append(features_30sec_ecg[i])
                    train_features_list_30sec_ppg.append(features_30sec_ppg[i])
                    train_features_list_30sec_ecg_ppg.append(features_30sec_ecg_ppg[i])
        train_features_30sec_ecg = np.stack(train_features_list_30sec_ecg, axis=0)
        train_features_30sec_ppg = np.stack(train_features_list_30sec_ppg, axis=0)
        train_features_30sec_ecg_ppg = np.stack(train_features_list_30sec_ecg_ppg, axis=0)
        print(len(train_buns))
        print(train_features_30sec_ecg.shape)
        print(train_features_30sec_ppg.shape)
        print(train_features_30sec_ecg_ppg.shape)
        print("train label stats:")
        print("min: %.2f" % np.min(train_buns))
        print("mean: %.2f" % np.mean(train_buns))
        print("max: %.2f" % np.max(train_buns))

        val_buns = []
        val_features_list_30sec_ecg = []
        val_features_list_30sec_ppg = []
        val_features_list_30sec_ecg_ppg = []
        for i, csn in enumerate(preprocessed_csns):
            if csn in val_csns:
                if (csn in csn_to_bun) and (csn_to_bun[csn] != '<2'):
                    val_buns.append(float(csn_to_bun[csn]))
                    val_features_list_30sec_ecg.append(features_30sec_ecg[i])
                    val_features_list_30sec_ppg.append(features_30sec_ppg[i])
                    val_features_list_30sec_ecg_ppg.append(features_30sec_ecg_ppg[i])
        val_features_30sec_ecg = np.stack(val_features_list_30sec_ecg, axis=0)
        val_features_30sec_ppg = np.stack(val_features_list_30sec_ppg, axis=0)
        val_features_30sec_ecg_ppg = np.stack(val_features_list_30sec_ecg_ppg, axis=0)
        print(len(val_buns))
        print(val_features_30sec_ecg.shape)
        print(val_features_30sec_ppg.shape)
        print(val_features_30sec_ecg_ppg.shape)
        print("val label stats:")
        print("min: %.2f" % np.min(val_buns))
        print("mean: %.2f" % np.mean(val_buns))
        print("max: %.2f" % np.max(val_buns))

        X_train_30sec_ecg = train_features_30sec_ecg
        X_train_30sec_ppg = train_features_30sec_ppg
        X_train_30sec_ecg_ppg = train_features_30sec_ecg_ppg
        X_train_30sec_ecg_ppg_mean = (X_train_30sec_ecg + X_train_30sec_ppg)/2.0
        y_train = np.array(train_buns)
        #######
        indices = np.arange(X_train_30sec_ecg.shape[0])
        train_inds = rng.choice(indices, size=int(train_prop*len(indices)), replace=True)
        X_train_30sec_ecg = X_train_30sec_ecg[train_inds]
        X_train_30sec_ppg = X_train_30sec_ppg[train_inds]
        X_train_30sec_ecg_ppg = X_train_30sec_ecg_ppg[train_inds]
        X_train_30sec_ecg_ppg_mean = X_train_30sec_ecg_ppg_mean[train_inds]
        y_train = y_train[train_inds]
        print(len(y_train))
        print(X_train_30sec_ecg.shape)
        print(X_train_30sec_ppg.shape)
        print(X_train_30sec_ecg_ppg.shape)
        print(X_train_30sec_ecg_ppg_mean.shape)

        X_val_30sec_ecg = val_features_30sec_ecg
        X_val_30sec_ppg = val_features_30sec_ppg
        X_val_30sec_ecg_ppg = val_features_30sec_ecg_ppg
        X_val_30sec_ecg_ppg_mean = (X_val_30sec_ecg + X_val_30sec_ppg)/2.0
        y_val = np.array(val_buns)

        pipe = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha_ecg))
        pipe.fit(X_train_30sec_ecg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg)
        mae = np.mean(np.abs(y_val - y_val_pred))
        pearson_corr, _ = pearsonr(y_val_pred, y_val)
        all_tasks_pearson_values_ecg.append(pearson_corr)
        all_tasks_mae_values_ecg.append(mae)
        print("30sec -- alpha: %f -- ECG-only -- Pearson: %.3f, MAE: %.3f" % (best_alpha_ecg, pearson_corr, mae))

        pipe = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha_ppg))
        pipe.fit(X_train_30sec_ppg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ppg)
        mae = np.mean(np.abs(y_val - y_val_pred))
        pearson_corr, _ = pearsonr(y_val_pred, y_val)
        all_tasks_pearson_values_ppg.append(pearson_corr)
        all_tasks_mae_values_ppg.append(mae)
        print("30sec -- alpha: %f -- PPG-only -- Pearson: %.3f, MAE: %.3f" % (best_alpha_ppg, pearson_corr, mae))

        pipe = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha_ecg_ppg))
        pipe.fit(X_train_30sec_ecg_ppg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg_ppg)
        mae = np.mean(np.abs(y_val - y_val_pred))
        pearson_corr, _ = pearsonr(y_val_pred, y_val)
        all_tasks_pearson_values_ecg_ppg.append(pearson_corr)
        all_tasks_mae_values_ecg_ppg.append(mae)
        print("30sec -- alpha: %f -- ECG + PPG -- Pearson: %.3f, MAE: %.3f" % (best_alpha_ecg_ppg, pearson_corr, mae))

        pipe = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha_ecg_ppg_mean))
        pipe.fit(X_train_30sec_ecg_ppg_mean, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg_ppg_mean)
        mae = np.mean(np.abs(y_val - y_val_pred))
        pearson_corr, _ = pearsonr(y_val_pred, y_val)
        all_tasks_pearson_values_ecg_ppg_mean.append(pearson_corr)
        all_tasks_mae_values_ecg_ppg_mean.append(mae)
        print("30sec -- alpha: %f -- ECG-only + PPG-only (mean of features) -- Pearson: %.3f, MAE: %.3f" % (best_alpha_ecg_ppg_mean, pearson_corr, mae))




        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("11. Sodium regression ('SODIUM'):")
        sodium_df = labs_df[labs_df["Component_name"] == "SODIUM"]
        # print(sodium_df)
        filtered = sodium_df.groupby("CSN").filter(lambda x: len(x) == 1) # (only keep visits with a single sodium value, NOTE! NOTE! NOTE!)
        # print(filtered)
        csn_to_sodium = dict(zip(filtered["CSN"], filtered["Component_value"]))
        sodium_values = [float(csn_to_sodium[csn]) for csn in preprocessed_csns if csn in csn_to_sodium]
        print(len(sodium_values))
        print(np.min(sodium_values))
        print(np.mean(sodium_values))
        print(np.max(sodium_values))

        train_sodiums = []
        train_features_list_30sec_ecg = []
        train_features_list_30sec_ppg = []
        train_features_list_30sec_ecg_ppg = []
        for i, csn in enumerate(preprocessed_csns):
            if csn in train_csns:
                if csn in csn_to_sodium:
                    train_sodiums.append(float(csn_to_sodium[csn]))
                    train_features_list_30sec_ecg.append(features_30sec_ecg[i])
                    train_features_list_30sec_ppg.append(features_30sec_ppg[i])
                    train_features_list_30sec_ecg_ppg.append(features_30sec_ecg_ppg[i])
        train_features_30sec_ecg = np.stack(train_features_list_30sec_ecg, axis=0)
        train_features_30sec_ppg = np.stack(train_features_list_30sec_ppg, axis=0)
        train_features_30sec_ecg_ppg = np.stack(train_features_list_30sec_ecg_ppg, axis=0)
        print(len(train_sodiums))
        print(train_features_30sec_ecg.shape)
        print(train_features_30sec_ppg.shape)
        print(train_features_30sec_ecg_ppg.shape)
        print("train label stats:")
        print("min: %.2f" % np.min(train_sodiums))
        print("mean: %.2f" % np.mean(train_sodiums))
        print("max: %.2f" % np.max(train_sodiums))

        val_sodiums = []
        val_features_list_30sec_ecg = []
        val_features_list_30sec_ppg = []
        val_features_list_30sec_ecg_ppg = []
        for i, csn in enumerate(preprocessed_csns):
            if csn in val_csns:
                if csn in csn_to_sodium:
                    val_sodiums.append(float(csn_to_sodium[csn]))
                    val_features_list_30sec_ecg.append(features_30sec_ecg[i])
                    val_features_list_30sec_ppg.append(features_30sec_ppg[i])
                    val_features_list_30sec_ecg_ppg.append(features_30sec_ecg_ppg[i])
        val_features_30sec_ecg = np.stack(val_features_list_30sec_ecg, axis=0)
        val_features_30sec_ppg = np.stack(val_features_list_30sec_ppg, axis=0)
        val_features_30sec_ecg_ppg = np.stack(val_features_list_30sec_ecg_ppg, axis=0)
        print(len(val_sodiums))
        print(val_features_30sec_ecg.shape)
        print(val_features_30sec_ppg.shape)
        print(val_features_30sec_ecg_ppg.shape)
        print("val label stats:")
        print("min: %.2f" % np.min(val_sodiums))
        print("mean: %.2f" % np.mean(val_sodiums))
        print("max: %.2f" % np.max(val_sodiums))

        X_train_30sec_ecg = train_features_30sec_ecg
        X_train_30sec_ppg = train_features_30sec_ppg
        X_train_30sec_ecg_ppg = train_features_30sec_ecg_ppg
        X_train_30sec_ecg_ppg_mean = (X_train_30sec_ecg + X_train_30sec_ppg)/2.0
        y_train = np.array(train_sodiums)
        #######
        indices = np.arange(X_train_30sec_ecg.shape[0])
        train_inds = rng.choice(indices, size=int(train_prop*len(indices)), replace=True)
        X_train_30sec_ecg = X_train_30sec_ecg[train_inds]
        X_train_30sec_ppg = X_train_30sec_ppg[train_inds]
        X_train_30sec_ecg_ppg = X_train_30sec_ecg_ppg[train_inds]
        X_train_30sec_ecg_ppg_mean = X_train_30sec_ecg_ppg_mean[train_inds]
        y_train = y_train[train_inds]
        print(len(y_train))
        print(X_train_30sec_ecg.shape)
        print(X_train_30sec_ppg.shape)
        print(X_train_30sec_ecg_ppg.shape)
        print(X_train_30sec_ecg_ppg_mean.shape)

        X_val_30sec_ecg = val_features_30sec_ecg
        X_val_30sec_ppg = val_features_30sec_ppg
        X_val_30sec_ecg_ppg = val_features_30sec_ecg_ppg
        X_val_30sec_ecg_ppg_mean = (X_val_30sec_ecg + X_val_30sec_ppg)/2.0
        y_val = np.array(val_sodiums)

        pipe = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha_ecg))
        pipe.fit(X_train_30sec_ecg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg)
        mae = np.mean(np.abs(y_val - y_val_pred))
        pearson_corr, _ = pearsonr(y_val_pred, y_val)
        all_tasks_pearson_values_ecg.append(pearson_corr)
        all_tasks_mae_values_ecg.append(mae)
        print("30sec -- alpha: %f -- ECG-only -- Pearson: %.3f, MAE: %.3f" % (best_alpha_ecg, pearson_corr, mae))

        pipe = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha_ppg))
        pipe.fit(X_train_30sec_ppg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ppg)
        mae = np.mean(np.abs(y_val - y_val_pred))
        pearson_corr, _ = pearsonr(y_val_pred, y_val)
        all_tasks_pearson_values_ppg.append(pearson_corr)
        all_tasks_mae_values_ppg.append(mae)
        print("30sec -- alpha: %f -- PPG-only -- Pearson: %.3f, MAE: %.3f" % (best_alpha_ppg, pearson_corr, mae))

        pipe = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha_ecg_ppg))
        pipe.fit(X_train_30sec_ecg_ppg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg_ppg)
        mae = np.mean(np.abs(y_val - y_val_pred))
        pearson_corr, _ = pearsonr(y_val_pred, y_val)
        all_tasks_pearson_values_ecg_ppg.append(pearson_corr)
        all_tasks_mae_values_ecg_ppg.append(mae)
        print("30sec -- alpha: %f -- ECG + PPG -- Pearson: %.3f, MAE: %.3f" % (best_alpha_ecg_ppg, pearson_corr, mae))

        pipe = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha_ecg_ppg_mean))
        pipe.fit(X_train_30sec_ecg_ppg_mean, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg_ppg_mean)
        mae = np.mean(np.abs(y_val - y_val_pred))
        pearson_corr, _ = pearsonr(y_val_pred, y_val)
        all_tasks_pearson_values_ecg_ppg_mean.append(pearson_corr)
        all_tasks_mae_values_ecg_ppg_mean.append(mae)
        print("30sec -- alpha: %f -- ECG-only + PPG-only (mean of features) -- Pearson: %.3f, MAE: %.3f" % (best_alpha_ecg_ppg_mean, pearson_corr, mae))








        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("12. I48 classification -- Prior Atrial Fibrillation Detection:")
        filtered = pmh_df[pmh_df["Code"].astype(str).str.startswith("I48")]
        mrns_with_code = filtered["MRN"].unique()
        csns_with_code = visits_df[visits_df["MRN"].isin(mrns_with_code)]["CSN"].unique()

        train_ICD10_I48s = []
        train_features_list_30sec_ecg = []
        train_features_list_30sec_ppg = []
        train_features_list_30sec_ecg_ppg = []
        for i, csn in enumerate(preprocessed_csns):
            if csn in train_csns:
                if csn in csns_with_code:
                    train_ICD10_I48s.append(1)
                else:
                    train_ICD10_I48s.append(0)
                train_features_list_30sec_ecg.append(features_30sec_ecg[i])
                train_features_list_30sec_ppg.append(features_30sec_ppg[i])
                train_features_list_30sec_ecg_ppg.append(features_30sec_ecg_ppg[i])
        train_features_30sec_ecg = np.stack(train_features_list_30sec_ecg, axis=0)
        train_features_30sec_ppg = np.stack(train_features_list_30sec_ppg, axis=0)
        train_features_30sec_ecg_ppg = np.stack(train_features_list_30sec_ecg_ppg, axis=0)
        print(len(train_ICD10_I48s))
        print(train_features_30sec_ecg.shape)
        print(train_features_30sec_ppg.shape)
        print(train_features_30sec_ecg_ppg.shape)
        print("proportion of positive class in train: %.4f" % (train_ICD10_I48s.count(1)/float(len(train_ICD10_I48s))))

        val_ICD10_I48s = []
        val_features_list_30sec_ecg = []
        val_features_list_30sec_ppg = []
        val_features_list_30sec_ecg_ppg = []
        for i, csn in enumerate(preprocessed_csns):
            if csn in val_csns:
                if csn in csns_with_code:
                    val_ICD10_I48s.append(1)
                else:
                    val_ICD10_I48s.append(0)
                val_features_list_30sec_ecg.append(features_30sec_ecg[i])
                val_features_list_30sec_ppg.append(features_30sec_ppg[i])
                val_features_list_30sec_ecg_ppg.append(features_30sec_ecg_ppg[i])
        val_features_30sec_ecg = np.stack(val_features_list_30sec_ecg, axis=0)
        val_features_30sec_ppg = np.stack(val_features_list_30sec_ppg, axis=0)
        val_features_30sec_ecg_ppg = np.stack(val_features_list_30sec_ecg_ppg, axis=0)
        print(len(val_ICD10_I48s))
        print(val_features_30sec_ecg.shape)
        print(val_features_30sec_ppg.shape)
        print(val_features_30sec_ecg_ppg.shape)
        print("proportion of positive class in val: %.4f" % (val_ICD10_I48s.count(1)/float(len(val_ICD10_I48s))))

        X_train_30sec_ecg = train_features_30sec_ecg
        X_train_30sec_ppg = train_features_30sec_ppg
        X_train_30sec_ecg_ppg = train_features_30sec_ecg_ppg
        X_train_30sec_ecg_ppg_mean = (X_train_30sec_ecg + X_train_30sec_ppg)/2.0
        y_train = np.array(train_ICD10_I48s)
        #######
        indices = np.arange(X_train_30sec_ecg.shape[0])
        train_inds = rng.choice(indices, size=int(train_prop*len(indices)), replace=True)
        X_train_30sec_ecg = X_train_30sec_ecg[train_inds]
        X_train_30sec_ppg = X_train_30sec_ppg[train_inds]
        X_train_30sec_ecg_ppg = X_train_30sec_ecg_ppg[train_inds]
        X_train_30sec_ecg_ppg_mean = X_train_30sec_ecg_ppg_mean[train_inds]
        y_train = y_train[train_inds]
        print(len(y_train))
        print(X_train_30sec_ecg.shape)
        print(X_train_30sec_ppg.shape)
        print(X_train_30sec_ecg_ppg.shape)
        print(X_train_30sec_ecg_ppg_mean.shape)

        X_val_30sec_ecg = val_features_30sec_ecg
        X_val_30sec_ppg = val_features_30sec_ppg
        X_val_30sec_ecg_ppg = val_features_30sec_ecg_ppg
        X_val_30sec_ecg_ppg_mean = (X_val_30sec_ecg + X_val_30sec_ppg)/2.0
        y_val = np.array(val_ICD10_I48s)

        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best_c_ecg, penalty="l2", solver="lbfgs", max_iter=2000))
        pipe.fit(X_train_30sec_ecg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg)
        y_val_proba = pipe.predict_proba(X_val_30sec_ecg)[:, 1]
        rocauc = roc_auc_score(y_val, y_val_proba)
        auprc = average_precision_score(y_val, y_val_proba)
        all_tasks_auroc_values_ecg.append(rocauc)
        all_tasks_auprc_values_ecg.append(auprc)
        print("30sec -- C: %f -- ECG-only -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ecg, rocauc, auprc))

        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best_c_ppg, penalty="l2", solver="lbfgs", max_iter=2000))
        pipe.fit(X_train_30sec_ppg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ppg)
        y_val_proba = pipe.predict_proba(X_val_30sec_ppg)[:, 1]
        rocauc = roc_auc_score(y_val, y_val_proba)
        auprc = average_precision_score(y_val, y_val_proba)
        all_tasks_auroc_values_ppg.append(rocauc)
        all_tasks_auprc_values_ppg.append(auprc)
        print("30sec -- C: %f -- PPG-only -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ppg, rocauc, auprc))

        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best_c_ecg_ppg, penalty="l2", solver="lbfgs", max_iter=2000))
        pipe.fit(X_train_30sec_ecg_ppg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg_ppg)
        y_val_proba = pipe.predict_proba(X_val_30sec_ecg_ppg)[:, 1]
        rocauc = roc_auc_score(y_val, y_val_proba)
        auprc = average_precision_score(y_val, y_val_proba)
        all_tasks_auroc_values_ecg_ppg.append(rocauc)
        all_tasks_auprc_values_ecg_ppg.append(auprc)
        print("30sec -- C: %f -- ECG + PPG -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ecg_ppg, rocauc, auprc))

        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best_c_ecg_ppg_mean, penalty="l2", solver="lbfgs", max_iter=2000))
        pipe.fit(X_train_30sec_ecg_ppg_mean, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg_ppg_mean)
        y_val_proba = pipe.predict_proba(X_val_30sec_ecg_ppg_mean)[:, 1]
        rocauc = roc_auc_score(y_val, y_val_proba)
        auprc = average_precision_score(y_val, y_val_proba)
        all_tasks_auroc_values_ecg_ppg_mean.append(rocauc)
        all_tasks_auprc_values_ecg_ppg_mean.append(auprc)
        print("30sec -- C: %f -- ECG-only + PPG-only (mean of features) -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ecg_ppg_mean, rocauc, auprc))




        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("13. I50 classification -- Prior Heart Failure Detection:")
        filtered = pmh_df[pmh_df["Code"].astype(str).str.startswith("I50")]
        mrns_with_code = filtered["MRN"].unique()
        csns_with_code = visits_df[visits_df["MRN"].isin(mrns_with_code)]["CSN"].unique()

        train_ICD10_I50s = []
        train_features_list_30sec_ecg = []
        train_features_list_30sec_ppg = []
        train_features_list_30sec_ecg_ppg = []
        for i, csn in enumerate(preprocessed_csns):
            if csn in train_csns:
                if csn in csns_with_code:
                    train_ICD10_I50s.append(1)
                else:
                    train_ICD10_I50s.append(0)
                train_features_list_30sec_ecg.append(features_30sec_ecg[i])
                train_features_list_30sec_ppg.append(features_30sec_ppg[i])
                train_features_list_30sec_ecg_ppg.append(features_30sec_ecg_ppg[i])
        train_features_30sec_ecg = np.stack(train_features_list_30sec_ecg, axis=0)
        train_features_30sec_ppg = np.stack(train_features_list_30sec_ppg, axis=0)
        train_features_30sec_ecg_ppg = np.stack(train_features_list_30sec_ecg_ppg, axis=0)
        print(len(train_ICD10_I50s))
        print(train_features_30sec_ecg.shape)
        print(train_features_30sec_ppg.shape)
        print(train_features_30sec_ecg_ppg.shape)
        print("proportion of positive class in train: %.4f" % (train_ICD10_I50s.count(1)/float(len(train_ICD10_I50s))))

        val_ICD10_I50s = []
        val_features_list_30sec_ecg = []
        val_features_list_30sec_ppg = []
        val_features_list_30sec_ecg_ppg = []
        for i, csn in enumerate(preprocessed_csns):
            if csn in val_csns:
                if csn in csns_with_code:
                    val_ICD10_I50s.append(1)
                else:
                    val_ICD10_I50s.append(0)
                val_features_list_30sec_ecg.append(features_30sec_ecg[i])
                val_features_list_30sec_ppg.append(features_30sec_ppg[i])
                val_features_list_30sec_ecg_ppg.append(features_30sec_ecg_ppg[i])
        val_features_30sec_ecg = np.stack(val_features_list_30sec_ecg, axis=0)
        val_features_30sec_ppg = np.stack(val_features_list_30sec_ppg, axis=0)
        val_features_30sec_ecg_ppg = np.stack(val_features_list_30sec_ecg_ppg, axis=0)
        print(len(val_ICD10_I50s))
        print(val_features_30sec_ecg.shape)
        print(val_features_30sec_ppg.shape)
        print(val_features_30sec_ecg_ppg.shape)
        print("proportion of positive class in val: %.4f" % (val_ICD10_I50s.count(1)/float(len(val_ICD10_I50s))))

        X_train_30sec_ecg = train_features_30sec_ecg
        X_train_30sec_ppg = train_features_30sec_ppg
        X_train_30sec_ecg_ppg = train_features_30sec_ecg_ppg
        X_train_30sec_ecg_ppg_mean = (X_train_30sec_ecg + X_train_30sec_ppg)/2.0
        y_train = np.array(train_ICD10_I50s)
        #######
        indices = np.arange(X_train_30sec_ecg.shape[0])
        train_inds = rng.choice(indices, size=int(train_prop*len(indices)), replace=True)
        X_train_30sec_ecg = X_train_30sec_ecg[train_inds]
        X_train_30sec_ppg = X_train_30sec_ppg[train_inds]
        X_train_30sec_ecg_ppg = X_train_30sec_ecg_ppg[train_inds]
        X_train_30sec_ecg_ppg_mean = X_train_30sec_ecg_ppg_mean[train_inds]
        y_train = y_train[train_inds]
        print(len(y_train))
        print(X_train_30sec_ecg.shape)
        print(X_train_30sec_ppg.shape)
        print(X_train_30sec_ecg_ppg.shape)
        print(X_train_30sec_ecg_ppg_mean.shape)

        X_val_30sec_ecg = val_features_30sec_ecg
        X_val_30sec_ppg = val_features_30sec_ppg
        X_val_30sec_ecg_ppg = val_features_30sec_ecg_ppg
        X_val_30sec_ecg_ppg_mean = (X_val_30sec_ecg + X_val_30sec_ppg)/2.0
        y_val = np.array(val_ICD10_I50s)

        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best_c_ecg, penalty="l2", solver="lbfgs", max_iter=2000))
        pipe.fit(X_train_30sec_ecg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg)
        y_val_proba = pipe.predict_proba(X_val_30sec_ecg)[:, 1]
        rocauc = roc_auc_score(y_val, y_val_proba)
        auprc = average_precision_score(y_val, y_val_proba)
        all_tasks_auroc_values_ecg.append(rocauc)
        all_tasks_auprc_values_ecg.append(auprc)
        print("30sec -- C: %f -- ECG-only -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ecg, rocauc, auprc))

        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best_c_ppg, penalty="l2", solver="lbfgs", max_iter=2000))
        pipe.fit(X_train_30sec_ppg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ppg)
        y_val_proba = pipe.predict_proba(X_val_30sec_ppg)[:, 1]
        rocauc = roc_auc_score(y_val, y_val_proba)
        auprc = average_precision_score(y_val, y_val_proba)
        all_tasks_auroc_values_ppg.append(rocauc)
        all_tasks_auprc_values_ppg.append(auprc)
        print("30sec -- C: %f -- PPG-only -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ppg, rocauc, auprc))

        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best_c_ecg_ppg, penalty="l2", solver="lbfgs", max_iter=2000))
        pipe.fit(X_train_30sec_ecg_ppg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg_ppg)
        y_val_proba = pipe.predict_proba(X_val_30sec_ecg_ppg)[:, 1]
        rocauc = roc_auc_score(y_val, y_val_proba)
        auprc = average_precision_score(y_val, y_val_proba)
        all_tasks_auroc_values_ecg_ppg.append(rocauc)
        all_tasks_auprc_values_ecg_ppg.append(auprc)
        print("30sec -- C: %f -- ECG + PPG -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ecg_ppg, rocauc, auprc))

        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best_c_ecg_ppg_mean, penalty="l2", solver="lbfgs", max_iter=2000))
        pipe.fit(X_train_30sec_ecg_ppg_mean, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg_ppg_mean)
        y_val_proba = pipe.predict_proba(X_val_30sec_ecg_ppg_mean)[:, 1]
        rocauc = roc_auc_score(y_val, y_val_proba)
        auprc = average_precision_score(y_val, y_val_proba)
        all_tasks_auroc_values_ecg_ppg_mean.append(rocauc)
        all_tasks_auprc_values_ecg_ppg_mean.append(auprc)
        print("30sec -- C: %f -- ECG-only + PPG-only (mean of features) -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ecg_ppg_mean, rocauc, auprc))




        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("14. {Z95.0, Z95.2, Z95.3} classification -- Prior Cardiac Device Detection (group17):")
        filtered = pmh_df[pmh_df["Code"].astype(str).str.startswith(("Z950", "Z952", "Z953"))]
        mrns_with_code = filtered["MRN"].unique()
        csns_with_code = visits_df[visits_df["MRN"].isin(mrns_with_code)]["CSN"].unique()

        train_ICD10_group17s = []
        train_features_list_30sec_ecg = []
        train_features_list_30sec_ppg = []
        train_features_list_30sec_ecg_ppg = []
        for i, csn in enumerate(preprocessed_csns):
            if csn in train_csns:
                if csn in csns_with_code:
                    train_ICD10_group17s.append(1)
                else:
                    train_ICD10_group17s.append(0)
                train_features_list_30sec_ecg.append(features_30sec_ecg[i])
                train_features_list_30sec_ppg.append(features_30sec_ppg[i])
                train_features_list_30sec_ecg_ppg.append(features_30sec_ecg_ppg[i])
        train_features_30sec_ecg = np.stack(train_features_list_30sec_ecg, axis=0)
        train_features_30sec_ppg = np.stack(train_features_list_30sec_ppg, axis=0)
        train_features_30sec_ecg_ppg = np.stack(train_features_list_30sec_ecg_ppg, axis=0)
        print(len(train_ICD10_group17s))
        print(train_features_30sec_ecg.shape)
        print(train_features_30sec_ppg.shape)
        print(train_features_30sec_ecg_ppg.shape)
        print("proportion of positive class in train: %.4f" % (train_ICD10_group17s.count(1)/float(len(train_ICD10_group17s))))

        val_ICD10_group17s = []
        val_features_list_30sec_ecg = []
        val_features_list_30sec_ppg = []
        val_features_list_30sec_ecg_ppg = []
        for i, csn in enumerate(preprocessed_csns):
            if csn in val_csns:
                if csn in csns_with_code:
                    val_ICD10_group17s.append(1)
                else:
                    val_ICD10_group17s.append(0)
                val_features_list_30sec_ecg.append(features_30sec_ecg[i])
                val_features_list_30sec_ppg.append(features_30sec_ppg[i])
                val_features_list_30sec_ecg_ppg.append(features_30sec_ecg_ppg[i])
        val_features_30sec_ecg = np.stack(val_features_list_30sec_ecg, axis=0)
        val_features_30sec_ppg = np.stack(val_features_list_30sec_ppg, axis=0)
        val_features_30sec_ecg_ppg = np.stack(val_features_list_30sec_ecg_ppg, axis=0)
        print(len(val_ICD10_group17s))
        print(val_features_30sec_ecg.shape)
        print(val_features_30sec_ppg.shape)
        print(val_features_30sec_ecg_ppg.shape)
        print("proportion of positive class in val: %.4f" % (val_ICD10_group17s.count(1)/float(len(val_ICD10_group17s))))

        X_train_30sec_ecg = train_features_30sec_ecg
        X_train_30sec_ppg = train_features_30sec_ppg
        X_train_30sec_ecg_ppg = train_features_30sec_ecg_ppg
        X_train_30sec_ecg_ppg_mean = (X_train_30sec_ecg + X_train_30sec_ppg)/2.0
        y_train = np.array(train_ICD10_group17s)
        #######
        indices = np.arange(X_train_30sec_ecg.shape[0])
        train_inds = rng.choice(indices, size=int(train_prop*len(indices)), replace=True)
        X_train_30sec_ecg = X_train_30sec_ecg[train_inds]
        X_train_30sec_ppg = X_train_30sec_ppg[train_inds]
        X_train_30sec_ecg_ppg = X_train_30sec_ecg_ppg[train_inds]
        X_train_30sec_ecg_ppg_mean = X_train_30sec_ecg_ppg_mean[train_inds]
        y_train = y_train[train_inds]
        print(len(y_train))
        print(X_train_30sec_ecg.shape)
        print(X_train_30sec_ppg.shape)
        print(X_train_30sec_ecg_ppg.shape)
        print(X_train_30sec_ecg_ppg_mean.shape)

        X_val_30sec_ecg = val_features_30sec_ecg
        X_val_30sec_ppg = val_features_30sec_ppg
        X_val_30sec_ecg_ppg = val_features_30sec_ecg_ppg
        X_val_30sec_ecg_ppg_mean = (X_val_30sec_ecg + X_val_30sec_ppg)/2.0
        y_val = np.array(val_ICD10_group17s)

        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best_c_ecg, penalty="l2", solver="lbfgs", max_iter=2000))
        pipe.fit(X_train_30sec_ecg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg)
        y_val_proba = pipe.predict_proba(X_val_30sec_ecg)[:, 1]
        rocauc = roc_auc_score(y_val, y_val_proba)
        auprc = average_precision_score(y_val, y_val_proba)
        all_tasks_auroc_values_ecg.append(rocauc)
        all_tasks_auprc_values_ecg.append(auprc)
        print("30sec -- C: %f -- ECG-only -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ecg, rocauc, auprc))

        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best_c_ppg, penalty="l2", solver="lbfgs", max_iter=2000))
        pipe.fit(X_train_30sec_ppg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ppg)
        y_val_proba = pipe.predict_proba(X_val_30sec_ppg)[:, 1]
        rocauc = roc_auc_score(y_val, y_val_proba)
        auprc = average_precision_score(y_val, y_val_proba)
        all_tasks_auroc_values_ppg.append(rocauc)
        all_tasks_auprc_values_ppg.append(auprc)
        print("30sec -- C: %f -- PPG-only -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ppg, rocauc, auprc))

        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best_c_ecg_ppg, penalty="l2", solver="lbfgs", max_iter=2000))
        pipe.fit(X_train_30sec_ecg_ppg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg_ppg)
        y_val_proba = pipe.predict_proba(X_val_30sec_ecg_ppg)[:, 1]
        rocauc = roc_auc_score(y_val, y_val_proba)
        auprc = average_precision_score(y_val, y_val_proba)
        all_tasks_auroc_values_ecg_ppg.append(rocauc)
        all_tasks_auprc_values_ecg_ppg.append(auprc)
        print("30sec -- C: %f -- ECG + PPG -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ecg_ppg, rocauc, auprc))

        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best_c_ecg_ppg_mean, penalty="l2", solver="lbfgs", max_iter=2000))
        pipe.fit(X_train_30sec_ecg_ppg_mean, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg_ppg_mean)
        y_val_proba = pipe.predict_proba(X_val_30sec_ecg_ppg_mean)[:, 1]
        rocauc = roc_auc_score(y_val, y_val_proba)
        auprc = average_precision_score(y_val, y_val_proba)
        all_tasks_auroc_values_ecg_ppg_mean.append(rocauc)
        all_tasks_auprc_values_ecg_ppg_mean.append(auprc)
        print("30sec -- C: %f -- ECG-only + PPG-only (mean of features) -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ecg_ppg_mean, rocauc, auprc))




        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("15. N18 classification -- Prior Chronic Kidney Disease Detection:")
        filtered = pmh_df[pmh_df["Code"].astype(str).str.startswith("N18")]
        mrns_with_code = filtered["MRN"].unique()
        csns_with_code = visits_df[visits_df["MRN"].isin(mrns_with_code)]["CSN"].unique()

        train_ICD10_N18s = []
        train_features_list_30sec_ecg = []
        train_features_list_30sec_ppg = []
        train_features_list_30sec_ecg_ppg = []
        for i, csn in enumerate(preprocessed_csns):
            if csn in train_csns:
                if csn in csns_with_code:
                    train_ICD10_N18s.append(1)
                else:
                    train_ICD10_N18s.append(0)
                train_features_list_30sec_ecg.append(features_30sec_ecg[i])
                train_features_list_30sec_ppg.append(features_30sec_ppg[i])
                train_features_list_30sec_ecg_ppg.append(features_30sec_ecg_ppg[i])
        train_features_30sec_ecg = np.stack(train_features_list_30sec_ecg, axis=0)
        train_features_30sec_ppg = np.stack(train_features_list_30sec_ppg, axis=0)
        train_features_30sec_ecg_ppg = np.stack(train_features_list_30sec_ecg_ppg, axis=0)
        print(len(train_ICD10_N18s))
        print(train_features_30sec_ecg.shape)
        print(train_features_30sec_ppg.shape)
        print(train_features_30sec_ecg_ppg.shape)
        print("proportion of positive class in train: %.4f" % (train_ICD10_N18s.count(1)/float(len(train_ICD10_N18s))))

        val_ICD10_N18s = []
        val_features_list_30sec_ecg = []
        val_features_list_30sec_ppg = []
        val_features_list_30sec_ecg_ppg = []
        for i, csn in enumerate(preprocessed_csns):
            if csn in val_csns:
                if csn in csns_with_code:
                    val_ICD10_N18s.append(1)
                else:
                    val_ICD10_N18s.append(0)
                val_features_list_30sec_ecg.append(features_30sec_ecg[i])
                val_features_list_30sec_ppg.append(features_30sec_ppg[i])
                val_features_list_30sec_ecg_ppg.append(features_30sec_ecg_ppg[i])
        val_features_30sec_ecg = np.stack(val_features_list_30sec_ecg, axis=0)
        val_features_30sec_ppg = np.stack(val_features_list_30sec_ppg, axis=0)
        val_features_30sec_ecg_ppg = np.stack(val_features_list_30sec_ecg_ppg, axis=0)
        print(len(val_ICD10_N18s))
        print(val_features_30sec_ecg.shape)
        print(val_features_30sec_ppg.shape)
        print(val_features_30sec_ecg_ppg.shape)
        print("proportion of positive class in val: %.4f" % (val_ICD10_N18s.count(1)/float(len(val_ICD10_N18s))))

        X_train_30sec_ecg = train_features_30sec_ecg
        X_train_30sec_ppg = train_features_30sec_ppg
        X_train_30sec_ecg_ppg = train_features_30sec_ecg_ppg
        X_train_30sec_ecg_ppg_mean = (X_train_30sec_ecg + X_train_30sec_ppg)/2.0
        y_train = np.array(train_ICD10_N18s)
        #######
        indices = np.arange(X_train_30sec_ecg.shape[0])
        train_inds = rng.choice(indices, size=int(train_prop*len(indices)), replace=True)
        X_train_30sec_ecg = X_train_30sec_ecg[train_inds]
        X_train_30sec_ppg = X_train_30sec_ppg[train_inds]
        X_train_30sec_ecg_ppg = X_train_30sec_ecg_ppg[train_inds]
        X_train_30sec_ecg_ppg_mean = X_train_30sec_ecg_ppg_mean[train_inds]
        y_train = y_train[train_inds]
        print(len(y_train))
        print(X_train_30sec_ecg.shape)
        print(X_train_30sec_ppg.shape)
        print(X_train_30sec_ecg_ppg.shape)
        print(X_train_30sec_ecg_ppg_mean.shape)

        X_val_30sec_ecg = val_features_30sec_ecg
        X_val_30sec_ppg = val_features_30sec_ppg
        X_val_30sec_ecg_ppg = val_features_30sec_ecg_ppg
        X_val_30sec_ecg_ppg_mean = (X_val_30sec_ecg + X_val_30sec_ppg)/2.0
        y_val = np.array(val_ICD10_N18s)

        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best_c_ecg, penalty="l2", solver="lbfgs", max_iter=2000))
        pipe.fit(X_train_30sec_ecg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg)
        y_val_proba = pipe.predict_proba(X_val_30sec_ecg)[:, 1]
        rocauc = roc_auc_score(y_val, y_val_proba)
        auprc = average_precision_score(y_val, y_val_proba)
        all_tasks_auroc_values_ecg.append(rocauc)
        all_tasks_auprc_values_ecg.append(auprc)
        print("30sec -- C: %f -- ECG-only -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ecg, rocauc, auprc))

        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best_c_ppg, penalty="l2", solver="lbfgs", max_iter=2000))
        pipe.fit(X_train_30sec_ppg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ppg)
        y_val_proba = pipe.predict_proba(X_val_30sec_ppg)[:, 1]
        rocauc = roc_auc_score(y_val, y_val_proba)
        auprc = average_precision_score(y_val, y_val_proba)
        all_tasks_auroc_values_ppg.append(rocauc)
        all_tasks_auprc_values_ppg.append(auprc)
        print("30sec -- C: %f -- PPG-only -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ppg, rocauc, auprc))

        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best_c_ecg_ppg, penalty="l2", solver="lbfgs", max_iter=2000))
        pipe.fit(X_train_30sec_ecg_ppg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg_ppg)
        y_val_proba = pipe.predict_proba(X_val_30sec_ecg_ppg)[:, 1]
        rocauc = roc_auc_score(y_val, y_val_proba)
        auprc = average_precision_score(y_val, y_val_proba)
        all_tasks_auroc_values_ecg_ppg.append(rocauc)
        all_tasks_auprc_values_ecg_ppg.append(auprc)
        print("30sec -- C: %f -- ECG + PPG -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ecg_ppg, rocauc, auprc))

        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best_c_ecg_ppg_mean, penalty="l2", solver="lbfgs", max_iter=2000))
        pipe.fit(X_train_30sec_ecg_ppg_mean, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg_ppg_mean)
        y_val_proba = pipe.predict_proba(X_val_30sec_ecg_ppg_mean)[:, 1]
        rocauc = roc_auc_score(y_val, y_val_proba)
        auprc = average_precision_score(y_val, y_val_proba)
        all_tasks_auroc_values_ecg_ppg_mean.append(rocauc)
        all_tasks_auprc_values_ecg_ppg_mean.append(auprc)
        print("30sec -- C: %f -- ECG-only + PPG-only (mean of features) -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ecg_ppg_mean, rocauc, auprc))




        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("16. {E10, E11} classification -- Prior Diabetes Mellitus Detection (group12):")
        filtered = pmh_df[pmh_df["Code"].astype(str).str.startswith(("E10", "E11"))]
        mrns_with_code = filtered["MRN"].unique()
        csns_with_code = visits_df[visits_df["MRN"].isin(mrns_with_code)]["CSN"].unique()

        train_ICD10_group12s = []
        train_features_list_30sec_ecg = []
        train_features_list_30sec_ppg = []
        train_features_list_30sec_ecg_ppg = []
        for i, csn in enumerate(preprocessed_csns):
            if csn in train_csns:
                if csn in csns_with_code:
                    train_ICD10_group12s.append(1)
                else:
                    train_ICD10_group12s.append(0)
                train_features_list_30sec_ecg.append(features_30sec_ecg[i])
                train_features_list_30sec_ppg.append(features_30sec_ppg[i])
                train_features_list_30sec_ecg_ppg.append(features_30sec_ecg_ppg[i])
        train_features_30sec_ecg = np.stack(train_features_list_30sec_ecg, axis=0)
        train_features_30sec_ppg = np.stack(train_features_list_30sec_ppg, axis=0)
        train_features_30sec_ecg_ppg = np.stack(train_features_list_30sec_ecg_ppg, axis=0)
        print(len(train_ICD10_group12s))
        print(train_features_30sec_ecg.shape)
        print(train_features_30sec_ppg.shape)
        print(train_features_30sec_ecg_ppg.shape)
        print("proportion of positive class in train: %.4f" % (train_ICD10_group12s.count(1)/float(len(train_ICD10_group12s))))

        val_ICD10_group12s = []
        val_features_list_30sec_ecg = []
        val_features_list_30sec_ppg = []
        val_features_list_30sec_ecg_ppg = []
        for i, csn in enumerate(preprocessed_csns):
            if csn in val_csns:
                if csn in csns_with_code:
                    val_ICD10_group12s.append(1)
                else:
                    val_ICD10_group12s.append(0)
                val_features_list_30sec_ecg.append(features_30sec_ecg[i])
                val_features_list_30sec_ppg.append(features_30sec_ppg[i])
                val_features_list_30sec_ecg_ppg.append(features_30sec_ecg_ppg[i])
        val_features_30sec_ecg = np.stack(val_features_list_30sec_ecg, axis=0)
        val_features_30sec_ppg = np.stack(val_features_list_30sec_ppg, axis=0)
        val_features_30sec_ecg_ppg = np.stack(val_features_list_30sec_ecg_ppg, axis=0)
        print(len(val_ICD10_group12s))
        print(val_features_30sec_ecg.shape)
        print(val_features_30sec_ppg.shape)
        print(val_features_30sec_ecg_ppg.shape)
        print("proportion of positive class in val: %.4f" % (val_ICD10_group12s.count(1)/float(len(val_ICD10_group12s))))

        X_train_30sec_ecg = train_features_30sec_ecg
        X_train_30sec_ppg = train_features_30sec_ppg
        X_train_30sec_ecg_ppg = train_features_30sec_ecg_ppg
        X_train_30sec_ecg_ppg_mean = (X_train_30sec_ecg + X_train_30sec_ppg)/2.0
        y_train = np.array(train_ICD10_group12s)
        #######
        indices = np.arange(X_train_30sec_ecg.shape[0])
        train_inds = rng.choice(indices, size=int(train_prop*len(indices)), replace=True)
        X_train_30sec_ecg = X_train_30sec_ecg[train_inds]
        X_train_30sec_ppg = X_train_30sec_ppg[train_inds]
        X_train_30sec_ecg_ppg = X_train_30sec_ecg_ppg[train_inds]
        X_train_30sec_ecg_ppg_mean = X_train_30sec_ecg_ppg_mean[train_inds]
        y_train = y_train[train_inds]
        print(len(y_train))
        print(X_train_30sec_ecg.shape)
        print(X_train_30sec_ppg.shape)
        print(X_train_30sec_ecg_ppg.shape)
        print(X_train_30sec_ecg_ppg_mean.shape)

        X_val_30sec_ecg = val_features_30sec_ecg
        X_val_30sec_ppg = val_features_30sec_ppg
        X_val_30sec_ecg_ppg = val_features_30sec_ecg_ppg
        X_val_30sec_ecg_ppg_mean = (X_val_30sec_ecg + X_val_30sec_ppg)/2.0
        y_val = np.array(val_ICD10_group12s)

        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best_c_ecg, penalty="l2", solver="lbfgs", max_iter=2000))
        pipe.fit(X_train_30sec_ecg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg)
        y_val_proba = pipe.predict_proba(X_val_30sec_ecg)[:, 1]
        rocauc = roc_auc_score(y_val, y_val_proba)
        auprc = average_precision_score(y_val, y_val_proba)
        all_tasks_auroc_values_ecg.append(rocauc)
        all_tasks_auprc_values_ecg.append(auprc)
        print("30sec -- C: %f -- ECG-only -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ecg, rocauc, auprc))

        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best_c_ppg, penalty="l2", solver="lbfgs", max_iter=2000))
        pipe.fit(X_train_30sec_ppg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ppg)
        y_val_proba = pipe.predict_proba(X_val_30sec_ppg)[:, 1]
        rocauc = roc_auc_score(y_val, y_val_proba)
        auprc = average_precision_score(y_val, y_val_proba)
        all_tasks_auroc_values_ppg.append(rocauc)
        all_tasks_auprc_values_ppg.append(auprc)
        print("30sec -- C: %f -- PPG-only -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ppg, rocauc, auprc))

        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best_c_ecg_ppg, penalty="l2", solver="lbfgs", max_iter=2000))
        pipe.fit(X_train_30sec_ecg_ppg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg_ppg)
        y_val_proba = pipe.predict_proba(X_val_30sec_ecg_ppg)[:, 1]
        rocauc = roc_auc_score(y_val, y_val_proba)
        auprc = average_precision_score(y_val, y_val_proba)
        all_tasks_auroc_values_ecg_ppg.append(rocauc)
        all_tasks_auprc_values_ecg_ppg.append(auprc)
        print("30sec -- C: %f -- ECG + PPG -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ecg_ppg, rocauc, auprc))

        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best_c_ecg_ppg_mean, penalty="l2", solver="lbfgs", max_iter=2000))
        pipe.fit(X_train_30sec_ecg_ppg_mean, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg_ppg_mean)
        y_val_proba = pipe.predict_proba(X_val_30sec_ecg_ppg_mean)[:, 1]
        rocauc = roc_auc_score(y_val, y_val_proba)
        auprc = average_precision_score(y_val, y_val_proba)
        all_tasks_auroc_values_ecg_ppg_mean.append(rocauc)
        all_tasks_auprc_values_ecg_ppg_mean.append(auprc)
        print("30sec -- C: %f -- ECG-only + PPG-only (mean of features) -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ecg_ppg_mean, rocauc, auprc))




        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("17. {G47.3, E66.2} classification -- Prior Sleep-Related Breathing Disorder Detection (group21):")
        filtered = pmh_df[pmh_df["Code"].astype(str).str.startswith(("G473", "E662"))]
        mrns_with_code = filtered["MRN"].unique()
        csns_with_code = visits_df[visits_df["MRN"].isin(mrns_with_code)]["CSN"].unique()

        train_ICD10_group21s = []
        train_features_list_30sec_ecg = []
        train_features_list_30sec_ppg = []
        train_features_list_30sec_ecg_ppg = []
        for i, csn in enumerate(preprocessed_csns):
            if csn in train_csns:
                if csn in csns_with_code:
                    train_ICD10_group21s.append(1)
                else:
                    train_ICD10_group21s.append(0)
                train_features_list_30sec_ecg.append(features_30sec_ecg[i])
                train_features_list_30sec_ppg.append(features_30sec_ppg[i])
                train_features_list_30sec_ecg_ppg.append(features_30sec_ecg_ppg[i])
        train_features_30sec_ecg = np.stack(train_features_list_30sec_ecg, axis=0)
        train_features_30sec_ppg = np.stack(train_features_list_30sec_ppg, axis=0)
        train_features_30sec_ecg_ppg = np.stack(train_features_list_30sec_ecg_ppg, axis=0)
        print(len(train_ICD10_group21s))
        print(train_features_30sec_ecg.shape)
        print(train_features_30sec_ppg.shape)
        print(train_features_30sec_ecg_ppg.shape)
        print("proportion of positive class in train: %.4f" % (train_ICD10_group21s.count(1)/float(len(train_ICD10_group21s))))

        val_ICD10_group21s = []
        val_features_list_30sec_ecg = []
        val_features_list_30sec_ppg = []
        val_features_list_30sec_ecg_ppg = []
        for i, csn in enumerate(preprocessed_csns):
            if csn in val_csns:
                if csn in csns_with_code:
                    val_ICD10_group21s.append(1)
                else:
                    val_ICD10_group21s.append(0)
                val_features_list_30sec_ecg.append(features_30sec_ecg[i])
                val_features_list_30sec_ppg.append(features_30sec_ppg[i])
                val_features_list_30sec_ecg_ppg.append(features_30sec_ecg_ppg[i])
        val_features_30sec_ecg = np.stack(val_features_list_30sec_ecg, axis=0)
        val_features_30sec_ppg = np.stack(val_features_list_30sec_ppg, axis=0)
        val_features_30sec_ecg_ppg = np.stack(val_features_list_30sec_ecg_ppg, axis=0)
        print(len(val_ICD10_group21s))
        print(val_features_30sec_ecg.shape)
        print(val_features_30sec_ppg.shape)
        print(val_features_30sec_ecg_ppg.shape)
        print("proportion of positive class in val: %.4f" % (val_ICD10_group21s.count(1)/float(len(val_ICD10_group21s))))

        X_train_30sec_ecg = train_features_30sec_ecg
        X_train_30sec_ppg = train_features_30sec_ppg
        X_train_30sec_ecg_ppg = train_features_30sec_ecg_ppg
        X_train_30sec_ecg_ppg_mean = (X_train_30sec_ecg + X_train_30sec_ppg)/2.0
        y_train = np.array(train_ICD10_group21s)
        #######
        indices = np.arange(X_train_30sec_ecg.shape[0])
        train_inds = rng.choice(indices, size=int(train_prop*len(indices)), replace=True)
        X_train_30sec_ecg = X_train_30sec_ecg[train_inds]
        X_train_30sec_ppg = X_train_30sec_ppg[train_inds]
        X_train_30sec_ecg_ppg = X_train_30sec_ecg_ppg[train_inds]
        X_train_30sec_ecg_ppg_mean = X_train_30sec_ecg_ppg_mean[train_inds]
        y_train = y_train[train_inds]
        print(len(y_train))
        print(X_train_30sec_ecg.shape)
        print(X_train_30sec_ppg.shape)
        print(X_train_30sec_ecg_ppg.shape)
        print(X_train_30sec_ecg_ppg_mean.shape)

        X_val_30sec_ecg = val_features_30sec_ecg
        X_val_30sec_ppg = val_features_30sec_ppg
        X_val_30sec_ecg_ppg = val_features_30sec_ecg_ppg
        X_val_30sec_ecg_ppg_mean = (X_val_30sec_ecg + X_val_30sec_ppg)/2.0
        y_val = np.array(val_ICD10_group21s)

        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best_c_ecg, penalty="l2", solver="lbfgs", max_iter=2000))
        pipe.fit(X_train_30sec_ecg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg)
        y_val_proba = pipe.predict_proba(X_val_30sec_ecg)[:, 1]
        rocauc = roc_auc_score(y_val, y_val_proba)
        auprc = average_precision_score(y_val, y_val_proba)
        all_tasks_auroc_values_ecg.append(rocauc)
        all_tasks_auprc_values_ecg.append(auprc)
        print("30sec -- C: %f -- ECG-only -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ecg, rocauc, auprc))

        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best_c_ppg, penalty="l2", solver="lbfgs", max_iter=2000))
        pipe.fit(X_train_30sec_ppg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ppg)
        y_val_proba = pipe.predict_proba(X_val_30sec_ppg)[:, 1]
        rocauc = roc_auc_score(y_val, y_val_proba)
        auprc = average_precision_score(y_val, y_val_proba)
        all_tasks_auroc_values_ppg.append(rocauc)
        all_tasks_auprc_values_ppg.append(auprc)
        print("30sec -- C: %f -- PPG-only -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ppg, rocauc, auprc))

        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best_c_ecg_ppg, penalty="l2", solver="lbfgs", max_iter=2000))
        pipe.fit(X_train_30sec_ecg_ppg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg_ppg)
        y_val_proba = pipe.predict_proba(X_val_30sec_ecg_ppg)[:, 1]
        rocauc = roc_auc_score(y_val, y_val_proba)
        auprc = average_precision_score(y_val, y_val_proba)
        all_tasks_auroc_values_ecg_ppg.append(rocauc)
        all_tasks_auprc_values_ecg_ppg.append(auprc)
        print("30sec -- C: %f -- ECG + PPG -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ecg_ppg, rocauc, auprc))

        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best_c_ecg_ppg_mean, penalty="l2", solver="lbfgs", max_iter=2000))
        pipe.fit(X_train_30sec_ecg_ppg_mean, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg_ppg_mean)
        y_val_proba = pipe.predict_proba(X_val_30sec_ecg_ppg_mean)[:, 1]
        rocauc = roc_auc_score(y_val, y_val_proba)
        auprc = average_precision_score(y_val, y_val_proba)
        all_tasks_auroc_values_ecg_ppg_mean.append(rocauc)
        all_tasks_auprc_values_ecg_ppg_mean.append(auprc)
        print("30sec -- C: %f -- ECG-only + PPG-only (mean of features) -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ecg_ppg_mean, rocauc, auprc))




        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("18. {D50, D51, D52, D53, D64} classification -- Prior Anemia Detection (group10):")
        filtered = pmh_df[pmh_df["Code"].astype(str).str.startswith(("D50", "D51", "D52", "D53", "D64"))]
        mrns_with_code = filtered["MRN"].unique()
        csns_with_code = visits_df[visits_df["MRN"].isin(mrns_with_code)]["CSN"].unique()

        train_ICD10_group10s = []
        train_features_list_30sec_ecg = []
        train_features_list_30sec_ppg = []
        train_features_list_30sec_ecg_ppg = []
        for i, csn in enumerate(preprocessed_csns):
            if csn in train_csns:
                if csn in csns_with_code:
                    train_ICD10_group10s.append(1)
                else:
                    train_ICD10_group10s.append(0)
                train_features_list_30sec_ecg.append(features_30sec_ecg[i])
                train_features_list_30sec_ppg.append(features_30sec_ppg[i])
                train_features_list_30sec_ecg_ppg.append(features_30sec_ecg_ppg[i])
        train_features_30sec_ecg = np.stack(train_features_list_30sec_ecg, axis=0)
        train_features_30sec_ppg = np.stack(train_features_list_30sec_ppg, axis=0)
        train_features_30sec_ecg_ppg = np.stack(train_features_list_30sec_ecg_ppg, axis=0)
        print(len(train_ICD10_group10s))
        print(train_features_30sec_ecg.shape)
        print(train_features_30sec_ppg.shape)
        print(train_features_30sec_ecg_ppg.shape)
        print("proportion of positive class in train: %.4f" % (train_ICD10_group10s.count(1)/float(len(train_ICD10_group10s))))

        val_ICD10_group10s = []
        val_features_list_30sec_ecg = []
        val_features_list_30sec_ppg = []
        val_features_list_30sec_ecg_ppg = []
        for i, csn in enumerate(preprocessed_csns):
            if csn in val_csns:
                if csn in csns_with_code:
                    val_ICD10_group10s.append(1)
                else:
                    val_ICD10_group10s.append(0)
                val_features_list_30sec_ecg.append(features_30sec_ecg[i])
                val_features_list_30sec_ppg.append(features_30sec_ppg[i])
                val_features_list_30sec_ecg_ppg.append(features_30sec_ecg_ppg[i])
        val_features_30sec_ecg = np.stack(val_features_list_30sec_ecg, axis=0)
        val_features_30sec_ppg = np.stack(val_features_list_30sec_ppg, axis=0)
        val_features_30sec_ecg_ppg = np.stack(val_features_list_30sec_ecg_ppg, axis=0)
        print(len(val_ICD10_group10s))
        print(val_features_30sec_ecg.shape)
        print(val_features_30sec_ppg.shape)
        print(val_features_30sec_ecg_ppg.shape)
        print("proportion of positive class in val: %.4f" % (val_ICD10_group10s.count(1)/float(len(val_ICD10_group10s))))

        X_train_30sec_ecg = train_features_30sec_ecg
        X_train_30sec_ppg = train_features_30sec_ppg
        X_train_30sec_ecg_ppg = train_features_30sec_ecg_ppg
        X_train_30sec_ecg_ppg_mean = (X_train_30sec_ecg + X_train_30sec_ppg)/2.0
        y_train = np.array(train_ICD10_group10s)
        #######
        indices = np.arange(X_train_30sec_ecg.shape[0])
        train_inds = rng.choice(indices, size=int(train_prop*len(indices)), replace=True)
        X_train_30sec_ecg = X_train_30sec_ecg[train_inds]
        X_train_30sec_ppg = X_train_30sec_ppg[train_inds]
        X_train_30sec_ecg_ppg = X_train_30sec_ecg_ppg[train_inds]
        X_train_30sec_ecg_ppg_mean = X_train_30sec_ecg_ppg_mean[train_inds]
        y_train = y_train[train_inds]
        print(len(y_train))
        print(X_train_30sec_ecg.shape)
        print(X_train_30sec_ppg.shape)
        print(X_train_30sec_ecg_ppg.shape)
        print(X_train_30sec_ecg_ppg_mean.shape)

        X_val_30sec_ecg = val_features_30sec_ecg
        X_val_30sec_ppg = val_features_30sec_ppg
        X_val_30sec_ecg_ppg = val_features_30sec_ecg_ppg
        X_val_30sec_ecg_ppg_mean = (X_val_30sec_ecg + X_val_30sec_ppg)/2.0
        y_val = np.array(val_ICD10_group10s)

        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best_c_ecg, penalty="l2", solver="lbfgs", max_iter=2000))
        pipe.fit(X_train_30sec_ecg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg)
        y_val_proba = pipe.predict_proba(X_val_30sec_ecg)[:, 1]
        rocauc = roc_auc_score(y_val, y_val_proba)
        auprc = average_precision_score(y_val, y_val_proba)
        all_tasks_auroc_values_ecg.append(rocauc)
        all_tasks_auprc_values_ecg.append(auprc)
        print("30sec -- C: %f -- ECG-only -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ecg, rocauc, auprc))

        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best_c_ppg, penalty="l2", solver="lbfgs", max_iter=2000))
        pipe.fit(X_train_30sec_ppg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ppg)
        y_val_proba = pipe.predict_proba(X_val_30sec_ppg)[:, 1]
        rocauc = roc_auc_score(y_val, y_val_proba)
        auprc = average_precision_score(y_val, y_val_proba)
        all_tasks_auroc_values_ppg.append(rocauc)
        all_tasks_auprc_values_ppg.append(auprc)
        print("30sec -- C: %f -- PPG-only -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ppg, rocauc, auprc))

        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best_c_ecg_ppg, penalty="l2", solver="lbfgs", max_iter=2000))
        pipe.fit(X_train_30sec_ecg_ppg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg_ppg)
        y_val_proba = pipe.predict_proba(X_val_30sec_ecg_ppg)[:, 1]
        rocauc = roc_auc_score(y_val, y_val_proba)
        auprc = average_precision_score(y_val, y_val_proba)
        all_tasks_auroc_values_ecg_ppg.append(rocauc)
        all_tasks_auprc_values_ecg_ppg.append(auprc)
        print("30sec -- C: %f -- ECG + PPG -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ecg_ppg, rocauc, auprc))

        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best_c_ecg_ppg_mean, penalty="l2", solver="lbfgs", max_iter=2000))
        pipe.fit(X_train_30sec_ecg_ppg_mean, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg_ppg_mean)
        y_val_proba = pipe.predict_proba(X_val_30sec_ecg_ppg_mean)[:, 1]
        rocauc = roc_auc_score(y_val, y_val_proba)
        auprc = average_precision_score(y_val, y_val_proba)
        all_tasks_auroc_values_ecg_ppg_mean.append(rocauc)
        all_tasks_auprc_values_ecg_ppg_mean.append(auprc)
        print("30sec -- C: %f -- ECG-only + PPG-only (mean of features) -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ecg_ppg_mean, rocauc, auprc))




        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("19. {J96, R09.02, R06.0} classification -- Prior Respiratory Compromise Detection (group16):")
        filtered = pmh_df[pmh_df["Code"].astype(str).str.startswith(("J96", "R0902", "R060"))]
        mrns_with_code = filtered["MRN"].unique()
        csns_with_code = visits_df[visits_df["MRN"].isin(mrns_with_code)]["CSN"].unique()

        train_ICD10_group16s = []
        train_features_list_30sec_ecg = []
        train_features_list_30sec_ppg = []
        train_features_list_30sec_ecg_ppg = []
        for i, csn in enumerate(preprocessed_csns):
            if csn in train_csns:
                if csn in csns_with_code:
                    train_ICD10_group16s.append(1)
                else:
                    train_ICD10_group16s.append(0)
                train_features_list_30sec_ecg.append(features_30sec_ecg[i])
                train_features_list_30sec_ppg.append(features_30sec_ppg[i])
                train_features_list_30sec_ecg_ppg.append(features_30sec_ecg_ppg[i])
        train_features_30sec_ecg = np.stack(train_features_list_30sec_ecg, axis=0)
        train_features_30sec_ppg = np.stack(train_features_list_30sec_ppg, axis=0)
        train_features_30sec_ecg_ppg = np.stack(train_features_list_30sec_ecg_ppg, axis=0)
        print(len(train_ICD10_group16s))
        print(train_features_30sec_ecg.shape)
        print(train_features_30sec_ppg.shape)
        print(train_features_30sec_ecg_ppg.shape)
        print("proportion of positive class in train: %.4f" % (train_ICD10_group16s.count(1)/float(len(train_ICD10_group16s))))

        val_ICD10_group16s = []
        val_features_list_30sec_ecg = []
        val_features_list_30sec_ppg = []
        val_features_list_30sec_ecg_ppg = []
        for i, csn in enumerate(preprocessed_csns):
            if csn in val_csns:
                if csn in csns_with_code:
                    val_ICD10_group16s.append(1)
                else:
                    val_ICD10_group16s.append(0)
                val_features_list_30sec_ecg.append(features_30sec_ecg[i])
                val_features_list_30sec_ppg.append(features_30sec_ppg[i])
                val_features_list_30sec_ecg_ppg.append(features_30sec_ecg_ppg[i])
        val_features_30sec_ecg = np.stack(val_features_list_30sec_ecg, axis=0)
        val_features_30sec_ppg = np.stack(val_features_list_30sec_ppg, axis=0)
        val_features_30sec_ecg_ppg = np.stack(val_features_list_30sec_ecg_ppg, axis=0)
        print(len(val_ICD10_group16s))
        print(val_features_30sec_ecg.shape)
        print(val_features_30sec_ppg.shape)
        print(val_features_30sec_ecg_ppg.shape)
        print("proportion of positive class in val: %.4f" % (val_ICD10_group16s.count(1)/float(len(val_ICD10_group16s))))

        X_train_30sec_ecg = train_features_30sec_ecg
        X_train_30sec_ppg = train_features_30sec_ppg
        X_train_30sec_ecg_ppg = train_features_30sec_ecg_ppg
        X_train_30sec_ecg_ppg_mean = (X_train_30sec_ecg + X_train_30sec_ppg)/2.0
        y_train = np.array(train_ICD10_group16s)
        #######
        indices = np.arange(X_train_30sec_ecg.shape[0])
        train_inds = rng.choice(indices, size=int(train_prop*len(indices)), replace=True)
        X_train_30sec_ecg = X_train_30sec_ecg[train_inds]
        X_train_30sec_ppg = X_train_30sec_ppg[train_inds]
        X_train_30sec_ecg_ppg = X_train_30sec_ecg_ppg[train_inds]
        X_train_30sec_ecg_ppg_mean = X_train_30sec_ecg_ppg_mean[train_inds]
        y_train = y_train[train_inds]
        print(len(y_train))
        print(X_train_30sec_ecg.shape)
        print(X_train_30sec_ppg.shape)
        print(X_train_30sec_ecg_ppg.shape)
        print(X_train_30sec_ecg_ppg_mean.shape)

        X_val_30sec_ecg = val_features_30sec_ecg
        X_val_30sec_ppg = val_features_30sec_ppg
        X_val_30sec_ecg_ppg = val_features_30sec_ecg_ppg
        X_val_30sec_ecg_ppg_mean = (X_val_30sec_ecg + X_val_30sec_ppg)/2.0
        y_val = np.array(val_ICD10_group16s)

        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best_c_ecg, penalty="l2", solver="lbfgs", max_iter=2000))
        pipe.fit(X_train_30sec_ecg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg)
        y_val_proba = pipe.predict_proba(X_val_30sec_ecg)[:, 1]
        rocauc = roc_auc_score(y_val, y_val_proba)
        auprc = average_precision_score(y_val, y_val_proba)
        all_tasks_auroc_values_ecg.append(rocauc)
        all_tasks_auprc_values_ecg.append(auprc)
        print("30sec -- C: %f -- ECG-only -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ecg, rocauc, auprc))

        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best_c_ppg, penalty="l2", solver="lbfgs", max_iter=2000))
        pipe.fit(X_train_30sec_ppg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ppg)
        y_val_proba = pipe.predict_proba(X_val_30sec_ppg)[:, 1]
        rocauc = roc_auc_score(y_val, y_val_proba)
        auprc = average_precision_score(y_val, y_val_proba)
        all_tasks_auroc_values_ppg.append(rocauc)
        all_tasks_auprc_values_ppg.append(auprc)
        print("30sec -- C: %f -- PPG-only -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ppg, rocauc, auprc))

        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best_c_ecg_ppg, penalty="l2", solver="lbfgs", max_iter=2000))
        pipe.fit(X_train_30sec_ecg_ppg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg_ppg)
        y_val_proba = pipe.predict_proba(X_val_30sec_ecg_ppg)[:, 1]
        rocauc = roc_auc_score(y_val, y_val_proba)
        auprc = average_precision_score(y_val, y_val_proba)
        all_tasks_auroc_values_ecg_ppg.append(rocauc)
        all_tasks_auprc_values_ecg_ppg.append(auprc)
        print("30sec -- C: %f -- ECG + PPG -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ecg_ppg, rocauc, auprc))

        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best_c_ecg_ppg_mean, penalty="l2", solver="lbfgs", max_iter=2000))
        pipe.fit(X_train_30sec_ecg_ppg_mean, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg_ppg_mean)
        y_val_proba = pipe.predict_proba(X_val_30sec_ecg_ppg_mean)[:, 1]
        rocauc = roc_auc_score(y_val, y_val_proba)
        auprc = average_precision_score(y_val, y_val_proba)
        all_tasks_auroc_values_ecg_ppg_mean.append(rocauc)
        all_tasks_auprc_values_ecg_ppg_mean.append(auprc)
        print("30sec -- C: %f -- ECG-only + PPG-only (mean of features) -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ecg_ppg_mean, rocauc, auprc))




        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("20. {I26, I80, I82} classification -- Prior Venous Thromboembolism Detection (group20):")
        filtered = pmh_df[pmh_df["Code"].astype(str).str.startswith(("I26", "I80", "I82"))]
        mrns_with_code = filtered["MRN"].unique()
        csns_with_code = visits_df[visits_df["MRN"].isin(mrns_with_code)]["CSN"].unique()

        train_ICD10_group20s = []
        train_features_list_30sec_ecg = []
        train_features_list_30sec_ppg = []
        train_features_list_30sec_ecg_ppg = []
        for i, csn in enumerate(preprocessed_csns):
            if csn in train_csns:
                if csn in csns_with_code:
                    train_ICD10_group20s.append(1)
                else:
                    train_ICD10_group20s.append(0)
                train_features_list_30sec_ecg.append(features_30sec_ecg[i])
                train_features_list_30sec_ppg.append(features_30sec_ppg[i])
                train_features_list_30sec_ecg_ppg.append(features_30sec_ecg_ppg[i])
        train_features_30sec_ecg = np.stack(train_features_list_30sec_ecg, axis=0)
        train_features_30sec_ppg = np.stack(train_features_list_30sec_ppg, axis=0)
        train_features_30sec_ecg_ppg = np.stack(train_features_list_30sec_ecg_ppg, axis=0)
        print(len(train_ICD10_group20s))
        print(train_features_30sec_ecg.shape)
        print(train_features_30sec_ppg.shape)
        print(train_features_30sec_ecg_ppg.shape)
        print("proportion of positive class in train: %.4f" % (train_ICD10_group20s.count(1)/float(len(train_ICD10_group20s))))

        val_ICD10_group20s = []
        val_features_list_30sec_ecg = []
        val_features_list_30sec_ppg = []
        val_features_list_30sec_ecg_ppg = []
        for i, csn in enumerate(preprocessed_csns):
            if csn in val_csns:
                if csn in csns_with_code:
                    val_ICD10_group20s.append(1)
                else:
                    val_ICD10_group20s.append(0)
                val_features_list_30sec_ecg.append(features_30sec_ecg[i])
                val_features_list_30sec_ppg.append(features_30sec_ppg[i])
                val_features_list_30sec_ecg_ppg.append(features_30sec_ecg_ppg[i])
        val_features_30sec_ecg = np.stack(val_features_list_30sec_ecg, axis=0)
        val_features_30sec_ppg = np.stack(val_features_list_30sec_ppg, axis=0)
        val_features_30sec_ecg_ppg = np.stack(val_features_list_30sec_ecg_ppg, axis=0)
        print(len(val_ICD10_group20s))
        print(val_features_30sec_ecg.shape)
        print(val_features_30sec_ppg.shape)
        print(val_features_30sec_ecg_ppg.shape)
        print("proportion of positive class in val: %.4f" % (val_ICD10_group20s.count(1)/float(len(val_ICD10_group20s))))

        X_train_30sec_ecg = train_features_30sec_ecg
        X_train_30sec_ppg = train_features_30sec_ppg
        X_train_30sec_ecg_ppg = train_features_30sec_ecg_ppg
        X_train_30sec_ecg_ppg_mean = (X_train_30sec_ecg + X_train_30sec_ppg)/2.0
        y_train = np.array(train_ICD10_group20s)
        #######
        indices = np.arange(X_train_30sec_ecg.shape[0])
        train_inds = rng.choice(indices, size=int(train_prop*len(indices)), replace=True)
        X_train_30sec_ecg = X_train_30sec_ecg[train_inds]
        X_train_30sec_ppg = X_train_30sec_ppg[train_inds]
        X_train_30sec_ecg_ppg = X_train_30sec_ecg_ppg[train_inds]
        X_train_30sec_ecg_ppg_mean = X_train_30sec_ecg_ppg_mean[train_inds]
        y_train = y_train[train_inds]
        print(len(y_train))
        print(X_train_30sec_ecg.shape)
        print(X_train_30sec_ppg.shape)
        print(X_train_30sec_ecg_ppg.shape)
        print(X_train_30sec_ecg_ppg_mean.shape)

        X_val_30sec_ecg = val_features_30sec_ecg
        X_val_30sec_ppg = val_features_30sec_ppg
        X_val_30sec_ecg_ppg = val_features_30sec_ecg_ppg
        X_val_30sec_ecg_ppg_mean = (X_val_30sec_ecg + X_val_30sec_ppg)/2.0
        y_val = np.array(val_ICD10_group20s)

        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best_c_ecg, penalty="l2", solver="lbfgs", max_iter=2000))
        pipe.fit(X_train_30sec_ecg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg)
        y_val_proba = pipe.predict_proba(X_val_30sec_ecg)[:, 1]
        rocauc = roc_auc_score(y_val, y_val_proba)
        auprc = average_precision_score(y_val, y_val_proba)
        all_tasks_auroc_values_ecg.append(rocauc)
        all_tasks_auprc_values_ecg.append(auprc)
        print("30sec -- C: %f -- ECG-only -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ecg, rocauc, auprc))

        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best_c_ppg, penalty="l2", solver="lbfgs", max_iter=2000))
        pipe.fit(X_train_30sec_ppg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ppg)
        y_val_proba = pipe.predict_proba(X_val_30sec_ppg)[:, 1]
        rocauc = roc_auc_score(y_val, y_val_proba)
        auprc = average_precision_score(y_val, y_val_proba)
        all_tasks_auroc_values_ppg.append(rocauc)
        all_tasks_auprc_values_ppg.append(auprc)
        print("30sec -- C: %f -- PPG-only -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ppg, rocauc, auprc))

        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best_c_ecg_ppg, penalty="l2", solver="lbfgs", max_iter=2000))
        pipe.fit(X_train_30sec_ecg_ppg, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg_ppg)
        y_val_proba = pipe.predict_proba(X_val_30sec_ecg_ppg)[:, 1]
        rocauc = roc_auc_score(y_val, y_val_proba)
        auprc = average_precision_score(y_val, y_val_proba)
        all_tasks_auroc_values_ecg_ppg.append(rocauc)
        all_tasks_auprc_values_ecg_ppg.append(auprc)
        print("30sec -- C: %f -- ECG + PPG -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ecg_ppg, rocauc, auprc))

        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best_c_ecg_ppg_mean, penalty="l2", solver="lbfgs", max_iter=2000))
        pipe.fit(X_train_30sec_ecg_ppg_mean, y_train)
        y_val_pred = pipe.predict(X_val_30sec_ecg_ppg_mean)
        y_val_proba = pipe.predict_proba(X_val_30sec_ecg_ppg_mean)[:, 1]
        rocauc = roc_auc_score(y_val, y_val_proba)
        auprc = average_precision_score(y_val, y_val_proba)
        all_tasks_auroc_values_ecg_ppg_mean.append(rocauc)
        all_tasks_auprc_values_ecg_ppg_mean.append(auprc)
        print("30sec -- C: %f -- ECG-only + PPG-only (mean of features) -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ecg_ppg_mean, rocauc, auprc))








        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print(len(all_tasks_pearson_values_ecg))
        print(len(all_tasks_pearson_values_ppg))
        print(len(all_tasks_pearson_values_ecg_ppg))
        print(len(all_tasks_pearson_values_ecg_ppg_mean))
        print(len(all_tasks_mae_values_ecg))
        print(len(all_tasks_mae_values_ppg))
        print(len(all_tasks_mae_values_ecg_ppg))
        print(len(all_tasks_mae_values_ecg_ppg_mean))
        #
        print("30sec -- alpha: %f -- Mean over all 9 regression tasks -- ECG-only -- Pearson: %.3f, MAE: %.3f" % (best_alpha_ecg, np.mean(all_tasks_pearson_values_ecg), np.mean(all_tasks_mae_values_ecg)))
        print("30sec -- alpha: %f -- Mean over all 9 regression tasks -- PPG-only -- Pearson: %.3f, MAE: %.3f" % (best_alpha_ppg, np.mean(all_tasks_pearson_values_ppg), np.mean(all_tasks_mae_values_ppg)))
        print("30sec -- alpha: %f -- Mean over all 9 regression tasks -- ECG + PPG -- Pearson: %.3f, MAE: %.3f" % (best_alpha_ecg_ppg, np.mean(all_tasks_pearson_values_ecg_ppg), np.mean(all_tasks_mae_values_ecg_ppg)))
        print("30sec -- alpha: %f -- Mean over all 9 regression tasks -- ECG-only + PPG-only (mean of features) -- Pearson: %.3f, MAE: %.3f" % (best_alpha_ecg_ppg_mean, np.mean(all_tasks_pearson_values_ecg_ppg_mean), np.mean(all_tasks_mae_values_ecg_ppg_mean)))

        labs_tasks_pearson_values_ecg = all_tasks_pearson_values_ecg[1:]
        labs_tasks_pearson_values_ppg = all_tasks_pearson_values_ppg[1:]
        labs_tasks_pearson_values_ecg_ppg = all_tasks_pearson_values_ecg_ppg[1:]
        labs_tasks_pearson_values_ecg_ppg_mean = all_tasks_pearson_values_ecg_ppg_mean[1:]
        labs_tasks_mae_values_ecg = all_tasks_mae_values_ecg[1:]
        labs_tasks_mae_values_ppg = all_tasks_mae_values_ppg[1:]
        labs_tasks_mae_values_ecg_ppg = all_tasks_mae_values_ecg_ppg[1:]
        labs_tasks_mae_values_ecg_ppg_mean = all_tasks_mae_values_ecg_ppg_mean[1:]
        print(len(labs_tasks_pearson_values_ecg))
        print(len(labs_tasks_pearson_values_ppg))
        print(len(labs_tasks_pearson_values_ecg_ppg))
        print(len(labs_tasks_pearson_values_ecg_ppg_mean))
        print(len(labs_tasks_mae_values_ecg))
        print(len(labs_tasks_mae_values_ppg))
        print(len(labs_tasks_mae_values_ecg_ppg))
        print(len(labs_tasks_mae_values_ecg_ppg_mean))
        #
        print("30sec -- alpha: %f -- Mean over the 8 labs regression tasks -- ECG-only -- Pearson: %.3f, MAE: %.3f" % (best_alpha_ecg, np.mean(labs_tasks_pearson_values_ecg), np.mean(labs_tasks_mae_values_ecg)))
        print("30sec -- alpha: %f -- Mean over the 8 labs regression tasks -- PPG-only -- Pearson: %.3f, MAE: %.3f" % (best_alpha_ppg, np.mean(labs_tasks_pearson_values_ppg), np.mean(labs_tasks_mae_values_ppg)))
        print("30sec -- alpha: %f -- Mean over the 8 labs regression tasks -- ECG + PPG -- Pearson: %.3f, MAE: %.3f" % (best_alpha_ecg_ppg, np.mean(labs_tasks_pearson_values_ecg_ppg), np.mean(labs_tasks_mae_values_ecg_ppg)))
        print("30sec -- alpha: %f -- Mean over the 8 labs regression tasks -- ECG-only + PPG-only (mean of features) -- Pearson: %.3f, MAE: %.3f" % (best_alpha_ecg_ppg_mean, np.mean(labs_tasks_pearson_values_ecg_ppg_mean), np.mean(labs_tasks_mae_values_ecg_ppg_mean)))

        print(len(all_tasks_auroc_values_ecg))
        print(len(all_tasks_auroc_values_ppg))
        print(len(all_tasks_auroc_values_ecg_ppg))
        print(len(all_tasks_auroc_values_ecg_ppg_mean))
        print(len(all_tasks_auprc_values_ecg))
        print(len(all_tasks_auprc_values_ppg))
        print(len(all_tasks_auprc_values_ecg_ppg))
        print(len(all_tasks_auprc_values_ecg_ppg_mean))
        #
        print("30sec -- C: %f -- Mean over all 11 classification tasks -- ECG-only -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ecg, np.mean(all_tasks_auroc_values_ecg), np.mean(all_tasks_auprc_values_ecg)))
        print("30sec -- C: %f -- Mean over all 11 classification tasks -- PPG-only -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ppg, np.mean(all_tasks_auroc_values_ppg), np.mean(all_tasks_auprc_values_ppg)))
        print("30sec -- C: %f -- Mean over all 11 classification tasks -- ECG + PPG -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ecg_ppg, np.mean(all_tasks_auroc_values_ecg_ppg), np.mean(all_tasks_auprc_values_ecg_ppg)))
        print("30sec -- C: %f -- Mean over all 11 classification tasks -- ECG-only + PPG-only (mean of features) -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ecg_ppg_mean, np.mean(all_tasks_auroc_values_ecg_ppg_mean), np.mean(all_tasks_auprc_values_ecg_ppg_mean)))

        ICD10_tasks_auroc_values_ecg = all_tasks_auroc_values_ecg[2:]
        ICD10_tasks_auroc_values_ppg = all_tasks_auroc_values_ppg[2:]
        ICD10_tasks_auroc_values_ecg_ppg = all_tasks_auroc_values_ecg_ppg[2:]
        ICD10_tasks_auroc_values_ecg_ppg_mean = all_tasks_auroc_values_ecg_ppg_mean[2:]
        ICD10_tasks_auprc_values_ecg = all_tasks_auprc_values_ecg[2:]
        ICD10_tasks_auprc_values_ppg = all_tasks_auprc_values_ppg[2:]
        ICD10_tasks_auprc_values_ecg_ppg = all_tasks_auprc_values_ecg_ppg[2:]
        ICD10_tasks_auprc_values_ecg_ppg_mean = all_tasks_auprc_values_ecg_ppg_mean[2:]
        print(len(ICD10_tasks_auroc_values_ecg))
        print(len(ICD10_tasks_auroc_values_ppg))
        print(len(ICD10_tasks_auroc_values_ecg_ppg))
        print(len(ICD10_tasks_auroc_values_ecg_ppg_mean))
        print(len(ICD10_tasks_auprc_values_ecg))
        print(len(ICD10_tasks_auprc_values_ppg))
        print(len(ICD10_tasks_auprc_values_ecg_ppg))
        print(len(ICD10_tasks_auprc_values_ecg_ppg_mean))
        #
        print("30sec -- C: %f -- Mean over the 9 ICD-10 code classification tasks -- ECG-only -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ecg, np.mean(ICD10_tasks_auroc_values_ecg), np.mean(ICD10_tasks_auprc_values_ecg)))
        print("30sec -- C: %f -- Mean over the 9 ICD-10 code classification tasks -- PPG-only -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ppg, np.mean(ICD10_tasks_auroc_values_ppg), np.mean(ICD10_tasks_auprc_values_ppg)))
        print("30sec -- C: %f -- Mean over the 9 ICD-10 code classification tasks -- ECG + PPG -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ecg_ppg, np.mean(ICD10_tasks_auroc_values_ecg_ppg), np.mean(ICD10_tasks_auprc_values_ecg_ppg)))
        print("30sec -- C: %f -- Mean over the 9 ICD-10 code classification tasks -- ECG-only + PPG-only (mean of features) -- AUROC: %.3f, AUPRC: %.3f" % (best_c_ecg_ppg_mean, np.mean(ICD10_tasks_auroc_values_ecg_ppg_mean), np.mean(ICD10_tasks_auprc_values_ecg_ppg_mean)))

        results_data[run_i] = {}
        results_data[run_i]["all_tasks_pearson_values_ecg"] = all_tasks_pearson_values_ecg
        results_data[run_i]["all_tasks_mae_values_ecg"] = all_tasks_mae_values_ecg
        results_data[run_i]["all_tasks_auroc_values_ecg"] = all_tasks_auroc_values_ecg
        results_data[run_i]["all_tasks_auprc_values_ecg"] = all_tasks_auprc_values_ecg
        #
        results_data[run_i]["all_tasks_pearson_values_ppg"] = all_tasks_pearson_values_ppg
        results_data[run_i]["all_tasks_mae_values_ppg"] = all_tasks_mae_values_ppg
        results_data[run_i]["all_tasks_auroc_values_ppg"] = all_tasks_auroc_values_ppg
        results_data[run_i]["all_tasks_auprc_values_ppg"] = all_tasks_auprc_values_ppg
        #
        results_data[run_i]["all_tasks_pearson_values_ecg_ppg"] = all_tasks_pearson_values_ecg_ppg
        results_data[run_i]["all_tasks_mae_values_ecg_ppg"] = all_tasks_mae_values_ecg_ppg
        results_data[run_i]["all_tasks_auroc_values_ecg_ppg"] = all_tasks_auroc_values_ecg_ppg
        results_data[run_i]["all_tasks_auprc_values_ecg_ppg"] = all_tasks_auprc_values_ecg_ppg
        #
        results_data[run_i]["all_tasks_pearson_values_ecg_ppg_mean"] = all_tasks_pearson_values_ecg_ppg_mean
        results_data[run_i]["all_tasks_mae_values_ecg_ppg_mean"] = all_tasks_mae_values_ecg_ppg_mean
        results_data[run_i]["all_tasks_auroc_values_ecg_ppg_mean"] = all_tasks_auroc_values_ecg_ppg_mean
        results_data[run_i]["all_tasks_auprc_values_ecg_ppg_mean"] = all_tasks_auprc_values_ecg_ppg_mean

        print("end of run!")
        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}")




    print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
    print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
    print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
    print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
    print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
    print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
    print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
    print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
    print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
    print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
    print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
    print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
    print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
    print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
    print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
    print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")

    age_pearson_values_ecg = []
    age_pearson_values_ppg = []
    age_pearson_values_ecg_ppg = []
    age_pearson_values_ecg_ppg_mean = []
    #
    age_mae_values_ecg = []
    age_mae_values_ppg = []
    age_mae_values_ecg_ppg = []
    age_mae_values_ecg_ppg_mean = []

    sex_auroc_values_ecg = []
    sex_auroc_values_ppg = []
    sex_auroc_values_ecg_ppg = []
    sex_auroc_values_ecg_ppg_mean = []
    #
    sex_auprc_values_ecg = []
    sex_auprc_values_ppg = []
    sex_auprc_values_ecg_ppg = []
    sex_auprc_values_ecg_ppg_mean = []

    dispo_auroc_values_ecg = []
    dispo_auroc_values_ppg = []
    dispo_auroc_values_ecg_ppg = []
    dispo_auroc_values_ecg_ppg_mean = []
    #
    dispo_auprc_values_ecg = []
    dispo_auprc_values_ppg = []
    dispo_auprc_values_ecg_ppg = []
    dispo_auprc_values_ecg_ppg_mean = []

    all11clstasks_auroc_values_ecg = []
    all11clstasks_auroc_values_ppg = []
    all11clstasks_auroc_values_ecg_ppg = []
    all11clstasks_auroc_values_ecg_ppg_mean = []
    #
    all11clstasks_auprc_values_ecg = []
    all11clstasks_auprc_values_ppg = []
    all11clstasks_auprc_values_ecg_ppg = []
    all11clstasks_auprc_values_ecg_ppg_mean = []

    icd10codetasks_auroc_values_ecg = []
    icd10codetasks_auroc_values_ppg = []
    icd10codetasks_auroc_values_ecg_ppg = []
    icd10codetasks_auroc_values_ecg_ppg_mean = []
    #
    icd10codetasks_auprc_values_ecg = []
    icd10codetasks_auprc_values_ppg = []
    icd10codetasks_auprc_values_ecg_ppg = []
    icd10codetasks_auprc_values_ecg_ppg_mean = []

    all9regtasks_pearson_values_ecg = []
    all9regtasks_pearson_values_ppg = []
    all9regtasks_pearson_values_ecg_ppg = []
    all9regtasks_pearson_values_ecg_ppg_mean = []
    #
    all9regtasks_mae_values_ecg = []
    all9regtasks_mae_values_ppg = []
    all9regtasks_mae_values_ecg_ppg = []
    all9regtasks_mae_values_ecg_ppg_mean = []

    labsregtasks_pearson_values_ecg = []
    labsregtasks_pearson_values_ppg = []
    labsregtasks_pearson_values_ecg_ppg = []
    labsregtasks_pearson_values_ecg_ppg_mean = []
    #
    labsregtasks_mae_values_ecg = []
    labsregtasks_mae_values_ppg = []
    labsregtasks_mae_values_ecg_ppg = []
    labsregtasks_mae_values_ecg_ppg_mean = []
    for run_i in range(number_of_runs):
        age_pearson_values_ecg.append(results_data[run_i]["all_tasks_pearson_values_ecg"][0])
        age_pearson_values_ppg.append(results_data[run_i]["all_tasks_pearson_values_ppg"][0])
        age_pearson_values_ecg_ppg.append(results_data[run_i]["all_tasks_pearson_values_ecg_ppg"][0])
        age_pearson_values_ecg_ppg_mean.append(results_data[run_i]["all_tasks_pearson_values_ecg_ppg_mean"][0])
        #
        age_mae_values_ecg.append(results_data[run_i]["all_tasks_mae_values_ecg"][0])
        age_mae_values_ppg.append(results_data[run_i]["all_tasks_mae_values_ppg"][0])
        age_mae_values_ecg_ppg.append(results_data[run_i]["all_tasks_mae_values_ecg_ppg"][0])
        age_mae_values_ecg_ppg_mean.append(results_data[run_i]["all_tasks_mae_values_ecg_ppg_mean"][0])

        sex_auroc_values_ecg.append(results_data[run_i]["all_tasks_auroc_values_ecg"][0])
        sex_auroc_values_ppg.append(results_data[run_i]["all_tasks_auroc_values_ppg"][0])
        sex_auroc_values_ecg_ppg.append(results_data[run_i]["all_tasks_auroc_values_ecg_ppg"][0])
        sex_auroc_values_ecg_ppg_mean.append(results_data[run_i]["all_tasks_auroc_values_ecg_ppg_mean"][0])
        #
        sex_auprc_values_ecg.append(results_data[run_i]["all_tasks_auprc_values_ecg"][0])
        sex_auprc_values_ppg.append(results_data[run_i]["all_tasks_auprc_values_ppg"][0])
        sex_auprc_values_ecg_ppg.append(results_data[run_i]["all_tasks_auprc_values_ecg_ppg"][0])
        sex_auprc_values_ecg_ppg_mean.append(results_data[run_i]["all_tasks_auprc_values_ecg_ppg_mean"][0])

        dispo_auroc_values_ecg.append(results_data[run_i]["all_tasks_auroc_values_ecg"][1])
        dispo_auroc_values_ppg.append(results_data[run_i]["all_tasks_auroc_values_ppg"][1])
        dispo_auroc_values_ecg_ppg.append(results_data[run_i]["all_tasks_auroc_values_ecg_ppg"][1])
        dispo_auroc_values_ecg_ppg_mean.append(results_data[run_i]["all_tasks_auroc_values_ecg_ppg_mean"][1])
        #
        dispo_auprc_values_ecg.append(results_data[run_i]["all_tasks_auprc_values_ecg"][1])
        dispo_auprc_values_ppg.append(results_data[run_i]["all_tasks_auprc_values_ppg"][1])
        dispo_auprc_values_ecg_ppg.append(results_data[run_i]["all_tasks_auprc_values_ecg_ppg"][1])
        dispo_auprc_values_ecg_ppg_mean.append(results_data[run_i]["all_tasks_auprc_values_ecg_ppg_mean"][1])

        all11clstasks_auroc_values_ecg.append(np.mean(results_data[run_i]["all_tasks_auroc_values_ecg"]))
        all11clstasks_auroc_values_ppg.append(np.mean(results_data[run_i]["all_tasks_auroc_values_ppg"]))
        all11clstasks_auroc_values_ecg_ppg.append(np.mean(results_data[run_i]["all_tasks_auroc_values_ecg_ppg"]))
        all11clstasks_auroc_values_ecg_ppg_mean.append(np.mean(results_data[run_i]["all_tasks_auroc_values_ecg_ppg_mean"]))
        #
        all11clstasks_auprc_values_ecg.append(np.mean(results_data[run_i]["all_tasks_auprc_values_ecg"]))
        all11clstasks_auprc_values_ppg.append(np.mean(results_data[run_i]["all_tasks_auprc_values_ppg"]))
        all11clstasks_auprc_values_ecg_ppg.append(np.mean(results_data[run_i]["all_tasks_auprc_values_ecg_ppg"]))
        all11clstasks_auprc_values_ecg_ppg_mean.append(np.mean(results_data[run_i]["all_tasks_auprc_values_ecg_ppg_mean"]))

        icd10codetasks_auroc_values_ecg.append(np.mean(results_data[run_i]["all_tasks_auroc_values_ecg"][2:]))
        icd10codetasks_auroc_values_ppg.append(np.mean(results_data[run_i]["all_tasks_auroc_values_ppg"][2:]))
        icd10codetasks_auroc_values_ecg_ppg.append(np.mean(results_data[run_i]["all_tasks_auroc_values_ecg_ppg"][2:]))
        icd10codetasks_auroc_values_ecg_ppg_mean.append(np.mean(results_data[run_i]["all_tasks_auroc_values_ecg_ppg_mean"][2:]))
        #
        icd10codetasks_auprc_values_ecg.append(np.mean(results_data[run_i]["all_tasks_auprc_values_ecg"][2:]))
        icd10codetasks_auprc_values_ppg.append(np.mean(results_data[run_i]["all_tasks_auprc_values_ppg"][2:]))
        icd10codetasks_auprc_values_ecg_ppg.append(np.mean(results_data[run_i]["all_tasks_auprc_values_ecg_ppg"][2:]))
        icd10codetasks_auprc_values_ecg_ppg_mean.append(np.mean(results_data[run_i]["all_tasks_auprc_values_ecg_ppg_mean"][2:]))

        all9regtasks_pearson_values_ecg.append(np.mean(results_data[run_i]["all_tasks_pearson_values_ecg"]))
        all9regtasks_pearson_values_ppg.append(np.mean(results_data[run_i]["all_tasks_pearson_values_ppg"]))
        all9regtasks_pearson_values_ecg_ppg.append(np.mean(results_data[run_i]["all_tasks_pearson_values_ecg_ppg"]))
        all9regtasks_pearson_values_ecg_ppg_mean.append(np.mean(results_data[run_i]["all_tasks_pearson_values_ecg_ppg_mean"]))
        #
        all9regtasks_mae_values_ecg.append(np.mean(results_data[run_i]["all_tasks_mae_values_ecg"]))
        all9regtasks_mae_values_ppg.append(np.mean(results_data[run_i]["all_tasks_mae_values_ppg"]))
        all9regtasks_mae_values_ecg_ppg.append(np.mean(results_data[run_i]["all_tasks_mae_values_ecg_ppg"]))
        all9regtasks_mae_values_ecg_ppg_mean.append(np.mean(results_data[run_i]["all_tasks_mae_values_ecg_ppg_mean"]))

        labsregtasks_pearson_values_ecg.append(np.mean(results_data[run_i]["all_tasks_pearson_values_ecg"][1:]))
        labsregtasks_pearson_values_ppg.append(np.mean(results_data[run_i]["all_tasks_pearson_values_ppg"][1:]))
        labsregtasks_pearson_values_ecg_ppg.append(np.mean(results_data[run_i]["all_tasks_pearson_values_ecg_ppg"][1:]))
        labsregtasks_pearson_values_ecg_ppg_mean.append(np.mean(results_data[run_i]["all_tasks_pearson_values_ecg_ppg_mean"][1:]))
        #
        labsregtasks_mae_values_ecg.append(np.mean(results_data[run_i]["all_tasks_mae_values_ecg"][1:]))
        labsregtasks_mae_values_ppg.append(np.mean(results_data[run_i]["all_tasks_mae_values_ppg"][1:]))
        labsregtasks_mae_values_ecg_ppg.append(np.mean(results_data[run_i]["all_tasks_mae_values_ecg_ppg"][1:]))
        labsregtasks_mae_values_ecg_ppg_mean.append(np.mean(results_data[run_i]["all_tasks_mae_values_ecg_ppg_mean"][1:]))

    print("train_prop: %.2f" % train_prop)

    print("########")
    print("30sec -- 1. Age regression -- ECG-only -- Pearson: %.3f +/- %.4f -- MAE: %.3f +/- %.4f" % (np.mean(age_pearson_values_ecg), np.std(age_pearson_values_ecg), np.mean(age_mae_values_ecg), np.std(age_mae_values_ecg)))
    print("30sec -- 1. Age regression -- PPG-only -- Pearson: %.3f +/- %.4f -- MAE: %.3f +/- %.4f" % (np.mean(age_pearson_values_ppg), np.std(age_pearson_values_ppg), np.mean(age_mae_values_ppg), np.std(age_mae_values_ppg)))
    print("30sec -- 1. Age regression -- ECG + PPG -- Pearson: %.3f +/- %.4f -- MAE: %.3f +/- %.4f" % (np.mean(age_pearson_values_ecg_ppg), np.std(age_pearson_values_ecg_ppg), np.mean(age_mae_values_ecg_ppg), np.std(age_mae_values_ecg_ppg)))
    print("30sec -- 1. Age regression -- ECG-only + PPG-only (mean of features) -- Pearson: %.3f +/- %.4f -- MAE: %.3f +/- %.4f" % (np.mean(age_pearson_values_ecg_ppg_mean), np.std(age_pearson_values_ecg_ppg_mean), np.mean(age_mae_values_ecg_ppg_mean), np.std(age_mae_values_ecg_ppg_mean)))

    print("########")
    print("30sec -- 2. Sex classification -- ECG-only -- AUROC: %.3f +/- %.4f -- AUPRC: %.3f +/- %.4f" % (np.mean(sex_auroc_values_ecg), np.std(sex_auroc_values_ecg), np.mean(sex_auprc_values_ecg), np.std(sex_auprc_values_ecg)))
    print("30sec -- 2. Sex classification -- PPG-only -- AUROC: %.3f +/- %.4f -- AUPRC: %.3f +/- %.4f" % (np.mean(sex_auroc_values_ppg), np.std(sex_auroc_values_ppg), np.mean(sex_auprc_values_ppg), np.std(sex_auprc_values_ppg)))
    print("30sec -- 2. Sex classification -- ECG + PPG -- AUROC: %.3f +/- %.4f -- AUPRC: %.3f +/- %.4f" % (np.mean(sex_auroc_values_ecg_ppg), np.std(sex_auroc_values_ecg_ppg), np.mean(sex_auprc_values_ecg_ppg), np.std(sex_auprc_values_ecg_ppg)))
    print("30sec -- 2. Sex classification -- ECG-only + PPG-only (mean of features) -- AUROC: %.3f +/- %.4f -- AUPRC: %.3f +/- %.4f" % (np.mean(sex_auroc_values_ecg_ppg_mean), np.std(sex_auroc_values_ecg_ppg_mean), np.mean(sex_auprc_values_ecg_ppg_mean), np.std(sex_auprc_values_ecg_ppg_mean)))

    print("########")
    print("30sec -- 3. ED_dispo classification -- ECG-only -- AUROC: %.3f +/- %.4f -- AUPRC: %.3f +/- %.4f" % (np.mean(dispo_auroc_values_ecg), np.std(dispo_auroc_values_ecg), np.mean(dispo_auprc_values_ecg), np.std(dispo_auprc_values_ecg)))
    print("30sec -- 3. ED_dispo classification -- PPG-only -- AUROC: %.3f +/- %.4f -- AUPRC: %.3f +/- %.4f" % (np.mean(dispo_auroc_values_ppg), np.std(dispo_auroc_values_ppg), np.mean(dispo_auprc_values_ppg), np.std(dispo_auprc_values_ppg)))
    print("30sec -- 3. ED_dispo classification -- ECG + PPG -- AUROC: %.3f +/- %.4f -- AUPRC: %.3f +/- %.4f" % (np.mean(dispo_auroc_values_ecg_ppg), np.std(dispo_auroc_values_ecg_ppg), np.mean(dispo_auprc_values_ecg_ppg), np.std(dispo_auprc_values_ecg_ppg)))
    print("30sec -- 3. ED_dispo classification -- ECG-only + PPG-only (mean of features) -- AUROC: %.3f +/- %.4f -- AUPRC: %.3f +/- %.4f" % (np.mean(dispo_auroc_values_ecg_ppg_mean), np.std(dispo_auroc_values_ecg_ppg_mean), np.mean(dispo_auprc_values_ecg_ppg_mean), np.std(dispo_auprc_values_ecg_ppg_mean)))

    print("########")
    print("30sec -- Mean over all 11 classification tasks -- ECG-only -- AUROC: %.3f +/- %.4f -- AUPRC: %.3f +/- %.4f" % (np.mean(all11clstasks_auroc_values_ecg), np.std(all11clstasks_auroc_values_ecg), np.mean(all11clstasks_auprc_values_ecg), np.std(all11clstasks_auprc_values_ecg)))
    print("30sec -- Mean over all 11 classification tasks -- PPG-only -- AUROC: %.3f +/- %.4f -- AUPRC: %.3f +/- %.4f" % (np.mean(all11clstasks_auroc_values_ppg), np.std(all11clstasks_auroc_values_ppg), np.mean(all11clstasks_auprc_values_ppg), np.std(all11clstasks_auprc_values_ppg)))
    print("30sec -- Mean over all 11 classification tasks -- ECG + PPG -- AUROC: %.3f +/- %.4f -- AUPRC: %.3f +/- %.4f" % (np.mean(all11clstasks_auroc_values_ecg_ppg), np.std(all11clstasks_auroc_values_ecg_ppg), np.mean(all11clstasks_auprc_values_ecg_ppg), np.std(all11clstasks_auprc_values_ecg_ppg)))
    print("30sec -- Mean over all 11 classification tasks -- ECG-only + PPG-only (mean of features) -- AUROC: %.3f +/- %.4f -- AUPRC: %.3f +/- %.4f" % (np.mean(all11clstasks_auroc_values_ecg_ppg_mean), np.std(all11clstasks_auroc_values_ecg_ppg_mean), np.mean(all11clstasks_auprc_values_ecg_ppg_mean), np.std(all11clstasks_auprc_values_ecg_ppg_mean)))

    print("########")
    print("30sec -- Mean over the 9 ICD-10 code classification tasks -- ECG-only -- AUROC: %.3f +/- %.4f -- AUPRC: %.3f +/- %.4f" % (np.mean(icd10codetasks_auroc_values_ecg), np.std(icd10codetasks_auroc_values_ecg), np.mean(icd10codetasks_auprc_values_ecg), np.std(icd10codetasks_auprc_values_ecg)))
    print("30sec -- Mean over the 9 ICD-10 code classification tasks -- PPG-only -- AUROC: %.3f +/- %.4f -- AUPRC: %.3f +/- %.4f" % (np.mean(icd10codetasks_auroc_values_ppg), np.std(icd10codetasks_auroc_values_ppg), np.mean(icd10codetasks_auprc_values_ppg), np.std(icd10codetasks_auprc_values_ppg)))
    print("30sec -- Mean over the 9 ICD-10 code classification tasks -- ECG + PPG -- AUROC: %.3f +/- %.4f -- AUPRC: %.3f +/- %.4f" % (np.mean(icd10codetasks_auroc_values_ecg_ppg), np.std(icd10codetasks_auroc_values_ecg_ppg), np.mean(icd10codetasks_auprc_values_ecg_ppg), np.std(icd10codetasks_auprc_values_ecg_ppg)))
    print("30sec -- Mean over the 9 ICD-10 code classification tasks -- ECG-only + PPG-only (mean of features) -- AUROC: %.3f +/- %.4f -- AUPRC: %.3f +/- %.4f" % (np.mean(icd10codetasks_auroc_values_ecg_ppg_mean), np.std(icd10codetasks_auroc_values_ecg_ppg_mean), np.mean(icd10codetasks_auprc_values_ecg_ppg_mean), np.std(icd10codetasks_auprc_values_ecg_ppg_mean)))

    print("########")
    print("30sec -- Mean over all 9 regression tasks -- ECG-only -- Pearson: %.3f +/- %.4f -- MAE: %.3f +/- %.4f" % (np.mean(all9regtasks_pearson_values_ecg), np.std(all9regtasks_pearson_values_ecg), np.mean(all9regtasks_mae_values_ecg), np.std(all9regtasks_mae_values_ecg)))
    print("30sec -- Mean over all 9 regression tasks -- PPG-only -- Pearson: %.3f +/- %.4f -- MAE: %.3f +/- %.4f" % (np.mean(all9regtasks_pearson_values_ppg), np.std(all9regtasks_pearson_values_ppg), np.mean(all9regtasks_mae_values_ppg), np.std(all9regtasks_mae_values_ppg)))
    print("30sec -- Mean over all 9 regression tasks -- ECG + PPG -- Pearson: %.3f +/- %.4f -- MAE: %.3f +/- %.4f" % (np.mean(all9regtasks_pearson_values_ecg_ppg), np.std(all9regtasks_pearson_values_ecg_ppg), np.mean(all9regtasks_mae_values_ecg_ppg), np.std(all9regtasks_mae_values_ecg_ppg)))
    print("30sec -- Mean over all 9 regression tasks -- ECG-only + PPG-only (mean of features) -- Pearson: %.3f +/- %.4f -- MAE: %.3f +/- %.4f" % (np.mean(all9regtasks_pearson_values_ecg_ppg_mean), np.std(all9regtasks_pearson_values_ecg_ppg_mean), np.mean(all9regtasks_mae_values_ecg_ppg_mean), np.std(all9regtasks_mae_values_ecg_ppg_mean)))

    print("########")
    print("30sec -- Mean over the 8 labs regression tasks -- ECG-only -- Pearson: %.3f +/- %.4f -- MAE: %.3f +/- %.4f" % (np.mean(labsregtasks_pearson_values_ecg), np.std(labsregtasks_pearson_values_ecg), np.mean(labsregtasks_mae_values_ecg), np.std(labsregtasks_mae_values_ecg)))
    print("30sec -- Mean over the 8 labs regression tasks -- PPG-only -- Pearson: %.3f +/- %.4f -- MAE: %.3f +/- %.4f" % (np.mean(labsregtasks_pearson_values_ppg), np.std(labsregtasks_pearson_values_ppg), np.mean(labsregtasks_mae_values_ppg), np.std(labsregtasks_mae_values_ppg)))
    print("30sec -- Mean over the 8 labs regression tasks -- ECG + PPG -- Pearson: %.3f +/- %.4f -- MAE: %.3f +/- %.4f" % (np.mean(labsregtasks_pearson_values_ecg_ppg), np.std(labsregtasks_pearson_values_ecg_ppg), np.mean(labsregtasks_mae_values_ecg_ppg), np.std(labsregtasks_mae_values_ecg_ppg)))
    print("30sec -- Mean over the 8 labs regression tasks -- ECG-only + PPG-only (mean of features) -- Pearson: %.3f +/- %.4f -- MAE: %.3f +/- %.4f" % (np.mean(labsregtasks_pearson_values_ecg_ppg_mean), np.std(labsregtasks_pearson_values_ecg_ppg_mean), np.mean(labsregtasks_mae_values_ecg_ppg_mean), np.std(labsregtasks_mae_values_ecg_ppg_mean)))




    all_results_data[train_prop] = results_data

    print(all_results_data)

    if not debug:
        with open(save_dir_path + "/all_results_data.pkl", "wb") as f:
            pickle.dump(all_results_data, f)
    else:
        with open(save_dir_path + "/debug_all_results_data.pkl", "wb") as f:
            pickle.dump(all_results_data, f)

    print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
    print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
    print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
    print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
    print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
    print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
    print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
    print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
    print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
    print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
    print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
    print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
    print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
    print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
    print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
    print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
    print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
    print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
    print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
    print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
