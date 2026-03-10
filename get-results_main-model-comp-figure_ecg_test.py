# camera-ready?
import numpy as np
import pickle


signalmcmed_dir_path = "INSERT-PATH-HERE" ###########################################################


train_props = [1.0, 0.50, 0.25, 0.10]
number_of_runs = 5


model_names = ["moment-base", "chronos-bolt-small", "xecg-10min", "ecgfounder", "d-beta", "papagei", "csfm-base", "ecg-domain-features"]

model_name_to_print_name = {}
model_name_to_print_name["moment-base"] = "MOMENT-base"
model_name_to_print_name["chronos-bolt-small"] = "Chronos-Bolt-small"
model_name_to_print_name["xecg-10min"] = "xECG-10min"
model_name_to_print_name["ecgfounder"] = "ECGFounder"
model_name_to_print_name["d-beta"] = "D-BETA"
model_name_to_print_name["papagei"] = "PaPaGei"
model_name_to_print_name["csfm-base"] = "CSFM-base"
model_name_to_print_name["ecg-domain-features"] = "ECG Domain Features"


model_name_to_all11clstasks_auroc_means_ecg = {}
model_name_to_all11clstasks_auroc_stds_ecg = {}
#
model_name_to_all9regtasks_pearson_means_ecg = {}
model_name_to_all9regtasks_pearson_stds_ecg = {}
#
model_name_to_sex_auroc_means_ecg = {}
model_name_to_sex_auroc_stds_ecg = {}
#
model_name_to_dispo_auroc_means_ecg = {}
model_name_to_dispo_auroc_stds_ecg = {}
#
model_name_to_icd10codetasks_auroc_means_ecg = {}
model_name_to_icd10codetasks_auroc_stds_ecg = {}
#
model_name_to_age_pearson_means_ecg = {}
model_name_to_age_pearson_stds_ecg = {}
#
model_name_to_labsregtasks_pearson_means_ecg = {}
model_name_to_labsregtasks_pearson_stds_ecg = {}
for model_name in model_names:
    with open(signalmcmed_dir_path + "/evaluation_test_10min_%s/all_results_data.pkl" % model_name, "rb") as f:
        all_results_data = pickle.load(f)

    all11clstasks_auroc_means_ecg = []
    all11clstasks_auroc_stds_ecg = []
    #
    all9regtasks_pearson_means_ecg = []
    all9regtasks_pearson_stds_ecg = []
    #
    sex_auroc_means_ecg = []
    sex_auroc_stds_ecg = []
    #
    dispo_auroc_means_ecg = []
    dispo_auroc_stds_ecg = []
    #
    icd10codetasks_auroc_means_ecg = []
    icd10codetasks_auroc_stds_ecg = []
    #
    age_pearson_means_ecg = []
    age_pearson_stds_ecg = []
    #
    labsregtasks_pearson_means_ecg = []
    labsregtasks_pearson_stds_ecg = []
    for train_prop in train_props:
        all11clstasks_auroc_values_ecg = []
        all9regtasks_pearson_values_ecg = []
        sex_auroc_values_ecg = []
        dispo_auroc_values_ecg = []
        icd10codetasks_auroc_values_ecg = []
        age_pearson_values_ecg = []
        labsregtasks_pearson_values_ecg = []
        for run_i in range(number_of_runs):
            all11clstasks_auroc_values_ecg.append(np.mean(all_results_data[train_prop][run_i]["all_tasks_auroc_values_ecg"]))
            all9regtasks_pearson_values_ecg.append(np.mean(all_results_data[train_prop][run_i]["all_tasks_pearson_values_ecg"]))
            sex_auroc_values_ecg.append(all_results_data[train_prop][run_i]["all_tasks_auroc_values_ecg"][0])
            dispo_auroc_values_ecg.append(all_results_data[train_prop][run_i]["all_tasks_auroc_values_ecg"][1])
            icd10codetasks_auroc_values_ecg.append(np.mean(all_results_data[train_prop][run_i]["all_tasks_auroc_values_ecg"][2:]))
            age_pearson_values_ecg.append(all_results_data[train_prop][run_i]["all_tasks_pearson_values_ecg"][0])
            labsregtasks_pearson_values_ecg.append(np.mean(all_results_data[train_prop][run_i]["all_tasks_pearson_values_ecg"][1:]))

        all11clstasks_auroc_means_ecg.append(np.mean(all11clstasks_auroc_values_ecg))
        all11clstasks_auroc_stds_ecg.append(np.std(all11clstasks_auroc_values_ecg))

        all9regtasks_pearson_means_ecg.append(np.mean(all9regtasks_pearson_values_ecg))
        all9regtasks_pearson_stds_ecg.append(np.std(all9regtasks_pearson_values_ecg))

        sex_auroc_means_ecg.append(np.mean(sex_auroc_values_ecg))
        sex_auroc_stds_ecg.append(np.std(sex_auroc_values_ecg))

        dispo_auroc_means_ecg.append(np.mean(dispo_auroc_values_ecg))
        dispo_auroc_stds_ecg.append(np.std(dispo_auroc_values_ecg))

        icd10codetasks_auroc_means_ecg.append(np.mean(icd10codetasks_auroc_values_ecg))
        icd10codetasks_auroc_stds_ecg.append(np.std(icd10codetasks_auroc_values_ecg))

        age_pearson_means_ecg.append(np.mean(age_pearson_values_ecg))
        age_pearson_stds_ecg.append(np.std(age_pearson_values_ecg))

        labsregtasks_pearson_means_ecg.append(np.mean(labsregtasks_pearson_values_ecg))
        labsregtasks_pearson_stds_ecg.append(np.std(labsregtasks_pearson_values_ecg))

    model_name_to_all11clstasks_auroc_means_ecg[model_name] = all11clstasks_auroc_means_ecg
    model_name_to_all11clstasks_auroc_stds_ecg[model_name] = all11clstasks_auroc_stds_ecg

    model_name_to_all9regtasks_pearson_means_ecg[model_name] = all9regtasks_pearson_means_ecg
    model_name_to_all9regtasks_pearson_stds_ecg[model_name] = all9regtasks_pearson_stds_ecg

    model_name_to_sex_auroc_means_ecg[model_name] = sex_auroc_means_ecg
    model_name_to_sex_auroc_stds_ecg[model_name] = sex_auroc_stds_ecg

    model_name_to_dispo_auroc_means_ecg[model_name] = dispo_auroc_means_ecg
    model_name_to_dispo_auroc_stds_ecg[model_name] = dispo_auroc_stds_ecg

    model_name_to_icd10codetasks_auroc_means_ecg[model_name] = icd10codetasks_auroc_means_ecg
    model_name_to_icd10codetasks_auroc_stds_ecg[model_name] = icd10codetasks_auroc_stds_ecg

    model_name_to_age_pearson_means_ecg[model_name] = age_pearson_means_ecg
    model_name_to_age_pearson_stds_ecg[model_name] = age_pearson_stds_ecg

    model_name_to_labsregtasks_pearson_means_ecg[model_name] = labsregtasks_pearson_means_ecg
    model_name_to_labsregtasks_pearson_stds_ecg[model_name] = labsregtasks_pearson_stds_ecg


print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
print("Average Classification Task Performance:")
for model_name in model_names:
    print("%% %s -- ECG-only" % model_name_to_print_name[model_name])
    for train_prop, score in zip(train_props, model_name_to_all11clstasks_auroc_means_ecg[model_name]):
        print("%d %f\\\\" % (train_prop*100, score))
    print("{{{{{{{{}}}}}}}}")

print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
print("Average Regression Task Performance:")
for model_name in model_names:
    print("%% %s -- ECG-only" % model_name_to_print_name[model_name])
    for train_prop, score in zip(train_props, model_name_to_all9regtasks_pearson_means_ecg[model_name]):
        print("%d %f\\\\" % (train_prop*100, score))
    print("{{{{{{{{}}}}}}}}")
