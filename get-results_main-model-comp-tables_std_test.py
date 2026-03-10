import numpy as np
import pickle


signalmcmed_dir_path = "INSERT-PATH-HERE" ###########################################################


train_props = [1.0, 0.50, 0.25, 0.10]
number_of_runs = 5


model_names = ["moment-base", "chronos-bolt-small", "xecg-10min", "ecgfounder", "d-beta", "papagei", "csfm-base", "ecg-domain-features", "ppg-domain-features-60sec", "ecg-domain-features-concat-ppg-domain-features-60sec"]

model_name_to_print_name = {}
model_name_to_print_name["moment-base"] = "MOMENT-base"
model_name_to_print_name["chronos-bolt-small"] = "Chronos-Bolt-small"
model_name_to_print_name["xecg-10min"] = "xECG-10min"
model_name_to_print_name["ecgfounder"] = "ECGFounder"
model_name_to_print_name["d-beta"] = "D-BETA"
model_name_to_print_name["papagei"] = "PaPaGei"
model_name_to_print_name["csfm-base"] = "CSFM-base"
model_name_to_print_name["ecg-domain-features"] = "ECG Domain Features"
model_name_to_print_name["ppg-domain-features-60sec"] = "PPG Domain Features - 60sec"
model_name_to_print_name["ecg-domain-features-concat-ppg-domain-features-60sec"] = "ECG Domain Features + PPG Domain Features - 60sec (concat)"


print("Model name ---- Perc ---- modality --------------- Age reg  Sex cls  ED_dispo cls  Labs reg  ICD-10 cls")
print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")

for model_name in model_names:
    with open(signalmcmed_dir_path + "/evaluation_test_10min_%s/all_results_data.pkl" % model_name, "rb") as f:
        all_results_data = pickle.load(f)

    if model_name == "ecg-domain-features-concat-ppg-domain-features-60sec":
        sex_auroc_means_ecg_ppg = []
        #
        dispo_auroc_means_ecg_ppg = []
        #
        icd10codetasks_auroc_means_ecg_ppg = []
        #
        #
        age_pearson_means_ecg_ppg = []
        #
        labsregtasks_pearson_means_ecg_ppg = []
        for run_i in range(number_of_runs):
            sex_auroc_values_ecg_ppg = []
            #
            dispo_auroc_values_ecg_ppg = []
            #
            icd10codetasks_auroc_values_ecg_ppg = []
            #
            #
            age_pearson_values_ecg_ppg = []
            #
            labsregtasks_pearson_values_ecg_ppg = []
            for train_prop in train_props:
                sex_auroc_values_ecg_ppg.append(all_results_data[train_prop][run_i]["all_tasks_auroc_values_ecg_ppg"][0])

                dispo_auroc_values_ecg_ppg.append(all_results_data[train_prop][run_i]["all_tasks_auroc_values_ecg_ppg"][1])

                icd10codetasks_auroc_values_ecg_ppg.append(np.mean(all_results_data[train_prop][run_i]["all_tasks_auroc_values_ecg_ppg"][2:]))

                age_pearson_values_ecg_ppg.append(all_results_data[train_prop][run_i]["all_tasks_pearson_values_ecg_ppg"][0])

                labsregtasks_pearson_values_ecg_ppg.append(np.mean(all_results_data[train_prop][run_i]["all_tasks_pearson_values_ecg_ppg"][1:]))

            sex_auroc_means_ecg_ppg.append(np.mean(sex_auroc_values_ecg_ppg))

            dispo_auroc_means_ecg_ppg.append(np.mean(dispo_auroc_values_ecg_ppg))

            icd10codetasks_auroc_means_ecg_ppg.append(np.mean(icd10codetasks_auroc_values_ecg_ppg))

            age_pearson_means_ecg_ppg.append(np.mean(age_pearson_values_ecg_ppg))

            labsregtasks_pearson_means_ecg_ppg.append(np.mean(labsregtasks_pearson_values_ecg_ppg))

        print("%s ---- Mean(100, 50, 25, 10) ---- ECG + PPG --------------   &%.3f $\pm$ %.3f    &%.3f $\pm$ %.3f    &%.3f $\pm$ %.3f    &%.3f $\pm$ %.3f    &%.3f $\pm$ %.3f\\\\" % (model_name_to_print_name[model_name], np.mean(age_pearson_means_ecg_ppg), np.std(age_pearson_means_ecg_ppg), np.mean(sex_auroc_means_ecg_ppg), np.std(sex_auroc_means_ecg_ppg), np.mean(dispo_auroc_means_ecg_ppg), np.std(dispo_auroc_means_ecg_ppg), np.mean(labsregtasks_pearson_means_ecg_ppg), np.std(labsregtasks_pearson_means_ecg_ppg), np.mean(icd10codetasks_auroc_means_ecg_ppg), np.std(icd10codetasks_auroc_means_ecg_ppg)))
        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")


    elif model_name == "ecg-domain-features":
        sex_auroc_means_ecg = []
        #
        dispo_auroc_means_ecg = []
        #
        icd10codetasks_auroc_means_ecg = []
        #
        #
        age_pearson_means_ecg = []
        #
        labsregtasks_pearson_means_ecg = []
        for run_i in range(number_of_runs):
            sex_auroc_values_ecg = []
            #
            dispo_auroc_values_ecg = []
            #
            icd10codetasks_auroc_values_ecg = []
            #
            #
            age_pearson_values_ecg = []
            #
            labsregtasks_pearson_values_ecg = []
            for train_prop in train_props:
                sex_auroc_values_ecg.append(all_results_data[train_prop][run_i]["all_tasks_auroc_values_ecg"][0])

                dispo_auroc_values_ecg.append(all_results_data[train_prop][run_i]["all_tasks_auroc_values_ecg"][1])

                icd10codetasks_auroc_values_ecg.append(np.mean(all_results_data[train_prop][run_i]["all_tasks_auroc_values_ecg"][2:]))

                age_pearson_values_ecg.append(all_results_data[train_prop][run_i]["all_tasks_pearson_values_ecg"][0])

                labsregtasks_pearson_values_ecg.append(np.mean(all_results_data[train_prop][run_i]["all_tasks_pearson_values_ecg"][1:]))

            sex_auroc_means_ecg.append(np.mean(sex_auroc_values_ecg))

            dispo_auroc_means_ecg.append(np.mean(dispo_auroc_values_ecg))

            icd10codetasks_auroc_means_ecg.append(np.mean(icd10codetasks_auroc_values_ecg))

            age_pearson_means_ecg.append(np.mean(age_pearson_values_ecg))

            labsregtasks_pearson_means_ecg.append(np.mean(labsregtasks_pearson_values_ecg))

        print("%s ---- Mean(100, 50, 25, 10) ---- ECG-only --------------    &%.3f $\pm$ %.3f    &%.3f $\pm$ %.3f    &%.3f $\pm$ %.3f    &%.3f $\pm$ %.3f    &%.3f $\pm$ %.3f\\\\" % (model_name_to_print_name[model_name], np.mean(age_pearson_means_ecg), np.std(age_pearson_means_ecg), np.mean(sex_auroc_means_ecg), np.std(sex_auroc_means_ecg), np.mean(dispo_auroc_means_ecg), np.std(dispo_auroc_means_ecg), np.mean(labsregtasks_pearson_means_ecg), np.std(labsregtasks_pearson_means_ecg), np.mean(icd10codetasks_auroc_means_ecg), np.std(icd10codetasks_auroc_means_ecg)))
        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")


    elif model_name == "ppg-domain-features-60sec":
        sex_auroc_means_ppg = []
        #
        dispo_auroc_means_ppg = []
        #
        icd10codetasks_auroc_means_ppg = []
        #
        #
        age_pearson_means_ppg = []
        #
        labsregtasks_pearson_means_ppg = []
        for run_i in range(number_of_runs):
            sex_auroc_values_ppg = []
            #
            dispo_auroc_values_ppg = []
            #
            icd10codetasks_auroc_values_ppg = []
            #
            #
            age_pearson_values_ppg = []
            #
            labsregtasks_pearson_values_ppg = []
            for train_prop in train_props:
                sex_auroc_values_ppg.append(all_results_data[train_prop][run_i]["all_tasks_auroc_values_ppg"][0])

                dispo_auroc_values_ppg.append(all_results_data[train_prop][run_i]["all_tasks_auroc_values_ppg"][1])

                icd10codetasks_auroc_values_ppg.append(np.mean(all_results_data[train_prop][run_i]["all_tasks_auroc_values_ppg"][2:]))

                age_pearson_values_ppg.append(all_results_data[train_prop][run_i]["all_tasks_pearson_values_ppg"][0])

                labsregtasks_pearson_values_ppg.append(np.mean(all_results_data[train_prop][run_i]["all_tasks_pearson_values_ppg"][1:]))

            sex_auroc_means_ppg.append(np.mean(sex_auroc_values_ppg))

            dispo_auroc_means_ppg.append(np.mean(dispo_auroc_values_ppg))

            icd10codetasks_auroc_means_ppg.append(np.mean(icd10codetasks_auroc_values_ppg))

            age_pearson_means_ppg.append(np.mean(age_pearson_values_ppg))

            labsregtasks_pearson_means_ppg.append(np.mean(labsregtasks_pearson_values_ppg))

        print("%s ---- Mean(100, 50, 25, 10) ---- PPG-only --------------    &%.3f $\pm$ %.3f    &%.3f $\pm$ %.3f    &%.3f $\pm$ %.3f    &%.3f $\pm$ %.3f    &%.3f $\pm$ %.3f\\\\" % (model_name_to_print_name[model_name], np.mean(age_pearson_means_ppg), np.std(age_pearson_means_ppg), np.mean(sex_auroc_means_ppg), np.std(sex_auroc_means_ppg), np.mean(dispo_auroc_means_ppg), np.std(dispo_auroc_means_ppg), np.mean(labsregtasks_pearson_means_ppg), np.std(labsregtasks_pearson_means_ppg), np.mean(icd10codetasks_auroc_means_ppg), np.std(icd10codetasks_auroc_means_ppg)))
        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")


    else:
        sex_auroc_means_ecg = []
        sex_auroc_means_ppg = []
        sex_auroc_means_ecg_ppg_mean = []
        #
        dispo_auroc_means_ecg = []
        dispo_auroc_means_ppg = []
        dispo_auroc_means_ecg_ppg_mean = []
        #
        icd10codetasks_auroc_means_ecg = []
        icd10codetasks_auroc_means_ppg = []
        icd10codetasks_auroc_means_ecg_ppg_mean = []
        #
        #
        age_pearson_means_ecg = []
        age_pearson_means_ppg = []
        age_pearson_means_ecg_ppg_mean = []
        #
        labsregtasks_pearson_means_ecg = []
        labsregtasks_pearson_means_ppg = []
        labsregtasks_pearson_means_ecg_ppg_mean = []
        for run_i in range(number_of_runs):
            sex_auroc_values_ecg = []
            sex_auroc_values_ppg = []
            sex_auroc_values_ecg_ppg_mean = []
            #
            dispo_auroc_values_ecg = []
            dispo_auroc_values_ppg = []
            dispo_auroc_values_ecg_ppg_mean = []
            #
            icd10codetasks_auroc_values_ecg = []
            icd10codetasks_auroc_values_ppg = []
            icd10codetasks_auroc_values_ecg_ppg_mean = []
            #
            #
            age_pearson_values_ecg = []
            age_pearson_values_ppg = []
            age_pearson_values_ecg_ppg_mean = []
            #
            labsregtasks_pearson_values_ecg = []
            labsregtasks_pearson_values_ppg = []
            labsregtasks_pearson_values_ecg_ppg_mean = []
            for train_prop in train_props:
                sex_auroc_values_ecg.append(all_results_data[train_prop][run_i]["all_tasks_auroc_values_ecg"][0])
                sex_auroc_values_ppg.append(all_results_data[train_prop][run_i]["all_tasks_auroc_values_ppg"][0])
                sex_auroc_values_ecg_ppg_mean.append(all_results_data[train_prop][run_i]["all_tasks_auroc_values_ecg_ppg_mean"][0])

                dispo_auroc_values_ecg.append(all_results_data[train_prop][run_i]["all_tasks_auroc_values_ecg"][1])
                dispo_auroc_values_ppg.append(all_results_data[train_prop][run_i]["all_tasks_auroc_values_ppg"][1])
                dispo_auroc_values_ecg_ppg_mean.append(all_results_data[train_prop][run_i]["all_tasks_auroc_values_ecg_ppg_mean"][1])

                icd10codetasks_auroc_values_ecg.append(np.mean(all_results_data[train_prop][run_i]["all_tasks_auroc_values_ecg"][2:]))
                icd10codetasks_auroc_values_ppg.append(np.mean(all_results_data[train_prop][run_i]["all_tasks_auroc_values_ppg"][2:]))
                icd10codetasks_auroc_values_ecg_ppg_mean.append(np.mean(all_results_data[train_prop][run_i]["all_tasks_auroc_values_ecg_ppg_mean"][2:]))

                age_pearson_values_ecg.append(all_results_data[train_prop][run_i]["all_tasks_pearson_values_ecg"][0])
                age_pearson_values_ppg.append(all_results_data[train_prop][run_i]["all_tasks_pearson_values_ppg"][0])
                age_pearson_values_ecg_ppg_mean.append(all_results_data[train_prop][run_i]["all_tasks_pearson_values_ecg_ppg_mean"][0])

                labsregtasks_pearson_values_ecg.append(np.mean(all_results_data[train_prop][run_i]["all_tasks_pearson_values_ecg"][1:]))
                labsregtasks_pearson_values_ppg.append(np.mean(all_results_data[train_prop][run_i]["all_tasks_pearson_values_ppg"][1:]))
                labsregtasks_pearson_values_ecg_ppg_mean.append(np.mean(all_results_data[train_prop][run_i]["all_tasks_pearson_values_ecg_ppg_mean"][1:]))

            sex_auroc_means_ecg.append(np.mean(sex_auroc_values_ecg))
            sex_auroc_means_ppg.append(np.mean(sex_auroc_values_ppg))
            sex_auroc_means_ecg_ppg_mean.append(np.mean(sex_auroc_values_ecg_ppg_mean))

            dispo_auroc_means_ecg.append(np.mean(dispo_auroc_values_ecg))
            dispo_auroc_means_ppg.append(np.mean(dispo_auroc_values_ppg))
            dispo_auroc_means_ecg_ppg_mean.append(np.mean(dispo_auroc_values_ecg_ppg_mean))

            icd10codetasks_auroc_means_ecg.append(np.mean(icd10codetasks_auroc_values_ecg))
            icd10codetasks_auroc_means_ppg.append(np.mean(icd10codetasks_auroc_values_ppg))
            icd10codetasks_auroc_means_ecg_ppg_mean.append(np.mean(icd10codetasks_auroc_values_ecg_ppg_mean))

            age_pearson_means_ecg.append(np.mean(age_pearson_values_ecg))
            age_pearson_means_ppg.append(np.mean(age_pearson_values_ppg))
            age_pearson_means_ecg_ppg_mean.append(np.mean(age_pearson_values_ecg_ppg_mean))

            labsregtasks_pearson_means_ecg.append(np.mean(labsregtasks_pearson_values_ecg))
            labsregtasks_pearson_means_ppg.append(np.mean(labsregtasks_pearson_values_ppg))
            labsregtasks_pearson_means_ecg_ppg_mean.append(np.mean(labsregtasks_pearson_values_ecg_ppg_mean))

        print("%s ---- Mean(100, 50, 25, 10) ---- ECG-only --------------    &%.3f $\pm$ %.3f    &%.3f $\pm$ %.3f    &%.3f $\pm$ %.3f    &%.3f $\pm$ %.3f    &%.3f $\pm$ %.3f\\\\" % (model_name_to_print_name[model_name], np.mean(age_pearson_means_ecg), np.std(age_pearson_means_ecg), np.mean(sex_auroc_means_ecg), np.std(sex_auroc_means_ecg), np.mean(dispo_auroc_means_ecg), np.std(dispo_auroc_means_ecg), np.mean(labsregtasks_pearson_means_ecg), np.std(labsregtasks_pearson_means_ecg), np.mean(icd10codetasks_auroc_means_ecg), np.std(icd10codetasks_auroc_means_ecg)))
        print("%s ---- Mean(100, 50, 25, 10) ---- PPG-only ---------------   &%.3f $\pm$ %.3f    &%.3f $\pm$ %.3f    &%.3f $\pm$ %.3f    &%.3f $\pm$ %.3f    &%.3f $\pm$ %.3f\\\\" % (model_name_to_print_name[model_name], np.mean(age_pearson_means_ppg), np.std(age_pearson_means_ppg), np.mean(sex_auroc_means_ppg), np.std(sex_auroc_means_ppg), np.mean(dispo_auroc_means_ppg), np.std(dispo_auroc_means_ppg), np.mean(labsregtasks_pearson_means_ppg), np.std(labsregtasks_pearson_means_ppg), np.mean(icd10codetasks_auroc_means_ppg), np.std(icd10codetasks_auroc_means_ppg)))
        print("%s ---- Mean(100, 50, 25, 10) ---- ECG-only + PPG-only ----   &%.3f $\pm$ %.3f    &%.3f $\pm$ %.3f    &%.3f $\pm$ %.3f    &%.3f $\pm$ %.3f    &%.3f $\pm$ %.3f\\\\" % (model_name_to_print_name[model_name], np.mean(age_pearson_means_ecg_ppg_mean), np.std(age_pearson_means_ecg_ppg_mean), np.mean(sex_auroc_means_ecg_ppg_mean), np.std(sex_auroc_means_ecg_ppg_mean), np.mean(dispo_auroc_means_ecg_ppg_mean), np.std(dispo_auroc_means_ecg_ppg_mean), np.mean(labsregtasks_pearson_means_ecg_ppg_mean), np.std(labsregtasks_pearson_means_ecg_ppg_mean), np.mean(icd10codetasks_auroc_means_ecg_ppg_mean), np.std(icd10codetasks_auroc_means_ecg_ppg_mean)))
        print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
