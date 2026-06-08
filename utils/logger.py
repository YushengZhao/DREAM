import os

import torch
import time
import pandas as pd


class MultiExpRecorder(object):
    def __init__(self, runs):
        self.results = {}
        self.runs_count = runs

    def add_result(self, run, result_dict):
        required_keys = ["train", "valid", "test", 'correct_labeled_train_accuracy',
                         'incorrect_labeled_train_accuracy', 'incorrect_labeled_mislead_train_accuracy',
                         'unlabeled_correct_supervised_accuracy', 'unlabeled_unsupervised_accuracy',
                         'unlabeled_incorrect_supervised_accuracy', 'total_time']
        for key in required_keys:
            assert key in result_dict, f"Key '{key}' missing in result_dict for run {run}"
            if result_dict[key] is None:
                print(f"Warning: result_dict['{key}'] is None for run {run}. Replacing with 0.0.")
                result_dict[key] = 0.0

        self.results[run] = [
            result_dict["train"],
            result_dict["valid"],
            result_dict["test"],
            result_dict['correct_labeled_train_accuracy'],
            result_dict['incorrect_labeled_train_accuracy'],
            result_dict['incorrect_labeled_mislead_train_accuracy'],
            result_dict['unlabeled_correct_supervised_accuracy'],
            result_dict['unlabeled_unsupervised_accuracy'],
            result_dict['unlabeled_incorrect_supervised_accuracy'],
            result_dict['total_time']
        ]

    def get_statistics(self):
        results_list = []
        missing_runs = []
        for i in range(self.runs_count):
            if i in self.results:
                if len(self.results[i]) == 10:
                    results_list.append(self.results[i])
                else:
                    print(f"Warning: Run {i} result has incorrect length ({len(self.results[i])}). Skipping.")
                    missing_runs.append(i)
            else:
                print(f"Warning: Result for run {i} is missing. Skipping.")
                missing_runs.append(i)

        if not results_list:
            print("Error: No valid results found to calculate statistics.")
            return {key: {"acc": 0.0, "std": 0.0} for key in [
                "train_accuracy", "valid_accuracy", "test_accuracy",
                "correct_labeled_train_accuracy", "incorrect_labeled_train_accuracy",
                "incorrect_labeled_mislead_train_accuracy", "unlabeled_correct_supervised_accuracy",
                "unlabeled_unsupervised_accuracy", "unlabeled_incorrect_supervised_accuracy"]} \
                | {"total_time": {"mean": 0.0, "std": 0.0}}

        best_result = 100 * torch.tensor(results_list, dtype=torch.float32)  # Ensure float tensor
        total_results = {}

        metrics = [
            "train_accuracy", "valid_accuracy", "test_accuracy",
            "correct_labeled_train_accuracy", "incorrect_labeled_train_accuracy",
            "incorrect_labeled_mislead_train_accuracy", "unlabeled_correct_supervised_accuracy",
            "unlabeled_unsupervised_accuracy", "unlabeled_incorrect_supervised_accuracy"
        ]

        for i, metric_name in enumerate(metrics):
            r = best_result[:, i]
            mean_val = torch.nanmean(r) if torch.isnan(r).any() else r.mean()
            std_val = torch.sqrt(torch.nanmean((r - mean_val) ** 2)) if torch.isnan(r).any() else r.std()
            total_results[metric_name] = {"acc": mean_val.item(), "std": std_val.item()}

        r_time = torch.tensor(results_list, dtype=torch.float32)[:, 9]
        mean_time = torch.nanmean(r_time) if torch.isnan(r_time).any() else r_time.mean()
        std_time = torch.sqrt(torch.nanmean((r_time - mean_time) ** 2)) if torch.isnan(r_time).any() else r_time.std()
        total_results['total_time'] = {"mean": mean_time.item(), "std": std_time.item()}

        if missing_runs:
            print(f"Statistics calculated based on {len(results_list)} successful runs. Missing runs: {missing_runs}")

        return total_results


class ResultLogger(object):
    def __init__(self, method_list, data_list, noise_list, runs):
        log_dir = './log/'
        os.makedirs(log_dir, exist_ok=True)
        self.file_name_base = str(time.strftime("%Y-%m-%d_%H-%M-%S"))
        self.log_path = os.path.join(log_dir, self.file_name_base + '.txt')
        self.tex_path = os.path.join(log_dir, self.file_name_base + '.tex')
        self.excel_path = os.path.join(log_dir, self.file_name_base + '.xlsx')
        self.runs = runs

        row_level_1 = data_list
        row_level_2_tex = [f'$ {int(n_rate * 100):2d} \\% $ {n_type}' for n_rate, n_type in noise_list]
        row_level_2_excel = [f'{int(n_rate * 100):2d} % {n_type}' for n_rate, n_type in noise_list]

        tex_index = pd.MultiIndex.from_product([row_level_1, row_level_2_tex], names=["Dataset", "Noise info"])
        excel_index = pd.MultiIndex.from_product([row_level_1, row_level_2_excel], names=["Dataset", "Noise info"])

        columns = method_list

        self.tex_result_tabel = pd.DataFrame(index=tex_index, columns=columns, dtype=object)
        self.excel_result_tabel_test_acc_main = pd.DataFrame(index=excel_index, columns=columns, dtype=object)
        self.excel_result_tabel_test_acc_ave = pd.DataFrame(index=excel_index, columns=columns, dtype=object)
        self.excel_result_tabel_test_acc_std = pd.DataFrame(index=excel_index, columns=columns, dtype=object)
        self.excel_result_tabel_aclt = pd.DataFrame(index=excel_index, columns=columns, dtype=object)
        self.excel_result_tabel_ailt = pd.DataFrame(index=excel_index, columns=columns, dtype=object)
        self.excel_result_tabel_aucs = pd.DataFrame(index=excel_index, columns=columns, dtype=object)
        self.excel_result_tabel_auis = pd.DataFrame(index=excel_index, columns=columns, dtype=object)
        self.excel_result_tabel_auu = pd.DataFrame(index=excel_index, columns=columns, dtype=object)
        self.excel_result_tabel_ailmt = pd.DataFrame(index=excel_index, columns=columns, dtype=object)
        self.excel_result_tabel_time = pd.DataFrame(index=excel_index, columns=columns, dtype=object)

    def dump_record(self, method_name, data_name, noise_type, noise_rate, total_results):
        test_stats = total_results.get('test_accuracy', {"acc": 0.0, "std": 0.0})
        test_acc_ave, test_acc_std = test_stats['acc'], test_stats['std']

        aclt_stats = total_results.get("correct_labeled_train_accuracy", {"acc": 0.0, "std": 0.0})
        aclt_ave, aclt_std = aclt_stats['acc'], aclt_stats['std']

        ailt_stats = total_results.get("incorrect_labeled_train_accuracy", {"acc": 0.0, "std": 0.0})
        ailt_ave, ailt_std = ailt_stats['acc'], ailt_stats['std']

        aucs_stats = total_results.get('unlabeled_correct_supervised_accuracy', {"acc": 0.0, "std": 0.0})
        aucs_ave, aucs_std = aucs_stats['acc'], aucs_stats['std']

        auu_stats = total_results.get('unlabeled_unsupervised_accuracy', {"acc": 0.0, "std": 0.0})
        auu_ave, auu_std = auu_stats['acc'], auu_stats['std']

        auis_stats = total_results.get('unlabeled_incorrect_supervised_accuracy', {"acc": 0.0, "std": 0.0})
        auis_ave, auis_std = auis_stats['acc'], auis_stats['std']

        ailmt_stats = total_results.get('incorrect_labeled_mislead_train_accuracy', {"acc": 0.0, "std": 0.0})
        ailmt_ave, ailmt_std = ailmt_stats['acc'], ailmt_stats['std']

        time_stats = total_results.get('total_time', {"mean": 0.0, "std": 0.0})
        time_ave, time_std = time_stats['mean'], time_stats['std']

        tex_row_label = f'$ {int(noise_rate * 100):2d} \\% $ {noise_type}'
        excel_row_label = f'{int(noise_rate * 100):2d} % {noise_type}'

        message = (f'| data: {data_name:12s} | method: {method_name:12s}'
                   f' | noise type: {noise_type:12s} | noise rate: {noise_rate:03.2f} '
                   f'| test acc: {test_acc_ave:.2f} ± {test_acc_std:.2f} |')
        print(message)

        try:
            with open(self.log_path, 'a') as f:
                f.write(message + '\n')

            self.tex_result_tabel.loc[(data_name, tex_row_label), method_name] = f'$ {test_acc_ave:.2f} \\pm {test_acc_std:.2f} $'
            tex_output_string = self.tex_result_tabel.to_latex(na_rep='-', escape=False, bold_rows=True, caption=f'RESULTS FOR {self.runs:d} RUNS')
            with open(self.tex_path, 'w') as f:
                f.write(tex_output_string)

            row_idx_excel = (data_name, excel_row_label)

            self.excel_result_tabel_test_acc_main.loc[row_idx_excel, method_name] = f'{test_acc_ave:.2f} ± {test_acc_std:.2f}'
            self.excel_result_tabel_test_acc_ave.loc[row_idx_excel, method_name] = f'{test_acc_ave:.2f}'
            self.excel_result_tabel_test_acc_std.loc[row_idx_excel, method_name] = f'{test_acc_std:.2f}'
            self.excel_result_tabel_aclt.loc[row_idx_excel, method_name] = f'{aclt_ave:.2f} ± {aclt_std:.2f}'
            self.excel_result_tabel_ailt.loc[row_idx_excel, method_name] = f'{ailt_ave:.2f} ± {ailt_std:.2f}'
            self.excel_result_tabel_aucs.loc[row_idx_excel, method_name] = f'{aucs_ave:.2f} ± {aucs_std:.2f}'
            self.excel_result_tabel_auis.loc[row_idx_excel, method_name] = f'{auis_ave:.2f} ± {auis_std:.2f}'
            self.excel_result_tabel_auu.loc[row_idx_excel, method_name] = f'{auu_ave:.2f} ± {auu_std:.2f}'
            self.excel_result_tabel_ailmt.loc[row_idx_excel, method_name] = f'{ailmt_ave:.2f} ± {ailmt_std:.2f}'
            self.excel_result_tabel_time.loc[row_idx_excel, method_name] = f'{time_ave:.2f} ± {time_std:.2f}'

            with pd.ExcelWriter(self.excel_path, engine='xlsxwriter') as writer:
                self.excel_result_tabel_test_acc_main.to_excel(excel_writer=writer, sheet_name='test_acc_main', na_rep='-')
                self.excel_result_tabel_test_acc_ave.to_excel(excel_writer=writer, sheet_name='test_acc_ave', na_rep='-')
                self.excel_result_tabel_test_acc_std.to_excel(excel_writer=writer, sheet_name='test_acc_std', na_rep='-')
                self.excel_result_tabel_aclt.to_excel(excel_writer=writer, sheet_name='aclt', na_rep='-')
                self.excel_result_tabel_ailt.to_excel(excel_writer=writer, sheet_name='ailt', na_rep='-')
                self.excel_result_tabel_aucs.to_excel(excel_writer=writer, sheet_name='aucs', na_rep='-')
                self.excel_result_tabel_auis.to_excel(excel_writer=writer, sheet_name='auis', na_rep='-')
                self.excel_result_tabel_auu.to_excel(excel_writer=writer, sheet_name='auu', na_rep='-')
                self.excel_result_tabel_ailmt.to_excel(excel_writer=writer, sheet_name='ailmt', na_rep='-')
                self.excel_result_tabel_time.to_excel(excel_writer=writer, sheet_name='time', na_rep='-')

        except Exception as e:
            print(f"Error occurred during dumping record for {data_name}/{method_name}/{noise_type}@{noise_rate}: {e}")
            import traceback
            traceback.print_exc()


class SingleExpRecorder:
    def __init__(self, patience=100, criterion=None):
        self.patience = patience
        self.criterion = criterion
        self.best_loss = 1e8
        self.best_metric = -float('inf') if criterion in ['metric', 'either', 'both'] else 0
        self.wait = 0
        self.best_metric_epoch = -1
        self.count_for_metric_epoch = 0

    def add(self, loss_val, metric_val):
        self.count_for_metric_epoch += 1

        current_loss_is_better = loss_val < self.best_loss
        current_metric_is_better = metric_val > self.best_metric

        if self.criterion is None:
            flag = True
        elif self.criterion == 'loss':
            flag = current_loss_is_better
        elif self.criterion == 'metric':
            flag = current_metric_is_better
        elif self.criterion == 'either':
            flag = current_loss_is_better or current_metric_is_better
        elif self.criterion == 'both':
            flag = current_loss_is_better and current_metric_is_better
        else:
            raise NotImplementedError(f"Criterion '{self.criterion}' is not implemented or invalid.")

        if flag:
            if self.criterion == 'loss':
                self.best_loss = loss_val
                self.best_metric = metric_val
            elif self.criterion == 'metric':
                self.best_metric = metric_val
                self.best_loss = loss_val
            elif self.criterion == 'either' or self.criterion == 'both' or self.criterion is None:
                if current_loss_is_better: self.best_loss = loss_val
                if current_metric_is_better: self.best_metric = metric_val
                if flag:
                    if current_loss_is_better: self.best_loss = loss_val
                    if current_metric_is_better: self.best_metric = metric_val
                    if self.criterion is None:
                        self.best_loss = loss_val
                        self.best_metric = metric_val

            self.best_metric_epoch = self.count_for_metric_epoch
            self.wait = 0
        else:
            self.wait += 1

        flag_earlystop = self.patience is not None and 0 < self.patience <= self.wait

        return flag, flag_earlystop

    def reset(self):
        self.best_loss = float('inf')
        self.best_metric = -float('inf') if self.criterion in ['metric', 'either', 'both'] else 0
        self.wait = 0
        self.best_metric_epoch = -1
        self.count_for_metric_epoch = 0
