import logging
import os
import pickle

import numpy as np
import pandas as pd
import traceback


def join_str_list(*args, delimiter='-'):
    assert isinstance(delimiter, str)
    args = filter(None, args)
    args = filter(lambda s: s != 'None', args)
    string = delimiter.join(args)
    string = string.replace(' ', '')
    return string


def name_from_path(file_path):
    base_name = os.path.basename(file_path)
    file_name = os.path.splitext(base_name)[0]
    return file_name


def visualization_choice(vis_rate):
    if vis_rate == 1:
        return True
    else:
        return np.random.choice([True, False], p=[vis_rate, 1 - vis_rate])


class CSVHelper:
    def __init__(self, save_dir: str, csv_date: str, train_mode: str, append_mode: bool = False,
                 excel: bool = True) -> None:
        save_dir = os.path.expanduser(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.csv_date = csv_date
        self.train_mode = train_mode
        self.append = append_mode
        if excel:
            self.file_ext = '.xlsx'
        else:
            raise NotImplementedError
        self.__csv_path = None
        self.__sheet_name = None

    @property
    def csv_path(self):
        if self.__csv_path is None:
            csv_path = os.path.join(self.save_dir,
                                    f'{self.csv_date}-{self.model_name}-{self.train_mode}{self.file_ext}')
            self.__csv_path = csv_path
        return self.__csv_path

    @property
    def sheet_name(self):
        if self.__sheet_name is None:
            if self.append and os.path.exists(self.csv_path):
                sheets = pd.read_excel(self.csv_path, sheet_name=None)
                n_sheet = len(sheets)
                sheet_name = f"Run{n_sheet + 1}"
            else:
                if os.path.exists(self.csv_path): os.remove(self.csv_path)
                sheet_name = "Run1"
            self.__sheet_name = sheet_name
        return self.__sheet_name

    def save(self, model_name: str, metrics_lists: list, append_mean_std=True):
        self.model_name = model_name
        train_metrics_list, dev_metrics_list, test_metrics_list = metrics_lists
        df_list = [pd.DataFrame(train_metrics_list), pd.DataFrame(dev_metrics_list), pd.DataFrame(test_metrics_list)]
        df_list = [idf for idf in df_list if not idf.empty]
        if append_mean_std:
            for i in range(len(df_list)):
                idf = df_list[i]
                mean_df = idf.groupby('pattern').mean()
                if 'SSIM' in mean_df.columns:
                    df_list[i] = idf.append([{'SCORE': mean_df['SCORE'].mean(), 'SSIM': mean_df['SSIM'].mean()},
                                             {'SCORE': mean_df['SCORE'].std(), 'SSIM': mean_df['SSIM'].std()}])
                else:
                    df_list[i] = idf.append([{'SCORE': mean_df['SCORE'].mean()},
                                             {'SCORE': mean_df['SCORE'].std()}])
        try:
            df = pd.concat(df_list, axis=1)
            df.to_excel(self.csv_path, index=False, sheet_name=self.sheet_name)
        except Exception:
            logging.error(traceback.print_exc())
            pkl_path = self.csv_path.replace(self.file_ext, '.pkl')
            with open(pkl_path, mode='wb') as f:
                pickle.dump({self.sheet_name: metrics_lists}, f)
            print(f'Encounter error when saving results, raw data are dumped into {pkl_path}')
