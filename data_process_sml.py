import re
import os
import random
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from collections import defaultdict
from io import StringIO
from shutil import copy
from datetime import timedelta
from glob import glob
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.stats import mode
from tqdm_joblib import tqdm_joblib  # Подключаем tqdm для joblib

from tsfresh import extract_features
from tsfresh import extract_relevant_features
from tsfresh.feature_extraction import MinimalFCParameters, EfficientFCParameters

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, normalize
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, roc_auc_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score, mean_squared_log_error, explained_variance_score

from df_addons import memory_compression
from df_addons import df_to_excel
from print_time import print_time, print_msg

__import__("warnings").filterwarnings('ignore')

WORK_PATH = Path('Z:/python-datasets/Siam_ML_Hack/task-1')
if not WORK_PATH.is_dir():
    WORK_PATH = Path('D:/python-datasets/Siam_ML_Hack/task-1')

if not WORK_PATH.is_dir():
    WORK_PATH = Path('.')

DATASET_PATH = WORK_PATH.joinpath('data')

if not WORK_PATH.is_dir():
    WORK_PATH = Path('.')
    DATASET_PATH = WORK_PATH

MODEL_PATH = WORK_PATH.joinpath('models')
MODEL_PATH.mkdir(parents=True, exist_ok=True)

WORK_PATH.joinpath('best_model').mkdir(parents=True, exist_ok=True)

PREDICTIONS_DIR = WORK_PATH.joinpath('predictions')
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

MODELS_LOGS = WORK_PATH.joinpath('scores_local.logs')
MODELS_LOGS_REG = WORK_PATH.joinpath('scores_local_reg.logs')

if not DATASET_PATH.exists():
    DATASET_PATH = Path('/kaggle/input/dfc-2025-4cast/data')

if not DATASET_PATH.exists():
    DATASET_PATH = Path('.')
    __file__ = Path('.')
    LOCAL_FILE = ''
else:
    LOCAL_FILE = '_local'

RANDOM_SEED = 127


def get_max_num(log_file=None):
    """Получение максимального номера итерации обучения моделей
    :param log_file: имя лог-файла с полным путем
    :return: максимальный номер
    """
    if log_file is None:
        log_file = MODELS_LOGS

    if not log_file.is_file():
        with open(log_file, mode='a', encoding='utf-8') as log:
            log.write('num;mdl;fold;mdl_score;auc_macro;auc_micro;auc_wght;f1_macro;f1_micro;'
                      'f1_wght;tst_score;model_columns;exclude_columns;cat_columns;comment\n')
        max_num = 0
    else:
        # Чтение файла как строки
        with open(log_file, encoding='utf-8') as file:
            file_rows = file.read()
        # Удаление переносов строки в кривых строках и загрузка файла в ДФ
        df = pd.read_csv(StringIO(file_rows.replace(',\n', ',')), sep=';', index_col=False)
        if df.empty:
            max_num = 0
        else:
            max_num = df.num.max()
    return int(max_num) if max_num is not None else 0


def clean_column_name(col_name):
    # Убираем лишние нули в конце десятичной части числа
    cleaned_string = re.sub(r'(\d+\.\d*?[1-9])0{2,}\d*', r'\1', col_name)
    # Убираем лишние нули после цифры 1-9
    cleaned_string = re.sub(r'([1-9]+)0{2,}\d*', r'\1', cleaned_string)
    # Заменяем все нежелательные символы на подчеркивание
    cleaned_string = re.sub(r'[^\w-]', '_', cleaned_string)
    # Убираем подряд идущие подчеркивания
    cleaned_string = re.sub(r'_+', '_', cleaned_string).strip('_')
    return cleaned_string


class DataTransform:
    def __init__(self, use_catboost=True, numeric_columns=None, category_columns=None,
                 features2drop=None, scaler=None, args_scaler=None, test_data_path=None,
                 **kwargs):
        """
        Преобразование данных
        :param use_catboost: данные готовятся для catboost
        :param numeric_columns: цифровые колонки
        :param category_columns: категориальные колонки
        :param drop_first: из dummy переменных удалить первую колонку
        :param scaler: какой скайлер будем использовать
        :param args_scaler: аргументы для скайлера, например: степень для полином.преобразов.
        :param test_data_path: Путь к каталогу с тестовыми данными
        """
        test_data = WORK_PATH.joinpath('validation_1')
        self.files_for_train = True

        self.use_catboost = use_catboost
        self.category_columns = [] if category_columns is None else category_columns
        self.numeric_columns = [] if numeric_columns is None else numeric_columns
        self.features2drop = [] if features2drop is None else features2drop
        self.exclude_columns = []
        self.comment = {}
        self.preprocessor = None
        self.scaler = scaler
        self.args_scaler = args_scaler
        self.test_data_path = test_data if test_data_path is None else Path(test_data_path)
        self.preprocess_files = 'preprocess_files_no_log.pkl'
        self.aggregate_path_file = 'aggregate_data_files_no_log.pkl'
        self.fillna_value = 0
        self.make_log10_features = True
        self.extract_features_from_ts_log = None

    def set_category(self, df):
        for col_name in self.category_columns:
            if col_name in df.columns:
                df[col_name] = df[col_name].astype('category')
        return df

    def fit(self, df):
        """
        Формирование фич
        :param df: исходный ФД
        :return: ДФ с агрегациями
        """
        # если нет цифровых колонок --> заполним их
        if self.category_columns and not self.numeric_columns:
            self.numeric_columns = [col_name for col_name in df.columns if col_name
                                    not in self.category_columns + self.features2drop]
        # если нет категориальных колонок --> заполним их
        if self.numeric_columns and not self.category_columns:
            self.category_columns = [col_name for col_name in df.columns if col_name
                                     not in self.numeric_columns + self.features2drop]

        start_time = print_msg('Группировка по целевому признаку...')

        print_time(start_time)

        return df

    def transform(self, df, model_columns=None):
        """
        Формирование остальных фич
        :param df: ДФ
        :param model_columns: список колонок, которые будут использованы в модели
        :return: ДФ с фичами
        """
        # Сохраняем исходный индекс ДФ
        original_index = df.index

        df = self.set_category(df)

        # Переводим типы данных в минимально допустимые - экономим ресурсы
        df = memory_compression(df)

        return df

    def fit_transform(self, df, model_columns=None):
        """
        Fit + transform data
        :param df: исходный ФД
        :param model_columns: список колонок, которые будут использованы в модели
        :return: ДФ с фичами
        """
        df = self.fit(df)
        df = self.transform(df, model_columns=model_columns)
        return df

    def find_bw_value(self, df_in, max_check_points=3, show_diag=False, current_row=None):
        """
        Функция поиска "Области влияния ствола скважины" и коэффициента wb_value
        :param df_in: входной датафрейм
        :param max_check_points:  Количество проверяемых точек вокруг
        :return: wb_value
        """
        if df_in.empty:
            return np.NaN, None

        df = df_in.copy()

        if not self.make_log10_features:
            for old_col, col in zip(('T', 'P', 'D'), ('time', "dP", 'Pder')):
                # Переход к логарифмическому масштабу времени
                df[old_col] = df[col]
                df[col] = df[col].map(np.log10)

        df['tDif'] = df['time'] - df['time'].min()
        df['PdDf'] = df['Pder'].diff().fillna(0)

        df_abs = df['PdDf'].abs()
        df_std = df['PdDf'].std()
        df_mean = df['PdDf'].mean()
        bad_all_diff = (df_abs > df_mean + 3 * df_std).sum()
        bad_all_diff = 0  # ??????

        if show_diag:
            print(df.head(20))

        # Нахождение индекса точки перегиба
        inflection_index = None
        # Ищем точку перегиба
        for i in range(1, len(df) - max_check_points):
            # Определяем количество предыдущих значений, которые нужно проверить
            prev_vc = min(max_check_points, i)  # Проверяем только доступные предыдущие точки
            next_vc = min(max_check_points, len(df) - i - 1)  # Проверяем последующие точки

            # Проверяем, что текущее значение Pder больше всех предыдущих значений
            if (all(df['Pder'].iloc[i] > df['Pder'].iloc[i - j] for j in
                    range(1, prev_vc + 1)) and
                    all(df['Pder'].iloc[i] >= df['Pder'].iloc[i + j] for j in
                        range(1, next_vc + 1))):
                inflection_index = i
                break

        if show_diag:
            print('inflection_index', inflection_index)

        result = []
        for first_row in range(3):
            # Если точка перегиба найдена
            if inflection_index is not None:
                sf = df.iloc[first_row:inflection_index + 3].copy()
            else:
                sf = df.iloc[first_row:].copy()
            # ограничиваем данные значением tDif < 2.0
            sf = sf[sf['tDif'] < 1.2].copy().reset_index(drop=True)

            sf['tDif'] = sf['time'] - sf['time'].min()

            # Сдвиг колонок 'time' и 'Pder'
            for shift in range(1, min(26, len(sf))):
                sf[f'time{shift}'] = sf['time'].shift(-shift)
                sf[f'Pder{shift}'] = sf['Pder'].shift(-shift)
                sf[f'diff{shift}'] = sf['time'].shift(-shift) - sf['time']

                # Вычисление k и b
                sf[f'k{shift}'] = (sf[f'Pder{shift}'] - sf['Pder']) / (
                        sf[f'time{shift}'] - sf['time'])
                sf[f'b{shift}'] = sf['Pder'] - sf[f'k{shift}'] * sf['time']

                # Если в ДФ есть 0.8 < k < 1.2
                if len(sf[(0.8 <= sf[f'k{shift}']) & (sf[f'k{shift}'] <= 1.2)]):

                    # Проверяем, есть ли хотя бы одно ненулевое значение
                    if not sf[f'k{shift}'].isna().all():
                        closest_k_idx = (sf[f'k{shift}'] - 1).abs().idxmin()

                        if pd.notna(closest_k_idx):  # Проверяем, что индекс не NaN
                            # Получаем значения diff, k и b для этого индекса
                            d = sf[f'diff{shift}'].iloc[closest_k_idx]
                            k = sf[f'k{shift}'].iloc[closest_k_idx]
                            b = sf[f'b{shift}'].iloc[closest_k_idx]

                            row_arr = (abs(1 - k), first_row, first_row + shift, k, b, d)
                            # Проверка на наличие NaN и период больше 0.5 лог.цикла
                            if (all(pd.notna(value) for value in row_arr)
                                    and d >= 0.13 and bad_all_diff < 1):
                                result.append(row_arr)

        if show_diag:
            sf.to_excel(WORK_PATH.joinpath(f'tmp/df_{current_row}_bw_{first_row}_shift.xlsx'))

        if result:
            result = sorted(result)[:5]

            if show_diag:
                print(*result, sep='\n')

            predicted = []
            for row in result:
                _, idx1, idx2, k, b, d = row

                # Извлечение реальных данных Pder между idx1 и idx2 из оригинального ДФ df
                real_values = df['Pder'].iloc[idx1:idx2 + 1].values
                time_values = df['time'].iloc[idx1:idx2 + 1].values

                # Выбор двух точек для построения прямой
                # Для примера выбираем первый и X элемент
                x1, y1 = df['time'].iloc[idx1], df['Pder'].iloc[idx1]  # Первая точка
                x2, y2 = df['time'].iloc[idx2], df['Pder'].iloc[idx2]  # Последняя точка

                if show_diag:
                    # Координаты точек
                    print(f'Первая точка: (x={x1}, y={y1})')
                    print(f'Вторая точка: (x={x2}, y={y2})')

                # Первая линия с k = 1, проходящая через (x1, y1)
                b1 = y1 - 1 * x1  # Свободный член для первой линии
                # Вторая линия с k = 1, проходящая через (x2, y2)
                b2 = y2 - 1 * x2  # Свободный член для второй линии

                for b_ in (b1, b2):
                    # Получение значений линий для тех же временных значений
                    real_values = np.nan_to_num(real_values, nan=0.0)
                    predicted_values = np.nan_to_num(1 * time_values + b_, nan=0.0)
                    if len(real_values) and len(predicted_values):
                        # Рассчитать ошибки
                        mae = mean_absolute_error(real_values, predicted_values).round(5)
                        predicted.append((mae, idx1, idx2, 1, b_, d))

            if predicted:
                # Группируем по (idx1, idx2)
                grouped = defaultdict(list)
                for mae, idx1, idx2, _, b, d in predicted:
                    grouped[(idx1, idx2)].append((mae, b, d))
                # Усредняем значения
                averaged_predicted = []
                for (idx1, idx2), values in grouped.items():
                    mae_avg = np.mean([v[0] for v in values])
                    b_avg = np.mean([v[1] for v in values])
                    d_avg = np.mean([v[2] for v in values])

                    averaged_predicted.append((mae_avg, idx1, idx2, 1, b_avg, d_avg))

                if show_diag:
                    print(*sorted(predicted), sep='\n')
                    print()
                    print(*sorted(averaged_predicted), sep='\n')

                # Получаем максимальный последний индекс линии
                last_index = max(row[2] for row in averaged_predicted)
                if inflection_index is None:
                    inflection_index = last_index

                wb_value = min(averaged_predicted)[4]
                last_index = min(last_index, inflection_index)
                return wb_value, last_index

        return np.NaN, None

    def find_kf_value(self, df_in, chk_k, idx=1, max_check_points=3, show_diag=False,
                      current_row=None):
        """
        Функция поиска "Области влияния ствола скважины" и коэффициента wb_value
        :param chk_k: Коэффициент, который будем искать
        :param idx: Начальниый индекс для поиска
        :param df_in: входной датафрейм
        :param max_check_points:  Количество проверяемых точек вокруг
        :return: wb_value
        """
        if idx is None:
            idx = 1

        if df_in.empty:
            return np.NaN, None

        df = df_in.iloc[idx:].copy().reset_index(drop=True)

        if not self.make_log10_features:
            for old_col, col in zip(('T', 'P', 'D'), ('time', "dP", 'Pder')):
                # Переход к логарифмическому масштабу времени
                df[old_col] = df[col]
                df[col] = df[col].map(np.log10)

        df['tDif'] = df['time'] - df['time'].min()
        df['PdDf'] = df['Pder'].diff().fillna(0)

        df_abs = df.loc[3:100, 'PdDf'].abs()
        df_std = df.loc[3:100, 'PdDf'].std()
        df_mean = df.loc[3:100, 'PdDf'].mean()
        bad_all_diff = (df_abs > df_mean + 3 * df_std).sum()

        start_idx = 2
        end_idx = min(len(df), 500)

        if show_diag:
            # print(df.iloc[idx:].head(20))
            print('idx', idx)

        if abs(chk_k) <= 0.01:
            # --- Параметры ---
            window = 3  # Размер окна для усреднения
            threshold_mean = 0.004  # Порог для горизонтальности
            # --- Вычисляем скользящее среднее и стандартное отклонение ---
            df['rolling_mean'] = df['PdDf'].rolling(window=window, center=True).mean()
            df['rolling_std'] = df['PdDf'].rolling(window=window, center=True).std()
            # Определяем горизонтальные участки (где стандартное отклонение мало)
            df['is_flat'] = df['rolling_mean'].abs() < threshold_mean
            # Применяем скользящее окно для сглаживания выбросов
            df["smoothed"] = df["is_flat"].rolling(3, center=True).mean() > 1 / 5

            # Добавляем колонку с направлением производной: 1 (возрастает), -1 (убывает), 0 (нулевая)
            df['sign'] = np.sign(df['PdDf'])
            # Определяем группы одинаковых знаков
            df['group'] = (df['sign'] != df['sign'].shift()).cumsum()
            # Считаем размер каждой группы
            group_sizes = df.groupby('group')['sign'].transform('count')
            # Определяем, была ли вся группа ИСТИНА в smoothed
            group_all_true = df.groupby('group')['smoothed'].transform('all')
            # Меняем smoothed на False, если у всей группы было True и длина > 5
            df.loc[(group_sizes > 5) & group_all_true, 'smoothed'] = False

            # Поиск границ интервалов
            valid_segments = []
            start = None
            for idx in range(len(df)):
                if df.loc[idx, "smoothed"]:  # Начинаем отрезок
                    if start is None:
                        start = idx
                else:  # Заканчиваем отрезок
                    if start is not None:
                        valid_segments.append((start, idx))
                        start = None

            if show_diag:
                print(valid_segments)
                # Выводим результаты
                # print(df[['PdDf', 'rolling_mean', 'rolling_std', 'smoothed']])
                xls_file = WORK_PATH.joinpath(f'tmp/df_{current_row}_chk_k_sft.xlsx')
                df_to_excel(df.reset_index(), xls_file, ins_col_width=[(1, 12)] * 9,
                            float_cells=[*range(1, 10)])

            # Порог для объединения
            gap_threshold = 5
            # Объединение участков
            merged_segments = []

            # Старый вариант
            # if valid_segments:
            #     start, end = valid_segments[0]
            #     for i in range(1, len(valid_segments)):
            #         next_start, next_end = valid_segments[i]
            #         # Проверяем разрыв между текущим концом и следующим началом
            #         if next_start - end <= gap_threshold:
            #             end = next_end  # Расширяем текущий отрезок
            #         else:
            #             merged_segments.append((start, end))
            #             start, end = next_start, next_end
            #     # Добавляем последний отрезок
            #     merged_segments.append((start, end))

            if valid_segments:
                start, end = valid_segments[0]
                length = end - start  # Длина текущего сегмента
                for i in range(1, len(valid_segments)):
                    next_start, next_end = valid_segments[i]
                    next_length = next_end - next_start  # Длина следующего сегмента
                    # Проверяем, если хотя бы один сегмент длиннее порога, объединяем
                    if (next_start - end <= gap_threshold and
                            (length > gap_threshold * 3 or next_length > gap_threshold * 3)):
                        end = next_end  # Расширяем текущий отрезок
                        length = end - start  # Пересчитываем длину
                    else:
                        merged_segments.append((start, end))
                        start, end = next_start, next_end
                        length = end - start  # Обновляем длину для нового сегмента

                # Добавляем последний отрезок
                merged_segments.append((start, end))

            valid_segments = []
            if merged_segments:
                for start_idx, end_idx in merged_segments:
                    d = df.loc[end_idx, 'time'] - df.loc[start_idx, 'time']
                    valid_segments.append((start_idx, end_idx, d))
                    if show_diag:
                        print(f'Найден участок: {start_idx} — {end_idx}, d={d}')
            else:
                return np.NaN, None

            start_idx, end_idx, d = max(valid_segments, key=lambda z: z[-1])
            # bad_all_diff = 0
            if show_diag:
                print(f'Горизонтальный участок: {start_idx} — {end_idx}, d={d}')
            if d < 0.3:
                return np.NaN, None

        # Нахождение индекса точки перегиба
        inflection_index = None
        # # Ищем точку перегиба
        # for i in range(idx, len(df) - max_check_points):
        #     # Определяем количество предыдущих значений, которые нужно проверить
        #     prev_vc = min(max_check_points, i)  # Проверяем только доступные предыдущие точки
        #     next_vc = min(max_check_points, len(df) - i - 1)  # Проверяем последующие точки
        #
        #     # Проверяем, что текущее значение Pder больше всех предыдущих значений
        #     if (all(df['Pder'].iloc[i] > df['Pder'].iloc[i - j] for j in
        #             range(1, prev_vc + 1)) and
        #             all(df['Pder'].iloc[i] >= df['Pder'].iloc[i + j] for j in
        #                 range(1, next_vc + 1))):
        #         inflection_index = i
        #         break

        if show_diag:
            print('inflection_index', inflection_index)

        result = []
        found_closest_k = False
        # TODO --> Не забыть поставить 500 вместо 100
        for first_row in range(start_idx, end_idx):
            if ((first_row > 200 and first_row % 2)
                    or (first_row > 400 and first_row % 3)
                    or (first_row > 600 and first_row % 4)
                    or (first_row > 1000 and first_row % 5)):
                continue
            # Если точка перегиба найдена
            if inflection_index is not None:
                sf = df.iloc[first_row:inflection_index + 3].copy().reset_index(drop=True)
            else:
                sf = df.iloc[first_row:].copy().reset_index(drop=True)
            # ограничиваем данные значением tDif < 2.0
            # sf = sf[sf['tDif'] < 1.2].copy().reset_index(drop=True)

            sf['tDif'] = sf['time'] - sf['time'].min()

            array_k = []

            # TODO --> Не забыть поставить 200 вместо 100
            # Сдвиг колонок 'time' и 'Pder'
            for shift in range(1, 200):
                if ((shift > 100 and shift % 2) or (shift > 200 and shift % 3) or
                        (shift > 400 and shift % 4) or (shift > 500 and shift % 5)):
                    continue

                sf[f'time{shift}'] = sf['time'].shift(-shift)
                sf[f'Pder{shift}'] = sf['Pder'].shift(-shift)
                sf[f'diff{shift}'] = sf['time'].shift(-shift) - sf['time']

                if show_diag and len(df) > 500 and not shift % 10:
                    print(f'first_row={first_row}, shift={shift}')

                # Вычисление k и b
                sf[f'k{shift}'] = (sf[f'Pder{shift}'] - sf['Pder']) / (
                        sf[f'time{shift}'] - sf['time'])
                sf[f'b{shift}'] = sf['Pder'] - sf[f'k{shift}'] * sf['time']

                m = 1.5
                period = df.iloc[first_row:first_row + shift + 1]['PdDf']
                std_period = period.std()
                mean_period = period.mean()
                bad_der_diff = (period.abs() > mean_period + m * std_period).sum()
                sf[f'bad{shift}'] = bad_der_diff

                # Если в ДФ есть k-0.2 < k < k+0.2
                beg_k, end_k = chk_k - 0.2, chk_k + 0.2
                if len(sf[(beg_k <= sf[f'k{shift}']) & (sf[f'k{shift}'] <= end_k)]):

                    # Проверяем, есть ли хотя бы одно ненулевое значение
                    if not sf[f'k{shift}'].isna().all():
                        closest_k_idx = (sf[f'k{shift}'] - chk_k).abs().idxmin()

                        if pd.notna(closest_k_idx):  # Проверяем, что индекс не NaN
                            # Получаем значения diff, k и b для этого индекса
                            d = sf[f'diff{shift}'].iloc[closest_k_idx]
                            k = sf[f'k{shift}'].iloc[closest_k_idx]
                            b = sf[f'b{shift}'].iloc[closest_k_idx]

                            # Проверка на плавность производной
                            # Уровень варьирования производной не больше m period.std()
                            m = 3.5
                            period = df.iloc[first_row:first_row + shift + 1]['PdDf']
                            std_period = period.std()
                            mean_period = period.mean()
                            bad_der_diff = (period.abs() > mean_period + m * std_period).sum()

                            row_arr = (abs(chk_k - k), first_row, first_row + shift, k, b, d)
                            # Проверка на наличие NaN и период больше 0.5 лог.цикла
                            if (all(pd.notna(value) for value in row_arr)
                                    and d >= 0.5 and bad_der_diff < 1 and bad_all_diff <= 3):
                                result.append(row_arr)
                                if row_arr[0] <= 0.05:
                                    found_closest_k = True
                                    break

                array_k.append((sf.loc[0, f'k{shift}'], bad_der_diff, d))

            if show_diag:
                print(f'bad_all_diff={bad_all_diff}, result', result)
                print(array_k[:50])

                xls_file = WORK_PATH.joinpath(
                    f'tmp/df_{current_row}_chk_k_{chk_k}_{first_row}_sft.xlsx')
                sf.to_excel(xls_file)

            if found_closest_k:
                break

        if result:
            result = sorted(result)[:7]

            if show_diag:
                print(*result, sep='\n')

            predicted = []
            for row in result:
                _, idx1, idx2, k, b, d = row

                # Извлечение реальных данных Pder между idx1 и idx2 из оригинального ДФ df
                real_values = df['Pder'].iloc[idx1:idx2 + 1].values
                time_values = df['time'].iloc[idx1:idx2 + 1].values

                # Выбор двух точек для построения прямой
                # Для примера выбираем первый и X элемент
                x1, y1 = df['time'].iloc[idx1], df['Pder'].iloc[idx1]  # Первая точка
                x2, y2 = df['time'].iloc[idx2], df['Pder'].iloc[idx2]  # Последняя точка

                if show_diag:
                    # Координаты точек
                    print(f'Первая точка: (x={x1}, y={y1})')
                    print(f'Вторая точка: (x={x2}, y={y2})')

                # Первая линия с k = chk_k, проходящая через (x1, y1)
                b1 = y1 - chk_k * x1  # Свободный член для первой линии
                # Вторая линия с k = chk_k, проходящая через (x2, y2)
                b2 = y2 - chk_k * x2  # Свободный член для второй линии

                for b_ in (b1, b2):
                    # Получение значений линий для тех же временных значений
                    real_values = np.nan_to_num(real_values, nan=0.0)
                    predicted_values = np.nan_to_num(chk_k * time_values + b_, nan=0.0)
                    if len(real_values) and len(predicted_values):
                        # Рассчитать ошибки
                        mae = mean_absolute_error(real_values, predicted_values).round(5)
                        predicted.append((mae, idx1, idx2, chk_k, b_, d))

            if predicted:
                # Группируем по (idx1, idx2)
                grouped = defaultdict(list)
                for mae, idx1, idx2, _, b, d in predicted:
                    # Если слишком большое расстояние - не будем группировать пары
                    if idx2 - idx1 > 69:
                        grouped[(idx1, idx2, mae)].append((mae, b, d))
                    else:
                        grouped[(idx1, idx2, 0)].append((mae, b, d))

                # Усредняем значения
                averaged_predicted = []
                for (idx1, idx2, _), values in grouped.items():
                    mae_avg = np.mean([v[0] for v in values])
                    b_avg = np.mean([v[1] for v in values])
                    d_avg = np.mean([v[2] for v in values])

                    averaged_predicted.append((mae_avg, idx1, idx2, chk_k, b_avg, d_avg))

                averaged_predicted = predicted

                if show_diag:
                    print(*sorted(predicted), sep='\n')
                    print()
                    print(*sorted(averaged_predicted), sep='\n')

                # Получаем максимальный последний индекс линии
                last_index = max(row[2] for row in averaged_predicted)
                if inflection_index is None:
                    inflection_index = last_index

                wb_value = min(averaged_predicted)[4]
                last_index = min(last_index, inflection_index)
                return wb_value, last_index

        return np.NaN, None

    def find_pc_value(self, df_in, points=7, show_diag=False, current_row=None):
        # Разворачиваем данные, чтобы двигаться с конца
        df_rev = df_in.iloc[::-1].reset_index()

        if show_diag:
            xls_file = WORK_PATH.joinpath(f'tmp/df_{current_row}_chk_k.xlsx')
            df_to_excel(df_rev.reset_index(), xls_file, ins_col_width=[(1, 12)] * 9,
                        float_cells=[*range(1, 10)])

        # Ищем локальный максимум, после которого идет устойчивое падение
        boundary_idx, boundary_time = None, None
        # Берём точку и проверяем следующие 5
        for i in range(points, min(30, len(df_rev) - points)):
            current_pder = df_rev.loc[i, "Pder"]
            prev_max = df_rev.loc[i - points: i, "Pder"].max()

            if show_diag:
                print(i, df_rev.loc[i, "index"], df_rev.loc[i, "time"], current_pder)
                print([df_rev.loc[i + j, "Pder"].round(5) for j in range(points)])

            # Условие: если после текущего значения есть минимум 5 точек подряд убывания
            if (all(current_pder >= df_rev.loc[i + 1: i + points, "Pder"]) and
                    all(prev_max >= df_rev.loc[i - points: i - 1, "Pder"])):

                boundary_idx = df_rev.loc[i, "index"]
                boundary_time = df_rev.loc[i, "time"]
                if i > points:
                    if show_diag:
                        print(f"Граница постоянного давления найдена: индекс {boundary_idx}, "
                              f"время {boundary_time} часов")

                    return boundary_time, boundary_idx

        return np.NaN, None

    def extract_features_from_ts(self, df, file_path=None):

        features = {}

        # TODO --> Сюда поставить расчет b_value
        features['wb_value'], last_idx_bw = self.find_bw_value(df)
        # features['pc_value'], last_idx_bw = self.find_pc_value(df)

        last_idx_kf = []
        for koeff, name_koeff in zip((0, 0.5, 0.25, -0.5),
                                     ('ra_value', 'li_value', 'bl_value', 'sp_value')):
            # TODO --> Не забыть это убрать !!!
            if name_koeff != '':
                continue
            # TODO --> Не забыть это ^^^^^^ убрать !!!
            features[name_koeff], last_idx = self.find_kf_value(df, koeff, idx=last_idx_bw)
            last_idx_kf.append(last_idx)

        # Вычисляем разности и статистики
        for col in ('time', 'dP', 'Pder', 'T', 'P', 'D'):
            if col not in df.columns:
                continue
            # Статистики
            features[f'{col}_std'] = df[col].std()
            features[f'{col}_mean'] = df[col].mean()
            features[f'{col}_median'] = df[col].median()

            diffs = df[col].diff().dropna()  # разности
            std_diffs = diffs.std()
            mean_diffs = diffs.mean()

            if col in ('dP', 'P'):
                # Вычисление скользящего среднего для dP
                smooth = diffs.rolling(window=3, min_periods=1).mean()
                # Проверка на монотонный рост или плато
                features[f'{col}_growth'] = (smooth.diff().dropna() >= 0).all().astype(int)

            features[f'{col}_diff_max'] = diffs.max()
            features[f'{col}_diff_std'] = std_diffs
            features[f'{col}_diff_mean'] = mean_diffs
            features[f'{col}_diff_median'] = diffs.median()
            # Сколько % выбросов в стандартных отклонениях
            for k in (1, 2, 3):
                features[f'{col}_{k}_std'] = (diffs.abs() > mean_diffs + k * std_diffs).sum()
                features[f'{col}_{k}_std'] = features[f'{col}_{k}_std'] / len(df) * 100

        features_add = self.extract_features_from_ts_add(df)
        features.update(features_add)

        if self.extract_features_from_ts_log is not None and file_path is not None:
            with open(self.extract_features_from_ts_log, "a", encoding="utf-8") as f:
                print(file_path.name, file=f)  # Запись в файл

        return features

    @staticmethod
    def extract_features_from_ts_add(df):

        features = {}

        # Вычисляем разности и статистики
        for col in ('Pder', 'D'):
            if col not in df.columns:
                continue

            df_col_diff = df[col].diff().fillna(0)  # разности

            for i, quantile_list in enumerate(([0.06, 0.15, 0.3, 0.42, 0.54, 0.66, 0.78, 0.9],
                                               [0.2, 0.4, 0.6, 0.8])):
                quantiles = np.quantile(df_col_diff, quantile_list).tolist()
                quantiles.insert(0, df_col_diff.min() - 1)
                quantiles.append(df_col_diff.max() + 1)
                quantile_list.append(1 + round((i + 1) / 10, 1))
                quantile_list = [str(q).replace('.', '_') for q in quantile_list]

                for q1, q2, cn in zip(quantiles, quantiles[1:], quantile_list):
                    diffs = df_col_diff[(q1 < df_col_diff) & (df_col_diff <= q2)]

                    std_diffs = diffs.std()
                    mean_diffs = diffs.mean()

                    features[f'{col}_q_{cn}_max'] = diffs.max()
                    features[f'{col}_q_{cn}_std'] = std_diffs
                    features[f'{col}_q_{cn}_mean'] = mean_diffs
                    features[f'{col}_q_{cn}_median'] = diffs.median()
                    # Сколько % выбросов в стандартных отклонениях
                    for k in (1, 2, 3):
                        ck = f'{col}_q_{cn}_{k}_std'
                        features[ck] = (diffs.abs() > mean_diffs + k * std_diffs).sum()
                        features[ck] = 0 if diffs.empty else features[ck] / len(diffs) * 100

        return features

    def read_data_file(self, file_path):
        try:
            df = pd.read_csv(file_path, sep='\t', names=['time', 'dP', 'Pder'])
            # Просто удаляем отрицательное время
            df = df[df['time'] >= 0]
            # Преобразуем все данные в логарифмы
            if self.make_log10_features:
                self.comment['log10_features'] = 'True'
                for old_col, col in zip(('T', 'P', 'D'), ('time', "dP", 'Pder')):
                    df[old_col] = df[col]
                    df[col] = df[col].map(np.log10)
        except:
            if self.make_log10_features:
                df = pd.DataFrame(columns=['time', 'dP', 'Pder', 'T', 'P', 'D'])
            else:
                df = pd.DataFrame(columns=['time', 'dP', 'Pder'])
        return df

    def process_files(self, files, desc=None, log_file=None):
        file_features = []
        total_df = pd.DataFrame(columns=['time', 'dP', 'Pder', 'file_name'])
        for file_path in tqdm(files, desc=desc, total=len(files)):
            # Загружаем временной ряд
            df = self.read_data_file(file_path)
            # Проверяем на пустые или слишком короткие файлы
            # if df.empty or len(df) < 2:
            if df.empty:
                # features = {'Некачественное ГДИС': 1}
                features = {}
                for col in ['time_std', 'time_mean', 'time_median', 'time_diff_max',
                            'time_diff_std', 'time_diff_mean', 'time_diff_median',
                            'time_1_std', 'time_2_std', 'time_3_std']:
                    features[col] = self.fillna_value  # Заполняем пропуски
            else:
                # Вычисляем признаки
                features = self.extract_features_from_ts(df, file_path=file_path)
                # features['Некачественное ГДИС'] = 0  # Данные нормальные

            features['file_name'] = file_path.name
            features['count_rows'] = len(df)
            file_features.append(features)

            df['file_name'] = file_path.name

            total_df = pd.concat([total_df, df], axis=0)

        return pd.DataFrame(file_features), total_df

    def process_file_joblib(self, file_path, log_file=None):
        # Загружаем временной ряд
        df = self.read_data_file(file_path)
        # Проверяем на пустые или слишком короткие файлы
        # if df.empty or len(df) < 2:
        if df.empty:
            # features = {'Некачественное ГДИС': 1}
            features = {}
            for col in ['time_std', 'time_mean', 'time_median', 'time_diff_max',
                        'time_diff_std', 'time_diff_mean', 'time_diff_median',
                        'time_1_std', 'time_2_std', 'time_3_std']:
                features[col] = self.fillna_value  # Заполняем пропуски
        else:
            # Вычисляем признаки
            features = self.extract_features_from_ts(df, file_path=file_path)
            # features['Некачественное ГДИС'] = 0  # Данные нормальные

        features['file_name'] = file_path.name
        features['count_rows'] = len(df)
        df.insert(0, 'file_name', file_path.name)

        return features, df

    def files_process_joblib(self, files, desc=None):
        # Добавляем поддержку tqdm для joblib
        with tqdm_joblib(tqdm(desc=desc, total=len(files), leave=True)) as pbar:
            results = Parallel(n_jobs=24)(delayed(self.process_file_joblib)(file)
                                          for file in files)
        # Разбираем результаты
        file_features, total_df = zip(*results)

        # Преобразуем в DataFrame
        features = pd.DataFrame(list(file_features)).sort_values('file_name')
        total_df = pd.concat(list(total_df), axis=0).sort_values(['file_name', 'time'])
        return features, total_df

    def preprocess_data(self, remake_file=False, fill_nan=True, sample=None, use_joblib=False,
                        file_with_target_class=None, **kwargs):
        """
        Предобработка данных
        :param remake_file: Переформировать файлы с агрегациями
        :param fill_nan: заполняем пропуски в данных
        :param sample: вернуть ДФ из указанного количества
        :param use_joblib: использовать многопоточную обработку файлов
        :param file_with_target_class: Используем предсказания классификатора
        :return: ДФ
        """
        preprocess_files = None

        if self.make_log10_features:
            self.comment['log10_features'] = 'True'

        self.comment['file_with_target_class'] = file_with_target_class

        if self.preprocess_files:
            if self.make_log10_features:
                self.preprocess_files = self.preprocess_files.replace('_no_log.pkl',
                                                                      '_log10.pkl')
            preprocess_files = WORK_PATH.joinpath(self.preprocess_files)

            if preprocess_files.is_file() and not remake_file and sample is None:
                start_time = print_msg('Читаю подготовленные данные...')
                with open(preprocess_files, 'rb') as in_file:
                    train_df, test_df, td_trn, td_tst = joblib.load(in_file)
                if file_with_target_class is not None:
                    test_df = self.merge_with_target_class(test_df, file_with_target_class)
                print_time(start_time)
                return train_df, test_df, td_trn, td_tst

        start_time = print_msg('Загрузка данных...')

        if use_joblib:
            process_func = self.files_process_joblib
        else:
            process_func = self.process_files

        # Определяем список бинарных колонок
        binary_columns = ['hq', 'Некачественное ГДИС', 'Влияние ствола скважины',
                          'Радиальный режим', 'Линейный режим', 'Билинейный режим',
                          'Сферический режим', 'Граница постоянного давления',
                          'Граница непроницаемый разлом']

        if self.files_for_train:
            mt = pd.read_csv(WORK_PATH.joinpath('markup_train.csv'))
            hq = pd.read_csv(WORK_PATH.joinpath('hq_markup_train.csv'))
            mt['hq'] = 0
            hq['hq'] = 1
            # Уберем из общего ДФ записи с файлами из hq_markup_train
            mt = mt[~mt.file_name.isin(hq.file_name)]
            # Объединим ДФ
            mt = pd.concat([mt, hq], axis=0)

            if sample is not None:
                mt = mt.sample(sample, random_state=RANDOM_SEED)

            # Создаем колонку labels, конкатенируя бинарные признаки в строку
            mt["labels"] = mt[binary_columns].astype(str).agg(''.join, axis=1)
            for label_len in (5, 3, 1):
                vc = mt['labels'].value_counts()
                # Получим метки, которые встречаются один раз
                unique_labels = vc[vc == 1].index
                # Преобразуем метки, которые встречаются один раз
                mt['labels'] = mt['labels'].apply(
                    lambda x: x[:label_len] if x in unique_labels else x)

            # Получаем список файлов для тренировки
            train_files = [DATASET_PATH.joinpath(file_name) for file_name in mt.file_name]
            # Загружаем данные и строим признаки
            train_df, td_trn = process_func(train_files, desc='Обработка файлов для трейна')
            # Объединяем с разметкой
            train_df = mt.merge(train_df, on='file_name', how='left')
            train_df.set_index('file_name', inplace=True)

            print(td_trn.info())

            if fill_nan:
                # Заполняем пропуски
                train_df.fillna(self.fillna_value, inplace=True)

        # Получаем список файлов для тестирования
        test_files = list(self.test_data_path.glob('*'))
        # Загружаем данные и строим признаки
        test_df, td_tst = process_func(test_files, desc='Обработка файлов для теста')
        test_df.set_index('file_name', inplace=True)

        if self.files_for_train:
            print('train_df.shape=', train_df.shape, 'test_df.shape=', test_df.shape)
            print_time(start_time)

        if fill_nan:
            # Заполняем пропуски
            test_df.fillna(self.fillna_value, inplace=True)

        if self.preprocess_files:
            save_time = print_msg('Сохраняем предобработанные данные...')
            test_df.to_excel(WORK_PATH.joinpath('test_df.xlsx'))
            if self.files_for_train:
                train_df.to_excel(WORK_PATH.joinpath('train_df.xlsx'))
                with open(preprocess_files, 'wb') as file:
                    joblib.dump((train_df, test_df, td_trn, td_tst), file, compress=7)
            print_time(save_time)

        if file_with_target_class is not None:
            test_df = self.merge_with_target_class(test_df, file_with_target_class)

        if self.files_for_train:
            return train_df, test_df, td_trn, td_tst

        return test_df, td_tst

    @staticmethod
    def merge_with_target_class(test_df, file_with_target_class):
        if PREDICTIONS_DIR.joinpath(file_with_target_class).is_file():
            subm_df = pd.read_csv(PREDICTIONS_DIR.joinpath(file_with_target_class),
                                  index_col='file_name')
            binary_columns = ['Некачественное ГДИС', 'Влияние ствола скважины',
                              'Радиальный режим', 'Линейный режим', 'Билинейный режим',
                              'Сферический режим', 'Граница постоянного давления',
                              'Граница непроницаемый разлом']
            test_df = subm_df[binary_columns].join(test_df, how="left")
        return test_df

    def make_agg_data(self, remake_file=False, sample=None, use_joblib=False,
                      add_agg_data=True, use_featuretools=False, file_with_target_class=None,
                      **kwargs):
        """
        Подсчет разных агрегированных статистик
        :param remake_file: Формируем файлы снова или читаем с диска
        :param sample: вернуть ДФ из указанного количества inn_id
        :param use_joblib: использовать многопоточную обработку файлов
        :param use_featuretools: Используем модуль featuretools
        :param add_agg_data: Добавляем самодельную аггрегацию
        :param file_with_target_class: Используем предсказания классификатора
        :return: ДФ трейна и теста с агрегированными данными
        """

        aggregate_path_file = None

        if self.make_log10_features:
            self.comment['log10_features'] = 'True'

        self.comment['file_with_target_class'] = file_with_target_class

        if self.aggregate_path_file and use_featuretools:
            if self.make_log10_features:
                self.aggregate_path_file = self.aggregate_path_file.replace('_no_log.pkl',
                                                                            '_log10.pkl')
            aggregate_path_file = WORK_PATH.joinpath(self.aggregate_path_file)

            if aggregate_path_file.is_file() and add_agg_data and not remake_file:
                start_time = print_msg('Читаю подготовленные данные...')
                with open(aggregate_path_file, 'rb') as in_file:
                    train_df, test_df = joblib.load(in_file)
                if file_with_target_class is not None:
                    test_df = self.merge_with_target_class(test_df, file_with_target_class)
                print_time(start_time)
                return train_df, test_df

        # Загрузка предобработанных данных
        if self.files_for_train:
            train_df, test_df, td_trn, td_tst = self.preprocess_data(remake_file=remake_file,
                                                                     sample=sample,
                                                                     use_joblib=use_joblib,
                                                                     )
            print(td_trn.shape, td_tst.shape)
            print(td_trn.columns, td_tst.columns)
        else:
            test_df, td_tst = self.preprocess_data(use_joblib=use_joblib)
            print(td_tst.shape, td_tst.columns)

        start_time = print_msg('Агрегация данных...')

        # # Находим строки с пропусками в поле 'time'
        # missing_time_files = td_trn[td_trn['time'].isnull()]['file_name'].unique()
        # print(missing_time_files)
        # for file in missing_time_files:
        #     copy(DATASET_PATH.joinpath(file), f'E:/temp/{file}')

        if add_agg_data and use_featuretools:
            # Извлечение признаков
            if self.files_for_train:
                ext_feat_trn = extract_features(td_trn,
                                                column_id='file_name',
                                                column_sort='time',
                                                )
                ext_feat_trn.columns = [clean_column_name(col) for col in
                                        ext_feat_trn.columns]

            ext_feat_tst = extract_features(td_tst,
                                            column_id='file_name',
                                            column_sort='time',
                                            )
            old_names = ext_feat_tst.columns.tolist()
            ext_feat_tst.columns = [clean_column_name(col) for col in ext_feat_tst.columns]

            if self.files_for_train:
                new_names = ext_feat_tst.columns.tolist()
                ndf = pd.DataFrame({'old_names': old_names, 'new_names': new_names})
                ndf.to_excel(WORK_PATH.joinpath('column_names.xlsx'))

                # удалим колонки, сформированые руками, если они есть в extract_features
                drop_cols = [col for col in train_df.columns if col in ext_feat_trn.columns]
                train_df = train_df.drop(columns=drop_cols).join(ext_feat_trn, how="left")
                # Заполняем пропуски
                train_df.fillna(self.fillna_value, inplace=True)

            # удалим колонки, которые сформированы руками, если они есть в extract_features
            drop_cols = [col for col in test_df.columns if col in ext_feat_tst.columns]
            test_df = test_df.drop(columns=drop_cols).join(ext_feat_tst, how="left")
            # Заполняем пропуски
            test_df.fillna(self.fillna_value, inplace=True)

        print_time(start_time)

        if self.aggregate_path_file and aggregate_path_file:
            save_time = print_msg('Сохраняем агрегированные данные...')
            with open(aggregate_path_file, 'wb') as file:
                joblib.dump((train_df, test_df), file, compress=7)
            print_time(save_time)

        if file_with_target_class is not None:
            test_df = self.merge_with_target_class(test_df, file_with_target_class)

        if self.files_for_train:
            return train_df, test_df

        return test_df

    @staticmethod
    def drop_constant_columns(df):
        # Ищем колонки с константным значением для удаления
        col_to_drop = []
        for col in df.columns:
            if df[col].nunique() == 1:
                col_to_drop.append(col)
        if col_to_drop:
            df.drop(columns=col_to_drop, inplace=True)
        return df


def set_all_seeds(seed=RANDOM_SEED):
    # python's seeds
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def add_info_to_log(prf, max_num, idx_fold, model, valid_scores, info_cols,
                    comment_dict=None, clf_lr=None, log_file=MODELS_LOGS):
    """
    Добавление информации об обучении модели
    :param prf: Префикс файла сабмита
    :param max_num: номер итерации обучения моделей
    :param idx_fold: номер фолда при обучении
    :param model: обученная модель
    :param valid_scores: скоры при обучении
    :param info_cols: информативные колонки
    :param comment_dict: комментарии
    :param clf_lr: список из learning_rate моделей
    :param log_file: полный путь к файлу с логами обучения моделей
    :return:
    """
    m_score, auc_macro, auc_micro, auc_wght, f1_macro, f1_micro, f1_wght, score = valid_scores

    model_columns, exclude_columns, cat_columns = info_cols

    if comment_dict is None:
        comment = {}
    else:
        comment = deepcopy(comment_dict)

    model_clf_lr = feature_imp = None
    if model is not None:
        if 'CatBoost' in model.__class__.__name__:
            model_clf_lr = model.get_all_params().get('learning_rate', 0)
            feature_imp = model.feature_importances_

        elif 'LGBM' in model.__class__.__name__:
            model_clf_lr = model.get_params().get('learning_rate', 0)

        elif 'XGB' in model.__class__.__name__:
            model_clf_lr = model.get_params().get('learning_rate', 0)

    if feature_imp is not None:
        try:
            use_cols = [col for col in model_columns if col not in exclude_columns]
            features = pd.DataFrame({'Feature': use_cols,
                                     'Importance': feature_imp}).sort_values('Importance',
                                                                             ascending=False)
            features.to_excel(MODEL_PATH.joinpath(f'features_{prf}{max_num}.xlsx'),
                              index=False)
        except:
            pass

    if model_clf_lr is not None:
        model_clf_lr = round(model_clf_lr, 8)

    if clf_lr is None:
        clf_lr = model_clf_lr

    comment['clf_lr'] = clf_lr
    if model is not None:
        comment.update(model.get_params())

    prf = prf.strip('_')

    with open(log_file, mode='a', encoding='utf-8') as log:
        # log.write('num;mdl;fold;mdl_score;auc_macro;auc_micro;auc_wght;f1_macro;f1_micro;'
        #           'f1_wght;tst_score;model_columns;exclude_columns;cat_columns;comment\n')
        log.write(f'{max_num};{prf};{idx_fold};{m_score:.6f};{auc_macro:.6f};{auc_micro:.6f};'
                  f'{auc_wght:.6f};{f1_macro:.6f};{f1_micro:.6f};{f1_wght:.6f};{score:.6f};'
                  f'{model_columns};{exclude_columns};{cat_columns};{comment}\n')


def merge_submits(max_num=0, submit_prefix='cb_', num_folds=5, exclude_folds=None,
                  use_proba=False, post_fix=''):
    """
    Объединение сабмитов
    :param max_num: номер итерации модели или список файлов, или список номеров сабмитов
    :param submit_prefix: префикс сабмита модели
    :param num_folds: количество фолдов модели для объединения
    :param exclude_folds: список списков для исключения фолдов из объединения:
                          длина списка exclude_folds должна быть равна длине списка max_num
    :param use_proba: использовать файлы с предсказаниями вероятностей
    :param post_fix: постфикс для регрессии
    :return: None
    """
    if use_proba:
        prob = '_proba'
    else:
        prob = ''

    # Список для хранения предсказаний
    predictions_list = []
    # Читаем каждый файл и добавляем его содержимое в список predictions_list
    if isinstance(max_num, int):
        for nfld in range(1, num_folds + 1):
            submit_csv = f'{submit_prefix}submit{prob}_{max_num:03}_{nfld}{LOCAL_FILE}{post_fix}.csv'
            df = pd.read_csv(PREDICTIONS_DIR.joinpath(submit_csv), index_col='file_name')

            if not predictions_list:
                df_columns = df.columns.tolist()
                df_index = df.index

            predictions_list.append(df.values)

        max_num = f'{max_num:03}'

    elif isinstance(max_num, (list, tuple)) and exclude_folds is None:
        for idx, file in enumerate(sorted(max_num)):
            df = pd.read_csv(PREDICTIONS_DIR.joinpath(file), index_col='file_name')

            if not predictions_list:
                df_columns = df.columns.tolist()
                df_index = df.index

            predictions_list.append(df.values)

        max_num = '-'.join(sorted(re.findall(r'\d{3,}(?:_\d)?', ' '.join(max_num)), key=int))

    elif isinstance(max_num, (list, tuple)) and isinstance(exclude_folds, (list, tuple)):
        # Список для хранения предсказаний
        predictions_list, str_nums = [], []
        for idx, (num, exc_folds) in enumerate(zip(max_num, exclude_folds), 1):
            str_num = str(num)
            for file in PREDICTIONS_DIR.glob(f'*submit{prob}_{num}_*.csv'):
                pool = re.findall(r'(?:(?<=\d{3}_)|(?<=\d{4}_))\d(?:(?=_local)|(?=\.csv))',
                                  file.name)
                if pool and int(pool[0]) not in exc_folds:
                    str_num += f'_{pool[0]}'
                    suffix = f'_{idx}_{pool[0]}'
                    df = pd.read_csv(file, index_col='file_name')

                    if not predictions_list:
                        df_columns = df.columns.tolist()
                        df_index = df.index

                    predictions_list.append(df.values)

            str_nums.append(str_num)
        max_num = '-'.join(sorted(str_nums))
        # print(df)
        print(max_num)

    # Определяем количество фолдов и целевых переменных
    num_folds = len(predictions_list)  # Количество файлов = количество фолдов
    num_samples, num_targets = predictions_list[0].shape  # Количество строк и колонок
    # Создаем трехмерный массив
    predictions_array = np.array(predictions_list).transpose(1, 0, 2)

    # df.to_excel(WORK_PATH.joinpath(f'{submit_prefix}submit_{max_num}{LOCAL_FILE}.xlsx'))

    if use_proba:
        # Нахождение среднего по строкам
        mode_result = (predictions_array.mean(axis=1) >= 0.5).astype(int)
    else:
        if not post_fix:
            # Вычисляем моду вдоль оси фолдов (axis=1)
            mode_result, _ = mode(predictions_array, axis=1)
        else:
            # Нахождение среднего по строкам
            mode_result = predictions_array.mean(axis=1)
            # # Нахождение медианы по строкам
            # mode_result = np.median(predictions_array, axis=1)

    submits = pd.DataFrame(data=mode_result, columns=df_columns, index=df_index)

    binary_columns = ['Некачественное ГДИС', 'Влияние ствола скважины', 'Радиальный режим',
                      'Линейный режим', 'Билинейный режим', 'Сферический режим',
                      'Граница постоянного давления', 'Граница непроницаемый разлом']
    binary_columns += ['bq', 'wb', 'ra', 'li', 'bi', 'sp', 'pc', 'im']

    for col in binary_columns:
        if col in submits.columns:
            submits[col] = submits[col].astype(int)

    submits_csv = f'{submit_prefix}submit_{max_num}{LOCAL_FILE}{prob}{post_fix}.csv'
    submits.to_csv(PREDICTIONS_DIR.joinpath(submits_csv))
    return submits_csv


def make_predict(idx_fold, model, datasets, max_num=0, submit_prefix='cb_', label_enc=None):
    """Предсказание для тестового датасета.
    Расчет метрик для модели: roc_auc и взвешенная F1-мера на валидации
    :param idx_fold: номер фолда при обучении
    :param model: обученная модель
    :param datasets: кортеж с тренировочной, валидационной и полной выборками
    :param max_num: максимальный порядковый номер обучения моделей
    :param submit_prefix: префикс для файла сабмита для каждой модели свой
    :param label_enc: используемый label_encоder для target'а
    :return: разные roc_auc и F1-мера
    """
    X_train, X_valid, y_train, y_valid, train, target, test_df, model_columns = datasets

    features2drop = ['hq', 'labels']

    test = test_df[model_columns].drop(columns=features2drop, errors='ignore').copy()

    # print('X_train.shape', X_train.shape)
    # print('train.shape', train.shape)
    # print('test.shape', test.shape)

    # постфикс если было обучение на отдельных фолдах
    nfld = f'_{idx_fold}' if idx_fold else ''

    predict_valid = model.predict(X_valid)
    predict_test = model.predict(test)

    predict_proba_classes = model.classes_
    print('predict_proba_classes:', predict_proba_classes)

    if label_enc:
        # преобразование обратно меток классов
        # predict_valid = label_enc.inverse_transform(predict_valid.reshape(-1, 1))
        predict_test = label_enc.inverse_transform(predict_test.reshape(-1, 1))
        predict_proba_classes = label_enc.inverse_transform(
            predict_proba_classes.reshape(-1, 1)).flatten()

    try:
        valid_proba = model.predict_proba(X_valid)
        predict_proba = model.predict_proba(test)
        if isinstance(valid_proba, list):
            # Преобразуем список массивов в 3D numpy-массив (8, :, 2)
            valid_proba = np.array(valid_proba)
            valid_proba = np.column_stack([arr[:, 1] for arr in valid_proba])  # (:, 8)
            predict_proba = np.array(predict_proba)
            predict_proba = np.column_stack([arr[:, 1] for arr in predict_proba])  # (500, 8)
    except:
        valid_proba = predict_valid
        predict_proba = predict_test

    valid_proba = np.nan_to_num(valid_proba, nan=0.0)
    predict_proba = np.nan_to_num(predict_proba, nan=0.0)

    print('proba.shape', valid_proba.shape, predict_proba.shape)

    # Сохранение предсказаний в файл
    submit_csv = f'{submit_prefix}submit_{max_num:03}{nfld}{LOCAL_FILE}.csv'
    file_submit_csv = PREDICTIONS_DIR.joinpath(submit_csv)
    # Преобразуем вероятности в бинарные метки (0 или 1)
    submission = pd.DataFrame(data=(predict_proba > 0.5).astype(int),
                              columns=y_train.columns,
                              index=test.index)
    submission.to_csv(file_submit_csv)

    # Сохранение вероятностей в файл
    submit_proba = f'{submit_prefix}submit_proba_{max_num:03}{nfld}{LOCAL_FILE}.csv'
    file_submit_proba = PREDICTIONS_DIR.joinpath(submit_proba)
    # Создаём DataFrame
    submission_proba = pd.DataFrame(data=predict_proba,
                                    columns=y_train.columns,
                                    index=test.index)
    submission_proba.to_csv(file_submit_proba)

    t_score = 0

    start_item = print_msg("Расчет ROC AUC...")
    # Для многоклассового ROC AUC, нужно указать multi_class
    auc_macro = roc_auc_score(y_valid, valid_proba, multi_class='ovr', average='macro')
    auc_micro = roc_auc_score(y_valid, valid_proba, multi_class='ovr', average='micro')
    auc_wght = roc_auc_score(y_valid, valid_proba, multi_class='ovr', average='weighted')
    print(f"auc_macro: {auc_macro:.6f}, auc_micro: {auc_micro:.6f}, auc_wght: {auc_wght:.6f}")
    print_time(start_item)

    start_item = print_msg("Расчет F1-score...")
    f1_macro = f1_micro = f1_wght = 0
    try:
        f1_macro = f1_score(y_valid, predict_valid, average='macro')
        f1_micro = f1_score(y_valid, predict_valid, average='micro')
        f1_wght = f1_score(y_valid, predict_valid, average='weighted')
    except:
        pass
    print(f'F1- f1_macro: {f1_wght:.6f}, f1_micro: {f1_wght:.6f}, f1_wght: {f1_wght:.6f}')
    print_time(start_item)

    model_score = 0

    try:
        if model.__class__.__name__ == 'CatBoostClassifier':
            eval_metric = model.get_params()['eval_metric']
            model_score = model.best_score_['validation'][eval_metric]
        elif model.__class__.__name__ == 'LGBMClassifier':
            model_score = model.best_score_['valid_0']['multi_logloss']
        elif model.__class__.__name__ == 'XGBClassifier':
            model_score = model.best_score
    except:
        pass

    return model_score, auc_macro, auc_micro, auc_wght, f1_macro, f1_micro, f1_wght, t_score


def make_predict_reg(idx_fold, model, datasets, max_num=0, submit_prefix='cb_'):
    """Предсказание для тестового датасета.
    Расчет метрик для модели: roc_auc и взвешенная F1-мера на валидации
    :param idx_fold: номер фолда при обучении
    :param model: обученная модель
    :param datasets: кортеж с тренировочной, валидационной и полной выборками
    :param max_num: максимальный порядковый номер обучения моделей
    :param submit_prefix: префикс для файла сабмита для каждой модели свой
    :return: разные roc_auc и F1-мера
    """
    X_train, X_valid, y_train, y_valid, train, target, test_df, model_columns = datasets

    features2drop = ['week', 'predict']

    test = test_df[model_columns].drop(columns=features2drop, errors='ignore').copy()

    # print('X_train.shape', X_train.shape)
    # print('train.shape', train.shape)
    # print('test.shape', test.shape)

    # постфикс если было обучение на отдельных фолдах
    nfld = f'_{idx_fold}' if idx_fold else ''

    predict_valid = model.predict(X_valid)
    predict_test = model.predict(test)

    predict_valid = np.nan_to_num(predict_valid, nan=0.0)
    predict_test = np.nan_to_num(predict_test, nan=0.0)

    # Заменяем значения меньше -33 на 0, т.к. пропуски были заполнены -99
    y_valid = y_valid.copy().values
    y_valid[y_valid < -33] = 0
    predict_valid[predict_valid < -33] = 0

    print(y_valid[:3, :].round(2))
    print(predict_valid[:3, :].round(2))
    print(predict_test[:3, :].round(2))

    binary_columns = ['Некачественное ГДИС', 'Влияние ствола скважины', 'Радиальный режим',
                      'Линейный режим', 'Билинейный режим', 'Сферический режим',
                      'Граница постоянного давления', 'Граница непроницаемый разлом']
    print(binary_columns)
    print(target.columns)

    # Сохранение предсказаний в файл
    submit_csv = f'{submit_prefix}submit_{max_num:03}{nfld}{LOCAL_FILE}_reg.csv'
    file_submit_csv = PREDICTIONS_DIR.joinpath(submit_csv)
    submission = pd.DataFrame(data=predict_test, columns=target.columns, index=test.index)
    submission = test[binary_columns].join(submission, how="left")
    # Постпроцессинг: зануляем значения, если binary_columns = 0
    for target, binary_col in zip(target.columns, binary_columns[1:]):
        submission.loc[submission[binary_col] == 0, target] = np.nan
    # new_cols = ['bq', 'wb', 'ra', 'li', 'bi', 'sp', 'pc', 'im']
    # submission.columns = new_cols + [f'{col}_d' for col in new_cols[1:]]
    submission.to_csv(file_submit_csv)

    t_score = 0

    # start_item = print_msg("Расчет scores...")
    # Root Mean Squared Error: квадратный корень среднеквадратичной ошибки
    # - среднее отклонение предсказанных значений от фактических значений без учета знака
    auc_macro = mean_squared_error(y_valid, predict_valid) ** 0.5
    # Mean Absolute Error: среднее абсолютное отклонение предсказанных значений от фактических
    auc_micro = mean_absolute_error(y_valid, predict_valid)
    # Mean Squared Error: средний квадрат отклонений предсказаний от фактических значений
    auc_wght = mean_squared_error(y_valid, predict_valid)
    # R² Score показывает, какую долю вариации зависимой переменной объясняет модель. Значение
    # R² варьируется от 0 до 1 (или может быть отрицательным, если модель плоха).
    # Значение 1 говорит о том, что модель идеально подходит к данным.
    f1_macro = r2_score(y_valid, predict_valid)
    # Explained Variance Score: метрика показывает долю вариации целевой переменной, которую
    # может объяснить модель. Значение равно 1, если модель идеально объясняет данные, и
    # 0, если модель не способна предсказать значение лучше, чем просто использовать среднее
    # значение целевой переменной.
    f1_wght = explained_variance_score(y_valid, predict_valid)
    # Mean Squared Logarithmic Error: измеряет среднюю разницу между логарифмами предсказанных
    # и фактических значений. Это полезно в случаях, когда хотим уменьшить влияние больших
    # ошибок, особенно при работе с экспоненциально растущими данными.
    # Заменяем значения меньше нуля на 0
    y_valid[y_valid < 0] = 0
    predict_valid[predict_valid < 0] = 0
    f1_micro = mean_squared_log_error(y_valid, predict_valid)
    # print_time(start_item)

    try:
        if 'CatBoost' in model.__class__.__name__:
            eval_metric = model.get_params()['eval_metric']
            model_score = model.best_score_['validation'][eval_metric]
        elif 'LGBM' in model.__class__.__name__:
            model_score = model.best_score_['valid_0']['rmse']
        elif 'XGB' in model.__class__.__name__:
            model_score = model.best_score
        else:
            model_score = auc_macro
    except:
        model_score = 0

    return model_score, auc_macro, auc_micro, auc_wght, f1_macro, f1_micro, f1_wght, t_score


import numpy as np
import pandas as pd


def custom_metric(y_true_bin, y_pred_bin, y_true_num, y_pred_num):
    """
    Расчёт сложной метрики по бинарным и численным признакам.

    Параметры:
    y_true_bin - DataFrame с истинными бинарными метками (0/1).
    y_pred_bin - DataFrame с предсказанными бинарными метками (0/1).
    y_true_num - DataFrame с истинными численными значениями.
    y_pred_num - DataFrame с предсказанными численными значениями.

    Возвращает:
    Долю True Positive предсказаний.
    """

    # Убеждаемся, что размеры данных совпадают
    assert y_true_bin.shape == y_pred_bin.shape, "Размеры бинарных меток не совпадают"
    assert y_true_num.shape == y_pred_num.shape, "Размеры численных меток не совпадают"

    # Имена столбцов
    binary_columns = y_true_bin.columns
    numeric_columns = y_true_num.columns

    # Инициализация счётчиков
    true_positives = 0
    total = 0

    # Проверяем бинарные метки
    for col_bin, col_num in zip(binary_columns, numeric_columns):
        # Берём только строки, где истинный бинарный признак = 1
        mask = (y_true_bin[col_bin] == 1)

        if mask.sum() == 0:  # Если нет ни одного случая с "1" - пропускаем
            continue

        # Достаём соответствующие значения
        true_values = y_true_num.loc[mask, col_num]
        pred_values = y_pred_num.loc[mask, col_num]

        # Проверяем критерии True Positive
        if col_bin in ['Влияние ствола скважины', 'Радиальный режим', 'Линейный режим',
                       'Билинейный режим']:
            # Проверяем коэффициент b
            correct_preds = np.abs(pred_values - true_values) <= 0.15
        else:
            # Проверяем логарифмическое отклонение
            correct_preds = np.abs(np.log10(pred_values) - np.log10(true_values)) <= 0.15

        # Считаем TP и общее число примеров
        true_positives += correct_preds.sum()
        total += len(correct_preds)

    # Итоговая метрика (если нет примеров, возвращаем 0)
    return true_positives / total if total > 0 else 0.0


if __name__ == "__main__":
    border_count = 254  # для кетбуста на ГПУ

    # Чтение и предобработка данных
    data_cls = DataTransform(use_catboost=True,
                             category_columns=[],
                             drop_first=False,
                             # numeric_columns=numeric_columns, scaler=StandardScaler,
                             )

    data_cls.make_log10_features = True

    # data_cls.files_for_train = None
    #
    # if data_cls.files_for_train:
    #     train_df, test_df, td_train, td_test = data_cls.preprocess_data(remake_file=True,
    #                                                                     use_joblib=True,
    #                                                                     )
    #     print(train_df.columns)
    # else:
    #     test_df, td_test = data_cls.preprocess_data(remake_file=True,
    #                                                 use_joblib=True,
    #                                                 )
    # print(test_df.columns)

    # train_df, test_df = data_cls.make_agg_data(remake_file=False,
    #                                            use_joblib=True,
    #                                            use_featuretools=True,
    #                                            )

    merge_submits(max_num=['lg_submit_040_1_local_reg.csv',
                           'lg_submit_040_2_local_reg.csv',
                           ],
                  submit_prefix='lg_', post_fix='_reg')
