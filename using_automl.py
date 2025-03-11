import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# LightAutoML presets, task and report generation
from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
# from lightautoml.automl.presets.whitebox_presets import WhiteBoxPreset
from lightautoml.tasks import Task
from lightautoml.report.report_deco import ReportDeco

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, log_loss
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
from sklearn.metrics import mean_squared_log_error, mean_absolute_error

from data_process_sml import (RANDOM_SEED, PREDICTIONS_DIR, get_max_num, DataTransform,
                              MODELS_LOGS, MODELS_LOGS_REG, MODEL_PATH, add_info_to_log)

from print_time import print_time, print_msg

from set_all_seeds import set_all_seeds

__import__("warnings").filterwarnings('ignore')


def acc_score(y_true, y_pred, **kwargs):
    # print("y_true.shape", y_true.shape, "y_pred.shape", y_pred.shape)
    return accuracy_score(y_true, (y_pred >= 0.5).astype(int), **kwargs)


def f1_metric(y_true, y_pred, **kwargs):
    return f1_score(y_true, (y_pred >= 0.5).astype(int), **kwargs)


set_all_seeds(seed=RANDOM_SEED)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

start_time = print_msg('Обучение TabularAutoML ...')

max_num_bin = get_max_num(log_file=MODELS_LOGS) + 1
max_num_reg = get_max_num(log_file=MODELS_LOGS_REG) + 1

sub_pref = 'la_'

numeric_columns = []

binary_targets = ['Некачественное ГДИС', 'Влияние ствола скважины', 'Радиальный режим',
                  'Линейный режим', 'Билинейный режим', 'Сферический режим',
                  'Граница постоянного давления', 'Граница непроницаемый разлом']

numeric_targets = ['Влияние ствола скважины_details', 'Радиальный режим_details',
                   'Линейный режим_details', 'Билинейный режим_details',
                   'Сферический режим_details', 'Граница постоянного давления_details',
                   'Граница непроницаемый разлом_details']

# binary_targets = [clean_column_name(col) for col in binary_targets]

cat_columns = []

# Чтение и предобработка данных
data_cls = DataTransform(use_catboost=True,
                         category_columns=cat_columns,
                         drop_first=False,
                         # numeric_columns=numeric_columns, scaler=StandardScaler,
                         )

data_cls.make_log10_features = True

train_df, test_df = data_cls.make_agg_data(use_featuretools=True,
                                           file_with_target_class=None)

# train_df = train_df.sample(n=300, random_state=RANDOM_SEED)

# # Добавление группировок от таргет-енкодинга
# train_df = data_cls.fit_transform(train_df)
# test_df = data_cls.transform(test_df)

features2drop = ['hq', 'labels']

exclude_columns = ['count_rows',
                   ]

# exclude_columns = [clean_column_name(col) for col in exclude_columns]

exclude_columns.extend(data_cls.exclude_columns)

model_columns = test_df.columns.to_list()

# Добавим в категориальные признаки те, что были посчитаны как мода
cat_columns.extend([col for col in model_columns if col.upper().startswith('MODE_')])

model_columns = [col for col in model_columns if col not in exclude_columns]
cat_columns = [col for col in cat_columns if col in model_columns]

exclude_columns = features2drop + exclude_columns

print('Обучаюсь на колонках:', model_columns)
print('Категорийные колонки:', cat_columns)
print('Исключенные колонки:', exclude_columns)

print(f'Размер train_df = {train_df.shape}, test = {test_df.shape}')

train = train_df[model_columns].drop(columns=features2drop, errors='ignore')
target = train_df[binary_targets + numeric_targets]
test_df = test_df[model_columns].copy()

print('train.shape', train.shape, 'пропусков:', train.isna().sum().sum())
print('test.shape', test_df.drop(columns=features2drop, errors='ignore').shape,
      'пропусков:', test_df.isna().sum().sum())

# # Добавление группировок от таргет-енкодинга
# train_df = data_cls.fit_transform(train_df)
# test_df = data_cls.transform(test_df)

# for label_len in (5, 3, 1):
#     vc = train_df['labels'].value_counts()
#     # Получим метки, которые встречаются один раз
#     unique_labels = vc[vc == 1].index
#     # Преобразуем метки, которые встречаются один раз
#     train_df['labels'] = train_df['labels'].apply(
#         lambda x: x[:label_len] if x in unique_labels else x)

# vc = train_df['labels'].value_counts()
# vc.to_excel(WORK_PATH.joinpath('vc.xlsx'))
# print(vc)
# exit()

if __name__ == '__main__':

    test_size = 0.2

    print(f'valid_size: {test_size} SEED={RANDOM_SEED}')

    # stratified = None
    stratified = 'labels'

    # Разделение на обучающую и валидационную выборки
    X_train, X_valid, y_train, y_valid = train_test_split(train, target,
                                                          test_size=test_size,
                                                          stratify=train_df[stratified],
                                                          random_state=RANDOM_SEED)

    splited = X_train, X_valid, y_train, y_valid

    print('X_train.shape', X_train.shape, 'пропусков:', X_train.isna().sum().sum())
    print('X_valid.shape', X_valid.shape, 'пропусков:', X_valid.isna().sum().sum())

    build_model = True
    write_log = True

    N_THREADS = 28  # threads cnt for lgbm and linear models
    G_MEMORY = 64  #
    N_FOLDS = 5  # folds cnt for AutoML
    TIMEOUT = 60 * 60  # Time in seconds for automl run

    full_submit = pd.DataFrame(index=test_df.index)
    models_scores = []
    models_to_save = []

    for idx_target in (0, 1):

        if idx_target:
            binary_task = False
            targets = y_train[numeric_targets].copy()
            roles = {'target': numeric_targets}
            task = Task('multi:reg')
            # Добавим предсказанные бинарные признаки для обучения регрессии
            X_train = X_train.join(y_train[binary_targets], how="left")
            test_df = test_df.join(full_submit[binary_targets], how="left")
            model_columns.extend(binary_targets)
        else:
            binary_task = True
            targets = y_train[binary_targets].copy()
            roles = {'target': binary_targets}
            task = Task('multilabel')

        train_with_target = X_train.join(targets, how="left")

        RD = ReportDeco(output_path='tabularAutoML_model_report')

        # Настройка автоматической модели
        automl = TabularAutoML(task=task,
                               timeout=TIMEOUT,
                               cpu_limit=N_THREADS,
                               memory_limit=G_MEMORY,
                               general_params={
                                   # 'use_algos': [[
                                   #     'rf',
                                   #     'lgb',
                                   #     'lgb_tuned',
                                   #     'cb_tuned',
                                   #     # 'linear_l2',
                                   #     'xgb_tuned',
                                   # ],
                                   #     [
                                   #         'cb_tuned',
                                   #         # 'rf',
                                   #         # 'linear_l2',
                                   #     ],
                                   # ],

                                   # 'use_algos': [[
                                   #     'rf',
                                   #     'lgb',
                                   #     'cb', ],
                                   # ],

                                   'use_algos': 'auto',

                                   'tuning_params': {
                                       # Время для подбора гиперпараметров (в секундах)
                                       'max_tuning_time': 600,
                                       # Максимальное количество итераций подбора
                                       'max_tuning_iter': 88,
                                   },

                                   # 'custom_params': custom_params,
                               },
                               reader_params={'n_jobs': N_THREADS,
                                              'cv': N_FOLDS,
                                              'random_state': RANDOM_SEED},
                               )

        # automl = RD(automl)

        oof_pred = automl.fit_predict(train_with_target, roles=roles, verbose=1)
        valid_proba = oof_pred.data

        print('oof_pred:\n{}\nShape = {}'.format(oof_pred[:7], oof_pred.shape))

        test_pred = automl.predict(test_df)
        predict_test = test_pred.data

        valid_proba = np.nan_to_num(valid_proba, nan=0.0)
        predict_test = np.nan_to_num(predict_test, nan=0.0)

        models_to_save.append(automl)

        print('Check scores...')

        y_valid = targets.values

        if binary_task:
            predict_valid = (valid_proba >= 0.5).astype(int)
            predict_test = (predict_test >= 0.5).astype(int)
            model_score = acc_score(y_valid, valid_proba)
        else:
            # Заменяем значения меньше -33 на 0, т.к. пропуски были заполнены -99
            y_valid[y_valid < -33] = 0
            valid_proba[valid_proba < -33] = 0
            predict_valid = valid_proba
            model_score = mean_squared_error(y_valid, predict_valid)

        print('OOF score: {}'.format(model_score))

        submission = test_df[test_df.columns.to_list()[:2]].copy()
        submission[targets.columns] = predict_test
        full_submit[targets.columns] = predict_test

        # Сохранение предсказаний в файл
        max_num = max_num_bin
        post_fix = ''
        if not binary_task:
            post_fix = '_reg'
            max_num = max_num_reg
        submit_csv = f'{sub_pref}submit_{max_num:03}_local{post_fix}.csv'

        submission[targets.columns].to_csv(PREDICTIONS_DIR.joinpath(submit_csv))

        print(automl.create_model_str_desc())

        info_cols = (model_columns, exclude_columns, cat_columns)

        comment = {}
        comment.update({'SEED': RANDOM_SEED,
                        'binary_task': binary_task,
                        })

        t_score = auc_macro = auc_micro = auc_wght = f1_macro = f1_micro = f1_wght = 0

        if binary_task:
            try:
                # Для многоклассового ROC AUC, нужно указать multi_class
                auc_macro = roc_auc_score(y_valid, valid_proba, multi_class='ovr',
                                          average='macro')
                auc_micro = roc_auc_score(y_valid, valid_proba, multi_class='ovr',
                                          average='micro')
                auc_wght = roc_auc_score(y_valid, valid_proba, multi_class='ovr',
                                         average='weighted')
                print(f"auc_macro: {auc_macro:.6f}, auc_micro: {auc_micro:.6f}, "
                      f"auc_wght: {auc_wght:.6f}")
            except:
                pass

            try:
                f1_macro = f1_score(y_valid, predict_valid, average='macro')
                f1_micro = f1_score(y_valid, predict_valid, average='micro')
                f1_wght = f1_score(y_valid, predict_valid, average='weighted')
                print(f'F1- f1_macro: {f1_wght:.6f}, f1_micro: {f1_wght:.6f}, '
                      f'f1_wght: {f1_wght:.6f}')
            except:
                pass

        else:
            try:
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
            except:
                pass
            try:
                y_valid[y_valid < 0] = 0
                predict_valid[predict_valid < 0] = 0
                f1_micro = mean_squared_log_error(y_valid, predict_valid)
            except:
                pass

        valid_scores = (model_score, auc_macro, auc_micro, auc_wght,
                        f1_macro, f1_micro, f1_wght, t_score)

        if binary_task:
            add_info_to_log(f'{sub_pref}', max_num_bin, 0, None, valid_scores,
                            info_cols, comment, log_file=MODELS_LOGS)
        else:
            add_info_to_log(f'{sub_pref}', max_num_reg, 0, None, valid_scores,
                            info_cols, comment, log_file=MODELS_LOGS_REG)

    # Постпроцессинг: зануляем значения, если binary_targets = 0
    for target, binary_col in zip(numeric_targets, binary_targets[1:]):
        full_submit.loc[full_submit[binary_col] == 0, target] = np.nan
    # Сохранение предсказаний в файл
    submit_csv = f'{sub_pref}submit_{max_num_reg:03}_reg.csv'
    full_submit.to_csv(PREDICTIONS_DIR.joinpath(submit_csv))

    save_time = print_msg('Сохранение моделей ...')
    with open(MODEL_PATH.joinpath(f'automl_{max_num_reg:03}_all.pkl'), 'wb') as file:
        joblib.dump((models_to_save, binary_targets, numeric_targets, info_cols), file,
                    compress=7)
    print_time(save_time)

    comment = {}
    comment.update({'SEED': RANDOM_SEED,
                    'target': f'Кол-во: {len(binary_targets)}',
                    })

    print_time(start_time)
