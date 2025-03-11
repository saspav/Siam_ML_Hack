import os
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from tqdm import tqdm

from data_process_sml import (RANDOM_SEED, LOCAL_FILE, MODEL_PATH, PREDICTIONS_DIR,
                              DataTransform, make_predict, make_predict_reg, merge_submits)

from print_time import print_time, print_msg

from set_all_seeds import set_all_seeds

__import__("warnings").filterwarnings('ignore')

set_all_seeds(seed=RANDOM_SEED)

if __name__ == '__main__':
    max_num = 1
    num_folds = 5
    max_num_bin = 52
    max_num_reg = 40
    info_cols = [''] * 3
    sub_pref = 'sp_'

    binary_targets = ['Некачественное ГДИС', 'Влияние ствола скважины', 'Радиальный режим',
                      'Линейный режим', 'Билинейный режим', 'Сферический режим',
                      'Граница постоянного давления', 'Граница непроницаемый разлом']

    numeric_targets = ['Влияние ствола скважины_details', 'Радиальный режим_details',
                       'Линейный режим_details', 'Билинейный режим_details',
                       'Сферический режим_details', 'Граница постоянного давления_details',
                       'Граница непроницаемый разлом_details']

    start_time = print_msg('Загрузка моделей ...')
    with open(MODEL_PATH.joinpath(f'lightgbm_{max_num_bin:03}.pkl'), 'rb') as in_file:
        models_bin, info_cols_bin = joblib.load(in_file)

    with open(MODEL_PATH.joinpath(f'lightgbm_{max_num_reg:03}_reg.pkl'), 'rb') as in_file:
        models_reg, info_cols_reg = joblib.load(in_file)
    print_time(start_time)

    model_columns, exclude_columns, cat_columns = info_cols_bin

    # Чтение и предобработка данных - тут указать каталог, где лежат файлы
    test_data_path = 'Z:/python-datasets/Siam_ML_Hack/task-1/validation_1'

    data_cls = DataTransform(test_data_path=test_data_path)
    data_cls.files_for_train = None
    data_cls.preprocess_files = None
    data_cls.aggregate_path_file = None

    data_cls.make_log10_features = True

    test_df = data_cls.make_agg_data(use_featuretools=True, use_joblib=True,
                                     file_with_target_class=None)

    features2drop = ['hq', 'labels']

    for idx_fold, model in enumerate(models_bin, 1):

        test = test_df[model_columns].drop(columns=features2drop, errors='ignore').copy()

        # постфикс если было обучение на отдельных фолдах
        nfld = f'_{idx_fold}' if idx_fold else ''

        predict_test = model.predict(test)

        try:
            predict_proba = model.predict_proba(test)
            if isinstance(predict_proba, list):
                # Преобразуем список массивов в 3D numpy-массив (8, :, 2)
                predict_proba = np.array(predict_proba)
                predict_proba = np.column_stack([arr[:, 1] for arr in predict_proba])
        except:
            predict_proba = predict_test

        # Сохранение предсказаний в файл
        submit_csv = f'{sub_pref}submit_{max_num:03}{nfld}{LOCAL_FILE}.csv'
        file_submit_csv = PREDICTIONS_DIR.joinpath(submit_csv)
        # Преобразуем вероятности в бинарные метки (0 или 1)
        submission = pd.DataFrame(data=(predict_proba > 0.5).astype(int),
                                  columns=binary_targets,
                                  index=test.index)
        submission.to_csv(file_submit_csv)

    # объединение сабмитов бинарной классификации
    submit_bin = merge_submits(max_num=max_num, submit_prefix=sub_pref, num_folds=num_folds)

    model_columns, exclude_columns, cat_columns = info_cols_reg

    data_cls = DataTransform(test_data_path=test_data_path)
    data_cls.files_for_train = None
    data_cls.preprocess_files = None
    data_cls.aggregate_path_file = None

    data_cls.make_log10_features = True

    # Предобработка данных, но уже с учетом бинарных признаков
    test_df = data_cls.make_agg_data(use_featuretools=True, use_joblib=True,
                                     file_with_target_class=submit_bin)

    for idx_fold, model in enumerate(models_reg, 1):

        test = test_df[model_columns].drop(columns=features2drop, errors='ignore').copy()

        # постфикс если было обучение на отдельных фолдах
        nfld = f'_{idx_fold}' if idx_fold else ''

        predict_test = model.predict(test)

        # Сохранение предсказаний в файл
        submit_csv = f'{sub_pref}submit_{max_num:03}{nfld}{LOCAL_FILE}_reg.csv'
        file_submit_csv = PREDICTIONS_DIR.joinpath(submit_csv)
        submission = pd.DataFrame(data=predict_test, columns=numeric_targets,
                                  index=test.index)
        submission = test[binary_targets].join(submission, how="left")
        # Постпроцессинг: зануляем значения, если binary_columns = 0
        for target, binary_col in zip(numeric_targets, binary_targets[1:]):
            submission.loc[submission[binary_col] == 0, target] = np.nan
        # # Постпроцессинг
        # test['wb_value'] = test['wb_value'].map(lambda z: z if z != 0 else np.NaN)
        # test['Влияние ствола скважины'] = test['wb_value'].map(
        #     lambda z: 0 if pd.isna(z) else 1)
        # submission['Влияние ствола скважины'] = test['Влияние ствола скважины']
        # submission['Влияние ствола скважины_details'] = test['wb_value']
        submission.to_csv(file_submit_csv)

    # объединение сабмитов регрессии
    submit_reg = merge_submits(max_num=max_num, submit_prefix=sub_pref,
                               num_folds=num_folds, post_fix='_reg')

    print_time(start_time)

    print(f'\nФайл самбита "{submit_reg}" находится в каталоге: {PREDICTIONS_DIR}')

    print('P.S. Лучший скор получился при смешивании результатов 1 и 2 фолдов')
    submit_reg = merge_submits(
        max_num=[f'{sub_pref}submit_{max_num:03}_1{LOCAL_FILE}_reg.csv',
                 f'{sub_pref}submit_{max_num:03}_2{LOCAL_FILE}_reg.csv'],
        submit_prefix='lg_', post_fix='_reg')
    print(f'\nФайл самбита "{submit_reg}" находится в каталоге: {PREDICTIONS_DIR}')
