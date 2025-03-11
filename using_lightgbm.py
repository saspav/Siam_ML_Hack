import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

import optuna
from optuna.integration import LightGBMPruningCallback
from lightgbm import LGBMClassifier, Dataset

from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, classification_report, auc

from data_process_sml import (RANDOM_SEED, MODEL_PATH, get_max_num, DataTransform,
                              set_all_seeds, make_predict, add_info_to_log, merge_submits)

from print_time import print_time, print_msg

__import__("warnings").filterwarnings('ignore')

set_all_seeds(seed=RANDOM_SEED)


# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial: optuna.Trial) -> float:
    """Оптимизируем гиперпараметры для MultiOutputClassifier"""

    # Гиперпараметры LGBM
    params = {
        "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart", "goss"]),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        "n_estimators": 1000,
        "random_state": RANDOM_SEED,
        "verbose": -1,
        "device": "gpu"
    }

    # Добавляем параметры bagging только для GBDT и DART
    if params["boosting_type"] in ["gbdt", "dart"]:
        params["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
        params["subsample_freq"] = trial.suggest_int("subsample_freq", 1, 7)

    loss_function = 'multiclass'  # Используем многоклассовую классификацию
    eval_metric = 'auc'

    # Базовая модель
    base_model = LGBMClassifier(loss_function=loss_function,
                                eval_metric=eval_metric,
                                random_seed=RANDOM_SEED,
                                # device='gpu',  # Используем GPU (если поддерживается)
                                # verbose=-1,
                                **params)

    # MultiOutputClassifier для многометочной классификации
    clf = MultiOutputClassifier(base_model)
    clf.fit(X_train, y_train)

    # Предсказания
    y_pred = clf.predict_proba(X_valid)

    # Рассчитываем средний ROC-AUC по всем выходам
    scores = []
    for i in range(y_train.shape[1]):
        # Берем вероятности класса 1
        scores.append(roc_auc_score(y_valid[:, i], y_pred[i][:, 1]))

    mean_auc = sum(scores) / len(scores)  # Усредняем метрики

    return mean_auc  # Оптимизируем средний ROC-AUC


max_num = get_max_num()
sub_pref = 'lg_'
start_time = print_msg('Обучение LGBMClassifier ...')

numeric_columns = []

targets = ['Некачественное ГДИС', 'Влияние ствола скважины', 'Радиальный режим',
           'Линейный режим', 'Билинейный режим', 'Сферический режим',
           'Граница постоянного давления', 'Граница непроницаемый разлом']

# targets = [clean_column_name(col) for col in targets]

cat_columns = []

# Чтение и предобработка данных
data_cls = DataTransform(use_catboost=True,
                         category_columns=cat_columns,
                         drop_first=False,
                         # numeric_columns=numeric_columns, scaler=StandardScaler,
                         )

data_cls.make_log10_features = True

# train_df, test_df, td_train, td_test = data_cls.preprocess_data(remake_file=False)

train_df, test_df = data_cls.make_agg_data(use_featuretools=True, remake_file=False, )

# train_df = train_df.sample(n=200, random_state=RANDOM_SEED)

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

features2drop = ['hq', 'labels']

exclude_columns = ['Влияние ствола скважины_details', 'Радиальный режим_details',
                   'Линейный режим_details', 'Билинейный режим_details',
                   'Сферический режим_details', 'Граница постоянного давления_details',
                   'Граница непроницаемый разлом_details',
                   # 'count_rows',
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
target = train_df[targets]
test_df = test_df[model_columns].copy()

target_encoder = None

# # таргет придется перевести из строкового типа в целочисленный
# target_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
# target = target_encoder.fit_transform(train_df[["target_class"]])

print('train.shape', train.shape, 'пропусков:', train.isna().sum().sum())
print('test.shape', test_df.drop(columns=features2drop, errors='ignore').shape,
      'пропусков:', test_df.isna().sum().sum())

# exit()

# Подготовка категориальных признаков (другой метод, чем использованный в лекции)
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

train[cat_columns] = encoder.fit_transform(train[cat_columns])
test_df[cat_columns] = encoder.transform(test_df[cat_columns])

# Указание категориальных признаков (если есть)
categorical_features = [train.columns.get_loc(col)
                        for col in cat_columns] if cat_columns else None

test_sizes = (0.2,)
# test_sizes = (0.4,)
# test_sizes = (0.2, 0.25)
# test_sizes = np.linspace(0.3, 0.4, 3)
# test_sizes = np.linspace(0.25, 0.35, 3)
# for num_iters in range(500, 701, 50):
# for SEED in range(100):
for test_size in test_sizes:

    max_num += 1

    # test_size = 0.3

    # num_iters = 7000
    # SEED = 17

    num_folds = 5

    test_size = round(test_size, 2)

    print(f'valid_size: {test_size} SEED={RANDOM_SEED}')

    # stratified = None
    stratified = 'labels'

    # Разделение на обучающую и валидационную выборки

    X_train, X_valid, y_train, y_valid = train_test_split(
        train, target,
        test_size=test_size,
        stratify=train_df[stratified] if len(train_df) > 40_000 else None,
        random_state=RANDOM_SEED)

    splited = X_train, X_valid, y_train, y_valid

    print('X_train.shape', X_train.shape, 'пропусков:', X_train.isna().sum().sum())
    print('X_valid.shape', X_valid.shape, 'пропусков:', X_valid.isna().sum().sum())

    pool_train = Dataset(data=X_train, label=y_train, categorical_feature=cat_columns)
    pool_valid = Dataset(data=X_valid, label=y_valid, categorical_feature=cat_columns)

    skf = StratifiedKFold(n_splits=num_folds, random_state=RANDOM_SEED, shuffle=True)
    split_kf = KFold(n_splits=num_folds, random_state=RANDOM_SEED, shuffle=True)

    fit_on_full_train = False
    use_grid_search = False
    use_cv_folds = True
    build_model = True
    write_log = True

    models, models_scores, predicts = [], [], []

    loss_function = 'multiclass'  # Используем многоклассовую классификацию
    eval_metric = 'auc'

    iterations = 2_000

    clf_params = dict(cat_feature=categorical_features,
                      # objective=loss_function,
                      eval_metric=eval_metric,
                      # num_class=8,
                      n_estimators=iterations,
                      # learning_rate=0.01,
                      # early_stopping_rounds=iterations // (10, 20)[iterations > 5_000],
                      random_seed=RANDOM_SEED,
                      device='gpu',  # Используем GPU (если поддерживается)
                      verbose=-1,
                      )

    base_model = LGBMClassifier(**clf_params)
    clf = MultiOutputClassifier(base_model)

    if use_grid_search:
        # Установить уровень логирования Optuna на WARNING
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Выполнить оптимизацию гиперпараметров
        study = optuna.create_study(
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
        )
        study.optimize(objective,
                       n_trials=100,
                       timeout=60 * 60,
                       show_progress_bar=True,
                       )

        print("Количество завершенных испытаний: {}".format(len(study.trials)))
        print("Лучшее испытание:")
        trial = study.best_trial
        print("  Значение: {}".format(trial.value))
        print("  Параметры: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        best_params = trial.params

        clf_params.update(best_params)
        if (clf_params.get('boosting_type', '') == 'Ordered'
                or clf_params.get('bootstrap_type', '') == 'MVS'):
            clf_params['task_type'] = 'CPU'

        print('clf_params', clf_params)

        clf = LGBMClassifier(**clf_params)

    info_cols = (model_columns, exclude_columns, cat_columns)

    comment = {}
    comment.update({'test_size': test_size,
                    'SEED': RANDOM_SEED,
                    'stratified': stratified,
                    })
    comment.update(data_cls.comment)

    if use_cv_folds:
        comment['num_folds'] = num_folds

        if stratified:
            skf_folds = skf.split(train, train_df[stratified])
        else:
            skf_folds = split_kf.split(train)

        for idx, (train_idx, valid_idx) in enumerate(skf_folds, 1):
            print(f'Фолд {idx} из {num_folds}')

            # подсчет статистик на трейне и заполнение их на валидации
            X_train, X_valid = train.iloc[train_idx], train.iloc[valid_idx]
            y_train, y_valid = target.iloc[train_idx], target.iloc[valid_idx]

            splited = X_train, X_valid, y_train, y_valid

            train_data = Dataset(data=X_train, label=y_train, categorical_feature=cat_columns)
            valid_data = Dataset(data=X_valid, label=y_valid, categorical_feature=cat_columns)

            base_model = LGBMClassifier(**clf_params)
            clf = MultiOutputClassifier(base_model)

            clf.fit(X_train, y_train)

            models.append(clf)

            save_time = print_msg('Сохранение моделей ...')
            with open(MODEL_PATH.joinpath(f'lightgbm_{max_num:03}.pkl'), 'wb') as file:
                joblib.dump((models, info_cols), file, compress=7)
            print_time(save_time)

            if build_model:
                DTS = (*splited, train, target, test_df, model_columns)
                valid_scores = make_predict(idx, clf, DTS, max_num, submit_prefix=sub_pref,
                                            label_enc=target_encoder)
                models_scores.append(valid_scores)

                try:
                    comment['clf_iters'] = clf.best_iteration_
                except AttributeError:
                    pass

                add_info_to_log(sub_pref, max_num, idx, clf, valid_scores, info_cols, comment)

        if build_model:
            # best_iterations = {'iterations': [clf.best_iteration_ for clf in models]}
            # comment.update(best_iterations)

            # объединение сабмитов
            merge_submits(max_num=max_num, submit_prefix=sub_pref, num_folds=num_folds)
            merge_submits(max_num=max_num, submit_prefix=sub_pref, num_folds=num_folds,
                          use_proba=True)

    else:
        DTS = (*splited, train, target, test_df, model_columns)

        clf.fit(X_train, y_train)

        models.append(clf)

        if build_model:
            DTS = (*splited, train, target, test_df, model_columns)
            valid_scores = make_predict(0, clf, DTS, max_num, submit_prefix=sub_pref,
                                        label_enc=target_encoder)
            models_scores.append(valid_scores)

    print('best_params:', clf.get_params())

    if build_model:
        if len(models) > 1:
            valid_scores = [np.mean(arg) for arg in zip(*models_scores)]
            try:
                clf_iters = [clf.best_iteration_ for clf in models]
            except AttributeError:
                clf_iters = []
            clf_lr = [round(clf.get_params().get('learning_rate', 0), 8)
                      for clf in models]
        else:
            try:
                clf_iters = models[0].best_iteration_
            except AttributeError:
                clf_iters = ''
            clf_lr = round(models[0].get_params().get('learning_rate', 0), 8)

        save_time = print_msg('Сохранение моделей ...')
        with open(MODEL_PATH.joinpath(f'lightgbm_{max_num:03}.pkl'), 'wb') as file:
            joblib.dump((models, info_cols), file, compress=7)
        print_time(save_time)

        comment['clf_iters'] = clf_iters

        add_info_to_log(sub_pref, max_num, 0, models[0], valid_scores, info_cols, comment)

print_time(start_time)
