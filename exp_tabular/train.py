from flaml import AutoML, CFO, tune, RandomSearch
from collections import defaultdict
import pandas as pd
import argparse
import pickle
import sys
import os
import numpy as np
from ray import tune as raytune
import sys
from flaml.data import load_openml_dataset
import time
import random
import arff
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from flaml.group import set_test_group

# def get_group(train_len, group_num):
#     length = train_len
#     fold_length = int(length / group_num)
#     group = []
#     for i in range(group_num):
#         group += [i] * fold_length
#         if i == group_num-1 and len(group) != length:
#             group += [group_num-1] * (length - len(group))
#     group = np.array(group)
#     return group
# def get_tolerances(args):
#     '''
#     Get five tolerances from CFO results.
#     '''
#     folder_name = f'./out/{args.dataset}_size{args.size}_{args.estimator}_{args.metric}_b{str(args.budget)}'
#     csv_file = f'{folder_name}/cfo/valid.csv' #!! fixed to cfo/valid
#     valid_df = pd.read_csv(csv_file)
#     mean = np.mean(np.array(valid_df['val_loss']))
#     tolerances = [round(mean*0.002*i, 4) for i in range(1,6)]
#     print(tolerances)
#     with open(f'{folder_name}/toler.txt', 'w') as f:
#         f.write(str(tolerances))
#     return tolerances


def cv_score_agg_func(val_loss_folds, log_metrics_folds):
    ''' Used in get_trainable_cv
    '''
    val_loss_folds.reverse()
    metric_to_minimize = sum(val_loss_folds) / len(val_loss_folds)
    metrics_to_log = {}
    n = len(val_loss_folds)
    metrics_to_log = (
        {k: v / n for k, v in metrics_to_log.items()}
        if isinstance(metrics_to_log, dict)
        else metrics_to_log / n
    )
    metrics_to_log["std"] = np.std(val_loss_folds)
    metrics_to_log["worst"] = np.max(val_loss_folds)
    metrics_to_log["best"] = np.min(val_loss_folds)
    metrics_to_log["folds"] = val_loss_folds
    return  metric_to_minimize, metrics_to_log

def get_trainable_cv(X_train,
                y_train,
                seed,
                group_num, 
                group_value,
                estimator, 
                metric,
                task):
    ''' Only used in training with holdout.
    '''
    set_test_group()
    length = len(X_train)

    if group_value is None:
        fold_length = int(length / group_num)
        group = []
        for i in range(group_num):
            group += [i] * fold_length
            if i == group_num-1 and len(group) != length:
                group += [group_num-1] * (length - len(group))
        group = np.array(group)
    else:
        group = group_value

    automl = AutoML()
    automl._state.eval_only = True

    eval_method = 'cv' # pseudo_cv
    settings = {
        "time_budget": 600,
        "estimator_list": [
            estimator,
        ],
        "metric": metric, # check
        "use_ray": False,
        "task": task,
        "max_iter": 0,
        "keep_search_state": True,
        "log_training_metric": False,
        "verbose": 0,
        "eval_method": eval_method, 
        "mem_thres": 128 * (1024**3),
        "seed": seed,
        "split_type": "group",
        "groups": group,
        "n_splits": group_num,
        "cv_score_agg_func": cv_score_agg_func,
        "early_stop": True,
    }
    print(settings, flush=True)
    if eval_method != 'cv':
        print(f'Caution: use {eval_method} in train.py line 108.', flush=True)
    automl.fit(X_train=X_train, y_train=y_train, **settings)
    trainable = automl.trainable
    return trainable

def get_lexico_objectives(algorithm, tolerance):
    if algorithm == "lexico_var":
        lexico_objectives = {}
        lexico_objectives["metrics"] = ["val_loss", "std"]
        lexico_objectives["tolerances"] = {"val_loss": tolerance, "std": 0.0} # check
        lexico_objectives["targets"] = {"val_loss": 0.0, "std": 0.0}
        lexico_objectives["modes"] = ["min", "min"]
    elif algorithm == "hypertime":
        lexico_objectives = {}
        lexico_objectives["metrics"] = ["val_loss", "worst"]
        lexico_objectives["tolerances"] = {"val_loss": tolerance, "worst": 0.0} # check
        # lexico_objectives["metrics"] = ["worst", "val_loss"]
        # lexico_objectives["tolerances"] = {"worst": tolerance, "val_loss": 0.0} # check
        if lexico_objectives["metrics"][0] != 'val_loss':
            print(lexico_objectives["metrics"], "Caution, val_loss is not the first objective.", flush=True)
        lexico_objectives["targets"] = {"val_loss": 0.0, "worst": 0.0}
        lexico_objectives["modes"] = ["min", "min"]
    else:
        lexico_objectives = None
    
    return lexico_objectives

def get_search_space(train_len, estimator='lgbm'):
    upper = max(5, min(32768, int(train_len)))

    if estimator == 'lgbm':
        search_space = {
            'n_estimators': raytune.lograndint(4, upper),
            'num_leaves': raytune.lograndint(4, upper),
            "min_child_samples": raytune.lograndint(2, 2**7 + 1),
            "learning_rate": raytune.loguniform(1 / 1024, 1.0),
            "log_max_bin": raytune.lograndint(3, 11),
            "colsample_bytree": raytune.uniform(0.01, 1.0),
            "reg_alpha": raytune.loguniform(1 / 1024, 1024),
            "reg_lambda": raytune.loguniform(1 / 1024, 1024),
            "learner": "lgbm",
        }
        low_cost_partial_config = {
            "n_estimators": 4,
            "num_leaves": 4,
        }
        points_to_evaluate = [{'n_estimators': 4,
                            'num_leaves': 4,
                            "min_child_samples": 20,
                            "learning_rate": 0.1,
                            "log_max_bin": 8,
                            "colsample_bytree": 1.0,
                            "reg_alpha": 1 / 1024,
                            "reg_lambda": 1.0,
                            "learner": "lgbm",
                            }]
    elif estimator == 'xgboost':
        search_space = {
            'max_leaves': raytune.lograndint(4, upper),
            'max_depth': raytune.choice([0, 6, 12]),
            "n_estimators": raytune.lograndint(4, upper),
            "min_child_weight": raytune.loguniform(0.001, 128),
            "learning_rate": raytune.loguniform(1 / 1024, 1.0),
            "subsample": raytune.uniform(0.1, 1.0),
            "colsample_bytree": raytune.uniform(0.01, 1.0),
            "colsample_bylevel": raytune.uniform(0.01, 1.0),
            "reg_alpha": raytune.loguniform(1 / 1024, 1024),
            "reg_lambda": raytune.loguniform(1 / 1024, 1024),
            "learner": "xgboost",
        }
        low_cost_partial_config = {
            "n_estimators": 4,
            "max_leaves": 4,
        }
        points_to_evaluate = [{"n_estimators": 4,
                            "max_leaves": 4,
                            "max_depth": 0,
                            "min_child_weight": 1.0,
                            "learning_rate": 0.1,
                            "subsample": 1.0,
                            "colsample_bylevel": 1.0,
                            "colsample_bytree": 1.0,
                            "reg_alpha": 1 / 1024,
                            "reg_lambda": 1.0,
                            }]
    elif estimator == 'rf':
        # https://blog.dataiku.com/narrowing-the-search-which-hyperparameters-really-matter
        # https://crunchingthedata.com/max-depth-in-random-forests/
        search_space = {
            'learner': 'rf',
            'criterion': raytune.choice(['gini', 'entropy']),
            "n_estimators": raytune.lograndint(4, upper),
            'max_features': raytune.randint(1,10),
            'max_leaves': raytune.lograndint(4, upper),
            'min_samples_leaf': raytune.lograndint(1, 100),
            'max_depth': raytune.randint(1, 20),
        }
        low_cost_partial_config = {
            "n_estimators": 4,
            "max_leaves": 4,
            'min_samples_leaf': 50
        }
        points_to_evaluate = [{
            'criterion': 'gini',
            "n_estimators": 4,
            'max_features': 1,
            'max_leaves': 4,
            'min_samples_leaf': 50,
            'max_depth': 1,
        }]
        
    # elif estimator == 'catbooost':
    #     search_space = {
    #         'early_stopping_rounds': raytune.lograndint(4, upper),
    #         'learning_rate': raytune.choice([0, 6, 12]),
    #         "n_estimators": raytune.lograndint(4, upper),
    #     }
    #     low_cost_partial_config = {
    #         "n_estimators": 4,
    #     }
    #     points_to_evaluate = [{"n_estimators": 4,
    #             'early_stopping_rounds': raytune.lograndint(4, upper),
    #             'learning_rate': raytune.choice([0, 6, 12]),
    #                         }]

    else:
        raise ValueError("Estimator not found.")

    return search_space, low_cost_partial_config, points_to_evaluate

def mytune(seed, budget, trainable, local_dir, algorithm, tolerance, train_len, estimator):
    '''
    seed: seed,
    budget: tune time in seconds,
    trainable: a predefined function returned by get_trainable_cv
    local_dir: local_dir for flaml to output log
    algorithm: in ['cfo', 'lexico_var', 'hypertime']
    tolerance: None for cfo, has to be set for other two
    train_len: used to set upper = max(5, min(32768, int(train_len)))
    estimator: choose in ['lgbm', 'xgboost']
    '''
    search_space, low_cost_partial_config, points_to_evaluate = get_search_space(train_len, estimator)
    lexico_objectives = get_lexico_objectives(algorithm, tolerance)

    def evaluate_function(config):
        trainable_return = trainable(config)
        result = {}
        result["val_loss"] = trainable_return["val_loss"]
        result["std"] = trainable_return["metric_for_logging"]["std"]
        result["worst"] = trainable_return["metric_for_logging"]["worst"]
        result["folds"] = trainable_return["metric_for_logging"]["folds"]
        # print(result)
        return result

    if algorithm in ["lexico_var", "hypertime"]:
        if tolerance is None:
            raise ValueError('Have to set tolerance for lexico.')
        analysis = tune.run(
            evaluate_function,
            num_samples=-1,
            time_budget_s=budget,
            config=search_space,
            use_ray=False,
            lexico_objectives=lexico_objectives,
            low_cost_partial_config=low_cost_partial_config,
            alg_seed=seed,
            points_to_evaluate=points_to_evaluate,
            local_dir=local_dir,
            verbose=3,
        )
    elif algorithm == 'cfo':
        algo = CFO(
            space=search_space,
            metric="val_loss",
            mode="min",
            seed=seed,
            low_cost_partial_config=low_cost_partial_config,
            points_to_evaluate=points_to_evaluate,
        )
        analysis = tune.run(
            evaluate_function,
            time_budget_s=budget,
            search_alg=algo,
            use_ray=False,
            num_samples=-1,
            metric="val_loss",
            local_dir=local_dir,
            verbose=3,
        )
    else:
        print(algorithm)
        algo = RandomSearch(
                space=search_space,
                metric="val_loss",
                mode="min",
                seed=seed,)
        analysis = tune.run(
            evaluate_function,
            time_budget_s=budget,
            search_alg=algo,
            use_ray=False,
            num_samples=-1,
            metric="val_loss",
            local_dir=local_dir,
            verbose=3,
        )
    return analysis

def get_trainable_holdout(X_train,
                y_train,
                X_test,
                y_test,
                group_num,
                group_value,
                estimator, 
                task, 
                metric):
    ''' Only used in testing.
    '''
    length = len(X_train)
    set_test_group(group_value=group_value, group_num=group_num, train_len=length)

    automl = AutoML()
    automl._state.eval_only = True
    settings = {
        "time_budget": 0,
        "estimator_list": [
            estimator,
        ],
        "metric": metric,
        "use_ray": False,
        "task": task,
        "max_iter": 0,
        "keep_search_state": True,
        "log_training_metric": False,
        "verbose": 0,
        "eval_method": "holdout",
        "mem_thres": 128 * (1024**3),
        "seed": 1,
        'early_stop': True,
        # "cv_score_agg_func": cv_score_agg_func,
    }
    automl.fit(X_train=X_train, y_train=y_train, X_val=X_test, y_val=y_test, **settings)
    trainable = automl.trainable
    return trainable

