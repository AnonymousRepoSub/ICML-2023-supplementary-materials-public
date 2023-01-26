from lightgbm import LGBMRegressor
from xgboost import XGBRegressor, XGBClassifier

from flaml.ml import get_val_loss
from data_loader import get_dataset
import numpy as np 
from utils import set_seed

set_seed(2)
def run_baseline(dataset, eval_metric, model):   
    if eval_metric == 'roc_auc':
        obj = 'binary'
    else:
        obj = 'regression'

    X_train, y_train, X_test, y_test, train_len, test_group_num, test_group_value, task = get_dataset(dataset, 'test')

    model.fit(X_train, y_train)
    print(len(X_test))

    val_loss, metric_for_logging, train_time, pred_time= get_val_loss(
        config=None,
        estimator=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        groups_val=test_group_value,
        weight_val=None,
        eval_metric=eval_metric,
        obj=obj,
        require_train=False,
    )
    print(dataset)
    print(metric_for_logging['month'])
    print(np.mean(metric_for_logging['month']))


run_baseline('electricity', 'roc_auc', XGBClassifier())
# run_baseline('sales', 'rmse', LGBMRegressor())
run_baseline('vessel', 'rmse', XGBRegressor())
run_baseline('temp', 'rmse', LGBMRegressor())
