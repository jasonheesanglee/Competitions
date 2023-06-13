import dacon_law_class as dlc
import optuna
from sklearn.model_selection import GridSearchCV as GSCV
import numpy as np
import pandas as pd
from xgboost.sklearn import XGBClassifier as xgb
import lightgbm as lgb

param_xgb_gscv = {
    'max_depth' : -1,
    'min_child_weight' : [i for i in range(1, 100)],
    'n_estimators' : [i for i in range(1, 3000)]
}

param_lgb_gscv = {
    'max_depth' : -1,
    'learning_rate' : [i for i in np.arange(0, 1,0.00000000001)],
    'num_leaves' : [i for i in range(1, 2000)],
    'n_estimators' : [i for i in range(1, 3000)]
}

class Jason_Classifier():
    
    def __init__(self, trial):
        self.trial = trial
    
    def optuna_model_selector(self, df, df1):

        X_train, X_val, y_train, y_val, test_X = dlc.test_val_separator(df, df1, 0.3)

        model_type = self.trial.suggest_categorical('model_type', ['xgb', 'lgbm'])
        random_state = 42
    
    
        if model_type == 'xgb':
            eval_metric = 'error'
            objective = self.trial.suggest_categorical('objective', ['binary:logistic', 'binary_hinge'])
            tree_method = self.trial.suggest_categorical('tree_method', ['exact', 'approx', 'hist'])
    
            if tree_method == 'exact':
                sampling_method = 'uniform'
                subsample = 0.5
                booster = self.trial.suggest_categorical('booster', ['dart', 'gbtree'])
                if booster == 'gbtree':
                    max_depth = self.trial.suggest_int('max_depth', 1, 300)
                    n_estimators = self.trial.suggest_int('n_estimators', 1, 1000)
                    if n_estimators < 500:
                        colsample_bytree = self.trial.suggest_loguniform('colsample_bytree', 0.3, 0.9)
                        learning_rate = self.trial.suggest_categorical('learning_rate', [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2,
                                                                                    1e-1])  # aka eta, from 0 to 1, loguniform
                        if learning_rate in [1e-4, 5e-4]:
                            min_split_loss = self.trial.suggest_loguniform('min_split_loss', 0.1,
                                                                      500)  # aka gamma, from 0 to 1, loguniform
                            min_child_weight = self.trial.suggest_int('min_child_weight', 1, 20)  # from 0 to infinite, int
                            alpha = self.trial.suggest_loguniform('alpha', 0.1, 20)  # from 0 to infinite, loguniform
                            reg_lambda = self.trial.suggest_loguniform('reg_lambda', 0.01, 20)  # from 0 to infinite, loguniform
                            model = xgb(
                                eval_metric=eval_metric,
                                objective=objective,
                                learning_rate=learning_rate,
                                sampling_method=sampling_method,
                                subsample=subsample,
                                colsample_bytree=colsample_bytree,
                                max_depth=max_depth,
                                tree_method=tree_method,
                                booster=booster,
                                min_split_loss=min_split_loss,
                                min_child_weight=min_child_weight,
                                alpha=alpha,
                                reg_lambda=reg_lambda,
                                random_state=random_state
                            )
    
                            model.fit(X_train, y_train)
                            preds = model.predict(X_val)
    
                            return accuracy_score(y_val, preds)
    
                        elif learning_rate in [1e-3, 5e-3, 1e-2]:
                            min_split_loss = self.trial.suggest_loguniform('min_split_loss', 200,
                                                                      700)  # aka gamma, from 0 to 1, loguniform
                            min_child_weight = self.trial.suggest_int('min_child_weight', 10, 30)  # from 0 to infinite, int
                            alpha = self.trial.suggest_loguniform('alpha', 10, 30)  # from 0 to infinite, loguniform
                            reg_lambda = self.trial.suggest_loguniform('reg_lambda', 10, 30)  # from 0 to infinite, loguniform
                            model = xgb(
                                eval_metric=eval_metric,
                                objective=objective,
                                learning_rate=learning_rate,
                                sampling_method=sampling_method,
                                subsample=subsample,
                                colsample_bytree=colsample_bytree,
                                max_depth=max_depth,
                                tree_method=tree_method,
                                booster=booster,
                                min_split_loss=min_split_loss,
                                min_child_weight=min_child_weight,
                                alpha=alpha,
                                reg_lambda=reg_lambda,
                                random_state=random_state
                            )
    
                            model.fit(X_train, y_train)
                            preds = model.predict(X_val)
    
                            return accuracy_score(y_val, preds)
    
                        elif learning_rate in [5e-2, 1e-1]:
                            min_split_loss = self.trial.suggest_loguniform('min_split_loss', 500,
                                                                      1000)  # aka gamma, from 0 to 1, loguniform
                            min_child_weight = self.trial.suggest_int('min_child_weight', 20, 40)  # from 0 to infinite, int
                            alpha = self.trial.suggest_loguniform('alpha', 20, 40)  # from 0 to infinite, loguniform
                            reg_lambda = self.trial.suggest_loguniform('reg_lambda', 20, 40)  # from 0 to infinite, loguniform
                            model = xgb(
                                eval_metric=eval_metric,
                                objective=objective,
                                learning_rate=learning_rate,
                                sampling_method=sampling_method,
                                subsample=subsample,
                                colsample_bytree=colsample_bytree,
                                max_depth=max_depth,
                                tree_method=tree_method,
                                booster=booster,
                                min_split_loss=min_split_loss,
                                min_child_weight=min_child_weight,
                                alpha=alpha,
                                reg_lambda=reg_lambda,
                                random_state=random_state
                            )
                            model.fit(X_train, y_train)
                            preds = model.predict(X_val)
    
                            return accuracy_score(y_val, preds)
    
                    elif n_estimators > 500:
                        colsample_bytree = self.trial.suggest_loguniform('colsample_bytree', 0.1, 0.7)
                        learning_rate = self.trial.suggest_categorical('learning_rate', [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2,
                                                                                1e-1])  # aka eta, from 0 to 1, loguniform
                        if learning_rate in [1e-4, 5e-4]:
                            min_split_loss = self.trial.suggest_loguniform('min_split_loss', 0.1,
                                                                      500)  # aka gamma, from 0 to 1, loguniform
                            min_child_weight = self.trial.suggest_int('min_child_weight', 1, 20)  # from 0 to infinite, int
                            alpha = self.trial.suggest_loguniform('alpha', 0.1, 20)  # from 0 to infinite, loguniform
                            reg_lambda = self.trial.suggest_loguniform('reg_lambda', 0.01, 20)  # from 0 to infinite, loguniform
                            model = xgb(
                                eval_metric=eval_metric,
                                objective=objective,
                                learning_rate=learning_rate,
                                sampling_method=sampling_method,
                                subsample=subsample,
                                colsample_bytree=colsample_bytree,
                                max_depth=max_depth,
                                tree_method=tree_method,
                                booster=booster,
                                min_split_loss=min_split_loss,
                                min_child_weight=min_child_weight,
                                alpha=alpha,
                                reg_lambda=reg_lambda,
                                random_state=random_state
                            )
    
                            model.fit(X_train, y_train)
                            preds = model.predict(X_val)
    
                            return accuracy_score(y_val, preds)
    
                        elif learning_rate in [1e-3, 5e-3, 1e-2]:
                            min_split_loss = self.trial.suggest_loguniform('min_split_loss', 200,
                                                                      700)  # aka gamma, from 0 to 1, loguniform
                            min_child_weight = self.trial.suggest_int('min_child_weight', 10, 30)  # from 0 to infinite, int
                            alpha = self.trial.suggest_loguniform('alpha', 10, 30)  # from 0 to infinite, loguniform
                            reg_lambda = self.trial.suggest_loguniform('reg_lambda', 10, 30)  # from 0 to infinite, loguniform
                            model = xgb(
                                eval_metric=eval_metric,
                                objective=objective,
                                learning_rate=learning_rate,
                                sampling_method=sampling_method,
                                subsample=subsample,
                                colsample_bytree=colsample_bytree,
                                max_depth=max_depth,
                                tree_method=tree_method,
                                booster=booster,
                                min_split_loss=min_split_loss,
                                min_child_weight=min_child_weight,
                                alpha=alpha,
                                reg_lambda=reg_lambda,
                                random_state=random_state
                            )
    
                            model.fit(X_train, y_train)
                            preds = model.predict(X_val)
    
                            return accuracy_score(y_val, preds)
    
                        elif learning_rate in [5e-2, 1e-1]:
                            min_split_loss = self.trial.suggest_loguniform('min_split_loss', 500,
                                                                      1000)  # aka gamma, from 0 to 1, loguniform
                            min_child_weight = self.trial.suggest_int('min_child_weight', 20, 40)  # from 0 to infinite, int
                            alpha = self.trial.suggest_loguniform('alpha', 20, 40)  # from 0 to infinite, loguniform
                            reg_lambda = self.trial.suggest_loguniform('reg_lambda', 20, 40)  # from 0 to infinite, loguniform
                            model = xgb(
                                eval_metric=eval_metric,
                                objective=objective,
                                learning_rate=learning_rate,
                                sampling_method=sampling_method,
                                subsample=subsample,
                                colsample_bytree=colsample_bytree,
                                max_depth=max_depth,
                                tree_method=tree_method,
                                booster=booster,
                                min_split_loss=min_split_loss,
                                min_child_weight=min_child_weight,
                                alpha=alpha,
                                reg_lambda=reg_lambda,
                                random_state=random_state
                            )
    
                            model.fit(X_train, y_train)
                            preds = model.predict(X_val)
    
                            return accuracy_score(y_val, preds)
    
                elif booster == 'dart':
                    max_depth = self.trial.suggest_int('max_depth', 1, 300)
                    colsample_bytree = self.trial.suggest_loguniform('colsample_bytree', 0.5, 0.9)
                    learning_rate = self.trial.suggest_categorical('learning_rate', [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2,
                                                                                1e-1])  # aka eta, from 0 to 1, loguniform
                    if learning_rate in [1e-4, 5e-4]:
                        rate_drop = self.trial.suggest_loguniform('rate_drop', 0.01, 0.5)
                        min_child_weight = self.trial.suggest_int('min_child_weight', 1, 20)  # from 0 to infinite, int
                        model = xgb(
                            eval_metric=eval_metric,
                            objective=objective,
                            learning_rate=learning_rate,
                            sampling_method=sampling_method,
                            subsample=subsample,
                            colsample_bytree=colsample_bytree,
                            rate_drop=rate_drop,
                            max_depth=max_depth,
                            tree_method=tree_method,
                            booster=booster,
                            min_child_weight=min_child_weight,
                            random_state=random_state
                        )
    
                        model.fit(X_train, y_train)
                        preds = model.predict(X_val)
    
                        return accuracy_score(y_val, preds)
    
                    elif learning_rate in [1e-3, 5e-3, 1e-2]:
                        rate_drop = self.trial.suggest_loguniform('rate_drop', 0.3, 0.7)
                        min_child_weight = self.trial.suggest_int('min_child_weight', 10, 30)  # from 0 to infinite, int
                        model = xgb(
                            eval_metric=eval_metric,
                            objective=objective,
                            learning_rate=learning_rate,
                            sampling_method=sampling_method,
                            subsample=subsample,
                            colsample_bytree=colsample_bytree,
                            rate_drop=rate_drop,
                            max_depth=max_depth,
                            tree_method=tree_method,
                            booster=booster,
                            min_child_weight=min_child_weight,
                            random_state=random_state
                        )
    
                        model.fit(X_train, y_train)
                        preds = model.predict(X_val)
    
                        return accuracy_score(y_val, preds)
    
                    elif learning_rate in [5e-2, 1e-1]:
                        rate_drop = self.trial.suggest_loguniform('rate_drop', 0.5, 1.0)
                        min_child_weight = self.trial.suggest_int('min_child_weight', 20, 40)  # from 0 to infinite, int
                        model = xgb(
                            eval_metric=eval_metric,
                            objective=objective,
                            learning_rate=learning_rate,
                            sampling_method=sampling_method,
                            subsample=subsample,
                            colsample_bytree=colsample_bytree,
                            rate_drop=rate_drop,
                            max_depth=max_depth,
                            tree_method=tree_method,
                            booster=booster,
                            min_child_weight=min_child_weight,
                            random_state=random_state
                        )
    
                        model.fit(X_train, y_train)
                        preds = model.predict(X_val)
    
                        return accuracy_score(y_val, preds)
    
                # elif booster == 'gblinear':
                #     learning_rate = self.trial.suggest_categorical('learning_rate', [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2,
                #                                                                 1e-1])  # aka eta, from 0 to 1, loguniform
                #     if learning_rate in [1e-4, 5e-4]:
                #         alpha = self.trial.suggest_loguniform('alpha', 0.1, 20)  # from 0 to infinite, loguniform
                #         reg_lambda = self.trial.suggest_loguniform('reg_lambda', 0.01, 20)  # from 0 to infinite, loguniform
                #         model = xgb(
                #             eval_metric=eval_metric,
                #             objective=objective,
                #             learning_rate=learning_rate,
                #             sampling_method=sampling_method,
                #             subsample=subsample,
                #             tree_method=tree_method,
                #             booster=booster,
                #             alpha=alpha,
                #             reg_lambda=reg_lambda,
                #             random_state=random_state
                #         )
                #
                #         model.fit(X_train, y_train)
                #         preds = model.predict(X_val)
                #
                #         return accuracy_score(y_val, preds)
                #
                #     elif learning_rate in [1e-3, 5e-3, 1e-2]:
                #         alpha = self.trial.suggest_loguniform('alpha', 10, 30)  # from 0 to infinite, loguniform
                #         reg_lambda = self.trial.suggest_loguniform('reg_lambda', 10, 30)  # from 0 to infinite, loguniform
                #         model = xgb(
                #             eval_metric=eval_metric,
                #             objective=objective,
                #             learning_rate=learning_rate,
                #             sampling_method=sampling_method,
                #             subsample=subsample,
                #             tree_method=tree_method,
                #             booster=booster,
                #             alpha=alpha,
                #             reg_lambda=reg_lambda,
                #             random_state=random_state
                #         )
                #
                #         model.fit(X_train, y_train)
                #         preds = model.predict(X_val)
                #
                #         return accuracy_score(y_val, preds)
                #
                #     elif learning_rate in [5e-2, 1e-1]:
                #         alpha = self.trial.suggest_loguniform('alpha', 20, 40)  # from 0 to infinite, loguniform
                #         reg_lambda = self.trial.suggest_loguniform('reg_lambda', 20, 40)  # from 0 to infinite, loguniform
                #         model = xgb(
                #             eval_metric=eval_metric,
                #             objective=objective,
                #             learning_rate=learning_rate,
                #             sampling_method=sampling_method,
                #             subsample=subsample,
                #             tree_method=tree_method,
                #             booster=booster,
                #             alpha=alpha,
                #             reg_lambda=reg_lambda,
                #             random_state=random_state
                #         )
                #
                #         model.fit(X_train, y_train)
                #         preds = model.predict(X_val)
                #
                #         return accuracy_score(y_val, preds)
    
            else:
                sampling_method = 'uniform'
                booster = self.trial.suggest_categorical('booster', ['dart', 'gbtree'])
                if booster == 'gbtree':
                    subsample = self.trial.suggest_loguniform('subsample', 0.1, 0.5)
                    max_depth = 0
                    n_estimators = self.trial.suggest_int('n_estimators', 1, 1000)
                    if n_estimators < 500:
                        colsample_bytree = self.trial.suggest_loguniform('colsample_bytree', 0.3, 0.9)
                        learning_rate = self.trial.suggest_categorical('learning_rate', [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2,
                                                                                    1e-1])  # aka eta, from 0 to 1, loguniform
                        if learning_rate in [1e-4, 5e-4]:
                            min_split_loss = self.trial.suggest_loguniform('min_split_loss', 0.1,
                                                                      400)  # aka gamma, from 0 to 1, loguniform
                            min_child_weight = self.trial.suggest_int('min_child_weight', 1, 20)  # from 0 to infinite, int
                            alpha = self.trial.suggest_loguniform('alpha', 0.1, 20)  # from 0 to infinite, loguniform
                            reg_lambda = self.trial.suggest_loguniform('reg_lambda', 0.01, 20)  # from 0 to infinite, loguniform
                            model = xgb(
                                eval_metric=eval_metric,
                                objective=objective,
                                learning_rate=learning_rate,
                                sampling_method=sampling_method,
                                subsample=subsample,
                                colsample_bytree=colsample_bytree,
                                max_depth=max_depth,
                                tree_method=tree_method,
                                booster=booster,
                                min_split_loss=min_split_loss,
                                min_child_weight=min_child_weight,
                                alpha=alpha,
                                reg_lambda=reg_lambda,
                                random_state=random_state
                            )
    
                            model.fit(X_train, y_train)
                            preds = model.predict(X_val)
    
                            return accuracy_score(y_val, preds)
    
                        elif learning_rate in [1e-3, 5e-3, 1e-2]:
                            min_split_loss = self.trial.suggest_loguniform('min_split_loss', 200,
                                                                      700)  # aka gamma, from 0 to 1, loguniform
                            min_child_weight = self.trial.suggest_int('min_child_weight', 10, 30)  # from 0 to infinite, int
                            alpha = self.trial.suggest_loguniform('alpha', 10, 30)  # from 0 to infinite, loguniform
                            reg_lambda = self.trial.suggest_loguniform('reg_lambda', 10, 30)  # from 0 to infinite, loguniform
                            model = xgb(
                                eval_metric=eval_metric,
                                objective=objective,
                                learning_rate=learning_rate,
                                sampling_method=sampling_method,
                                subsample=subsample,
                                colsample_bytree=colsample_bytree,
                                max_depth=max_depth,
                                tree_method=tree_method,
                                booster=booster,
                                min_split_loss=min_split_loss,
                                min_child_weight=min_child_weight,
                                alpha=alpha,
                                reg_lambda=reg_lambda,
                                random_state=random_state
                            )
    
                            model.fit(X_train, y_train)
                            preds = model.predict(X_val)
    
                            return accuracy_score(y_val, preds)
    
                        elif learning_rate in [5e-2, 1e-1]:
                            min_split_loss = self.trial.suggest_loguniform('min_split_loss', 400,
                                                                      1000)  # aka gamma, from 0 to 1, loguniform
                            min_child_weight = self.trial.suggest_int('min_child_weight', 20, 40)  # from 0 to infinite, int
                            alpha = self.trial.suggest_loguniform('alpha', 20, 40)  # from 0 to infinite, loguniform
                            reg_lambda = self.trial.suggest_loguniform('reg_lambda', 20, 40)  # from 0 to infinite, loguniform
                            model = xgb(
                                eval_metric=eval_metric,
                                objective=objective,
                                learning_rate=learning_rate,
                                sampling_method=sampling_method,
                                subsample=subsample,
                                colsample_bytree=colsample_bytree,
                                max_depth=max_depth,
                                tree_method=tree_method,
                                booster=booster,
                                min_split_loss=min_split_loss,
                                min_child_weight=min_child_weight,
                                alpha=alpha,
                                reg_lambda=reg_lambda,
                                random_state=random_state
                            )
    
                            model.fit(X_train, y_train)
                            preds = model.predict(X_val)
    
                            return accuracy_score(y_val, preds)
    
                    elif n_estimators > 500:
                        colsample_bytree = self.trial.suggest_loguniform('colsample_bytree', 0.1, 0.7)
                        learning_rate = self.trial.suggest_categorical('learning_rate', [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2,
                                                                                    1e-1])  # aka eta, from 0 to 1, loguniform
                        if learning_rate in [1e-4, 5e-4]:
                            min_split_loss = self.trial.suggest_loguniform('min_split_loss', 0.1,
                                                                      400)  # aka gamma, from 0 to 1, loguniform
                            min_child_weight = self.trial.suggest_int('min_child_weight', 1, 20)  # from 0 to infinite, int
                            alpha = self.trial.suggest_loguniform('alpha', 0.1, 20)  # from 0 to infinite, loguniform
                            reg_lambda = self.trial.suggest_loguniform('reg_lambda', 0.01, 20)  # from 0 to infinite, loguniform
                            model = xgb(
                                eval_metric=eval_metric,
                                objective=objective,
                                learning_rate=learning_rate,
                                sampling_method=sampling_method,
                                subsample=subsample,
                                colsample_bytree=colsample_bytree,
                                max_depth=max_depth,
                                tree_method=tree_method,
                                booster=booster,
                                min_split_loss=min_split_loss,
                                min_child_weight=min_child_weight,
                                alpha=alpha,
                                reg_lambda=reg_lambda,
                                random_state=random_state
                            )
    
                            model.fit(X_train, y_train)
                            preds = model.predict(X_val)
    
                            return accuracy_score(y_val, preds)
    
                        elif learning_rate in [1e-3, 5e-3, 1e-2]:
                            min_split_loss = self.trial.suggest_loguniform('min_split_loss', 200,
                                                                      700)  # aka gamma, from 0 to 1, loguniform
                            min_child_weight = self.trial.suggest_int('min_child_weight', 10, 30)  # from 0 to infinite, int
                            alpha = self.trial.suggest_loguniform('alpha', 10, 30)  # from 0 to infinite, loguniform
                            reg_lambda = self.trial.suggest_loguniform('reg_lambda', 10, 30)  # from 0 to infinite, loguniform
                            model = xgb(
                                eval_metric=eval_metric,
                                objective=objective,
                                learning_rate=learning_rate,
                                sampling_method=sampling_method,
                                subsample=subsample,
                                colsample_bytree=colsample_bytree,
                                max_depth=max_depth,
                                tree_method=tree_method,
                                booster=booster,
                                min_split_loss=min_split_loss,
                                min_child_weight=min_child_weight,
                                alpha=alpha,
                                reg_lambda=reg_lambda,
                                random_state=random_state
                            )
    
                            model.fit(X_train, y_train)
                            preds = model.predict(X_val)
    
                            return accuracy_score(y_val, preds)
    
                        elif learning_rate in [5e-2, 1e-1]:
                            min_split_loss = self.trial.suggest_loguniform('min_split_loss', 400,
                                                                      1000)  # aka gamma, from 0 to 1, loguniform
                            min_child_weight = self.trial.suggest_int('min_child_weight', 20, 40)  # from 0 to infinite, int
                            alpha = self.trial.suggest_loguniform('alpha', 20, 40)  # from 0 to infinite, loguniform
                            reg_lambda = self.trial.suggest_loguniform('reg_lambda', 20, 40)  # from 0 to infinite, loguniform
                            model = xgb(
                                eval_metric=eval_metric,
                                objective=objective,
                                learning_rate=learning_rate,
                                sampling_method=sampling_method,
                                subsample=subsample,
                                colsample_bytree=colsample_bytree,
                                max_depth=max_depth,
                                tree_method=tree_method,
                                booster=booster,
                                min_split_loss=min_split_loss,
                                min_child_weight=min_child_weight,
                                alpha=alpha,
                                reg_lambda=reg_lambda,
                                random_state=random_state
                            )
    
                            model.fit(X_train, y_train)
                            preds = model.predict(X_val)
    
                            return accuracy_score(y_val, preds)
    
                elif booster == 'dart':
                    subsample = 0.5
                    max_depth = 0
                    colsample_bytree = self.trial.suggest_loguniform('colsample_bytree', 0.5, 0.9)
                    learning_rate = self.trial.suggest_categorical('learning_rate', [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2,
                                                                                1e-1])  # aka eta, from 0 to 1, loguniform
                    if learning_rate in [1e-4, 5e-4]:
                        rate_drop = self.trial.suggest_loguniform('rate_drop', 0.01, 0.5)
                        min_child_weight = self.trial.suggest_int('min_child_weight', 1, 20)  # from 0 to infinite, int
                        model = xgb(
                            eval_metric=eval_metric,
                            objective=objective,
                            learning_rate=learning_rate,
                            sampling_method=sampling_method,
                            subsample=subsample,
                            colsample_bytree=colsample_bytree,
                            max_depth=max_depth,
                            tree_method=tree_method,
                            booster=booster,
                            rate_drop=rate_drop,
                            min_child_weight=min_child_weight,
                            random_state=random_state
                        )
    
                        model.fit(X_train, y_train)
                        preds = model.predict(X_val)
    
                        return accuracy_score(y_val, preds)
    
    
                    elif learning_rate in [1e-3, 5e-3, 1e-2]:
                        rate_drop = self.trial.suggest_loguniform('rate_drop', 0.3, 0.7)
                        min_child_weight = self.trial.suggest_int('min_child_weight', 10, 30)  # from 0 to infinite, int
                        model = xgb(
                            eval_metric=eval_metric,
                            objective=objective,
                            learning_rate=learning_rate,
                            sampling_method=sampling_method,
                            subsample=subsample,
                            colsample_bytree=colsample_bytree,
                            max_depth=max_depth,
                            tree_method=tree_method,
                            booster=booster,
                            rate_drop=rate_drop,
                            min_child_weight=min_child_weight,
                            random_state=random_state
                        )
    
                        model.fit(X_train, y_train)
                        preds = model.predict(X_val)
    
                        return accuracy_score(y_val, preds)
    
                    elif learning_rate in [5e-2, 1e-1]:
                        rate_drop = self.trial.suggest_loguniform('rate_drop', 0.5, 1.0)
                        min_child_weight = self.trial.suggest_int('min_child_weight', 20, 40)  # from 0 to infinite, int
                        model = xgb(
                            eval_metric=eval_metric,
                            objective=objective,
                            learning_rate=learning_rate,
                            sampling_method=sampling_method,
                            subsample=subsample,
                            colsample_bytree=colsample_bytree,
                            max_depth=max_depth,
                            tree_method=tree_method,
                            booster=booster,
                            rate_drop=rate_drop,
                            min_child_weight=min_child_weight,
                            random_state=random_state
                        )
    
                        model.fit(X_train, y_train)
                        preds = model.predict(X_val)
    
                        return accuracy_score(y_val, preds)
    
    
                # elif booster == 'gblinear':
                #     learning_rate = self.trial.suggest_categorical('learning_rate', [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2,
                #                                                                 1e-1])  # aka eta, from 0 to 1, loguniform
                #     if learning_rate in [1e-4, 5e-4]:
                #         alpha = self.trial.suggest_loguniform('alpha', 0.1, 20)  # from 0 to infinite, loguniform
                #         reg_lambda = self.trial.suggest_loguniform('reg_lambda', 0.01, 20)  # from 0 to infinite, loguniform
                #         model = xgb(
                #             eval_metric=eval_metric,
                #             objective=objective,
                #             learning_rate=learning_rate,
                #             sampling_method=sampling_method,
                #             tree_method=tree_method,
                #             booster=booster,
                #             alpha=alpha,
                #             reg_lambda=reg_lambda,
                #             random_state=random_state
                #         )
                #         model.fit(X_train, y_train)
                #         preds = model.predict(X_val)
                #
                #         return accuracy_score(y_val, preds)
                #
                #     elif learning_rate in [1e-3, 5e-3, 1e-2]:
                #         alpha = self.trial.suggest_loguniform('alpha', 10, 30)  # from 0 to infinite, loguniform
                #         reg_lambda = self.trial.suggest_loguniform('reg_lambda', 10, 30)  # from 0 to infinite, loguniform
                #         model = xgb(
                #             eval_metric=eval_metric,
                #             objective=objective,
                #             learning_rate=learning_rate,
                #             sampling_method=sampling_method,
                #             tree_method=tree_method,
                #             booster=booster,
                #             alpha=alpha,
                #             reg_lambda=reg_lambda,
                #             random_state=random_state
                #         )
                #         model.fit(X_train, y_train)
                #         preds = model.predict(X_val)
                #
                #         return accuracy_score(y_val, preds)
                #
                #     elif learning_rate in [5e-2, 1e-1]:
                #         alpha = self.trial.suggest_loguniform('alpha', 20, 40)  # from 0 to infinite, loguniform
                #         reg_lambda = self.trial.suggest_loguniform('reg_lambda', 20, 40)  # from 0 to infinite, loguniform
                #         model = xgb(
                #             eval_metric=eval_metric,
                #             objective=objective,
                #             learning_rate=learning_rate,
                #             sampling_method=sampling_method,
                #             tree_method=tree_method,
                #             booster=booster,
                #             alpha=alpha,
                #             reg_lambda=reg_lambda,
                #             random_state=random_state
                #         )
                #         model.fit(X_train, y_train)
                #         preds = model.predict(X_val)
                #
                #         return accuracy_score(y_val, preds)
    
        elif model_type == 'lgbm':
            objective = 'binary'
            metric = 'accuracy'
            num_threads = 0
            max_depth = -1
    
            num_leaves = self.trial.suggest_int('num_leaves', 1, 1000)
            boosting_type = self.trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'rf', 'goss'])
            if boosting_type == 'gbdt':
                n_estimators = self.trial.suggest_int('n_estimators', 1, 500)
                learning_rate = self.trial.suggest_categorical('learning_rate', [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2,
                                                                            1e-1])  # aka eta, from 0 to 1, loguniform
                if learning_rate in [1e-4, 5e-4]:
                    min_data_in_leaf = self.trial.suggest_int('min_data_in_leaf', 1, 20)
                    feature_fraction = self.trial.suggest_uniform('feature_fraction', 0.5, 0.999)
                    bagging_fraction = self.trial.suggest_uniform('bagging_fraction', 0.5, 0.999)
                    bagging_freq = self.trial.suggest_int('bagging_freq', 1, n_estimators)
                    alpha = self.trial.suggest_loguniform('alpha', 0.1, 20)  # from 0 to infinite, loguniform
                    reg_lambda = self.trial.suggest_loguniform('reg_lambda', 0.01, 20)  # from 0 to infinite, loguniform
                    model = lgb.LGBMClassifier(
                        learning_rate=learning_rate,
                        boosting_type=boosting_type,
                        num_leaves=num_leaves,
                        num_threads=num_threads,
                        objective=objective,
                        metric=metric,
                        n_estimators=n_estimators,
                        min_data_in_leaf=min_data_in_leaf,
                        feature_fraction=feature_fraction,
                        bagging_fraction=bagging_fraction,
                        bagging_freq=bagging_freq,
                        max_depth=max_depth,
                        alpha=alpha,
                        reg_lambda=reg_lambda,
                        random_state=random_state
                    )
                    model.fit(X_train, y_train)
                    preds = model.predict(X_val)
    
                    return accuracy_score(y_val, preds)
    
                elif learning_rate in [1e-3, 5e-3, 1e-2]:
                    min_data_in_leaf = self.trial.suggest_int('min_data_in_leaf', 10, 30)
                    feature_fraction = self.trial.suggest_uniform('feature_fraction', 0.3, 0.7)
                    bagging_fraction = self.trial.suggest_uniform('bagging_fraction', 0.3, 0.7)
                    bagging_freq = self.trial.suggest_int('bagging_freq', 1, n_estimators)
                    alpha = self.trial.suggest_loguniform('alpha', 10, 30)  # from 0 to infinite, loguniform
                    reg_lambda = self.trial.suggest_loguniform('reg_lambda', 10, 30)  # from 0 to infinite, loguniform
                    model = lgb.LGBMClassifier(
                        learning_rate=learning_rate,
                        boosting_type=boosting_type,
                        num_leaves=num_leaves,
                        num_threads=num_threads,
                        objective=objective,
                        metric=metric,
                        n_estimators=n_estimators,
                        min_data_in_leaf=min_data_in_leaf,
                        feature_fraction=feature_fraction,
                        bagging_fraction=bagging_fraction,
                        bagging_freq=bagging_freq,
                        max_depth=max_depth,
                        alpha=alpha,
                        reg_lambda=reg_lambda,
                        random_state=random_state
                    )
                    model.fit(X_train, y_train)
                    preds = model.predict(X_val)
    
                    return accuracy_score(y_val, preds)
    
                elif learning_rate in [5e-2, 1e-1]:
                    min_data_in_leaf = self.trial.suggest_int('min_data_in_leaf', 20, 40)
                    feature_fraction = self.trial.suggest_uniform('feature_fraction', 0.1, 0.5)
                    bagging_fraction = self.trial.suggest_uniform('bagging_fraction', 0.1, 0.5)
                    bagging_freq = self.trial.suggest_int('bagging_freq', 1, n_estimators)
                    alpha = self.trial.suggest_loguniform('alpha', 20, 40)  # from 0 to infinite, loguniform
                    reg_lambda = self.trial.suggest_loguniform('reg_lambda', 20, 40)  # from 0 to infinite, loguniform
                    model = lgb.LGBMClassifier(
                        learning_rate=learning_rate,
                        boosting_type=boosting_type,
                        num_leaves=num_leaves,
                        num_threads=num_threads,
                        objective=objective,
                        metric=metric,
                        n_estimators=n_estimators,
                        min_data_in_leaf=min_data_in_leaf,
                        feature_fraction=feature_fraction,
                        bagging_fraction=bagging_fraction,
                        bagging_freq=bagging_freq,
                        max_depth=max_depth,
                        alpha=alpha,
                        reg_lambda=reg_lambda,
                        random_state=random_state
                    )
                    model.fit(X_train, y_train)
                    preds = model.predict(X_val)
    
                    return accuracy_score(y_val, preds)
    
            elif boosting_type == 'dart':
                n_estimators = self.trial.suggest_int('n_estimators', 1, 500)
                learning_rate = self.trial.suggest_categorical('learning_rate', [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2,
                                                                            1e-1])  # aka eta, from 0 to 1, loguniform
                if learning_rate in [1e-4, 5e-4]:
                    min_data_in_leaf = self.trial.suggest_int('min_data_in_leaf', 1, 20)
                    feature_fraction = self.trial.suggest_uniform('feature_fraction', 0.5, 0.999)
                    bagging_fraction = self.trial.suggest_uniform('bagging_fraction', 0.5, 0.999)
                    bagging_freq = self.trial.suggest_int('bagging_freq', 1, n_estimators)
                    drop_rate = self.trial.suggest_loguniform('drop_rate', 0.01, 0.5)
                    model = lgb.LGBMClassifier(
                        learning_rate=learning_rate,
                        boosting_type=boosting_type,
                        num_leaves=num_leaves,
                        num_threads=num_threads,
                        n_estimators=n_estimators,
                        objective=objective,
                        metric=metric,
                        min_data_in_leaf=min_data_in_leaf,
                        feature_fraction=feature_fraction,
                        bagging_fraction=bagging_fraction,
                        bagging_freq=bagging_freq,
                        max_depth=max_depth,
                        drop_rate=drop_rate,
                        random_state=random_state
                    )
                    model.fit(X_train, y_train)
                    preds = model.predict(X_val)
    
                    return accuracy_score(y_val, preds)
    
                elif learning_rate in [1e-3, 5e-3, 1e-2]:
                    min_data_in_leaf = self.trial.suggest_int('min_data_in_leaf', 10, 30)
                    feature_fraction = self.trial.suggest_uniform('feature_fraction', 0.3, 0.7)
                    bagging_fraction = self.trial.suggest_uniform('bagging_fraction', 0.3, 0.7)
                    bagging_freq = self.trial.suggest_int('bagging_freq', 1, n_estimators)
                    drop_rate = self.trial.suggest_loguniform('drop_rate', 0.3, 0.7)
                    model = lgb.LGBMClassifier(
                        learning_rate=learning_rate,
                        boosting_type=boosting_type,
                        num_leaves=num_leaves,
                        num_threads=num_threads,
                        n_estimators=n_estimators,
                        objective=objective,
                        metric=metric,
                        min_data_in_leaf=min_data_in_leaf,
                        feature_fraction=feature_fraction,
                        bagging_fraction=bagging_fraction,
                        bagging_freq=bagging_freq,
                        max_depth=max_depth,
                        drop_rate=drop_rate,
                        random_state=random_state
                    )
                    model.fit(X_train, y_train)
                    preds = model.predict(X_val)
    
                    return accuracy_score(y_val, preds)
    
                elif learning_rate in [5e-2, 1e-1]:
                    min_data_in_leaf = self.trial.suggest_int('min_data_in_leaf', 20, 40)
                    feature_fraction = self.trial.suggest_uniform('feature_fraction', 0.5, 0.999)
                    bagging_fraction = self.trial.suggest_uniform('bagging_fraction', 0.5, 0.999)
                    bagging_freq = self.trial.suggest_int('bagging_freq', 1, n_estimators)
                    drop_rate = self.trial.suggest_loguniform('drop_rate', 0.5, 1)
                    model = lgb.LGBMClassifier(
                        learning_rate=learning_rate,
                        boosting_type=boosting_type,
                        num_leaves=num_leaves,
                        num_threads=num_threads,
                        n_estimators=n_estimators,
                        objective=objective,
                        metric=metric,
                        min_data_in_leaf=min_data_in_leaf,
                        feature_fraction=feature_fraction,
                        bagging_fraction=bagging_fraction,
                        bagging_freq=bagging_freq,
                        max_depth=max_depth,
                        drop_rate=drop_rate,
                        random_state=random_state
                    )
                    model.fit(X_train, y_train)
                    preds = model.predict(X_val)
    
                    return accuracy_score(y_val, preds)
    
            elif boosting_type == 'rf':
                n_estimators = self.trial.suggest_int('n_estimators', 1, 500)
                feature_fraction_bynode = self.trial.suggest_loguniform('feature_fraction_bynode', 0.01, 0.999)
                if feature_fraction_bynode < 0.33:
                    bagging_fraction = self.trial.suggest_uniform('bagging_fraction', 0.5, 0.999)
                    bagging_freq = self.trial.suggest_int('bagging_freq', 1, n_estimators)
                    min_data_in_leaf = self.trial.suggest_int('min_data_in_leaf', 0, 20)
                    alpha = self.trial.suggest_loguniform('alpha', 0.1, 20)  # from 0 to infinite, loguniform
                    reg_lambda = self.trial.suggest_loguniform('reg_lambda', 0.01, 20)  # from 0 to infinite, loguniform
                    model = lgb.LGBMClassifier(
                        boosting_type=boosting_type,
                        num_leaves=num_leaves,
                        num_threads=num_threads,
                        feature_fraction_bynode=feature_fraction_bynode,
                        bagging_fraction=bagging_fraction,
                        bagging_freq=bagging_freq,
                        objective=objective,
                        metric=metric,
                        n_estimators=n_estimators,
                        min_data_in_leaf=min_data_in_leaf,
                        max_depth=max_depth,
                        alpha=alpha,
                        reg_lambda=reg_lambda,
                        random_state=random_state
                    )
                    model.fit(X_train, y_train)
                    preds = model.predict(X_val)
    
                    return accuracy_score(y_val, preds)
    
                elif feature_fraction_bynode > 0.33 and feature_fraction_bynode < 0.66:
                    min_data_in_leaf = self.trial.suggest_int('min_data_in_leaf', 10, 30)
                    alpha = self.trial.suggest_loguniform('alpha', 10, 30)  # from 0 to infinite, loguniform
                    bagging_fraction = self.trial.suggest_uniform('bagging_fraction', 0.5, 0.999)
                    bagging_freq = self.trial.suggest_int('bagging_freq', 1, n_estimators)
                    reg_lambda = self.trial.suggest_loguniform('reg_lambda', 10, 30)  # from 0 to infinite, loguniform
                    model = lgb.LGBMClassifier(
                        boosting_type=boosting_type,
                        num_leaves=num_leaves,
                        num_threads=num_threads,
                        feature_fraction_bynode=feature_fraction_bynode,
                        bagging_fraction=bagging_fraction,
                        bagging_freq=bagging_freq,
                        objective=objective,
                        metric=metric,
                        n_estimators=n_estimators,
                        min_data_in_leaf=min_data_in_leaf,
                        max_depth=max_depth,
                        alpha=alpha,
                        reg_lambda=reg_lambda,
                        random_state=random_state
                    )
    
                    model.fit(X_train, y_train)
                    preds = model.predict(X_val)
    
                    return accuracy_score(y_val, preds)
    
                elif feature_fraction_bynode > 0.66:
                    min_data_in_leaf = self.trial.suggest_int('min_data_in_leaf', 20, 40)
                    alpha = self.trial.suggest_loguniform('alpha', 20, 40)  # from 0 to infinite, loguniform
                    reg_lambda = self.trial.suggest_loguniform('reg_lambda', 20, 40)  # from 0 to infinite, loguniform
                    bagging_fraction = self.trial.suggest_uniform('bagging_fraction', 0.5, 0.999)
                    bagging_freq = self.trial.suggest_int('bagging_freq', 1, n_estimators)
                    model = lgb.LGBMClassifier(
                        boosting_type=boosting_type,
                        num_leaves=num_leaves,
                        num_threads=num_threads,
                        feature_fraction_bynode=feature_fraction_bynode,
                        objective=objective,
                        metric=metric,
                        bagging_freq=bagging_freq,
                        bagging_fraction=bagging_fraction,
                        n_estimators=n_estimators,
                        min_data_in_leaf=min_data_in_leaf,
                        max_depth=max_depth,
                        alpha=alpha,
                        reg_lambda=reg_lambda,
                        random_state=random_state
                    )
    
                    model.fit(X_train, y_train)
                    preds = model.predict(X_val)
    
                    return accuracy_score(y_val, preds)
    
            elif boosting_type == 'goss':
                learning_rate = self.trial.suggest_categorical('learning_rate', [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2,
                                                                            1e-1])  # aka eta, from 0 to 1, loguniform
                if learning_rate in [1e-4, 5e-4]:
                    min_data_in_leaf = self.trial.suggest_int('min_data_in_leaf', 0, 20)
                    top_rate = self.trial.suggest_float('top_rate', 0.1, 0.5)
                    model = lgb.LGBMClassifier(
                        learning_rate=learning_rate,
                        boosting_type=boosting_type,
                        num_leaves=num_leaves,
                        num_threads=num_threads,
                        objective=objective,
                        metric=metric,
                        top_rate=top_rate,
                        min_data_in_leaf=min_data_in_leaf,
                        max_depth=max_depth,
                        random_state=random_state
                    )
                    model.fit(X_train, y_train)
                    preds = model.predict(X_val)
    
                    return accuracy_score(y_val, preds)
    
                elif learning_rate in [1e-3, 5e-3, 1e-2]:
                    min_data_in_leaf = self.trial.suggest_int('min_data_in_leaf', 10, 30)
                    top_rate = self.trial.suggest_float('top_rate', 0.3, 0.7)
                    model = lgb.LGBMClassifier(
                        learning_rate=learning_rate,
                        boosting_type=boosting_type,
                        num_leaves=num_leaves,
                        num_threads=num_threads,
                        objective=objective,
                        metric=metric,
                        top_rate=top_rate,
                        min_data_in_leaf=min_data_in_leaf,
                        max_depth=max_depth,
                        random_state=random_state
                    )
                    model.fit(X_train, y_train)
                    preds = model.predict(X_val)
    
                    return accuracy_score(y_val, preds)
    
                elif learning_rate in [5e-2, 1e-1]:
                    min_data_in_leaf = self.trial.suggest_int('min_data_in_leaf', 2, 40)
                    top_rate = self.trial.suggest_float('top_rate', 0.5, 0.9)
                    model = lgb.LGBMClassifier(
                        learning_rate=learning_rate,
                        boosting_type=boosting_type,
                        num_leaves=num_leaves,
                        num_threads=num_threads,
                        objective=objective,
                        top_rate=top_rate,
                        min_data_in_leaf=min_data_in_leaf,
                        max_depth=max_depth,
                        random_state=random_state
                    )
    
                    model.fit(X_train, y_train)
                    preds = model.predict(X_val)
    
                    return accuracy_score(y_val, preds)