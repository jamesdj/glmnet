import numbers

import pandas as pd
import numpy as np
from sklearn.utils import compute_class_weight
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from collections import Counter

import readline
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
glmnet = importr("glmnet")
base = importr("base")
dollar = base.__dict__["$"]
stats = importr('stats')
from rpy2.robjects import numpy2ri

"""
class GLMNet(BaseEstimator, RegressorMixin):
    # Todo: flesh out this class for non-CV fitting
    # may be useful for CCA

    def __init__(self, alpha=0, l1_ratio=0.5):
        self.coef_ = None
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def fit(self, x, y, upper=None, lower=None):
        pass

    def predict(self, x):
        pass
"""

class GLMNetCV(BaseEstimator, RegressorMixin, ClassifierMixin):
    # Todo: enable parallel CV (details in vignette)

    def __init__(self, lower=None, upper=None, loss_metric='mse', alpha=0.5):
        self.coef_ = None
        self.model = None
        self.lower = lower
        self.upper = upper
        self.loss_metric = loss_metric
        self.lambda_min = None
        self.loss = None
        self.reg_type = None
        self.alpha = alpha

    def fit(self, x, y, classes=None, sample_weights=None):
        self.model = glmnet_fit(np.array(x), np.array(y), self.reg_type,
                                self.lower, self.upper, cv=True, loss_metric=self.loss_metric,
                                classes=classes, alpha=self.alpha, sample_weights=sample_weights)

        self.lambda_min = np.array(dollar(self.model, 'lambda.min'))[0]
        self.coef_ = get_coeffs(self.model, lmda=self.lambda_min)
        cv_losses = get_values_from_glmnet_fit(self.model, 'cvm')
        self.loss = np.min(cv_losses)
        # Todo: some valid loss_metric arguments such as 'auc' are not losses,
        # and should be maximized, not minimized
        return self


class GLMNetLinearRegressionCV(GLMNetCV):

    def __init__(self, lower=None, upper=None, loss_metric='mse', alpha=0.5):
        super().__init__(lower=lower, upper=upper, loss_metric=loss_metric, alpha=alpha)
        self.reg_type = 'linear'

    def predict(self, x):
        return np.array(ro.r['predict'](self.model, newx=np.array(x), s=self.lambda_min)).T[0]


class GLMNetLogisticRegressionCV(GLMNetCV):
    def __init__(self, lower=None, upper=None, loss_metric='mse', alpha=0.5):
        super().__init__(lower=lower, upper=upper, loss_metric=loss_metric, alpha=alpha)
        self.reg_type = 'logistic'

    def predict(self, x):
        return np.array(list(map(int, ro.r['predict'](self.model, newx=np.array(x), s=self.lambda_min, type='class'))))

    def predict_proba(self, x):
        return np.array(ro.r['predict'](self.model, newx=np.array(x), s=self.lambda_min, type='response')).T[0]


def get_coeffs(cvfit, lmda='min'):

    if not isinstance(lmda, numbers.Number):
        if isinstance(lmda, str):
            if lmda not in ['min', '1se']:
                raise ValueError("{} not an accepted lmda; try 'min', '1se', or a number")
            else:
                lmda = get_values_from_glmnet_fit(cvfit, 'lambda.{}'.format(lmda))[0]
        else:
            raise ValueError("lmda must be a string or number")
    r = ro.r
    coeffs = np.array(r['as.matrix'](stats.coef(cvfit, s=lmda)))
    return coeffs[1:].T[0]


def glmnet_cv(x, y, reg_type='linear', lower=None, upper=None):
    cvfit = glmnet_fit(x, y, reg_type=reg_type, lower=lower, upper=upper, cv=True)
    coeffs = get_coeffs(cvfit)
    return coeffs


def fold_generator_to_foldid(fg):
    fold_idx = 1
    sample_to_fold = {}
    for train, test in fg:
        for sample in test:
            sample_to_fold[sample] = fold_idx
        fold_idx += 1
    return list(zip(*sorted(sample_to_fold.items())))[1]


def glmnet_fit(x, y, reg_type='linear', lower=None, upper=None, cv=False,
               loss_metric='mse', classes=None, alpha=0.5, sample_weights=None):
    fit_func = glmnet.cv_glmnet if cv else glmnet.glmnet
    lower = float('-inf') if lower is None else lower
    upper = float('inf') if upper is None else upper
    x_used = x.values if isinstance(x, pd.DataFrame) else x
    y_used = y.values if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series) else y
    numpy2ri.activate()
    if reg_type == 'linear':
        if sample_weights is None:
            sample_weights = np.ones((x.shape[0],))
        if classes is not None:
            class_counter = Counter(classes)
            n_in_least_frequent_class = class_counter.most_common()[-1][1]
            n_splits = min(n_in_least_frequent_class, 10)
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
            foldid = np.array(fold_generator_to_foldid(skf.split(x, classes)))
            fit = fit_func(x_used, np.atleast_2d(y_used), lower=lower, upper=upper, alpha=alpha, weights=sample_weights,
                           **{"type.measure": loss_metric, 'foldid': foldid})
        else:
            fit = fit_func(x_used, np.atleast_2d(y_used), lower=lower, upper=upper, alpha=alpha, weights=sample_weights,
                           **{"type.measure": loss_metric})
    else:
        class_weights = compute_class_weight('balanced', np.unique(y), y)
        sample_weights = [class_weights[int(y[i])] for i in range(len(y))]
        if reg_type == 'logistic':
            fit = fit_func(x_used, np.atleast_2d(y_used), family='binomial', lower=lower, upper=upper,
                           weights=sample_weights, **{"type.measure":loss_metric})
        elif reg_type == 'multinomial':
            fit = fit_func(x_used, np.atleast_2d(y_used), family='multinomial', lower=lower, upper=upper,
                           weights=sample_weights, **{"type.measure": loss_metric})
        else:
            raise ValueError('{} is not a supported regression type; try "linear", "logistic", or "multinomial"')
    numpy2ri.deactivate()
    return fit


def get_values_from_glmnet_fit(fit, field):
    names = list(fit.names)
    #print names
    if field not in names:
        raise ValueError("{} not a field of glmnet fit object".format(field))
    return np.array(fit[names.index(field)])


def glmnet_rank(x, y, reg_type='linear', lower=None, upper=None):
    """
    Returns indices of variables according to the order in which they appear in the model
    (most relevant first). There can be ties; it doesn't attempt to settle that, but returns them
    in arbitrary order
    It may never include some variables. I was hoping it would include them all at some regularization,
    but if they simply are not relevant, they don't get included...
    :param x:
    :param y:
    :param reg_type:
    :param positive:
    :return:
    """
    fit = glmnet_fit(x, y, reg_type=reg_type, lower=lower, upper=upper, cv=False)
    r = ro.r
    dfs = get_values_from_glmnet_fit(fit, 'df')
    lambdas = get_values_from_glmnet_fit(fit, 'lambda')
    # why isn't it one var out per lambda? i thought that's what the path is about...
    df_change_idxs = []
    cum_nonzero_idxs = []
    for i in range(1, len(dfs)):
        if dfs[i] > dfs[i - 1]:
            df_change_idxs.append(i)
    for idx in df_change_idxs:
        coeffs = np.array(r['as.matrix'](stats.coef(fit, s=lambdas[idx])))
        coeffs[cum_nonzero_idxs] = 0
        nonzero_idxs = np.nonzero(coeffs)[0]
        #print nonzero_idxs
        cum_nonzero_idxs.extend(nonzero_idxs)
    if 0 in cum_nonzero_idxs:
        cum_nonzero_idxs.remove(0)  # intercept
    return list(np.array(cum_nonzero_idxs) - 1)

