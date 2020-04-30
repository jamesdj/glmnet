import numbers
import readline  # sometimes necessary to import before rpy2

import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import ListVector
from rpy2.robjects import numpy2ri
glmnet = importr("glmnet")
base = importr("base")
dollar = base.__dict__["$"]
stats = importr('stats')
as_null = ro.r['as.null']

"""
TODO: 
- use intercept and coefficients to predict etc. without invoking R?
   May allow use of a pre-fit model on a machine where rpy2 is not set up.
   Though I'm not sure because the self.model attribute is an R object.
- more input validation etc.
- enable multi-target regression
- enable grouped option in multinomial
- enable other regression models (Poisson, Cox)
- expose other features such as the regularization path info
- some valid loss_metric arguments such as 'auc' are not losses,
   and should be maximized, not minimized
"""


REG_TYPE2FAMILY = {
    'linear': 'gaussian',
    'logistic': 'binomial',
    'multinomial': 'multinomial',
}


def with_numpy2ri(func):
    """
    Activates numpy2ri, executes function, deactivates numpy2ri

    Parameters
    ----------
    func: callable
        Function to wrap

    Returns
    -------
    numpy2ri_wrapper : func
        Wrapped function

    """
    def numpy2ri_wrapper(*args, **kwargs):
        numpy2ri.activate()
        result = func(*args, **kwargs)
        numpy2ri.deactivate()
        return result
    return numpy2ri_wrapper


class GLMNetCV(BaseEstimator):

    def __init__(self,
                 lower=None,
                 upper=None,
                 loss_metric='mse',
                 alpha=0.5):
        self.lower = lower
        self.upper = upper
        self.loss_metric = loss_metric
        self.alpha = alpha

        # Assigned in inheriting classes
        self.intercept_ = None
        self.coef_ = None
        self.model = None
        self.lambda_min = None
        self.loss = None
        self.reg_type = None
        self.sample_weight = None
        self.class_weight = None

    def fit(self, x, y):
        x_used = x.values if isinstance(x, pd.DataFrame) else np.array(x)
        y_used = y.values if isinstance(y, (pd.DataFrame, pd.Series)) else np.array(y)
        self.model = glmnet_fit(x_used,
                                y_used,
                                self.reg_type,
                                self.lower,
                                self.upper,
                                loss_metric=self.loss_metric,
                                alpha=self.alpha,
                                class_weight=self.class_weight,
                                sample_weight=self.sample_weight)

        self.lambda_min = np.array(dollar(self.model, 'lambda.min'))[0]
        self.intercept_, self.coef_ = get_coeffs(self.model, lmda=self.lambda_min)
        cv_losses = get_values_from_glmnet_fit(self.model, 'cvm')
        self.loss = np.min(cv_losses)
        return self

    @with_numpy2ri
    def _pred(self,
              x,
              pred_type=None):
        x_used = x.values if isinstance(x, pd.DataFrame) else np.array(x)
        if pred_type is None:
            pred_type = as_null()
        return ro.r['predict'](self.model,
                               newx=x_used,
                               s=self.lambda_min,
                               type=pred_type)


class GLMNetClassifierCV(GLMNetCV, ClassifierMixin):

    def __init__(self,
                 class_weight=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.reg_type = None
        self.class_weight = class_weight

    def predict(self, x):
        raw_pred = self._pred(x, pred_type='class')
        pred = np.array(list(map(int, raw_pred)))
        return pred

    def predict_proba(self, x):
        raw_pred = self._pred(x, pred_type='response')
        return np.array(raw_pred).T[0]


class GLMNetRegressionCV(GLMNetCV, RegressorMixin):
    def __init__(self,
                 sample_weight=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.reg_type = None
        self.sample_weight = sample_weight

    def predict(self, x):
        raw_pred = self._pred(x)
        pred = np.array(raw_pred).T[0]
        return pred


class GLMNetLinearRegressionCV(GLMNetRegressionCV):

    def __init__(self,
                 lower=None,
                 upper=None,
                 loss_metric='mse',
                 alpha=0.5,
                 sample_weight=None):
        super().__init__(lower=lower,
                         upper=upper,
                         loss_metric=loss_metric,
                         alpha=alpha,
                         sample_weight=sample_weight)
        self.reg_type = 'linear'


class GLMNetLogisticRegressionCV(GLMNetClassifierCV):
    def __init__(self,
                 lower=None,
                 upper=None,
                 loss_metric='mse',
                 alpha=0.5,
                 class_weight=None):
        super().__init__(lower=lower,
                         upper=upper,
                         loss_metric=loss_metric,
                         alpha=alpha,
                         class_weight=class_weight)
        self.reg_type = 'logistic'


class GLMNetMultinomialRegressionCV(GLMNetClassifierCV):
    def __init__(self,
                 lower=None,
                 upper=None,
                 loss_metric='mse',
                 alpha=0.5,
                 class_weight=None):
        super().__init__(lower=lower,
                         upper=upper,
                         loss_metric=loss_metric,
                         alpha=alpha,
                         class_weight=class_weight)
        self.reg_type = 'multinomial'


def get_coeffs(cvfit, lmda='min'):
    if not isinstance(lmda, numbers.Number):
        if isinstance(lmda, str):
            if lmda not in ['min', '1se']:
                raise ValueError(f"{lmda} not an accepted lmda; try 'min', '1se', or a number")
            else:
                lmda = get_values_from_glmnet_fit(cvfit, f'lambda.{lmda}')[0]
        else:
            raise ValueError("lmda must be a string or number")
    r = ro.r
    coeffs_obj = stats.coef(cvfit, s=lmda)
    if isinstance(coeffs_obj, ListVector):
        coeffs = np.hstack([np.array(r['as.matrix'](sub_coeffs_obj)) for sub_coeffs_obj in coeffs_obj]).T
    else:
        coeffs = np.array(r['as.matrix'](coeffs_obj)).T
    intercept = coeffs[:, 0]
    coeffs = coeffs[:, 1:]
    return intercept, coeffs


@with_numpy2ri
def glmnet_fit(x,
               y,
               reg_type,
               lower=None,
               upper=None,
               loss_metric='mse',
               alpha=0.5,
               class_weight=None,
               sample_weight=None):
    lower = float('-inf') if lower is None else lower
    upper = float('inf') if upper is None else upper
    if reg_type in ['logistic', 'multinomial']:
        class_weights = compute_class_weight(class_weight, np.unique(y), y)
        sample_weight = [class_weights[int(y[i])] for i in range(len(y))]
    else:
        if sample_weight is None:
            sample_weight = np.ones((x.shape[0],))
    family = REG_TYPE2FAMILY[reg_type]
    fit = glmnet.cv_glmnet(x,
                           np.atleast_2d(y),
                           family=family,
                           lower=lower,
                           upper=upper,
                           alpha=alpha,
                           weights=sample_weight,
                           **{"type.measure": loss_metric})
    return fit


def get_values_from_glmnet_fit(fit, field):
    names = list(fit.names)
    if field not in names:
        raise ValueError("{} not a field of glmnet fit object".format(field))
    return np.array(fit[names.index(field)])


