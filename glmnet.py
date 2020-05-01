import numbers
import readline  # sometimes necessary to import before rpy2

import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.model_selection import check_cv

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
- support passing in fold indices and selecting numbers of folds
- Add more detail about parameters in docstrings
  - list what options are supported for each model, such as losses
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

CLASSIFICATION_REG_TYPES = ['logistic', 'multinomial']


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
    """
    Base class for all wrapped models.
    """

    def __init__(self,
                 lower=None,
                 upper=None,
                 loss_metric='mse',
                 alpha=0.5,
                 cv=None):
        self.lower = lower
        self.upper = upper
        self.loss_metric = loss_metric
        self.alpha = alpha
        self.cv = cv

        # Assigned in inheriting classes
        self.intercept_ = None
        self.coef_ = None
        self.model = None
        self.lambda_min = None
        self.loss = None
        self.reg_type = None
        self.class_weight = None

    def fit(self, x, y, sample_weight=None):
        """
        Fit generalized linear model.

        Parameters
        ----------
        x: array-like of shape (n_samples, n_features)
            Training data
        y: array-like of shape (n_samples,)
            Target values
        sample_weight: array-like of shape (n_samples,), default=None
            Individual weights for each sample

        Returns
        -------
        self: GLMNetCV object
            Fitted estimator
        """
        x_used = x.values if isinstance(x, pd.DataFrame) else np.array(x)
        y_used = y.values if isinstance(y, (pd.DataFrame, pd.Series)) else np.array(y)
        self.model = glmnet_fit(x_used,
                                y_used,
                                self.reg_type,
                                self.lower,
                                self.upper,
                                loss_metric=self.loss_metric,
                                alpha=self.alpha,
                                cv=self.cv,
                                class_weight=self.class_weight,
                                sample_weight=sample_weight)

        self.lambda_min = np.array(dollar(self.model, 'lambda.min'))[0]
        self.intercept_, self.coef_ = get_coeffs(self.model, lmda=self.lambda_min)
        cv_losses = get_attr_from_glmnet_fit(self.model, 'cvm')
        self.loss = np.min(cv_losses)
        return self

    @with_numpy2ri
    def _pred(self,
              x,
              pred_type=None):
        """
        Base function for prediction
        
        Parameters
        ----------
        x: array-like of shape (n_samples, n_features)
            Data to predict from.
        pred_type: {'class', 'response', None}
            glmnet param for whether to predict class, probability, or value.

        Returns
        -------
        raw_pred:
            Raw prediction results
        """
        x_used = x.values if isinstance(x, pd.DataFrame) else np.array(x)
        if pred_type is None:
            pred_type = as_null()
        raw_pred = ro.r['predict'](self.model,
                                   newx=x_used,
                                   s=self.lambda_min,
                                   type=pred_type)
        return raw_pred


class GLMNetClassifierCV(GLMNetCV, ClassifierMixin):
    """
    Base class for classifier models.
    """

    def __init__(self,
                 class_weight=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.reg_type = None
        self.class_weight = class_weight

    def predict(self, x):
        """
        Predict class membership for each sample.
        
        
        Parameters
        ----------
        x: array-like of shape (n_samples, n_features)
            Data to predict from.

        Returns
        -------
        pred : array-like
            Predicted probabilities of class membership for each sample.

        """
        raw_pred = self._pred(x, pred_type='class')
        pred = np.array(list(map(int, raw_pred)))
        return pred

    def predict_proba(self, x):
        """
        Predict probabilities of class membership for each sample.
        
        Parameters
        ----------
        x: array-like of shape (n_samples, n_features)
            Data to predict from.

        Returns
        -------
        pred_proba : array-like
            Predicted probabilities of class membership for each sample.
        """
        raw_pred_proba = self._pred(x, pred_type='response')
        pred_proba = np.squeeze(np.array(raw_pred_proba))
        return pred_proba


class GLMNetRegressionCV(GLMNetCV, RegressorMixin):
    """
    Base class for regression models.
    """
    def __init__(self,
                 sample_weight=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.reg_type = None
        self.sample_weight = sample_weight

    def predict(self, x):
        """
        Predict values for each sample.
        
        Parameters
        ----------
        x: array-like of shape (n_samples, n_features)
            Data to predict from.

        Returns
        -------
        pred : array-like
            Predicted values for each sample
        """
        raw_pred = self._pred(x)
        pred = np.array(raw_pred).T[0]
        return pred


class GLMNetLinearRegressionCV(GLMNetRegressionCV):
    """
    Regularized linear regression tuned by cross-validation.
    
    Parameters
    ----------
    lower: float, default=None
        Lower bound on coefficients. E.g. lower=0 is non-negative regression
    upper: float, default=None
        Upper bound on coefficients
    loss_metric: str, default='mse'
        Which loss metric to minimize.
    alpha: float, default=0.5
        Balance of L1 and L2 regularization
    cv: int, cross-validation generator or an iterable, optional, default=None
        Determines the cross-validation splitting strategy. See documentation
        for sklearn.model_selection.check_cv. Cross-validation strategies with
        overlapping folds are not supported.


    See Also
    --------
    https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html#lin
    """

    def __init__(self,
                 lower=None,
                 upper=None,
                 loss_metric='mse',
                 alpha=0.5,
                 cv=None):
        super().__init__(lower=lower,
                         upper=upper,
                         loss_metric=loss_metric,
                         alpha=alpha,
                         cv=cv)
        self.reg_type = 'linear'


class GLMNetLogisticRegressionCV(GLMNetClassifierCV):
    """
    Regularized logistic regression tuned by cross-validation.
    
    For binary classification tasks only.
    
    Parameters
    ----------
    lower: float, default=None
        Lower bound on coefficients. E.g. lower=0 is non-negative regression
    upper: float, default=None
        Upper bound on coefficients
    loss_metric: str, default='mse'
        Which loss metric to minimize.
    alpha: float, default=0.5
        Balance of L1 and L2 regularization
    cv: int, cross-validation generator or an iterable, optional, default=None
        Determines the cross-validation splitting strategy. See documentation
        for sklearn.model_selection.check_cv. Cross-validation strategies with
        overlapping folds are not supported.
    class_weight: dict or ‘balanced’, default=None
        Weights associated with classes. As in sklearn.

    See Also
    --------
    https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html#log
    """
    def __init__(self,
                 lower=None,
                 upper=None,
                 loss_metric='mse',
                 alpha=0.5,
                 cv=None,
                 class_weight=None):
        super().__init__(lower=lower,
                         upper=upper,
                         loss_metric=loss_metric,
                         alpha=alpha,
                         cv=cv,
                         class_weight=class_weight)
        self.reg_type = 'logistic'


class GLMNetMultinomialRegressionCV(GLMNetClassifierCV):
    """
    Regularized multinomial regression tuned by cross-validation.
    
    For multi-class classification tasks.
    
    Parameters
    ----------
    lower: float, default=None
        Lower bound on coefficients. E.g. lower=0 is non-negative regression
    upper: float, default=None
        Upper bound on coefficients
    loss_metric: str, default='mse'
        Which loss metric to minimize.
    alpha: float, default=0.5
        Balance of L1 and L2 regularization
    cv: int, cross-validation generator or an iterable, optional, default=None
        Determines the cross-validation splitting strategy. See documentation
        for sklearn.model_selection.check_cv. Cross-validation strategies with
        overlapping folds are not supported.
    class_weight: dict or ‘balanced’, default=None
        Weights associated with classes. As in sklearn.

    See Also
    --------
    https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html#log
    """
    def __init__(self,
                 lower=None,
                 upper=None,
                 loss_metric='mse',
                 alpha=0.5,
                 cv=None,
                 class_weight=None):
        super().__init__(lower=lower,
                         upper=upper,
                         loss_metric=loss_metric,
                         alpha=alpha,
                         cv=cv,
                         class_weight=class_weight)
        self.reg_type = 'multinomial'


def get_coeffs(cvfit, lmda='min'):
    """
    Retrieve coefficients, including intercept(s), from a fitted glmnet model.
    
    Parameters
    ----------
    cvfit: rpy2 GLMNet model object
        Fit model
    lmda: {'min', '1se', or float}, default='min'
        Which regularization strength to retrieve coefficients for.
        'min' gives the regularization with minimum loss.
        '1se' gives the strongest regularization within 1 std. dev. of the min.
        A float specifies the actual value of the regularization parameter. 

    Returns
    -------
    intercept: array-like
        Intercept(s) of the model
    coeffs : array-like
        Coefficients of the model

    """
    if not isinstance(lmda, numbers.Number):
        if isinstance(lmda, str):
            if lmda not in ['min', '1se']:
                raise ValueError(f"{lmda} not an accepted lmda; try 'min', '1se', or a number")
            else:
                lmda = get_attr_from_glmnet_fit(cvfit, f'lambda.{lmda}')[0]
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
               cv=None,
               class_weight=None,
               sample_weight=None):
    """
    Fit generalized linear model in glmnet, tuning with cross-validation

    Parameters
    ----------
    x: array-like of shape (n_samples, n_features)
        Training data
    y: array-like of shape (n_samples,)
        Target values
    reg_type: {'linear', 'logistic', 'multinomial'}
        Type of regression model.
    lower: float, default=None
        Lower bound on coefficients. E.g. lower=0 is non-negative regression
    upper: float, default=None
        Upper bound on coefficients
    loss_metric: str
        Which loss metric to minimize. Supported options and their meaning may
        vary by model.
    alpha: float, default=0.5
        Balance of L1 and L2 regularization
    class_weight: dict or ‘balanced’, default=None
        Weights associated with classes (classification only). As in sklearn.
    sample_weight: array-like of shape (n_samples,), default=None
        Individual weights for each sample

    Returns
    -------
    fit: rpy2 GLMNet model object
        Fit model

    See Also
    --------
    https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html#qs
    """
    lower = float('-inf') if lower is None else lower
    upper = float('inf') if upper is None else upper
    foldid = make_foldid(cv, reg_type, x, y)
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
                           #nfolds=nfolds,
                           foldid=foldid,
                           **{'type.measure': loss_metric})
    return fit


def make_foldid(cv, reg_type, x, y):
    classifier = reg_type in CLASSIFICATION_REG_TYPES
    checked_cv = check_cv(cv=cv,
                          y=y if classifier else None,
                          classifier=classifier)
    fg = checked_cv.split(x, y) if classifier else checked_cv.split(x)
    fold_idx = 1
    sample_to_fold = {}
    for train, test in fg:
        for sample in test:
            sample_to_fold[sample] = fold_idx
        fold_idx += 1
    return np.array(list(zip(*sorted(sample_to_fold.items())))[1])


def get_attr_from_glmnet_fit(fit, attr):
    """
    Retrieve a particular attribute from a glmnet model object.
    
    Parameters
    ----------
    fit: rpy2 GLMNet model object
        Fit model
    attr: str
        Attribute of model to retrieve

    Returns
    -------
    values: array-like
        Values of specified attribute
    """
    names = list(fit.names)
    if attr not in names:
        raise ValueError("{} not a field of glmnet fit object".format(attr))
    values = np.array(fit[names.index(attr)])
    return values


