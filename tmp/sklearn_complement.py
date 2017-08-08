# -*- coding: utf-8 -*-
"""
Created on Thu Aug 03 11:35:26 2017

@author: s.trebeschi
"""
import numpy as np
import sklearn

from sklearn.base import BaseEstimator 
from sklearn.base import TransformerMixin
from sklearn.base import MetaEstimatorMixin
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection.base import SelectorMixin
from sklearn.feature_selection.from_model import _get_feature_importances
from sklearn.utils.metaestimators import if_delegate_has_method

###############################################################################

class IdentityTransformer(BaseEstimator, TransformerMixin):
    """
    Identity transformer returns X
    """
    def __init__(self):
        pass
    
    def fit(self, input_array, y=None):
        return self
    
    def transform(self, input_array, y=None):
        return input_array * 1

###############################################################################

def _calculate_k(importances, k):
    """Interpret the k value"""

    if isinstance(k, int):
        if k >= 1:
            return k

    if isinstance(k, float):
        if (k > 0) and (k < 1):
            return max(int(np.floor(len(importances)*k)), 1)

    raise ValueError("Unknown k value type")
    
    
class SelectBestKFromModel(BaseEstimator, SelectorMixin, MetaEstimatorMixin):
    """Meta-transformer for selecting best k features based on importance weights.

    Parameters
    ----------
    estimator : object
        The base estimator from which the transformer is built.
        This can be both a fitted (if ``prefit`` is set to True)
        or a non-fitted estimator. The estimator must have either a
        ``feature_importances_`` or ``coef_`` attribute after fitting.
    k : int or float
        The number of features to be selected from the estimator. If int,
        the best k features are selected. If float in the interval (0, 1), 
        the best floor(k * num_features) are selected. 
    prefit : bool, default False
        Whether a prefit model is expected to be passed into the constructor
        directly or not. If True, ``transform`` must be called directly
        and SelectFromModel cannot be used with ``cross_val_score``,
        ``GridSearchCV`` and similar utilities that clone the estimator.
        Otherwise train the model using ``fit`` and then ``transform`` to do
        feature selection.
    Attributes
    ----------
    estimator_ : an estimator, or a string
        The base estimator from which the transformer is built.
        This is stored only when a non-fitted estimator is passed to the
        ``SelectFromModel``, i.e when prefit is False.
        If string it will used the name of the classifier in sklearn
    k_ : int or float
        The k value used for feature selection.
    """
    
    def __init__(self, estimator, k=10, prefit=False, norm_order=1):
        
        if isinstance(estimator, str):
            estimator = eval(estimator)
            
        self.estimator = estimator
        self.k = k
        self.prefit = prefit
        self.norm_order = norm_order

    def _get_support_mask(self):

        if self.prefit:
            estimator = self.estimator
        elif hasattr(self, 'estimator_'):
            estimator = self.estimator_
        else:
            raise ValueError(
                'Either fit SelectBestKFromModel before transform or set "prefit='
                'True" and pass a fitted estimator to the constructor.')
        scores = _get_feature_importances(estimator)
        k = _calculate_k(scores, self.k)
        support_mask = np.zeros(len(scores), dtype='bool')
        support_mask[np.argsort(scores)[-k:]] = True
        return support_mask


    def fit(self, X, y=None, **fit_params):
        """Fit the SelectFromModel meta-transformer.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values (integers that correspond to classes in
            classification, real numbers in regression).
        **fit_params : Other estimator specific parameters
        Returns
        -------
        self : object
            Returns self.
        """
 
        if isinstance(self.estimator, str):
            self.estimator = eval(self.estimator)
           
        if self.prefit:
            raise NotFittedError(
                "Since 'prefit=True', call transform directly")
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y, **fit_params)
        return self

    @property
    def k_(self):
        scores = _get_feature_importances(self.estimator_)
        return _calculate_k(scores, self.k)

    @if_delegate_has_method('estimator')
    def partial_fit(self, X, y=None, **fit_params):
        """Fit the SelectFromModel meta-transformer only once.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values (integers that correspond to classes in
            classification, real numbers in regression).
        **fit_params : Other estimator specific parameters
        Returns
        -------
        self : object
            Returns self.
        """
        if self.prefit:
            raise NotFittedError(
                "Since 'prefit=True', call transform directly")
        if not hasattr(self, "estimator_"):
            self.estimator_ = clone(self.estimator)
        self.estimator_.partial_fit(X, y, **fit_params)
        return self
    
###############################################################################
    
class KMeansEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self, n_components=10):
        self.n_components = n_components

    def fit(self, X, y=None, **fit_params):
        self.estimator = sklearn.cluster.KMeans(n_clusters=self.n_components)
        self.estimator.fit(X)
        self.encoder = sklearn.preprocessing.LabelBinarizer()
        self.encoder.fit(self.estimator.predict(X))
        
    def transform(self, X):
        return self.encoder.transform(self.estimator.predict(X))
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

        
        